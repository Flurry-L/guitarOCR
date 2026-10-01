//! Native ownership of a pinned llama-server sidecar. No Python, shell, or installer.
//! Image normalization and musical interpretation belong to their own stages.
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    fs::File,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

/// Packaged inference must not inherit model-download, RPC, or plugin-loading
/// options from a developer shell. Device visibility masks remain respected by
/// both probing and startup; the selected CLI device is always explicit.
pub(crate) fn restrict_runtime_environment(command: &mut Command) {
    for (key, _) in std::env::vars_os() {
        if key
            .to_str()
            .is_some_and(|name| name.starts_with("LLAMA_ARG_"))
        {
            command.env_remove(key);
        }
    }
    for name in [
        "GGML_BACKEND_PATH",
        "MTMD_BACKEND_DEVICE",
        "LLAMA_API_KEY",
        "LLAMA_API_KEY_FILE",
    ] {
        command.env_remove(name);
    }
}

pub const PINNED_COMMIT: &str = "8019dc563b1ecbae6b161a70c3a1359f1b206c1e";

/// Check the packaged runtime without loading a model. Artifact hash validation
/// remains the installer's responsibility; a version string is not a signature.
pub fn verify_revision(executable: &Path) -> Result<(), String> {
    let mut command = Command::new(executable);
    restrict_runtime_environment(&mut command);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("Cannot probe llama-server: {e}"))?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("llama-server version probe failed or timed out".into());
            }
        }
    }
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let version = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() || !matches_pinned_revision(&version) {
        return Err("llama-server does not match the pinned native runtime revision".into());
    }
    Ok(())
}

fn matches_pinned_revision(version: &str) -> bool {
    // Git abbreviates the same pinned commit differently in shallow/full clones.
    // Parse one complete version record, never an arbitrary hex substring.
    let pattern =
        regex::Regex::new(r"^version: [^\r\n]*\(build [0-9]+, commit ([0-9a-fA-F]{7,40})\)$")
            .unwrap();
    let mut revisions = version
        .lines()
        .filter_map(|line| pattern.captures(line.trim()));
    let Some(found) = revisions.next() else {
        return false;
    };
    revisions.next().is_none() && PINNED_COMMIT.starts_with(&found[1].to_ascii_lowercase())
}

pub struct Config {
    pub executable: PathBuf,
    pub model: PathBuf,
    pub projector: PathBuf,
    pub log: PathBuf,
    pub device: Device,
    pub device_selector: Option<String>,
    pub threads: usize,
    pub parallel: usize,
    pub context_per_slot: usize,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        match self.device {
            Device::Cpu if self.device_selector.as_deref().is_none_or(|s| s == "none") => {}
            Device::Cuda | Device::Metal
                if self.device_selector.as_deref().is_some_and(|s| {
                    crate::device_selection::valid_device_selector(self.device, s)
                }) => {}
            _ => {
                return Err("Device selector must match the selected verified local backend".into())
            }
        }

        if !(1..=256).contains(&self.threads)
            || !(1..=4).contains(&self.parallel)
            || !(2048..=131072).contains(&self.context_per_slot)
        {
            return Err("Invalid llama-server threads, parallel slots, or context size".into());
        }
        if self.startup_timeout.is_zero() || self.request_timeout.is_zero() {
            return Err("llama-server timeouts must be positive".into());
        }
        for path in [&self.executable, &self.model, &self.projector] {
            if !path.is_file() {
                return Err(format!(
                    "Missing local model/runtime file: {}",
                    path.display()
                ));
            }
        }
        Ok(())
    }
    fn arguments(&self, port: u16) -> Vec<String> {
        vec![
            "--log-verbosity".into(),
            "3".into(),
            "--no-webui".into(),
            "--cors-origins".into(),
            format!("http://127.0.0.1:{port}"),
            "--no-cors-credentials".into(),
            "-m".into(),
            self.model.to_string_lossy().into_owned(),
            "--mmproj".into(),
            self.projector.to_string_lossy().into_owned(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            "-c".into(),
            (self.context_per_slot * self.parallel).to_string(),
            "-np".into(),
            self.parallel.to_string(),
            "-t".into(),
            self.threads.to_string(),
            "--device".into(),
            self.device_selector.as_deref().unwrap_or("none").into(),
            "--mmproj-device".into(),
            self.device_selector.as_deref().unwrap_or("none").into(),
            "-ngl".into(),
            if matches!(self.device, Device::Cpu) {
                "0"
            } else {
                "99"
            }
            .into(),
        ]
    }
}

// Deliberately no Debug: the ephemeral key must not appear in error/debug logs.
pub struct Engine {
    child: Arc<Mutex<Child>>,
    client: Client,
    endpoint: String,
    key: String,
    request_timeout: Duration,
    parallel: usize,
    async_client: reqwest::Client,
    runtime: Option<tokio::runtime::Runtime>,
}
impl Engine {
    pub fn start(config: Config) -> Result<Self, String> {
        Self::start_cancellable(config, &|| false)
    }
    pub fn start_cancellable(config: Config, cancelled: &dyn Fn() -> bool) -> Result<Self, String> {
        config.validate()?;
        verify_revision(&config.executable)?;
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| e.to_string())?;
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes)
            .map_err(|e| format!("Cannot create temporary local authentication: {e}"))?;
        let key = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let socket = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
        let port = socket.local_addr().map_err(|e| e.to_string())?.port();
        let log = File::create(&config.log).map_err(|e| format!("Cannot open engine log: {e}"))?;
        let mut command = Command::new(&config.executable);
        restrict_runtime_environment(&mut command);
        command
            .args(config.arguments(port))
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
            .stderr(Stdio::from(log))
            .env_remove("LLAMA_ARG_DEVICE")
            .env_remove("LLAMA_ARG_RPC")
            .env_remove("MTMD_BACKEND_DEVICE")
            .env_remove("LLAMA_API_KEY_FILE")
            .env_remove("LLAMA_ARG_API_KEY_FILE")
            .env_remove("LLAMA_ARG_LOG_VERBOSITY")
            .env_remove("LLAMA_LOG_VERBOSITY")
            .env("LLAMA_API_KEY", &key)
            .env("GGML_CUDA_DISABLE_GRAPHS", "1");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        // llama-server cannot inherit a listening socket; a bind conflict fails
        // closed rather than reusing an existing unauthenticated service.
        drop(socket);
        let child = command
            .spawn()
            .map_err(|e| format!("Cannot start llama-server: {e}"))?;
        let engine = Self {
            child: Arc::new(Mutex::new(child)),
            client,
            endpoint: format!("http://127.0.0.1:{port}"),
            key,
            request_timeout: config.request_timeout,
            parallel: config.parallel,
            async_client: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(2))
                .build()
                .map_err(|e| e.to_string())?,
            runtime: Some(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?,
            ),
        };
        let deadline = Instant::now() + config.startup_timeout;
        while Instant::now() < deadline {
            if cancelled() {
                return Err("Model startup cancelled".into());
            }
            if let Some(status) = engine
                .child
                .lock()
                .map_err(|_| "Engine process lock failed")?
                .try_wait()
                .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "llama-server exited ({status}); see {}",
                    config.log.display()
                ));
            }
            // /health is public upstream; require an authenticated endpoint too.
            if engine
                .client
                .get(format!("{}/health", engine.endpoint))
                .timeout(Duration::from_secs(2))
                .send()
                .is_ok_and(|r| r.status().is_success())
                && engine
                    .client
                    .get(format!("{}/v1/models", engine.endpoint))
                    .bearer_auth(&engine.key)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .is_ok_and(|r| r.status().is_success())
            {
                return Ok(engine);
            }
            thread::sleep(Duration::from_millis(100));
        }
        Err(format!(
            "llama-server startup timed out; see {}",
            config.log.display()
        ))
    }

    /// Messages contain normalized image data URLs; cancellation drops only this
    /// HTTP request so concurrent generations keep their slots and KV cache.
    pub fn generate(
        &self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<Generation, String> {
        self.generate_cancellable(messages, max_tokens, schema, grammar, &|| false)
    }
    pub fn generate_cancellable(
        &self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Generation, String> {
        let body = generation_body(messages, max_tokens, schema, grammar)?;
        self.generate_bodies(vec![body], cancelled)?
            .pop()
            .ok_or("Empty model response".into())
    }
    pub fn generate_batch_cancellable(
        &self,
        messages: Vec<(Value, Option<Value>)>,
        max_tokens: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Generation>, String> {
        let bodies = messages
            .into_iter()
            .map(|(m, s)| generation_body(m, max_tokens, s, None))
            .collect::<Result<Vec<_>, _>>()?;
        self.generate_bodies(bodies, cancelled)
    }
    fn generate_bodies(
        &self,
        bodies: Vec<Value>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Generation>, String> {
        if !self.alive() {
            return Err("llama-server is no longer running".into());
        }
        self.runtime.as_ref().unwrap().block_on(async {
            let mut results = Vec::with_capacity(bodies.len());
            // The model's slots bound memory and continuous batching. Dropping
            // the task set on cancellation aborts every request in this batch.
            for batch in bodies.chunks(self.parallel) {
                let mut pending = tokio::task::JoinSet::new();
                let mut ordered: Vec<Option<Generation>> = (0..batch.len()).map(|_| None).collect();
                for (index, body) in batch.iter().enumerate() {
                    let request = self
                        .async_client
                        .post(format!("{}/v1/chat/completions", self.endpoint))
                        .bearer_auth(&self.key)
                        .timeout(self.request_timeout)
                        .json(body);
                    pending.spawn(async move {
                        let response = request.send().await.map_err(|e| {
                            format!("Local model request failed: {}", e.without_url())
                        })?;
                        if !response.status().is_success() {
                            return Err(format!("Local model returned HTTP {}", response.status()));
                        }
                        let value = response
                            .json()
                            .await
                            .map_err(|_| "Invalid local model response".to_string())?;
                        Ok((index, Self::parse_generation(value)?))
                    });
                }
                while !pending.is_empty() {
                    if cancelled() {
                        return Err("Inference cancelled".into());
                    }
                    tokio::select! {
                        completed = pending.join_next() => {
                            let (index, output) = completed.ok_or("Missing batch result")?
                                .map_err(|e| format!("Local model request task failed: {e}"))??;
                            ordered[index] = Some(output);
                        }
                        _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                    }
                }
                results.extend(
                    ordered
                        .into_iter()
                        .map(|v| v.ok_or("Missing batch response"))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            Ok(results)
        })
    }
    fn parse_generation(result: Value) -> Result<Generation, String> {
        let message = result
            .pointer("/choices/0/message")
            .ok_or("Missing model response choice")?;
        let content = match message.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) => String::new(),
            _ => return Err("Invalid model response content".into()),
        };
        let completion_tokens = result
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .ok_or("Missing completion token count")?;
        Ok(Generation {
            content,
            completion_tokens,
        })
    }
    /// The app task controller may stop the sidecar from another thread to
    /// interrupt a blocking completion. A cancelled engine must be restarted.
    pub fn alive(&self) -> bool {
        self.child
            .lock()
            .is_ok_and(|mut child| matches!(child.try_wait(), Ok(None)))
    }
    pub fn cancellation_handle(&self) -> StopHandle {
        StopHandle(self.child.clone())
    }
    pub fn stop(&mut self) {
        self.cancellation_handle().stop();
        self.key.clear();
    }
}
#[derive(Clone)]
pub struct StopHandle(Arc<Mutex<Child>>);
impl StopHandle {
    pub fn stop(&self) {
        if let Ok(mut child) = self.0.lock() {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.stop();
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

pub struct Generation {
    pub content: String,
    pub completion_tokens: u64,
}
fn generation_body(
    messages: Value,
    max_tokens: usize,
    schema: Option<Value>,
    grammar: Option<&str>,
) -> Result<Value, String> {
    if !messages.is_array()
        || messages.as_array().is_some_and(Vec::is_empty)
        || !(1..=131072).contains(&max_tokens)
    {
        return Err("Messages must be non-empty and output tokens bounded".into());
    }
    if schema.is_some() && grammar.is_some() {
        return Err("Choose JSON schema or musical grammar, not both".into());
    }
    let mut body = json!({"messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": false});
    if let Some(schema) = schema {
        body["response_format"] = json!({"type": "json_schema", "json_schema": {"name": "score_annotation", "schema": schema}});
    }
    if let Some(grammar) = grammar {
        body["grammar"] = Value::String(grammar.into());
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_revision_accepts_complete_git_abbreviations_only() {
        for length in [7, 9, 12, 40] {
            assert!(matches_pinned_revision(&format!(
                "version: 0.5.0-dev (build 1, commit {})\n",
                &PINNED_COMMIT[..length]
            )));
        }
        for value in [
            "8019dc",
            "8019dc0",
            "8019dc5dirty",
            "8019dc5-deadbeef",
            "8019dc563b1ecbae6b161a70c3a1359f1b206c1e0",
        ] {
            assert!(
                !matches_pinned_revision(&format!("version: 0.5.0-dev (build 1, commit {value})")),
                "{value}"
            );
        }
        assert!(!matches_pinned_revision("unrelated log commit 8019dc5)"));
        assert!(!matches_pinned_revision(
            "version: 0.5.0-dev (build 1, commit 8019dc5) trailing"
        ));
    }
    #[test]
    fn payload_matches_python_backend_contract() {
        let messages = json!([{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}]}]);
        let body =
            generation_body(messages.clone(), 512, Some(json!({"type":"object"})), None).unwrap();
        assert_eq!(body["messages"], messages);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["cache_prompt"], false);
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "score_annotation"
        );
        assert!(generation_body(messages, 512, Some(json!({})), Some("root ::= \"x\" ")).is_err());
        assert!(generation_body(json!([]), 512, None, None).is_err());
    }
    #[test]
    fn local_arguments_bound_resources_without_credentials_or_python() {
        let config = Config {
            executable: "llama-server".into(),
            model: "model.gguf".into(),
            projector: "vision.gguf".into(),
            log: "engine.log".into(),
            device: Device::Cpu,
            device_selector: None,
            threads: 2,
            parallel: 2,
            context_per_slot: 4096,
            startup_timeout: Duration::from_secs(60),
            request_timeout: Duration::from_secs(60),
        };
        let args = config.arguments(34567);
        assert!(args.windows(2).any(|a| a == ["-c", "8192"]));
        assert!(args.windows(2).any(|a| a == ["-ngl", "0"]));
        assert!(args.windows(2).any(|a| a == ["--device", "none"]));
        assert!(args.windows(2).any(|a| a == ["--mmproj-device", "none"]));
        assert!(args.windows(2).any(|a| a == ["--host", "127.0.0.1"]));
        assert!(!args
            .iter()
            .any(|a| a.contains("api-key") || a.contains("python")));
        let mut gpu = config;
        gpu.device = Device::Metal;
        gpu.device_selector = Some("MTL0".into());
        let args = gpu.arguments(34567);
        assert!(args.windows(2).any(|a| a == ["--device", "MTL0"]));
        assert!(args.windows(2).any(|a| a == ["--mmproj-device", "MTL0"]));
        assert!(args.windows(2).any(|a| a == ["-ngl", "99"]));
        gpu.device_selector = Some("RPC0".into());
        assert!(gpu.validate().unwrap_err().contains("Device selector"));
    }
}
