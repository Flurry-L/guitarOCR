//! Native ownership of a llama-server sidecar. No Python, shell, or installer.
//! Image normalization and musical interpretation belong to their own stages.
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    fs::File,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
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

#[derive(Clone)]
pub struct Config {
    pub executable: PathBuf,
    pub model: PathBuf,
    pub projector: PathBuf,
    pub adapter: Option<PathBuf>,
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
        for path in [&self.executable, &self.model, &self.projector]
            .into_iter()
            .chain(self.adapter.iter())
        {
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
        let mut args = vec![
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
        ];
        if let Some(adapter) = &self.adapter {
            args.extend([
                "--lora".into(),
                adapter.to_string_lossy().into_owned(),
                "--lora-init-without-apply".into(),
            ]);
        }
        args
    }
}

// Deliberately no Debug: the ephemeral key must not appear in error/debug logs.
pub struct Engine {
    child: Arc<Mutex<Child>>,
    client: Client,
    endpoint: String,
    key: String,
    request_timeout: Duration,
    requests: Arc<RequestPool>,
    has_adapter: bool,
    async_client: reqwest::Client,
    runtime: Option<tokio::runtime::Runtime>,
}
impl Engine {
    pub fn has_adapter(&self) -> bool {
        self.has_adapter
    }
    pub fn stats(&self) -> Value {
        self.requests.stats()
    }
    pub fn start_cancellable(config: Config, cancelled: &dyn Fn() -> bool) -> Result<Self, String> {
        config.validate()?;
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
            requests: Arc::new(RequestPool::new(config.parallel)),
            has_adapter: config.adapter.is_some(),
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

    pub fn generate_with_adapter(
        &self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        adapted: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Generation, String> {
        self.generate_batch_with_adapters(vec![(messages, schema, adapted)], max_tokens, cancelled)?
            .pop()
            .ok_or("Empty model response".into())
    }
    pub fn generate_batch_with_adapters(
        &self,
        messages: Vec<(Value, Option<Value>, bool)>,
        max_tokens: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Generation>, String> {
        let mut ordered: Vec<Option<Generation>> = (0..messages.len()).map(|_| None).collect();
        self.generate_batch_with_adapters_each(
            &mut messages.into_iter().map(Ok),
            max_tokens,
            cancelled,
            &mut |index, output| {
                ordered[index] = Some(output);
                Ok(())
            },
        )?;
        ordered
            .into_iter()
            .map(|v| v.ok_or("Missing model response".into()))
            .collect()
    }
    /// Prepare only the current window and report each result as it finishes.
    /// Indices refer to input order; callbacks execute on the calling thread.
    pub fn generate_batch_with_adapters_each(
        &self,
        messages: &mut dyn Iterator<Item = Result<(Value, Option<Value>, bool), String>>,
        max_tokens: usize,
        cancelled: &dyn Fn() -> bool,
        on_complete: &mut dyn FnMut(usize, Generation) -> Result<(), String>,
    ) -> Result<(), String> {
        if !self.alive() {
            return Err("llama-server is no longer running".into());
        }
        let mut requests = messages.map(|message| {
            let (messages, schema, adapted) = message?;
            let mut body = generation_body(messages, max_tokens, schema)?;
            select_adapter(&mut body, adapted, self.has_adapter);
            Ok(self
                .async_client
                .post(format!("{}/v1/chat/completions", self.endpoint))
                .bearer_auth(&self.key)
                .timeout(self.request_timeout)
                .json(&body))
        });
        self.runtime
            .as_ref()
            .unwrap()
            .block_on(self.requests.generate(
                &mut requests,
                self.request_timeout,
                cancelled,
                on_complete,
            ))
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

/// All callers share these request slots. Tokio's semaphore serves waiters FIFO;
/// each caller submits at most one window before other callers can take a turn.
struct RequestPool {
    slots: usize,
    permits: tokio::sync::Semaphore,
    queued: AtomicUsize,
    completed: AtomicU64,
}
struct Waiting<'a>(&'a AtomicUsize);
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}
impl RequestPool {
    fn new(slots: usize) -> Self {
        Self {
            slots,
            permits: tokio::sync::Semaphore::new(slots),
            queued: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
        }
    }
    fn stats(&self) -> Value {
        json!({
            "active": self.slots - self.permits.available_permits(),
            "queued": self.queued.load(Ordering::Relaxed),
            "completed": self.completed.load(Ordering::Relaxed),
            "slots": self.slots,
        })
    }
    async fn generate(
        self: &Arc<Self>,
        requests: &mut dyn Iterator<Item = Result<reqwest::RequestBuilder, String>>,
        queue_timeout: Duration,
        cancelled: &dyn Fn() -> bool,
        on_complete: &mut dyn FnMut(usize, Generation) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut source = requests.enumerate();
        let mut exhausted = false;
        let mut pending = tokio::task::JoinSet::new();
        let result = async {
            loop {
                if cancelled() {
                    return Err("Inference cancelled".into());
                }
                while !exhausted && pending.len() < self.slots {
                    let Some((index, request)) = source.next() else {
                        exhausted = true;
                        break;
                    };
                    let request = request?;
                    let pool = self.clone();
                    pending.spawn(async move {
                        pool.queued.fetch_add(1, Ordering::Relaxed);
                        let waiting = Waiting(&pool.queued);
                        let _permit = tokio::time::timeout(queue_timeout, pool.permits.acquire())
                            .await
                            .map_err(|_| "Inference queue timeout")?
                            .map_err(|_| "Inference queue closed")?;
                        drop(waiting);
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
                        let output = Engine::parse_generation(value)?;
                        pool.completed.fetch_add(1, Ordering::Relaxed);
                        Ok((index, output))
                    });
                }
                if pending.is_empty() {
                    return Ok(());
                }
                tokio::select! {
                    completed = pending.join_next() => {
                        let (index, output) = completed.ok_or("Missing model result")?
                            .map_err(|e| format!("Local model request task failed: {e}"))??;
                        on_complete(index, output)?;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                }
            }
        }
        .await;
        // Abort only this call's requests, and release their permits before
        // returning. Completed callbacks have already reached the caller.
        pending.shutdown().await;
        result
    }
}
fn select_adapter(body: &mut Value, adapted: bool, loaded: bool) {
    // Per-request selection avoids shared mutable adapter state and keeps base
    // generations unchanged even when adapted requests run in another slot.
    // In the server an empty array restores the global default; it does
    // not reliably disable a loaded adapter. Base calls require explicit zero.
    body["lora"] = if loaded {
        json!([{"id":0,"scale":if adapted {1.0} else {0.0}}])
    } else {
        json!([])
    };
}
fn generation_body(
    messages: Value,
    max_tokens: usize,
    schema: Option<Value>,
) -> Result<Value, String> {
    if !messages.is_array()
        || messages.as_array().is_some_and(Vec::is_empty)
        || !(1..=131072).contains(&max_tokens)
    {
        return Err("Messages must be non-empty and output tokens bounded".into());
    }
    let mut body = json!({"messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": false});
    if let Some(schema) = schema {
        body["response_format"] = json!({"type": "json_schema", "json_schema": {"name": "score_annotation", "schema": schema}});
    }
    Ok(body)
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        sync::atomic::AtomicBool,
    };

    struct Server {
        address: String,
        release: Arc<AtomicBool>,
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicUsize>,
        thread: Option<thread::JoinHandle<()>>,
    }
    impl Server {
        fn new() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let address = format!("http://{}", listener.local_addr().unwrap());
            let release = Arc::new(AtomicBool::new(false));
            let stop = Arc::new(AtomicBool::new(false));
            let peak = Arc::new(AtomicUsize::new(0));
            let active = Arc::new(AtomicUsize::new(0));
            let (released, stopped, maximum) = (release.clone(), stop.clone(), peak.clone());
            let thread = thread::spawn(move || {
                let mut handlers = Vec::new();
                while !stopped.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut socket, _)) => {
                            let (release, peak, active) =
                                (released.clone(), maximum.clone(), active.clone());
                            handlers.push(thread::spawn(move || {
                                // macOS accepted sockets inherit the listener's nonblocking mode.
                                socket.set_nonblocking(false).unwrap();
                                socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                                let mut reader = BufReader::new(socket.try_clone().unwrap());
                                let mut line = String::new();
                                reader.read_line(&mut line).unwrap();
                                let route = line.split_whitespace().nth(1).unwrap().to_owned();
                                loop {
                                    line.clear();
                                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                                        break;
                                    }
                                }
                                peak.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                                if route.ends_with("/hold") {
                                    let deadline = Instant::now() + Duration::from_secs(3);
                                    while !release.load(Ordering::SeqCst) && Instant::now() < deadline {
                                        thread::sleep(Duration::from_millis(1));
                                    }
                                }
                                let content = route.split('/').nth(1).unwrap();
                                let body = json!({"choices":[{"message":{"content":content}}],
                                    "usage":{"completion_tokens":1}}).to_string();
                                active.fetch_sub(1, Ordering::SeqCst);
                                let _ = write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
                            }));
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("Mock model listener: {error}"),
                    }
                }
                for handler in handlers {
                    handler.join().unwrap();
                }
            });
            Self {
                address,
                release,
                stop,
                peak,
                thread: Some(thread),
            }
        }
        fn requests(&self, names: &[&str]) -> Vec<Result<reqwest::RequestBuilder, String>> {
            let client = reqwest::Client::builder().no_proxy().build().unwrap();
            names
                .iter()
                .map(|name| {
                    Ok(client
                        .get(format!("{}/{name}", self.address))
                        .timeout(Duration::from_secs(4)))
                })
                .collect()
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            self.release.store(true, Ordering::SeqCst);
            self.stop.store(true, Ordering::SeqCst);
            self.thread.take().unwrap().join().unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rolling_requests_keep_indices_and_prepare_only_the_window() {
        let server = Server::new();
        let pool = Arc::new(RequestPool::new(2));
        let prepared = AtomicUsize::new(0);
        let mut requests = server
            .requests(&["0/hold", "1", "2"])
            .into_iter()
            .inspect(|_| {
                prepared.fetch_add(1, Ordering::SeqCst);
            });
        let mut order = Vec::new();
        pool.generate(
            &mut requests,
            Duration::from_secs(3),
            &|| false,
            &mut |index, output| {
                assert_eq!(output.content, index.to_string());
                if order.is_empty() {
                    assert_eq!(prepared.load(Ordering::SeqCst), 2);
                }
                order.push(index);
                if index == 2 {
                    server.release.store(true, Ordering::SeqCst);
                }
                Ok(())
            },
        )
        .await
        .unwrap();
        assert_eq!(order, vec![1, 2, 0]);
        assert_eq!(pool.stats()["completed"], 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_calls_share_slots_and_waiters_get_a_turn() {
        let server = Server::new();
        let pool = Arc::new(RequestPool::new(2));
        let order = Mutex::new(Vec::new());
        let mut a = server
            .requests(&["a0/hold", "a1/hold", "a2", "a3", "a4"])
            .into_iter();
        let mut b = server.requests(&["b0", "b1"]).into_iter();
        let mut save_a = |_: usize, output: Generation| {
            order.lock().unwrap().push(output.content);
            Ok(())
        };
        let mut save_b = |_: usize, output: Generation| {
            order.lock().unwrap().push(output.content);
            Ok(())
        };
        let first = pool.generate(&mut a, Duration::from_secs(3), &|| false, &mut save_a);
        let second = async {
            while pool.stats()["active"] != 2 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            pool.generate(&mut b, Duration::from_secs(3), &|| false, &mut save_b)
                .await
        };
        let release = async {
            // Occupied pool slots can still be establishing their HTTP connections.
            while pool.stats()["queued"].as_u64().unwrap() < 2
                || server.peak.load(Ordering::SeqCst) < 2
            {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            server.release.store(true, Ordering::SeqCst);
        };
        let (a, b, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(first, second, release)
        })
        .await
        .unwrap();
        a.unwrap();
        b.unwrap();
        assert_eq!(server.peak.load(Ordering::SeqCst), 2);
        let order = order.into_inner().unwrap();
        for name in ["b0", "b1"] {
            assert!(
                order.iter().position(|s| s == name).unwrap()
                    < order.iter().position(|s| s == "a4").unwrap()
            );
        }
        assert_eq!(
            pool.stats(),
            json!({"active":0,"queued":0,"completed":7,"slots":2})
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_keeps_completed_results_and_other_calls_running() {
        let server = Server::new();
        let pool = Arc::new(RequestPool::new(2));
        let cancelled = AtomicBool::new(false);
        let mut saved = Vec::new();
        let mut a = server
            .requests(&["saved", "cancelled/hold", "never-sent"])
            .into_iter();
        let mut b = server.requests(&["other-project"]).into_iter();
        let mut save_a = |_: usize, output: Generation| {
            saved.push(output.content);
            cancelled.store(true, Ordering::SeqCst);
            Ok(())
        };
        let mut other = None;
        let mut save_b = |_: usize, output: Generation| {
            other = Some(output.content);
            Ok(())
        };
        let is_cancelled = || cancelled.load(Ordering::SeqCst);
        let (a, b) = tokio::join!(
            pool.generate(&mut a, Duration::from_secs(3), &is_cancelled, &mut save_a),
            pool.generate(&mut b, Duration::from_secs(3), &|| false, &mut save_b),
        );
        assert_eq!(a.unwrap_err(), "Inference cancelled");
        b.unwrap();
        assert_eq!(saved, vec!["saved"]);
        assert_eq!(other.as_deref(), Some("other-project"));
        assert_eq!(pool.stats()["active"], 0);
        assert_eq!(pool.stats()["queued"], 0);
    }
}
