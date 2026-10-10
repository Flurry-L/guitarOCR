//! One model process, bounded admission and independent request cancellation.
//! llama.cpp owns continuous batching and the KV cache inside the process.
use crate::{
    device_selection::{self, MemoryBudget, RuntimeCapabilities},
    llama::{self, Config, Engine},
};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

type Result<T> = std::result::Result<T, String>;
pub struct Options {
    pub executable: PathBuf,
    pub accelerator: Option<PathBuf>,
    pub model: PathBuf,
    pub projector: PathBuf,
    pub adapter: Option<PathBuf>,
    pub adapter_tasks: Vec<String>,
    pub log: PathBuf,
    pub capabilities: RuntimeCapabilities,
    pub slots: usize,
    pub context: usize,
}
struct State {
    engine: Option<Arc<Engine>>,
    selection: Value,
}
pub struct Runtime {
    options: Options,
    state: Mutex<State>,
    stopping: AtomicBool,
}
pub struct Session<'a> {
    engine: Arc<Engine>,
    adapter_tasks: &'a [String],
    cancelled: &'a dyn Fn() -> bool,
    report: &'a dyn Fn(&str) -> Result<()>,
}
impl Session<'_> {
    pub fn adapter_enabled_for(&self, task: &str) -> bool {
        self.engine.has_adapter() && self.adapter_tasks.iter().any(|allowed| allowed == task)
    }
    pub fn generate_with_adapter(
        &mut self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        adapted: bool,
    ) -> Result<llama::Generation> {
        self.engine
            .generate_with_adapter(messages, max_tokens, schema, adapted, self.cancelled)
    }
    pub fn generate_batch_with_adapters(
        &mut self,
        messages: Vec<(Value, Option<Value>, bool)>,
        max_tokens: usize,
    ) -> Result<Vec<llama::Generation>> {
        self.engine
            .generate_batch_with_adapters(messages, max_tokens, self.cancelled)
    }
    pub fn generate_batch_with_adapters_each(
        &mut self,
        messages: &mut dyn Iterator<Item = Result<(Value, Option<Value>, bool)>>,
        max_tokens: usize,
        on_complete: &mut dyn FnMut(usize, llama::Generation) -> Result<()>,
    ) -> Result<()> {
        self.engine.generate_batch_with_adapters_each(
            messages,
            max_tokens,
            self.cancelled,
            on_complete,
        )
    }
    pub fn report(&self, message: &str) -> Result<()> {
        (self.report)(message)
    }
    pub fn generate(
        &mut self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
    ) -> Result<llama::Generation> {
        self.engine
            .generate_with_adapter(messages, max_tokens, schema, false, self.cancelled)
    }
}
impl Runtime {
    pub fn new(mut options: Options) -> Self {
        options.slots = options.slots.clamp(1, 4);
        Self {
            options,
            state: Mutex::new(State {
                engine: None,
                selection: json!({"device":"auto","selector":null}),
            }),
            stopping: AtomicBool::new(false),
        }
    }
    pub fn device(&self) -> Value {
        self.state
            .try_lock()
            .map(|s| s.selection.clone())
            .unwrap_or_else(|_| json!({"device":"auto","loading":true}))
    }
    pub fn stats(&self) -> Value {
        self.state
            .try_lock()
            .map(|state| {
                state.engine.as_ref().map_or_else(
                    || json!({"active":0,"queued":0,"completed":0,"slots":self.options.slots}),
                    |engine| engine.stats(),
                )
            })
            .unwrap_or(Value::Null)
    }
    pub fn run<T>(
        &self,
        cancelled: &dyn Fn() -> bool,
        report: &dyn Fn(&str) -> Result<()>,
        operation: impl FnOnce(&mut Session<'_>) -> Result<T>,
    ) -> Result<T> {
        let is_cancelled = || cancelled() || self.stopping.load(Ordering::Relaxed);
        let cancelled: &dyn Fn() -> bool = &is_cancelled;
        let engine = {
            let mut state = loop {
                if cancelled() {
                    return Err("Inference cancelled".into());
                }
                match self.state.try_lock() {
                    Ok(state) => break state,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        std::thread::sleep(Duration::from_millis(100))
                    }
                    Err(_) => return Err("Model state unavailable".into()),
                }
            };
            if state.engine.as_ref().is_none_or(|engine| !engine.alive()) {
                state.engine = None;
                report("正在加载识别模型")?;
                let mut budget = MemoryBudget::from_files(
                    &self.options.model,
                    &self.options.projector,
                    self.options.context,
                    self.options.slots,
                )?;
                if let Some(adapter) = &self.options.adapter {
                    let bytes = std::fs::metadata(adapter).map_err(|e| e.to_string())?.len();
                    budget.required_bytes = budget
                        .required_bytes
                        .checked_add(bytes + bytes / 4)
                        .ok_or("Adapter memory estimate overflow")?;
                }
                let mut capabilities = self.options.capabilities.clone();
                if self.options.accelerator.is_some() {
                    capabilities.cuda = true;
                }
                let selected = device_selection::choose_runtime(
                    self.options
                        .accelerator
                        .as_ref()
                        .unwrap_or(&self.options.executable),
                    &capabilities,
                    &budget,
                    cancelled,
                )?;
                let config = Config {
                    executable: if selected.device == llama::Device::Cuda {
                        self.options
                            .accelerator
                            .as_ref()
                            .unwrap_or(&self.options.executable)
                            .clone()
                    } else {
                        self.options.executable.clone()
                    },
                    model: self.options.model.clone(),
                    projector: self.options.projector.clone(),
                    adapter: self.options.adapter.clone(),
                    log: self.options.log.clone(),
                    device: selected.device,
                    device_selector: selected.device_selector.clone(),
                    threads: std::thread::available_parallelism()
                        .map(|n| n.get().min(8))
                        .unwrap_or(2),
                    parallel: self.options.slots,
                    context_per_slot: self.options.context,
                    startup_timeout: Duration::from_secs(600),
                    request_timeout: Duration::from_secs(1800),
                };
                report(&selected.reason)?;
                let process = Engine::start_cancellable(config, cancelled)?;
                state.selection = json!({"device":format!("{:?}",selected.device).to_lowercase(),"selector":selected.device_selector});
                state.engine = Some(Arc::new(process));
            }
            state.engine.as_ref().unwrap().clone()
        };
        operation(&mut Session {
            engine,
            adapter_tasks: &self.options.adapter_tasks,
            cancelled,
            report,
        })
    }
    pub fn shutdown(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        // Loading observes the same flag; taking its lock waits for cleanup.
        if let Ok(mut state) = self.state.lock() {
            if let Some(engine) = state.engine.take() {
                engine.cancellation_handle().stop();
            }
        }
    }
}
