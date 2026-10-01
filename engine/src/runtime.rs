//! One model process, bounded admission and independent request cancellation.
//! llama.cpp owns continuous batching and the KV cache inside the process.
use crate::{
    device_selection::{self, MemoryBudget, Preference, RuntimeCapabilities, Selection},
    llama::{self, Config, Engine},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, String>;
pub struct Options {
    pub executable: PathBuf,
    pub accelerator: Option<PathBuf>,
    pub model: PathBuf,
    pub projector: PathBuf,
    pub log: PathBuf,
    pub capabilities: RuntimeCapabilities,
    pub slots: usize,
    pub context: usize,
}
struct State {
    engine: Option<Arc<Engine>>,
    selection: Value,
}
#[derive(Default)]
struct Queue {
    next: u64,
    waiting: VecDeque<u64>,
    active: usize,
    completed: u64,
}
pub struct Runtime {
    options: Options,
    state: Mutex<State>,
    queue: Mutex<Queue>,
    changed: Condvar,
    stopping: AtomicBool,
}
struct Permit<'a>(&'a Runtime);
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        if let Ok(mut queue) = self.0.queue.lock() {
            queue.active -= 1;
            queue.completed += 1;
            self.0.changed.notify_all();
        }
    }
}
pub struct Session<'a> {
    engine: Arc<Engine>,
    cancelled: &'a dyn Fn() -> bool,
}
impl Session<'_> {
    pub fn generate(
        &mut self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<llama::Generation> {
        self.engine
            .generate_cancellable(messages, max_tokens, schema, grammar, self.cancelled)
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
            queue: Mutex::new(Queue::default()),
            changed: Condvar::new(),
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
        self.queue.lock().map(|q| json!({"active":q.active,"queued":q.waiting.len(),"completed":q.completed,"slots":self.options.slots})).unwrap_or(Value::Null)
    }
    fn admit(&self, cancelled: &dyn Fn() -> bool) -> Result<Permit<'_>> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| "Inference queue unavailable")?;
        if queue.waiting.len() >= 128 {
            return Err("Inference queue is full; retry later".into());
        }
        let ticket = queue.next;
        queue.next += 1;
        queue.waiting.push_back(ticket);
        let deadline = Instant::now() + Duration::from_secs(1800);
        loop {
            if cancelled() || Instant::now() >= deadline {
                queue.waiting.retain(|t| *t != ticket);
                self.changed.notify_all();
                return Err(if cancelled() {
                    "Inference cancelled"
                } else {
                    "Inference queue timeout"
                }
                .into());
            }
            if queue.active < self.options.slots && queue.waiting.front() == Some(&ticket) {
                queue.waiting.pop_front();
                queue.active += 1;
                self.changed.notify_all();
                return Ok(Permit(self));
            }
            queue = self
                .changed
                .wait_timeout(queue, Duration::from_millis(100))
                .map_err(|_| "Inference queue unavailable")?
                .0;
        }
    }
    pub fn run<T>(
        &self,
        cancelled: &dyn Fn() -> bool,
        report: &dyn Fn(&str) -> Result<()>,
        operation: impl FnOnce(&mut Session<'_>) -> Result<T>,
    ) -> Result<T> {
        let is_cancelled = || cancelled() || self.stopping.load(Ordering::Relaxed);
        let cancelled: &dyn Fn() -> bool = &is_cancelled;
        let _permit = self.admit(cancelled)?;
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
                let budget = MemoryBudget::from_files(
                    &self.options.model,
                    &self.options.projector,
                    self.options.context,
                    self.options.slots,
                )?;
                let mut capabilities = self.options.capabilities.clone();
                if self.options.accelerator.is_some() {
                    capabilities.cuda = Some(true);
                }
                let mut selected = device_selection::choose_runtime(
                    self.options
                        .accelerator
                        .as_ref()
                        .unwrap_or(&self.options.executable),
                    &capabilities,
                    &budget,
                    Preference::Auto,
                    cancelled,
                )?;
                let config = |selected: &Selection| Config {
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
                let process = match Engine::start_cancellable(config(&selected), cancelled) {
                    Ok(engine) => engine,
                    Err(original) => {
                        if cancelled() {
                            return Err("Inference cancelled".into());
                        }
                        selected = device_selection::cpu_fallback_after_start_failure(
                            Preference::Auto,
                            &self.options.capabilities,
                            &selected,
                        )
                        .map_err(|_| original)?;
                        report(&selected.reason)?;
                        Engine::start_cancellable(config(&selected), cancelled)?
                    }
                };
                state.selection = json!({"device":format!("{:?}",selected.device).to_lowercase(),"selector":selected.device_selector});
                state.engine = Some(Arc::new(process));
            }
            state.engine.as_ref().unwrap().clone()
        };
        operation(&mut Session { engine, cancelled })
    }
    pub fn shutdown(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        self.changed.notify_all();
        // Loading observes the same flag; taking its lock waits for cleanup.
        if let Ok(mut state) = self.state.lock() {
            if let Some(engine) = state.engine.take() {
                engine.cancellation_handle().stop();
            }
        }
    }
    pub fn unload(&self) -> Result<()> {
        let queue = self
            .queue
            .lock()
            .map_err(|_| "Inference queue unavailable")?;
        if queue.active > 0 || !queue.waiting.is_empty() {
            return Err("模型正在使用，请先停止任务".into());
        }
        self.state
            .lock()
            .map_err(|_| "Model state unavailable")?
            .engine = None;
        Ok(())
    }
}
