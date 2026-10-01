//! Durable application jobs; model and score code know nothing about HTTP.
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

// Bound preprocessing as well as model admission across all hosted accounts.
static ACTIVE_TASKS: AtomicUsize = AtomicUsize::new(0);
struct TaskPermit;
impl Drop for TaskPermit {
    fn drop(&mut self) {
        ACTIVE_TASKS.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct Tasks {
    root: PathBuf,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
}
struct Job {
    state: Mutex<Value>,
    cancelled: AtomicBool,
}
#[derive(Clone)]
pub struct Progress {
    sid: String,
    owner: Arc<Tasks>,
    job: Arc<Job>,
}
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
fn persist(path: &Path, state: &Value) -> Result<(), String> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("Invalid job path")?)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut file, state).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
    file.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}
impl Tasks {
    pub fn new(root: PathBuf) -> Result<Arc<Self>, String> {
        let tasks = Arc::new(Self {
            root,
            jobs: Mutex::new(HashMap::new()),
        });
        for entry in fs::read_dir(&tasks.root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let sid = entry.file_name().to_string_lossy().to_string();
            if crate::view::directory(&tasks.root, &sid).is_err() {
                continue;
            }
            let path = entry.path().join("job.json");
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(mut state) = serde_json::from_slice::<Value>(&bytes) else {
                continue;
            };
            if matches!(state["status"].as_str(), Some("queued" | "running")) {
                state["status"] = json!("interrupted");
                state["message"] = json!("应用上次退出，已保存的结果仍保留，可继续处理");
                state["cancellable"] = json!(false);
                persist(&path, &state)?;
            }
            tasks.jobs.lock().map_err(|_| "Job lock failed")?.insert(
                sid,
                Arc::new(Job {
                    state: Mutex::new(state),
                    cancelled: AtomicBool::new(false),
                }),
            );
        }
        Ok(tasks)
    }
    pub fn snapshot(&self, sid: &str) -> Option<Value> {
        self.jobs
            .lock()
            .ok()?
            .get(sid)?
            .state
            .lock()
            .ok()
            .map(|v| v.clone())
    }
    pub fn snapshots(&self) -> Vec<(String, Value)> {
        let Ok(jobs) = self.jobs.lock() else {
            return Vec::new();
        };
        jobs.iter()
            .filter_map(|(sid, job)| {
                job.state
                    .lock()
                    .ok()
                    .map(|state| (sid.clone(), state.clone()))
            })
            .collect()
    }
    pub fn idle(&self, sid: &str) -> Result<(), String> {
        if self
            .snapshot(sid)
            .is_some_and(|j| matches!(j["status"].as_str(), Some("queued" | "running")))
        {
            Err("这个项目正在处理，请等待当前步骤结束".into())
        } else {
            Ok(())
        }
    }
    pub fn clear(&self, sid: &str) -> Result<(), String> {
        self.idle(sid)?;
        self.jobs.lock().map_err(|_| "Job lock failed")?.remove(sid);
        let path = self.root.join(sid).join("job.json");
        if path.exists() {
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn submit(
        self: &Arc<Self>,
        sid: String,
        label: &str,
        action: &str,
        operation: impl FnOnce(Progress) -> Result<(), String> + Send + 'static,
    ) -> Result<Value, String> {
        crate::view::directory(&self.root, &sid)?;
        let mut jobs = self.jobs.lock().map_err(|_| "Job lock failed")?;
        if jobs.get(&sid).is_some_and(|j| {
            j.state
                .lock()
                .is_ok_and(|s| matches!(s["status"].as_str(), Some("queued" | "running")))
        }) {
            return Err("这个项目正在处理".into());
        }
        let active = jobs
            .values()
            .filter(|job| {
                job.state.lock().is_ok_and(|state| {
                    matches!(state["status"].as_str(), Some("queued" | "running"))
                })
            })
            .count();
        if active >= 4 {
            return Err("已有4个任务正在处理，请等待完成或停止部分任务".into());
        }
        ACTIVE_TASKS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n < 32 {
                    Some(n + 1)
                } else {
                    None
                }
            })
            .map_err(|_| "服务当前任务已满，请稍后重试")?;
        let permit = TaskPermit;
        let state = json!({"status":"queued","message":label,"done":0,"total":0,"error":null,"cancellable":true,"action":action,"created":now()});
        persist(&self.root.join(&sid).join("job.json"), &state)?;
        let job = Arc::new(Job {
            state: Mutex::new(state.clone()),
            cancelled: AtomicBool::new(false),
        });
        jobs.insert(sid.clone(), job.clone());
        drop(jobs);
        let progress = Progress {
            sid: sid.clone(),
            owner: self.clone(),
            job,
        };
        std::thread::spawn(move || {
            let _permit = permit;
            if progress.cancelled() {
                return;
            }
            if progress.update(json!({"status":"running"})).is_err() {
                return;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                progress.checkpoint()?;
                operation(progress.clone())
            }))
            .unwrap_or_else(|_| Err("处理过程异常退出，原有结果仍保留".into()));
            let final_state = if progress.cancelled() {
                json!({"status":"cancelled","error":null,"message":"任务已停止，原有结果与已完成阶段保留"})
            } else {
                match result {
                    Ok(()) => json!({"status":"complete","message":"处理完成"}),
                    Err(error) => json!({"status":"failed","error":error}),
                }
            };
            let _ = progress.update(final_state);
            let _ = progress.update(json!({"cancellable":false,"finished":now()}));
        });
        Ok(json!({"id":sid,"job":state}))
    }
    pub fn cancel(&self, sid: &str) -> Result<Value, String> {
        let jobs = self.jobs.lock().map_err(|_| "Job lock failed")?;
        let Some(job) = jobs.get(sid) else {
            return Ok(json!({"id":sid,"job":null}));
        };
        let mut state = job.state.lock().map_err(|_| "Job state failed")?;
        if matches!(state["status"].as_str(), Some("queued" | "running")) {
            job.cancelled.store(true, Ordering::SeqCst);
            if state["status"] == "queued" {
                state["status"] = json!("cancelled");
                state["cancellable"] = json!(false);
            }
            state["message"] = json!("正在停止，已保存的结果保留");

            persist(&self.root.join(sid).join("job.json"), &state)?;
        }
        Ok(json!({"id":sid,"job":state.clone()}))
    }
}
impl Progress {
    pub fn cancelled(&self) -> bool {
        self.job.cancelled.load(Ordering::SeqCst)
    }
    pub fn checkpoint(&self) -> Result<(), String> {
        if self.cancelled() {
            Err("任务已取消".into())
        } else {
            Ok(())
        }
    }
    pub fn update(&self, fields: Value) -> Result<(), String> {
        let jobs = self.owner.jobs.lock().map_err(|_| "Job lock failed")?;
        if !jobs
            .get(&self.sid)
            .is_some_and(|current| Arc::ptr_eq(current, &self.job))
        {
            return Ok(());
        }
        let mut state = self.job.state.lock().map_err(|_| "Job state failed")?;
        for (key, value) in fields.as_object().ok_or("Invalid progress")? {
            state[key] = value.clone();
        }
        persist(&self.owner.root.join(&self.sid).join("job.json"), &state)
    }
}
