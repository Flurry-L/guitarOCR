//! Shared inference models, independent of device and application version.
use crate::tasks::Progress;
use fs2::FileExt;
use serde_json::Value;
use std::{
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

pub struct Models {
    pub root: PathBuf,
    pub runtime_root: PathBuf,
    pub policy: Value,
    pub capabilities: Value,
    catalog: Value,
    verified: Mutex<Option<Vec<(u64, std::time::SystemTime)>>>,
}
impl Models {
    pub fn new(data_root: &Path, override_root: Option<PathBuf>) -> Result<Self, String> {
        // Compile-time inclusion keeps model metadata sourced from the existing
        // research artifacts without shipping Python/tokenizers/weights in app.
        let catalog: Value = serde_json::from_str(include_str!("../../weights/distribution.json"))
            .map_err(|e| e.to_string())?;
        let generation = catalog["generation"]
            .as_str()
            .ok_or("Missing model generation")?;
        if generation.is_empty() || generation.contains(['/', '\\', ':']) || generation == ".." {
            return Err("Unsafe model generation".into());
        }
        let root = if let Some(base) = override_root {
            // Keep the Python research cache environment variable's base-dir
            // meaning; an explicit generation directory is also accepted.
            if base.file_name().and_then(|n| n.to_str()) == Some(generation) {
                base
            } else {
                base.join(generation)
            }
        } else {
            let current = data_root.join("models").join(generation);
            let legacy = data_root.join("backend/tools/model-cache").join(generation);
            if !current.exists() && legacy.is_dir() {
                legacy
            } else {
                current
            }
        };
        Ok(Self {
            root,
            runtime_root: data_root.join("runtimes"),
            policy: serde_json::from_str(include_str!(
                "../../weights/score_ocr/merged/score_image_policy.json"
            ))
            .map_err(|e| e.to_string())?,
            capabilities: serde_json::from_str(include_str!(
                "../../weights/score_ocr/merged/capabilities.json"
            ))
            .map_err(|e| e.to_string())?,
            catalog,
            verified: Mutex::new(None),
        })
    }
    fn files(&self) -> Vec<&Value> {
        ["auxiliary", "llamacpp"]
            .iter()
            .flat_map(|key| self.catalog["files"][*key].as_array().into_iter().flatten())
            .collect()
    }
    fn cache_stamp(&self) -> Option<Vec<(u64, std::time::SystemTime)>> {
        self.files()
            .iter()
            .map(|file| {
                let metadata = self.root.join(file["path"].as_str()?).metadata().ok()?;
                if Some(metadata.len()) != file["bytes"].as_u64() {
                    return None;
                }
                Some((metadata.len(), metadata.modified().ok()?))
            })
            .collect()
    }
    pub fn cached(&self) -> bool {
        self.cache_stamp().is_some()
    }
    pub fn ready(&self) -> bool {
        let Some(stamp) = self.cache_stamp() else {
            return false;
        };
        self.verified
            .try_lock()
            .is_ok_and(|verified| verified.as_ref() == Some(&stamp))
    }
    pub fn layout(&self) -> PathBuf {
        self.root.join("auxiliary/layout.onnx")
    }
    pub fn signature(&self) -> PathBuf {
        self.root.join("auxiliary/signature.onnx")
    }
    pub fn model(&self) -> PathBuf {
        self.root.join("weights/score_ocr/gguf/model-Q8_0.gguf")
    }
    pub fn projector(&self) -> PathBuf {
        self.root.join("weights/score_ocr/gguf/vision-F16.gguf")
    }
    pub fn catalog_identity(&self) -> Value {
        self.catalog.clone()
    }
    pub fn total_bytes(&self) -> u64 {
        self.files()
            .iter()
            .filter_map(|f| f["bytes"].as_u64())
            .sum()
    }
    pub fn ensure(&self, allow_download: bool, progress: &Progress) -> Result<(), String> {
        let mut verified = loop {
            progress.checkpoint()?;
            match self.verified.try_lock() {
                Ok(lock) => break lock,
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                Err(_) => return Err("Model cache lock failed".into()),
            }
        };
        if verified.is_some() && *verified == self.cache_stamp() {
            return Ok(());
        }
        *verified = None;
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.root.join(".prepare.lock"))
            .map_err(|e| e.to_string())?;
        // OS-owned lock is released after a crash; wait remains cancellable.
        loop {
            match lock.try_lock_exclusive() {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    progress.checkpoint()?;
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(e.to_string()),
            }
        }
        crate::downloads::prepare(
            &self.root,
            &self.catalog,
            &self.files(),
            "识别模型",
            allow_download,
            progress,
        )?;
        *verified = self.cache_stamp();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::Tasks;
    use serde_json::json;
    #[test]
    fn a_cancelled_job_does_not_wait_for_another_models_lock() {
        let root = tempfile::tempdir().unwrap();
        let sid = "11111111111111111111111111111111".to_owned();
        fs::create_dir(root.path().join(&sid)).unwrap();
        let models = std::sync::Arc::new(Models::new(root.path(), None).unwrap());
        let held = models.verified.lock().unwrap();
        let tasks = Tasks::new(root.path().to_owned()).unwrap();
        let (made, result) = std::sync::mpsc::channel();
        let shared = models.clone();
        tasks
            .submit(sid.clone(), "prepare", "full", move |p| {
                let cancelled = shared.ensure(false, &p);
                made.send(cancelled.clone()).unwrap();
                cancelled
            })
            .unwrap();
        // Give the worker a real chance to enter the contended preparation path.
        std::thread::sleep(Duration::from_millis(30));
        tasks.cancel(&sid).unwrap();
        assert!(result
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .is_err());
        drop(held);
    }
    #[test]
    fn same_size_files_are_not_claimed_verified_on_startup() {
        let root = tempfile::tempdir().unwrap();
        let mut models = Models::new(root.path(), Some(root.path().to_owned())).unwrap();
        fs::create_dir_all(&models.root).unwrap();
        fs::write(models.root.join("weight.bin"), b"corrupt!").unwrap();
        models.catalog = json!({"files":{"auxiliary":[{"path":"weight.bin","bytes":8,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}],"llamacpp":[]}});
        assert!(models.cached());
        assert!(!models.ready());
        // A size-only match must still request permission to repair if SHA fails.
        assert!(models.cache_stamp().is_some());
        assert!(models.verified.lock().unwrap().is_none());
    }
}
