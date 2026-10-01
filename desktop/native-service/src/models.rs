//! Native model cache. No environment installer and no executable downloads.
use crate::{assets::check_hash_cancellable, tasks::Progress};
use fs2::FileExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

pub struct Models {
    pub root: PathBuf,
    pub policy: Value,
    pub capabilities: Value,
    catalog: Value,
    verified: Mutex<Option<Vec<(u64, std::time::SystemTime)>>>,
}
impl Models {
    pub fn new(data_root: &Path, override_root: Option<PathBuf>) -> Result<Self, String> {
        // Compile-time inclusion keeps model metadata sourced from the existing
        // research artifacts without shipping Python/tokenizers/weights in app.
        let catalog: Value =
            serde_json::from_str(include_str!("../../../weights/distribution.json"))
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
            policy: serde_json::from_str(include_str!(
                "../../../weights/score_ocr/merged/score_image_policy.json"
            ))
            .map_err(|e| e.to_string())?,
            capabilities: serde_json::from_str(include_str!(
                "../../../weights/score_ocr/merged/capabilities.json"
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
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        runtime.block_on(self.prepare_files(allow_download, progress))?;
        *verified = self.cache_stamp();
        Ok(())
    }
    async fn prepare_files(&self, allow_download: bool, progress: &Progress) -> Result<(), String> {
        let client = Client::builder()
            .https_only(true)
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(3600))
            .build()
            .map_err(|e| e.to_string())?;
        let total = self.total_bytes();
        let mut completed = 0;
        for file in self.files() {
            progress.checkpoint()?;
            let relative = file["path"].as_str().ok_or("Model path missing")?;
            if Path::new(relative)
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
                || relative.contains(['\\', ':'])
            {
                return Err("Unsafe model path".into());
            }
            let expected = file["sha256"].as_str().ok_or("Model checksum missing")?;
            let size = file["bytes"].as_u64().ok_or("Model size missing")?;
            let destination = self.root.join(relative);
            progress.update(json!({"message":format!("正在校验模型：{relative}"),"done":completed,"total":total}))?;
            if destination.is_file()
                && check_hash_cancellable(&destination, expected, &|| progress.cancelled()).is_ok()
            {
                completed += size;
                continue;
            }
            progress.checkpoint()?;
            if !allow_download {
                return Err("识别模型尚未准备，请先确认模型下载，或指定已有模型缓存".into());
            }
            fs::create_dir_all(destination.parent().ok_or("Invalid model destination")?)
                .map_err(|e| e.to_string())?;
            let base = self.root.canonicalize().map_err(|e| e.to_string())?;
            if !destination
                .parent()
                .unwrap()
                .canonicalize()
                .map_err(|e| e.to_string())?
                .starts_with(&base)
            {
                return Err("Model path escapes cache".into());
            }
            let part = destination.with_extension(format!(
                "{}.part",
                destination
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
            ));
            if fs::symlink_metadata(&part).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err("Model partial file cannot be a symbolic link".into());
            }
            let mut offset = part.metadata().map(|m| m.len()).unwrap_or(0);
            if offset >= size {
                if offset == size
                    && check_hash_cancellable(&part, expected, &|| progress.cancelled()).is_ok()
                {
                    fs::rename(&part, &destination).map_err(|e| e.to_string())?;
                    completed += size;
                    continue;
                }
                // Cancelling a hash check must preserve the completed download.
                progress.checkpoint()?;
                fs::write(&part, []).map_err(|e| e.to_string())?;
                offset = 0;
            }
            let required = size
                .saturating_sub(offset)
                .saturating_add(128 * 1024 * 1024);
            if fs2::available_space(&self.root).map_err(|e| e.to_string())? < required {
                return Err(format!(
                    "模型准备磁盘空间不足：本文件至少还需 {} MiB，请释放空间后续传",
                    required / 1024 / 1024
                ));
            }
            let asset = file["asset"].as_str().ok_or("Model asset missing")?;
            if asset.contains(['/', '\\', '?', '#']) {
                return Err("Unsafe model asset".into());
            }
            let repository = self.catalog["repository"]
                .as_str()
                .ok_or("Model repository missing")?;
            let release = self.catalog["release"]
                .as_str()
                .ok_or("Model release missing")?;
            if repository != "Flurry-L/guitarOCR"
                || !release.starts_with('v')
                || release.contains(['/', '?', '#'])
            {
                return Err("Unrecognized model download source".into());
            }
            let url =
                format!("https://github.com/{repository}/releases/download/{release}/{asset}");
            let mut request = client.get(url);
            if offset > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
            }
            let mut response = cancellable(request.send(), progress).await?;
            if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
                let range = response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|h| h.to_str().ok())
                    .unwrap_or("");
                if !range.starts_with(&format!("bytes {offset}-"))
                    || !range.ends_with(&format!("/{size}"))
                {
                    return Err("模型续传响应范围不匹配".into());
                }
            } else if response.status().is_success() {
                if offset > 0
                    && fs2::available_space(&self.root).map_err(|e| e.to_string())?
                        < size.saturating_add(128 * 1024 * 1024)
                {
                    return Err("下载源未接受续传，需要完整文件空间；请释放磁盘空间后重试".into());
                }
                offset = 0;
            } else {
                return Err(format!("模型下载返回HTTP {}", response.status()));
            }
            let mut output = OpenOptions::new()
                .write(true)
                .create(true)
                .append(offset > 0)
                .truncate(offset == 0)
                .open(&part)
                .map_err(|e| e.to_string())?;
            let mut written = offset;
            let mut last_progress = std::time::Instant::now();
            loop {
                let Some(buffer) = cancellable(response.chunk(), progress).await? else {
                    break;
                };
                let count = buffer.len();
                if written + count as u64 > size {
                    return Err("模型下载超过清单大小".into());
                }
                output
                    .write_all(&buffer[..count])
                    .map_err(|e| e.to_string())?;
                written += count as u64;
                if last_progress.elapsed() >= Duration::from_millis(100) || written == size {
                    progress.update(json!({"message":format!("正在准备模型：{asset}"),"done":completed+written,"total":total}))?;
                    last_progress = std::time::Instant::now();
                }
            }
            output.sync_all().map_err(|e| e.to_string())?;
            drop(output);
            if written != size {
                return Err("模型下载未完成，可重新启动续传".into());
            }
            progress.update(json!({"message":format!("正在校验下载：{asset}"),"done":completed+size,"total":total}))?;
            check_hash_cancellable(&part, expected, &|| progress.cancelled())?;
            fs::rename(&part, &destination).map_err(|e| e.to_string())?;
            completed += size;
        }
        Ok(())
    }
}

async fn cancellable<T>(
    future: impl std::future::Future<Output = std::result::Result<T, reqwest::Error>>,
    progress: &Progress,
) -> Result<T, String> {
    progress.checkpoint()?;
    tokio::pin!(future);
    loop {
        tokio::select! {result=&mut future=>return result.map_err(|e|format!("模型下载失败：{}",e.without_url())),_=tokio::time::sleep(Duration::from_millis(100))=>progress.checkpoint()?}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::Tasks;
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
