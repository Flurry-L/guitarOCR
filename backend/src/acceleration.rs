//! Optional Windows GPU runtime; CPU and Metal remain part of the application.
use crate::{assets::check_hash_cancellable, downloads, models::Models, tasks::Progress};
use fs2::FileExt;
use guitarocr_engine::{device_selection::MemoryBudget, llama::PINNED_COMMIT};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

pub struct Acceleration {
    root: PathBuf,
    entry: Value,
}
impl Acceleration {
    pub fn for_device(models: &Models, slots: usize) -> Option<Self> {
        let catalog = models.catalog_identity();
        let sizes: Vec<u64> = catalog["files"]["llamacpp"]
            .as_array()?
            .iter()
            .filter_map(|v| v["bytes"].as_u64())
            .collect();
        if sizes.len() != 2 {
            return None;
        }
        let budget = MemoryBudget::estimate(sizes[0], sizes[1], 8192, slots.clamp(1, 4)).ok()?;
        if !guitarocr_engine::cuda::available(budget.required_bytes) {
            return None;
        }
        let runtimes: Value =
            serde_json::from_str(include_str!("../../scripts/llamacpp-runtime.json")).ok()?;
        if runtimes["commit"] != PINNED_COMMIT {
            return None;
        }
        let entry = runtimes["artifacts"]["windows-x64-cuda"].as_object()?;
        Some(Self {
            root: models
                .runtime_root
                .join(format!("{PINNED_COMMIT}-windows-cuda")),
            entry: json!(entry),
        })
    }
    pub fn bytes(&self) -> u64 {
        self.entry["bytes"].as_u64().unwrap_or(0)
    }
    fn folder(&self) -> PathBuf {
        self.root.join("installed")
    }
    pub fn cached(&self) -> bool {
        self.entry["files"].as_array().is_some_and(|files| {
            !files.is_empty()
                && files.iter().all(|file| {
                    self.folder()
                        .join(file["name"].as_str().unwrap_or(""))
                        .metadata()
                        .is_ok_and(|m| Some(m.len()) == file["bytes"].as_u64())
                })
        })
    }
    fn verified(&self, progress: &Progress) -> Result<PathBuf, String> {
        let root = self.folder().canonicalize().map_err(|e| e.to_string())?;
        for file in self.entry["files"]
            .as_array()
            .ok_or("Missing acceleration files")?
        {
            let name = file["name"].as_str().ok_or("Invalid acceleration file")?;
            safe_path(name)?;
            let path = root.join(name).canonicalize().map_err(|e| e.to_string())?;
            if !path.starts_with(&root) {
                return Err("Acceleration file escapes cache".into());
            }
            check_hash_cancellable(
                &path,
                file["sha256"]
                    .as_str()
                    .ok_or("Missing acceleration checksum")?,
                &|| progress.cancelled(),
            )?;
        }
        Ok(root.join("llama-server.exe"))
    }
    pub fn prepare(
        &self,
        allow_download: bool,
        progress: &Progress,
    ) -> Result<Option<PathBuf>, String> {
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.root.join(".prepare.lock"))
            .map_err(|e| e.to_string())?;
        loop {
            progress.checkpoint()?;
            match lock.try_lock_exclusive() {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                Err(e) => return Err(e.to_string()),
            }
        }
        if self.cached() {
            if let Ok(executable) = self.verified(progress) {
                return Ok(Some(executable));
            }
            progress.checkpoint()?;
        }
        if !allow_download {
            return Ok(None);
        }
        let prefix = "https://github.com/Flurry-L/guitarOCR/releases/download/";
        let (release, asset) = self.entry["url"]
            .as_str()
            .and_then(|s| s.strip_prefix(prefix))
            .and_then(|s| s.split_once('/'))
            .ok_or("Invalid acceleration source")?;
        let file = json!({"path":"runtime.zip","asset":asset,"bytes":self.entry["bytes"],"sha256":self.entry["sha256"]});
        let catalog = json!({"repository":"Flurry-L/guitarOCR","release":release});
        downloads::prepare(
            &self.root,
            &catalog,
            &[&file],
            "显卡加速组件",
            true,
            progress,
        )?;
        let staging = tempfile::Builder::new()
            .prefix("install-")
            .tempdir_in(&self.root)
            .map_err(|e| e.to_string())?;
        let source = fs::File::open(self.root.join("runtime.zip")).map_err(|e| e.to_string())?;
        let mut archive = zip::ZipArchive::new(source).map_err(|e| e.to_string())?;
        for file in self.entry["files"]
            .as_array()
            .ok_or("Missing acceleration files")?
        {
            progress.checkpoint()?;
            let name = file["name"].as_str().ok_or("Invalid acceleration file")?;
            safe_path(name)?;
            let mut member = archive.by_name(name).map_err(|e| e.to_string())?;
            let size = file["bytes"]
                .as_u64()
                .ok_or("Missing acceleration file size")?;
            if member.size() != size
                || member.is_dir()
                || member.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000)
            {
                return Err("Invalid acceleration archive entry".into());
            }
            let path = staging.path().join(name);
            fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
            let mut output = fs::File::create(&path).map_err(|e| e.to_string())?;
            let mut bytes = [0u8; 65536];
            loop {
                progress.checkpoint()?;
                let n = member.read(&mut bytes).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                output.write_all(&bytes[..n]).map_err(|e| e.to_string())?;
            }
            output.sync_all().map_err(|e| e.to_string())?;
            check_hash_cancellable(
                &path,
                file["sha256"]
                    .as_str()
                    .ok_or("Missing acceleration checksum")?,
                &|| progress.cancelled(),
            )?;
        }
        progress.checkpoint()?;
        if self.folder().exists() {
            fs::remove_dir_all(self.folder()).map_err(|e| e.to_string())?;
        }
        fs::rename(staging.path(), self.folder()).map_err(|e| e.to_string())?;
        let _ = fs::remove_file(self.root.join("runtime.zip"));
        Ok(Some(self.folder().join("llama-server.exe")))
    }
}
fn safe_path(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.contains(['\\', ':'])
        || Path::new(name)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("Unsafe acceleration file path".into());
    }
    Ok(())
}
