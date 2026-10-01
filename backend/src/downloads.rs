//! Resumable downloads shared by models and optional acceleration libraries.
use crate::{assets::check_hash_cancellable, tasks::Progress};
use reqwest::Client;
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Duration,
};

pub fn prepare(
    root: &Path,
    catalog: &Value,
    files: &[&Value],
    label: &str,
    allow_download: bool,
    progress: &Progress,
) -> Result<(), String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?
        .block_on(prepare_files(
            root,
            catalog,
            files,
            label,
            allow_download,
            progress,
        ))
}

async fn prepare_files(
    root: &Path,
    catalog: &Value,
    files: &[&Value],
    label: &str,
    allow_download: bool,
    progress: &Progress,
) -> Result<(), String> {
    let client = Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;
    let total: u64 = files.iter().filter_map(|file| file["bytes"].as_u64()).sum();
    let mut completed = 0;
    for file in files {
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
        let destination = root.join(relative);
        progress
            .update(json!({"message":format!("正在准备{label}"),"done":completed,"total":total}))?;
        if destination.is_file()
            && check_hash_cancellable(&destination, expected, &|| progress.cancelled()).is_ok()
        {
            completed += size;
            continue;
        }
        progress.checkpoint()?;
        if !allow_download {
            return Err("识别资源尚未准备，请先确认下载".into());
        }
        fs::create_dir_all(destination.parent().ok_or("Invalid model destination")?)
            .map_err(|e| e.to_string())?;
        let base = root.canonicalize().map_err(|e| e.to_string())?;
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
        if fs2::available_space(&root).map_err(|e| e.to_string())? < required {
            return Err(format!(
                "资源准备磁盘空间不足：本文件至少还需 {} MiB，请释放空间后续传",
                required / 1024 / 1024
            ));
        }
        let asset = file["asset"].as_str().ok_or("Model asset missing")?;
        if asset.contains(['/', '\\', '?', '#']) {
            return Err("Unsafe model asset".into());
        }
        let repository = catalog["repository"]
            .as_str()
            .ok_or("Model repository missing")?;
        let release = catalog["release"].as_str().ok_or("Model release missing")?;
        if repository != "Flurry-L/guitarOCR"
            || !release.starts_with('v')
            || release.contains(['/', '?', '#'])
        {
            return Err("Unrecognized model download source".into());
        }
        let url = format!("https://github.com/{repository}/releases/download/{release}/{asset}");
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
                return Err("资源续传响应范围不匹配".into());
            }
        } else if response.status().is_success() {
            if offset > 0
                && fs2::available_space(&root).map_err(|e| e.to_string())?
                    < size.saturating_add(128 * 1024 * 1024)
            {
                return Err("下载源未接受续传，需要完整文件空间；请释放磁盘空间后重试".into());
            }
            offset = 0;
        } else {
            return Err(format!("资源下载返回HTTP {}", response.status()));
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
                return Err("资源下载超过清单大小".into());
            }
            output
                .write_all(&buffer[..count])
                .map_err(|e| e.to_string())?;
            written += count as u64;
            if last_progress.elapsed() >= Duration::from_millis(100) || written == size {
                progress.update(json!({"message":format!("正在下载{label}"),"done":completed+written,"total":total}))?;
                last_progress = std::time::Instant::now();
            }
        }
        output.sync_all().map_err(|e| e.to_string())?;
        drop(output);
        if written != size {
            return Err("资源下载未完成，可重新启动续传".into());
        }
        progress.update(
            json!({"message":format!("正在完成{label}下载"),"done":completed+size,"total":total}),
        )?;
        check_hash_cancellable(&part, expected, &|| progress.cancelled())?;
        fs::rename(&part, &destination).map_err(|e| e.to_string())?;
        completed += size;
    }
    Ok(())
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
