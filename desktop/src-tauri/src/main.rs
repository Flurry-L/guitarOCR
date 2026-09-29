#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::{
    collections::HashSet,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, OnceLock,
    },
    thread,
    time::Duration,
};
use tauri::{Emitter, Manager, WebviewUrl, WebviewWindow};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};
use url::Url;

#[derive(Default)]
struct Runtime(Mutex<Option<Arc<Process>>>);
struct Process {
    child: Mutex<Child>,
    stopped: AtomicBool,
    mode: String,
    url: Mutex<Option<Url>>,
    #[cfg(windows)]
    job: usize,
}
impl Process {
    fn stop(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut child = self.child.lock().unwrap();
        #[cfg(unix)]
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        #[cfg(windows)]
        unsafe {
            windows_sys::Win32::System::JobObjects::TerminateJobObject(self.job as _, 1);
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        self.stop();
        #[cfg(windows)]
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.job as _);
        }
    }
}

// Commands are available only to the bundled launcher, never a remote workbench.
fn authorize(window: &WebviewWindow) -> Result<(), String> {
    let url = window.url().map_err(|e| e.to_string())?;
    if window.label() == "main"
        && (url.scheme() == "tauri" || matches!(url.host_str(), Some("tauri.localhost")))
    {
        Ok(())
    } else {
        Err("此页面不能调用桌面功能".into())
    }
}
fn data(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let path = app.path().app_data_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    Ok(path)
}
fn server_url(address: &str) -> Result<Url, String> {
    let url = Url::parse(address.trim())
        .map_err(|_| "请输入完整的服务地址，例如 https://ocr.example.com")?;
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if url.scheme() != "https" && !(url.scheme() == "http" && local) {
        return Err("远程服务请使用 HTTPS；本机服务可使用 HTTP。".into());
    }
    if url.host_str().is_none() || !url.username().is_empty() || url.password().is_some() {
        return Err("服务地址不能包含用户名或密码".into());
    }
    Ok(url)
}
static DOWNLOADS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

fn workspace(app: &tauri::AppHandle, label: &str, url: Url) -> Result<(), String> {
    if let Some(existing) = app.get_webview_window(label) {
        existing.set_focus().map_err(|e| e.to_string())?;
        return Ok(());
    }
    let origin = url.origin();
    tauri::WebviewWindowBuilder::new(app, label, WebviewUrl::External(url))
        .title("GuitarOCR 工作台")
        .inner_size(1280., 860.)
        .min_inner_size(640., 480.)
        .disable_drag_drop_handler()
        .on_navigation(move |url| url.origin() == origin)
        .on_download(|webview, event| {
            match event {
                tauri::webview::DownloadEvent::Requested { destination, .. } => {
                    let app = webview.app_handle();
                    let dir = app.path().download_dir().ok().or_else(|| data(app).ok());
                    let Some(dir) = dir else {
                        return false;
                    };
                    if fs::create_dir_all(&dir).is_err() {
                        return false;
                    }
                    let name = destination
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let name = if name.is_empty() {
                        "guitarocr-download".to_string()
                    } else {
                        name
                    };
                    for i in 0..10000 {
                        let path = dir.join(if i == 0 {
                            name.clone()
                        } else {
                            let p = Path::new(&name);
                            format!(
                                "{} ({i}).{}",
                                p.file_stem().unwrap_or_default().to_string_lossy(),
                                p.extension().unwrap_or_default().to_string_lossy()
                            )
                        });
                        let mut downloads = DOWNLOADS.get_or_init(Default::default).lock().unwrap();
                        if !path.exists() && downloads.insert(path.clone()) {
                            *destination = path;
                            return true;
                        }
                    }
                    return false;
                }
                tauri::webview::DownloadEvent::Finished { path, success, .. } => {
                    if let Some(path) = &path {
                        DOWNLOADS
                            .get_or_init(Default::default)
                            .lock()
                            .unwrap()
                            .remove(path);
                    }
                    let message = if success {
                        format!(
                            "文件已保存到 {}",
                            path.map(|p| p.display().to_string()).unwrap_or_default()
                        )
                    } else {
                        "下载失败，请重试。".to_string()
                    };
                    webview
                        .app_handle()
                        .dialog()
                        .message(message)
                        .title("GuitarOCR")
                        .show(|_| {});
                }
                _ => {}
            }
            true
        })
        .build()
        .map_err(|e| e.to_string())?;
    Ok(())
}
#[derive(Serialize)]
struct Settings {
    server: String,
    local_gpu: bool,
}
#[tauri::command]
fn settings(app: tauri::AppHandle, window: WebviewWindow) -> Result<Settings, String> {
    authorize(&window)?;
    Ok(Settings {
        server: fs::read_to_string(data(&app)?.join("server.txt")).unwrap_or_default(),
        local_gpu: cfg!(all(
            any(target_os = "linux", target_os = "windows"),
            target_arch = "x86_64"
        )),
    })
}
#[tauri::command]
fn connect_server(
    app: tauri::AppHandle,
    window: WebviewWindow,
    address: String,
) -> Result<(), String> {
    authorize(&window)?;
    let url = server_url(&address)?;
    if let Some(old) = app.get_webview_window("remote") {
        old.close().map_err(|e| e.to_string())?;
    }
    workspace(&app, "remote", url.clone())?;
    fs::write(data(&app)?.join("server.txt"), url.as_str()).map_err(|e| e.to_string())
}
fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), to.join(entry.file_name()))?;
        }
    }
    Ok(())
}
fn log(app: &tauri::AppHandle, line: &str) {
    if let Some(message) = line.strip_prefix("GUITAROCR_PROGRESS ") {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(message) {
            let _ = app.emit_to("main", "runtime-progress", value);
            return;
        }
    }
    let _ = app.emit_to("main", "runtime-log", line);
    if let Ok(dir) = data(app) {
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("runtime.log"))
        {
            let _ = writeln!(f, "{line}");
        }
    }
}
fn command(path: &Path) -> Command {
    let mut command = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
}
#[cfg(windows)]
fn owned_process(mut child: Child, mode: &str) -> Result<Process, String> {
    use std::{mem, os::windows::io::AsRawHandle};
    use windows_sys::Win32::{Foundation::CloseHandle, System::JobObjects::*};
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if job.is_null()
            || SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as _,
                mem::size_of_val(&info) as u32,
            ) == 0
            || AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0
        {
            let _ = child.kill();
            let _ = child.wait();
            if !job.is_null() {
                CloseHandle(job);
            }
            return Err("无法管理本机服务进程，请重新启动应用".into());
        }
        Ok(Process {
            child: Mutex::new(child),
            stopped: AtomicBool::new(false),
            mode: mode.into(),
            url: Mutex::new(None),
            job: job as usize,
        })
    }
}
#[cfg(not(windows))]
fn owned_process(child: Child, mode: &str) -> Result<Process, String> {
    Ok(Process {
        child: Mutex::new(child),
        stopped: AtomicBool::new(false),
        mode: mode.into(),
        url: Mutex::new(None),
    })
}

fn start(app: &tauri::AppHandle, mode: &str) -> Result<Url, String> {
    if !matches!(mode, "gpu" | "edit") {
        return Err("未知运行方式".into());
    }
    let runtime = app.state::<Runtime>();
    let mut slot = runtime.0.lock().unwrap();
    if let Some(process) = slot.as_ref() {
        if process.mode != mode {
            return Err("切换方式前请停止本机服务。".into());
        }
        return process
            .url
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "本机环境正在准备，请稍候。".into());
    }
    let root = data(app)?;
    let resource = app.path().resource_dir().map_err(|e| e.to_string())?;
    let backend = root.join("backend");
    copy_tree(&resource.join("backend"), &backend).map_err(|e| e.to_string())?;
    let uv = resource
        .join("uv")
        .join(if cfg!(windows) { "uv.exe" } else { "uv" });
    let _ = fs::write(root.join("runtime.log"), "");
    let mut cmd = command(&uv);
    cmd.args([
        "run",
        "--no-project",
        "--python",
        "3.11",
        "--config-file",
        "scripts/bootstrap-uv.toml",
        "scripts/desktop_runtime.py",
        "--mode",
        mode,
        "--output",
    ])
    .arg(root.join("projects"))
    .current_dir(&backend)
    .env("GUITAROCR_UV", &uv)
    .env("GUITAROCR_DESKTOP", "1")
    .env("HF_HUB_OFFLINE", "1")
    .env("TRANSFORMERS_OFFLINE", "1")
    .env("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
    .env("UV_NO_PROGRESS", "false")
    .env("UV_COLOR", "never")
    .env("PYTHONUTF8", "1")
    .env("PYTHONUNBUFFERED", "1")
    .env("UV_PYTHON_INSTALL_DIR", root.join("python"))
    .env(
        "UV_PYTHON_INSTALL_MIRROR",
        std::env::var("UV_PYTHON_INSTALL_MIRROR").unwrap_or_else(|_| {
            "https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone".into()
        }),
    )
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .stdin(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let process = Arc::new(owned_process(child, mode)?);
    *slot = Some(process.clone());
    drop(slot);
    let (tx, rx) = mpsc::channel();
    let app_out = app.clone();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(url) = line.strip_prefix("GUITAROCR_READY ") {
                let _ = tx.send(url.to_string());
            } else {
                log(&app_out, &line);
            }
        }
    });
    let app_err = app.clone();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            log(&app_err, &line);
        }
    });
    let ready_process = process.clone();
    let app_watch = app.clone();
    thread::spawn(move || {
        loop {
            let finished = process
                .child
                .lock()
                .unwrap()
                .try_wait()
                .ok()
                .flatten()
                .is_some();
            if finished {
                break;
            }
            thread::sleep(Duration::from_millis(250));
        }
        let runtime = app_watch.state::<Runtime>();
        let mut slot = runtime.0.lock().unwrap();
        if slot.as_ref().is_some_and(|p| Arc::ptr_eq(p, &process)) {
            slot.take();
            let _ = app_watch.emit_to(
                "main",
                "runtime-exit",
                "本机服务已退出，详情见运行日志。项目已保留。",
            );
        }
    });
    let url = rx
        .recv()
        .map_err(|_| "本机服务未能启动，详情见运行日志。可重试或连接 GPU 服务。")?;
    let url = server_url(&url)?;
    if ready_process.stopped.load(Ordering::SeqCst) {
        return Err("本机服务已停止。".into());
    }
    *ready_process.url.lock().unwrap() = Some(url.clone());
    Ok(url)
}
#[tauri::command]
async fn start_local(
    app: tauri::AppHandle,
    window: WebviewWindow,
    mode: String,
) -> Result<(), String> {
    authorize(&window)?;
    let handle = app.clone();
    let url = tauri::async_runtime::spawn_blocking(move || start(&handle, &mode))
        .await
        .map_err(|e| e.to_string())??;
    workspace(&app, "local", url)
}
fn stop(app: &tauri::AppHandle) {
    if let Some(process) = app.state::<Runtime>().0.lock().unwrap().take() {
        process.stop();
    }
}
#[tauri::command]
fn stop_local(app: tauri::AppHandle, window: WebviewWindow) -> Result<(), String> {
    authorize(&window)?;
    if let Some(local) = app.get_webview_window("local") {
        let _ = local.close();
    }
    stop(&app);
    Ok(())
}
#[tauri::command]
fn open_data(app: tauri::AppHandle, window: WebviewWindow) -> Result<(), String> {
    authorize(&window)?;
    let dir = data(&app)?;
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    Command::new(program)
        .arg(dir)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .manage(Runtime::default())
        .setup(|app| {
            // WebKit falls back to the working directory when the OS has no Downloads folder.
            std::env::set_current_dir(data(app.handle()).map_err(std::io::Error::other)?)?;
            tauri::WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("GuitarOCR")
                .inner_size(780., 680.)
                .min_inner_size(560., 500.)
                .on_navigation(|url| {
                    url.scheme() == "tauri" || url.host_str() == Some("tauri.localhost")
                })
                .build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            settings,
            connect_server,
            start_local,
            stop_local,
            open_data
        ])
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    if window.state::<Runtime>().0.lock().unwrap().is_some() {
                        api.prevent_close();
                        let app = window.app_handle().clone();
                        window
                            .dialog()
                            .message("退出会停止本机任务。已保存的项目会保留。")
                            .title("退出 GuitarOCR？")
                            .buttons(MessageDialogButtons::OkCancelCustom(
                                "退出".into(),
                                "继续使用".into(),
                            ))
                            .show(move |confirmed| {
                                if confirmed {
                                    stop(&app);
                                    app.exit(0);
                                }
                            });
                    } else {
                        window.app_handle().exit(0);
                    }
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("无法启动 GuitarOCR")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                stop(app);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn server_addresses_restrict_schemes_and_credentials() {
        for address in [
            "https://ocr.example.com",
            "http://127.0.0.1:7860",
            "http://[::1]:7860",
        ] {
            assert!(server_url(address).is_ok());
        }
        for address in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "http://example.com",
            "https://user:pass@example.com",
        ] {
            assert!(server_url(address).is_err());
        }
    }
}
