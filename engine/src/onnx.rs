//! Shared ONNX Runtime initialization and bounded CPU sessions.
use ort::session::Session;
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};
type Result<T> = std::result::Result<T, String>;
static ORT_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Initialize exactly once with an application-verified bundled ORT library.
/// No implicit search path/download is allowed. rc.10 uses the ORT 1.22 ABI.
pub fn initialize_runtime(path: &Path) -> Result<()> {
    if !path.is_absolute() || !path.is_file() {
        return Err("A verified absolute ONNX Runtime library path is required".into());
    }
    let path = path.canonicalize().map_err(err)?;
    let mut current = ORT_PATH
        .lock()
        .map_err(|_| "ORT initialization lock poisoned")?;
    if let Some(existing) = current.as_ref() {
        return if existing == &path {
            Ok(())
        } else {
            Err("ONNX Runtime is already initialized with another library".into())
        };
    }
    let path_text = path
        .to_str()
        .ok_or("ORT library path must be valid Unicode")?;
    // ort's dynamic loader panics on incompatible shared libraries. Convert this
    // boundary failure to a visible error rather than killing the local service.
    let initialized =
        std::panic::catch_unwind(|| ort::init_from(path_text).with_name("guitarocr").commit())
            .map_err(|_| {
                "ONNX Runtime could not initialize; check bundled library ABI/dependencies"
                    .to_string()
            })?
            .map_err(err)?;
    if !initialized {
        return Err("ONNX Runtime was initialized outside this runtime boundary; restart with the verified library path".into());
    }
    *current = Some(path);
    Ok(())
}

pub fn session(path: &Path) -> Result<Session> {
    if ORT_PATH
        .lock()
        .map_err(|_| "ORT initialization lock poisoned")?
        .is_none()
    {
        return Err("Call initialize_runtime with the verified runtime library first".into());
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8);
    Session::builder()
        .map_err(err)?
        .with_intra_threads(threads)
        .map_err(err)?
        .with_inter_threads(1)
        .map_err(err)?
        .commit_from_file(path)
        .map_err(err)
}
