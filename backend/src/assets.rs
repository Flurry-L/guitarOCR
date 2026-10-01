//! Locate only hash-verified packaged native code. Never search PATH/site-packages.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Default, Clone)]
pub struct NativeAssets {
    pub ort: Option<PathBuf>,
    pub pdfium: Option<PathBuf>,
    pub llama: Option<PathBuf>,
    pub capabilities: guitarocr_engine::device_selection::RuntimeCapabilities,
}
impl NativeAssets {
    pub fn load(root: Option<&Path>) -> Result<Self, String> {
        let Some(root) = root else {
            return Ok(Self::default());
        };
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let manifest: Value = serde_json::from_slice(
            &fs::read(root.join("manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let os = match std::env::consts::OS {
            "macos" => "darwin",
            "windows" => "win32",
            other => other,
        };
        let arch = match std::env::consts::ARCH {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            other => other,
        };
        if manifest["schema"] != 1 || manifest["target"] != format!("{os}-{arch}") {
            return Err("Native runtime manifest target/schema mismatch".into());
        }
        let mut result = Self::default();
        for component in manifest["components"]
            .as_array()
            .ok_or("Native component list is missing")?
        {
            let name = component["name"]
                .as_str()
                .ok_or("Invalid native component")?;
            let source = match name {
                "onnxruntime" => "https://github.com/microsoft/onnxruntime",
                "pdfium" => "https://pdfium.googlesource.com/pdfium",
                "llama.cpp" => "https://github.com/ggml-org/llama.cpp",
                _ => return Err("Unrecognized native component".into()),
            };
            if component["source"] != source {
                return Err(format!("Unrecognized native source: {name}"));
            }
            if name == "llama.cpp" && component["version"] != guitarocr_engine::llama::PINNED_COMMIT
            {
                return Err("Native llama revision mismatch".into());
            }
            if name == "llama.cpp" {
                result.capabilities =
                    guitarocr_engine::device_selection::RuntimeCapabilities::from_component(
                        component,
                    )?;
            }
            for file in component["files"]
                .as_array()
                .ok_or("Missing native files")?
            {
                let relative = file["path"].as_str().ok_or("Invalid native file path")?;
                if relative.is_empty()
                    || relative.contains('\\')
                    || relative.contains(':')
                    || Path::new(relative)
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                {
                    return Err("Unsafe native file path".into());
                }
                let path = root
                    .join(name)
                    .join(relative)
                    .canonicalize()
                    .map_err(|e| e.to_string())?;
                if !path.starts_with(root.join(name)) {
                    return Err("Native file escapes package".into());
                }
                check_hash(
                    &path,
                    file["sha256"].as_str().ok_or("Missing native file hash")?,
                )?;
                let basename = path.file_name().and_then(|p| p.to_str()).unwrap_or("");
                // Transitive libraries are also inventoried and verified, but must
                // never replace the actual engine entry point based on file order.
                if primary_binary(name, basename, file["role"].as_str().unwrap_or("")) {
                    let entry = match name {
                        "onnxruntime" => &mut result.ort,
                        "pdfium" => &mut result.pdfium,
                        _ => &mut result.llama,
                    };
                    if entry.replace(path).is_some() {
                        return Err(format!("Multiple native entry points: {name}"));
                    }
                }
            }
            let found = match name {
                "onnxruntime" => result.ort.is_some(),
                "pdfium" => result.pdfium.is_some(),
                _ => result.llama.is_some(),
            };
            if !found {
                return Err(format!("Native entry point missing: {name}"));
            }
        }
        Ok(result)
    }
}
fn primary_binary(component: &str, name: &str, role: &str) -> bool {
    match (component, role) {
        ("onnxruntime", "library") => {
            name == "onnxruntime.dll"
                || name == "libonnxruntime.dylib"
                || name == "libonnxruntime.so"
                || name
                    .strip_prefix("libonnxruntime.so.")
                    .is_some_and(version_suffix)
                || name
                    .strip_prefix("libonnxruntime.")
                    .and_then(|s| s.strip_suffix(".dylib"))
                    .is_some_and(version_suffix)
        }
        ("pdfium", "library") => matches!(name, "pdfium.dll" | "libpdfium.so" | "libpdfium.dylib"),
        ("llama.cpp", "executable") => matches!(name, "llama-server" | "llama-server.exe"),
        _ => false,
    }
}
fn version_suffix(value: &str) -> bool {
    !value.is_empty()
        && value
            .split('.')
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

pub fn check_hash(path: &Path, expected: &str) -> Result<(), String> {
    check_hash_cancellable(path, expected, &|| false)
}
pub fn check_hash_cancellable(
    path: &Path,
    expected: &str,
    cancelled: &dyn Fn() -> bool,
) -> Result<(), String> {
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid file hash".into());
    }
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut block = [0u8; 64 * 1024];
    loop {
        if cancelled() {
            return Err("校验已取消".into());
        }
        let count = file.read(&mut block).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&block[..count]);
    }
    if format!("{:x}", hasher.finalize()) != expected.to_lowercase() {
        return Err(format!(
            "Native file failed integrity check: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependencies_cannot_be_selected_as_engine_entry_points() {
        for name in [
            "libonnxruntime.so",
            "libonnxruntime.so.1.23.2",
            "libonnxruntime.dylib",
            "libonnxruntime.1.23.2.dylib",
            "onnxruntime.dll",
        ] {
            assert!(primary_binary("onnxruntime", name, "library"), "{name}");
        }
        for name in [
            "libonnxruntime_providers_shared.so",
            "libomp.dylib",
            "libonnxruntime.so.bad",
            "libonnxruntime..dylib",
        ] {
            assert!(!primary_binary("onnxruntime", name, "library"), "{name}");
        }
        assert!(primary_binary("pdfium", "libpdfium.dylib", "library"));
        assert!(!primary_binary("pdfium", "libc++.1.dylib", "library"));
        assert!(primary_binary(
            "llama.cpp",
            "llama-server.exe",
            "executable"
        ));
        assert!(!primary_binary("llama.cpp", "llama-cli", "executable"));
    }
}
