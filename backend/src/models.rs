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

const COMPILED_CATALOG: &str = include_str!(concat!(env!("OUT_DIR"), "/model-distribution.json"));
const COMPILED_POLICY: &str =
    include_str!("../../weights/score_ocr/merged/score_image_policy.json");
const COMPILED_CAPABILITIES: &str =
    include_str!("../../weights/score_ocr/merged/capabilities.json");

fn parse_catalog(contents: &str) -> Result<Value, String> {
    let catalog: Value = serde_json::from_str(contents).map_err(|e| e.to_string())?;
    let generation = catalog["generation"]
        .as_str()
        .ok_or("Missing model generation")?;
    if generation.is_empty()
        || generation.contains(['/', '\\', ':'])
        || [".", ".."].contains(&generation)
    {
        return Err("Unsafe model generation".into());
    }
    let mut seen = std::collections::HashSet::new();
    for (group, required, optional) in [
        (
            "auxiliary",
            &[
                "auxiliary/layout.onnx",
                "auxiliary/layout_quality.onnx",
                "auxiliary/signature.onnx",
            ][..],
            &[
                "auxiliary/chord_evidence.onnx",
                "auxiliary/structure_relations.onnx",
                "auxiliary/staff_evidence.onnx",
                "auxiliary/staff_label.onnx",
                "auxiliary/instrument_text.onnx",
                "auxiliary/part_continuation.onnx",
                "auxiliary/clef_spatial.onnx",
                "auxiliary/text_regions.onnx",
                "auxiliary/chord_text_roles.onnx",
            ][..],
        ),
        (
            "llamacpp",
            &[
                "weights/score_ocr/gguf/model-Q8_0.gguf",
                "weights/score_ocr/gguf/vision-F16.gguf",
                "weights/information/gguf/model-Q8_0.gguf",
                "weights/information/gguf/vision-F16.gguf",
                "weights/information/gguf/information-adapter.gguf",
            ][..],
            &[][..],
        ),
    ] {
        let files = catalog["files"][group]
            .as_array()
            .ok_or_else(|| format!("Missing model files: {group}"))?;
        if files.len() < required.len() || files.len() > required.len() + optional.len() {
            return Err(format!("Unexpected number of {group} model files"));
        }
        for file in files {
            let path = file["path"].as_str().ok_or("Missing model path")?;
            if (!required.contains(&path) && !optional.contains(&path))
                || !seen.insert(path.to_owned())
            {
                return Err(format!("Unexpected or duplicate model path: {path}"));
            }
            let local = catalog["availability"] == "local_unpublished";
            if (!local || file.get("bytes").is_some())
                && file["bytes"].as_u64().is_none_or(|size| size == 0)
            {
                return Err(format!("Model size must be a positive integer: {path}"));
            }
            let hash = file["sha256"].as_str().unwrap_or("");
            if (!local || file.get("sha256").is_some())
                && (hash.len() != 64
                    || !hash
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
            {
                return Err(format!(
                    "Model checksum must contain 64 lowercase hex digits: {path}"
                ));
            }
        }
        for path in required {
            if !seen.contains(*path) {
                return Err(format!("Missing required model: {path}"));
            }
        }
    }
    Ok(catalog)
}

/// Relocate a model-relative declaration into the packaged auxiliary directory.
/// Source models and standalone exports can have different parent directories.
fn auxiliary_asset(value: &Value) -> Option<&str> {
    let path = value.as_str()?;
    if path.starts_with("auxiliary/") {
        Some(path)
    } else {
        path.find("/auxiliary/").map(|index| &path[index + 1..])
    }
}

fn validate_information_model(capabilities: &Value) -> Result<(), String> {
    let declaration = &capabilities["information_model"];
    let tasks: std::collections::BTreeSet<_> = declaration["adapter_tasks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if declaration["model"] != "../../information/gguf/model-Q8_0.gguf"
        || declaration["projector"] != "../../information/gguf/vision-F16.gguf"
        || declaration["adapter"] != "../../information/gguf/information-adapter.gguf"
        || tasks != ["staff", "clef", "tempo"].into_iter().collect()
        || declaration["adapter_tasks"]
            .as_array()
            .is_none_or(|v| v.len() != tasks.len())
    {
        return Err(
            "Information recognition requires its model, projector and task adapter".into(),
        );
    }
    Ok(())
}

/// Validate the complete compiled metadata before opening a cache or starting
/// inference. A capability cannot enable an asset absent from the selected
/// catalog, even if a stray copy exists in the cache.
fn parse_metadata(
    catalog: &str,
    policy: &str,
    capabilities: &str,
) -> Result<(Value, Value, Value), String> {
    let catalog = parse_catalog(catalog)?;
    let policy: Value = serde_json::from_str(policy)
        .map_err(|error| format!("Invalid score_image_policy.json: {error}"))?;
    crate::recognition::ImagePolicy::from_value(&policy)?;
    let capabilities: Value = serde_json::from_str(capabilities)
        .map_err(|error| format!("Invalid capabilities.json: {error}"))?;
    let declarations = capabilities
        .as_object()
        .ok_or("Invalid capabilities.json: expected an object")?;
    if capabilities["measure_output"] != "observation" || capabilities["measure_images"] != 1 {
        return Err("Measure recognition requires single-image observation weights".into());
    }
    if auxiliary_asset(&capabilities["layout_quality_model"])
        != Some("auxiliary/layout_quality.onnx")
    {
        return Err("Layout detection requires the declared quality model".into());
    }
    for (field, path, description) in [
        (
            "staff_label_model",
            "auxiliary/staff_label.onnx",
            "staff label",
        ),
        (
            "instrument_text_model",
            "auxiliary/instrument_text.onnx",
            "instrument text",
        ),
        (
            "part_continuation_model",
            "auxiliary/part_continuation.onnx",
            "part continuation",
        ),
        (
            "clef_spatial_model",
            "auxiliary/clef_spatial.onnx",
            "spatial clef",
        ),
    ] {
        if !capabilities[field].is_null() && auxiliary_asset(&capabilities[field]) != Some(path) {
            return Err(format!("Unsupported {description} model declaration"));
        }
    }
    validate_information_model(&capabilities)?;
    if capabilities["staff_label_model"].is_null()
        != capabilities["instrument_text_model"].is_null()
    {
        return Err("staff_label_model and instrument_text_model must be declared together".into());
    }
    if !capabilities["part_continuation_model"].is_null()
        && (capabilities["staff_label_model"].is_null() || capabilities["staff_profile"] != true)
    {
        return Err("Part continuation requires observed staff label profiles".into());
    }
    let assets: std::collections::HashSet<_> = ["auxiliary", "llamacpp"]
        .iter()
        .flat_map(|group| catalog["files"][*group].as_array().unwrap())
        .filter_map(|file| file["path"].as_str())
        .collect();
    for (capability, asset) in declarations {
        let Some(asset) = auxiliary_asset(asset) else {
            continue;
        };
        if !assets.contains(asset) {
            return Err(format!(
                "Model catalog {} omits declared {capability}: {asset}. \
                 Rebuild with GUITAROCR_BUILD_MODEL_CATALOG=<matching-catalog.json>, \
                 or prepare desktop resources with --build --model-catalog <matching-catalog.json>. \
                 See README.md; --models only selects the cache directory.",
                catalog["generation"].as_str().unwrap()
            ));
        }
    }
    Ok((catalog, policy, capabilities))
}

/// Inspect the catalog compiled into this exact binary without opening a cache
/// or initializing any inference runtime. Preserve bytes for metadata comparison.
pub fn compiled_catalog() -> Result<&'static str, String> {
    parse_catalog(COMPILED_CATALOG)?;
    Ok(COMPILED_CATALOG)
}

/// Exact metadata bytes used by this binary. Packaging verifies and stages one
/// snapshot so newer source metadata cannot be paired with an older executable.
pub fn compiled_metadata() -> Result<Value, String> {
    parse_metadata(COMPILED_CATALOG, COMPILED_POLICY, COMPILED_CAPABILITIES)?;
    Ok(serde_json::json!({
        "distribution.json": COMPILED_CATALOG,
        "score_image_policy.json": COMPILED_POLICY,
        "capabilities.json": COMPILED_CAPABILITIES,
    }))
}

pub struct Models {
    pub root: PathBuf,
    pub runtime_root: PathBuf,
    pub policy: Value,
    pub capabilities: Value,
    pub(crate) catalog: Value,
    verified: Mutex<Option<Vec<(u64, std::time::SystemTime)>>>,
}
impl Models {
    pub fn new(data_root: &Path, override_root: Option<PathBuf>) -> Result<Self, String> {
        // Compile-time inclusion keeps model metadata sourced from the existing
        // research artifacts without shipping Python/tokenizers/weights in app.
        let (catalog, policy, capabilities) =
            parse_metadata(COMPILED_CATALOG, COMPILED_POLICY, COMPILED_CAPABILITIES)?;
        let generation = catalog["generation"].as_str().unwrap();
        let root = if let Some(base) = override_root {
            // Keep the Python research cache environment variable's base-dir
            // meaning; an explicit generation directory is also accepted.
            if base.file_name().and_then(|n| n.to_str()) == Some(generation) {
                base
            } else {
                base.join(generation)
            }
        } else {
            data_root.join("models").join(generation)
        };
        Ok(Self {
            root,
            runtime_root: data_root.join("runtimes"),
            policy,
            capabilities,
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
                if metadata.len() == 0
                    || file["bytes"]
                        .as_u64()
                        .is_some_and(|size| metadata.len() != size)
                {
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
    pub fn layout_quality(&self) -> PathBuf {
        self.root.join("auxiliary/layout_quality.onnx")
    }
    pub fn signature(&self) -> PathBuf {
        self.root.join("auxiliary/signature.onnx")
    }
    /// Optional models are enabled only by the declared role inventory.
    /// A stray file in the cache cannot silently change inference behavior.
    fn optional_auxiliary(&self, path: &str) -> Option<PathBuf> {
        self.catalog["files"]["auxiliary"]
            .as_array()?
            .iter()
            .any(|file| file["path"].as_str() == Some(path))
            .then(|| self.root.join(path))
    }
    pub fn chord_evidence(&self) -> Option<PathBuf> {
        self.optional_auxiliary("auxiliary/chord_evidence.onnx")
    }
    pub fn structure_relations(&self) -> Option<PathBuf> {
        self.optional_auxiliary("auxiliary/structure_relations.onnx")
    }
    pub fn staff_evidence(&self) -> Option<PathBuf> {
        self.optional_auxiliary("auxiliary/staff_evidence.onnx")
    }
    pub fn staff_label(&self) -> Option<PathBuf> {
        (auxiliary_asset(&self.capabilities["staff_label_model"])
            == Some("auxiliary/staff_label.onnx"))
        .then(|| self.root.join("auxiliary/staff_label.onnx"))
    }
    pub fn instrument_text(&self) -> Option<PathBuf> {
        (auxiliary_asset(&self.capabilities["instrument_text_model"])
            == Some("auxiliary/instrument_text.onnx"))
        .then(|| self.root.join("auxiliary/instrument_text.onnx"))
    }
    pub fn part_continuation(&self) -> Option<PathBuf> {
        (auxiliary_asset(&self.capabilities["part_continuation_model"])
            == Some("auxiliary/part_continuation.onnx"))
        .then(|| self.root.join("auxiliary/part_continuation.onnx"))
    }
    pub fn clef_spatial(&self) -> Option<PathBuf> {
        (auxiliary_asset(&self.capabilities["clef_spatial_model"])
            == Some("auxiliary/clef_spatial.onnx"))
        .then(|| self.root.join("auxiliary/clef_spatial.onnx"))
    }
    pub fn chord_text_roles(&self) -> Option<PathBuf> {
        self.optional_auxiliary("auxiliary/chord_text_roles.onnx")
    }
    pub fn text_regions(&self) -> Option<PathBuf> {
        self.optional_auxiliary("auxiliary/text_regions.onnx")
    }
    pub fn model(&self) -> PathBuf {
        self.root.join("weights/score_ocr/gguf/model-Q8_0.gguf")
    }
    pub fn projector(&self) -> PathBuf {
        self.root.join("weights/score_ocr/gguf/vision-F16.gguf")
    }
    pub fn information_model(&self) -> PathBuf {
        self.root.join("weights/information/gguf/model-Q8_0.gguf")
    }
    pub fn information_projector(&self) -> PathBuf {
        self.root.join("weights/information/gguf/vision-F16.gguf")
    }
    pub fn information_adapter(&self) -> Option<PathBuf> {
        Some(
            self.root
                .join("weights/information/gguf/information-adapter.gguf"),
        )
    }
    pub fn information_adapter_tasks(&self) -> Vec<String> {
        self.capabilities["information_model"]["adapter_tasks"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    }
    pub fn download_bytes(&self) -> u64 {
        self.files()
            .iter()
            .filter(|file| {
                let existing = file["path"]
                    .as_str()
                    .and_then(|path| self.root.join(path).metadata().ok())
                    .map(|m| m.len());
                existing != file["bytes"].as_u64()
            })
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
