//! Import the version-1 portable ZIP format emitted by `pipeline/archive.py`.
//!
//! JSON objects are kept as `serde_json::Value` so newly added score fields survive
//! an import. `session.json` is written last; an unsuccessful import removes only
//! the new directory it created, never an existing session.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use unicode_casefold::UnicodeCaseFold;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_PAGES: usize = 100;
const MAX_FILES: usize = 25_000;
const MAX_MANIFEST_BYTES: u64 = 20 * 1024 * 1024;
const PATH_FIELDS: &[&str] = &[
    "image",
    "source_pdf",
    "source_page",
    "inputs",
    "layout",
    "info",
    "recognition",
    "export",
    "m2",
    "score_text",
    "score_document",
    "musicxml",
    "structure_predictions",
    "previous_image",
    "next_image",
    "recognition_log",
    "predictions",
    "gp5",
    "encoding_report",
];
const STAGES: &[(&str, &str)] = &[
    ("layout", "layout"),
    ("info", "document_info"),
    ("recognition", "measure_ocr"),
    ("export", "gp5_export"),
];

/// Caller storage/page policy, capped at the portable format's hard limits.
#[derive(Clone, Debug)]
pub struct ImportOptions {
    /// A fresh lowercase 32-digit hexadecimal ID, or None to generate one.
    pub session_id: Option<String>,
    pub max_bytes: u64,
    pub max_pages: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            session_id: None,
            max_bytes: MAX_BYTES,
            max_pages: MAX_PAGES,
        }
    }
}

#[derive(Debug)]
pub struct ImportedProject {
    pub session_id: String,
    /// Absolute project directory inside the caller-selected workspace.
    pub directory: PathBuf,
    /// The exact persisted state, with portable references remapped.
    pub session: Value,
    pub imported_files: usize,
}

#[derive(Debug)]
pub enum ImportError {
    Invalid(String),
    Io(io::Error),
    Zip(zip::result::ZipError),
    Json(serde_json::Error),
}
impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Io(error) => write!(f, "Project I/O: {error}"),
            Self::Zip(error) => write!(f, "Invalid project ZIP: {error}"),
            Self::Json(error) => write!(f, "Invalid project JSON: {error}"),
        }
    }
}
impl std::error::Error for ImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Zip(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for ImportError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<zip::result::ZipError> for ImportError {
    fn from(e: zip::result::ZipError) -> Self {
        Self::Zip(e)
    }
}
impl From<serde_json::Error> for ImportError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
type Result<T> = std::result::Result<T, ImportError>;
fn invalid(message: impl Into<String>) -> ImportError {
    ImportError::Invalid(message.into())
}

/// Restore a project using only Rust. No model, Python, HTTP server, or Tauri is needed.
///
/// The workspace is created if absent. Existing session directories (including
/// broken symlinks) are never overwritten. An import error cleans up the newly
/// created directory. A trusted caller must keep the workspace private to this
/// process while importing; this is not a defense against concurrent local users
/// replacing directories. ZIP64, split, self-extracting, encrypted, and non-portable
/// Windows filenames are rejected intentionally; the Python exporter emits none
/// of these for projects within the import limits.
pub fn import_project(
    source: impl AsRef<Path>,
    workspace: impl AsRef<Path>,
    options: &ImportOptions,
) -> Result<ImportedProject> {
    let sid = match &options.session_id {
        Some(id) => id.clone(),
        None => {
            let mut bytes = [0u8; 16];
            getrandom::fill(&mut bytes).map_err(|e| invalid(format!("Session ID entropy: {e}")))?;
            bytes.iter().map(|byte| format!("{byte:02x}")).collect()
        }
    };
    if sid.len() != 32
        || !sid
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid(
            "Invalid project ID: expected 32 lowercase hexadecimal digits",
        ));
    }
    let limit = options.max_bytes.min(MAX_BYTES);
    let mut input = File::open(source)?;
    let (count, directory_start) = preflight_zip(&mut input, limit)?;
    let mut archive = ZipArchive::new(input)?;
    // zip's name lookup collapses exact duplicate names. Compare to the raw
    // directory count before inspecting its otherwise trustworthy entry metadata.
    if archive.len() != count
        || archive.central_directory_start() != directory_start
        || archive.offset() != 0
    {
        return Err(invalid("Duplicate files or inconsistent ZIP directory"));
    }
    let mut names = HashSet::new();
    let mut folded = HashMap::new();
    let mut total = 0u64;
    for index in 0..archive.len() {
        let file = archive.by_index(index)?;
        let name = file.name();
        safe_relative(name)?;
        register_path(name, &mut folded)?;
        if !names.insert(name.to_owned()) {
            return Err(invalid("Duplicate project file"));
        }
        if ["session.json", "job.json"].contains(&name.to_lowercase().as_str()) {
            return Err(invalid("Reserved project filename"));
        }
        let kind = file.unix_mode().unwrap_or(0) & 0o170000;
        if file.is_dir() || file.is_symlink() || !matches!(kind, 0 | 0o100000) || file.encrypted() {
            return Err(invalid(
                "Project ZIP must contain regular, unencrypted files only",
            ));
        }
        total = total
            .checked_add(file.size())
            .ok_or_else(|| invalid("Project ZIP size overflow"))?;
        if total > limit {
            return Err(invalid("Project ZIP exceeds the uncompressed byte limit"));
        }
    }
    // A file cannot also be an ancestor directory on any supported filesystem.
    for name in &names {
        for (index, _) in name.match_indices('/') {
            if names.contains(&name[..index]) {
                return Err(invalid("Project file/directory collision"));
            }
        }
    }
    let manifest = read_entry(&mut archive, "project.json", MAX_MANIFEST_BYTES.min(limit))?;
    let package: Value = serde_json::from_slice(&manifest)?;
    if package.get("format").and_then(Value::as_str) != Some("guitarocr-project")
        || package.get("version").and_then(Value::as_u64) != Some(1)
    {
        return Err(invalid("Expected GuitarOCR project format version 1"));
    }
    let files = package
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Project file list must be an array"))?;
    let mut state = package
        .get("session")
        .filter(|s| s.is_object())
        .cloned()
        .ok_or_else(|| invalid("Project session must be an object"))?;
    let pages = state
        .get("pages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Project pages must be an array"))?;
    let page_count = pages.len();
    if page_count == 0 || page_count > options.max_pages.min(MAX_PAGES) {
        return Err(invalid("Project page count exceeds the allowed range"));
    }
    validate_page_references(&state, page_count)?;
    for (key, _) in STAGES {
        if let Some(reference) = optional_reference(&state, key)? {
            if !reference.to_ascii_lowercase().ends_with(".json") {
                return Err(invalid(
                    "Project stage manifests must use a .json extension",
                ));
            }
        }
    }
    let mut entries = Vec::with_capacity(files.len());
    let mut listed = HashSet::new();
    for item in files {
        let path = item
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("Project file entry has no path"))?;
        safe_relative(path)?;
        let hash = item
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("Project file entry has no SHA-256"))?;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(invalid("Invalid project SHA-256"));
        }
        if path == "project.json" || !listed.insert(path.to_owned()) {
            return Err(invalid("Duplicate or reserved entry in project file list"));
        }
        entries.push((path, hash));
    }
    let mut expected = listed.clone();
    expected.insert("project.json".to_owned());
    if expected != names {
        return Err(invalid("Project ZIP members do not match its manifest"));
    }
    // Old native exports included model/device/input fingerprints. They are not
    // portable results or paths and cannot be resumed after importing a project.
    // Keep verifying their archive bytes, but neither restore nor reference them.
    let portable_files: HashSet<String> = listed
        .iter()
        .filter(|name| !machine_checkpoint(Path::new(name)))
        .cloned()
        .collect();

    fs::create_dir_all(workspace.as_ref())?;
    let workspace = fs::canonicalize(workspace)?;
    let root = workspace.join(&sid);
    // create_dir is exclusive even for broken symlinks; never remove a preexisting
    // directory on the error path.
    fs::create_dir(&root)?;
    let mut cleanup = NewProject {
        root: root.clone(),
        committed: false,
    };
    let operation = (|| {
        for (name, expected_hash) in &entries {
            let contents = read_entry(&mut archive, name, limit)?;
            let hash = format!("{:x}", Sha256::digest(&contents));
            if &hash != expected_hash {
                return Err(invalid(format!("Project SHA-256 mismatch: {name}")));
            }
            if machine_checkpoint(Path::new(name)) {
                continue;
            }
            let contents = transform_file(&contents, name, &root, &portable_files, page_count)?;
            let path = root.join(name);
            fs::create_dir_all(path.parent().expect("validated project path has a parent"))?;
            let mut out = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            out.write_all(&contents)?;
        }
        remap(&mut state, "", &root, &portable_files)?;
        state["id"] = Value::String(sid.clone());
        state["revision"] = Value::from(0);
        state
            .as_object_mut()
            .expect("validated object")
            .remove("ocr_task");
        validate_saved_state(&state, page_count)?;
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("session.json"))?;
        serde_json::to_writer_pretty(&mut out, &state)?;
        out.write_all(b"\n")?;
        out.sync_all()?;
        Ok(())
    })();
    if let Err(error) = operation {
        // Report cleanup failures rather than claiming a rollback we could not do.
        if let Err(cleanup_error) = fs::remove_dir_all(&root) {
            return Err(invalid(format!(
                "{error}; could not remove incomplete project {}: {cleanup_error}",
                root.display()
            )));
        }
        cleanup.committed = true;
        return Err(error);
    }
    cleanup.committed = true;
    Ok(ImportedProject {
        session_id: sid,
        directory: root,
        session: state,
        imported_files: portable_files.len(),
    })
}

struct NewProject {
    root: PathBuf,
    committed: bool,
}
impl Drop for NewProject {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn safe_relative(name: &str) -> Result<()> {
    if name.is_empty() || name.contains(['\\', ':', '\0']) || name.starts_with('/') {
        return Err(invalid(format!("Invalid project path: {name:?}")));
    }
    for component in name.split('/') {
        if component.is_empty()
            || component == "."
            || component == ".."
            || component.ends_with(['.', ' '])
            || component
                .chars()
                .any(|c| c.is_control() || "<>\"|?*".contains(c))
        {
            return Err(invalid(format!("Non-portable project path: {name:?}")));
        }
        let stem = component
            .split('.')
            .next()
            .unwrap_or("")
            .to_ascii_uppercase();
        if ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
            || (stem.len() == 4
                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        {
            return Err(invalid(format!("Reserved device path: {name:?}")));
        }
    }
    Ok(())
}

fn register_path(name: &str, folded: &mut HashMap<String, String>) -> Result<()> {
    for end in name
        .match_indices('/')
        .map(|(i, _)| i)
        .chain(std::iter::once(name.len()))
    {
        let prefix = &name[..end];
        let key: String = prefix.to_lowercase().case_fold().collect();
        if let Some(previous) = folded.insert(key, prefix.to_owned()) {
            if previous != prefix {
                return Err(invalid("Case-aliased project paths"));
            }
        }
    }
    Ok(())
}

fn read_entry(archive: &mut ZipArchive<File>, name: &str, limit: u64) -> Result<Vec<u8>> {
    let mut file = archive.by_name(name)?;
    if file.size() > limit {
        return Err(invalid(format!("Project entry exceeds byte limit: {name}")));
    }
    let size = file.size();
    let mut contents = Vec::new();
    (&mut file).take(size + 1).read_to_end(&mut contents)?;
    if contents.len() as u64 != size {
        return Err(invalid("ZIP entry has inconsistent uncompressed size"));
    }
    Ok(contents)
}

fn remap(value: &mut Value, key: &str, root: &Path, files: &HashSet<String>) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                remap(value, key, root, files)?;
            }
        }
        Value::Array(array) => {
            for value in array {
                remap(value, key, root, files)?;
            }
        }
        Value::String(text) => {
            if let Some(relative) = text.strip_prefix("project://") {
                safe_relative(relative)?;
                if !files.contains(relative) {
                    return Err(invalid(format!("Missing project reference: {relative}")));
                }
                *text = root
                    .join(relative)
                    .to_str()
                    .ok_or_else(|| invalid("Workspace path is not valid Unicode"))?
                    .to_owned();
            } else if !text.is_empty() && PATH_FIELDS.contains(&key) {
                return Err(invalid(format!(
                    "Project path must refer inside its ZIP: {key}"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn transform_file(
    contents: &[u8],
    name: &str,
    root: &Path,
    files: &HashSet<String>,
    pages: usize,
) -> Result<Vec<u8>> {
    let suffix = Path::new(name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if suffix != "json" && suffix != "jsonl" {
        return Ok(contents.to_owned());
    }
    let mut result = Vec::new();
    let records: Vec<&[u8]> = if suffix == "json" {
        vec![contents]
    } else {
        contents.split(|c| *c == b'\n' || *c == b'\r').collect()
    };
    for record in records {
        if suffix == "jsonl" && record.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let mut value: Value = serde_json::from_slice(record)?;
        validate_page_references(&value, pages)?;
        remap(&mut value, "", root, files)?;
        serde_json::to_writer(&mut result, &value)?;
        result.push(b'\n');
    }
    Ok(result)
}

fn validate_page_references(value: &Value, pages: usize) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "page" && !value.as_u64().is_some_and(|n| n >= 1 && n <= pages as u64) {
                    return Err(invalid(
                        "Invalid page reference: expected an integer within the project",
                    ));
                }
                validate_page_references(value, pages)?;
            }
        }
        Value::Array(array) => {
            for value in array {
                validate_page_references(value, pages)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn optional_reference<'a>(state: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match state.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        _ => Err(invalid(format!(
            "Project stage reference must be a string: {key}"
        ))),
    }
}

fn require_file(value: &Value, key: &str) -> Result<()> {
    let path = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("Missing project {key}")))?;
    if !Path::new(path).is_file() {
        return Err(invalid(format!("Project file does not exist: {key}")));
    }
    Ok(())
}

fn validate_saved_state(state: &Value, pages: usize) -> Result<()> {
    for page in state["pages"].as_array().expect("validated pages") {
        require_file(page, "image")?;
    }
    for (key, stage) in STAGES {
        if let Some(path) = optional_reference(state, key)? {
            let result: Value = serde_json::from_reader(File::open(path)?)?;
            if result.get("schema_version").and_then(Value::as_str) != Some("1.0")
                || result.get("stage").and_then(Value::as_str) != Some(*stage)
            {
                return Err(invalid(format!(
                    "Expected a {stage} stage manifest (schema 1.0)"
                )));
            }
            validate_page_references(&result, pages)?;
            if *key == "recognition" {
                let records = result
                    .get("records")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid("Recognition records must be an array"))?;
                for row in records {
                    require_file(row, "image")?;
                    let target = row
                        .get("target")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("Recognition record has no M2 target"))?;
                    crate::score::parse_measure_target(target).map_err(invalid)?;
                }
            }
        }
    }
    Ok(())
}

/// Bound the classic ZIP directory before the ZIP crate allocates its entry list.
/// Also retain the physical count, since ZIP name lookup deduplicates exact names.
fn preflight_zip(file: &mut File, limit: u64) -> Result<(usize, u64)> {
    let length = file.metadata()?.len();
    if length < 22 {
        return Err(invalid("Truncated project ZIP"));
    }
    let tail_len = length.min(22 + 65_535) as usize;
    file.seek(SeekFrom::End(-(tail_len as i64)))?;
    let mut tail = vec![0; tail_len];
    file.read_exact(&mut tail)?;
    let end = (0..=tail_len - 22)
        .rev()
        .find(|&i| {
            tail[i..i + 4] == *b"PK\x05\x06" && i + 22 + u16_at(&tail, i + 20) as usize == tail_len
        })
        .ok_or_else(|| invalid("Missing ZIP end record"))?;
    let disk = u16_at(&tail, end + 4);
    let cd_disk = u16_at(&tail, end + 6);
    let on_disk = u16_at(&tail, end + 8);
    let count = u16_at(&tail, end + 10) as usize;
    let size = u32_at(&tail, end + 12) as u64;
    let offset = u32_at(&tail, end + 16) as u64;
    let end_offset = length - tail_len as u64 + end as u64;
    if disk != 0
        || cd_disk != 0
        || on_disk as usize != count
        || count == 0
        || count > MAX_FILES
        || size == u32::MAX as u64
        || offset == u32::MAX as u64
        || offset.checked_add(size) != Some(end_offset)
    {
        return Err(invalid("Unsupported ZIP layout or too many project files (ZIP64 and split ZIPs are not supported)"));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut total = 0u64;
    for _ in 0..count {
        let mut header = [0u8; 46];
        file.read_exact(&mut header)?;
        if &header[..4] != b"PK\x01\x02" {
            return Err(invalid("Invalid ZIP central directory"));
        }
        let kind = (u32_at(&header, 38) >> 16) & 0o170000;
        if !matches!(kind, 0 | 0o100000) {
            return Err(invalid(
                "Project ZIP contains a link, directory, or special file",
            ));
        }
        let size = u32_at(&header, 24) as u64;
        total = total
            .checked_add(size)
            .ok_or_else(|| invalid("Project size overflow"))?;
        if total > limit {
            return Err(invalid("Project ZIP exceeds the uncompressed byte limit"));
        }
        let extra =
            u16_at(&header, 28) as u64 + u16_at(&header, 30) as u64 + u16_at(&header, 32) as u64;
        let position = file
            .stream_position()?
            .checked_add(extra)
            .ok_or_else(|| invalid("ZIP offset overflow"))?;
        if position > end_offset {
            return Err(invalid("Truncated ZIP central directory"));
        }
        file.seek(SeekFrom::Start(position))?;
    }
    if file.stream_position()? != end_offset {
        return Err(invalid("ZIP directory count does not match its entries"));
    }
    file.rewind()?;
    Ok((count, offset))
}
fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed header"))
}

/// Load a persisted session without reducing it to a fixed schema.
pub fn load_session(workspace: impl AsRef<Path>, sid: &str) -> Result<Value> {
    let root = existing_session_directory(workspace.as_ref(), sid)?;
    let path = root.join("session.json");
    if !fs::symlink_metadata(&path)?.file_type().is_file() {
        return Err(invalid("Session must be a regular JSON file"));
    }
    let state: Value = serde_json::from_reader(File::open(path)?)?;
    check_session_identity(&state, sid)?;
    Ok(state)
}

/// Atomically replace session.json. Revision comparison/locking belongs to the
/// caller; this function never invents a revision or drops unknown fields.
pub fn write_session_atomic(workspace: impl AsRef<Path>, sid: &str, state: &Value) -> Result<()> {
    check_session_identity(state, sid)?;
    let root = existing_session_directory(workspace.as_ref(), sid)?;
    let path = root.join("session.json");
    if let Ok(metadata) = fs::symlink_metadata(&path) {
        if !metadata.file_type().is_file() {
            return Err(invalid("Session must be a regular JSON file"));
        }
    }
    atomic_write(&path, |file| {
        serde_json::to_writer_pretty(&mut *file, state)?;
        file.write_all(b"\n")?;
        Ok(())
    })
}

/// Export the Python-compatible project ZIP, preserving all JSON fields and
/// rewriting references below the session root to project:// paths.
///
/// Source files are never changed. Machine checkpoint state, job/session files,
/// symlinks, tmp directories, .tmp files, and ZIPs are excluded. Only an
/// interrupted final JSONL record without a final LF can be discarded, matching
/// the Python exporter's journal recovery. Existing destination files are
/// atomically replaced after the ZIP is finished successfully.
pub fn export_project(
    workspace: impl AsRef<Path>,
    sid: &str,
    destination: impl AsRef<Path>,
) -> Result<PathBuf> {
    let root = existing_session_directory(workspace.as_ref(), sid)?;
    let mut state = load_session(workspace, sid)?;
    state
        .as_object_mut()
        .expect("checked session object")
        .remove("ocr_task");
    encode_paths(&mut state, &root);
    let mut paths = Vec::new();
    collect_project_files(&root, &root, &mut paths)?;
    paths.sort();
    if paths.len() >= MAX_FILES {
        return Err(invalid("Too many project files"));
    }
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let destination = fs::canonicalize(parent)?.join(
        destination
            .file_name()
            .ok_or_else(|| invalid("ZIP destination has no filename"))?,
    );
    if destination.starts_with(&root) {
        return Err(invalid(
            "Export destination must be outside the project directory",
        ));
    }
    atomic_write(&destination, |file| {
        let mut zip = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        let mut files = Vec::with_capacity(paths.len());
        let mut total = 0u64;
        for path in &paths {
            let relative = path.strip_prefix(&root).expect("collected inside root");
            let name = relative
                .to_str()
                .ok_or_else(|| invalid("Project filename is not valid Unicode"))?
                .replace('\\', "/");
            safe_relative(&name)?;
            zip.start_file(&name, options)?;
            let mut hash = Sha256::new();
            let suffix = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if suffix == "json" || suffix == "jsonl" {
                let contents = encode_json_file(&fs::read(path)?, &suffix, &root)?;
                total = total
                    .checked_add(contents.len() as u64)
                    .ok_or_else(|| invalid("Project size overflow"))?;
                if total > MAX_BYTES {
                    return Err(invalid("Project exceeds the portable ZIP size limit"));
                }
                hash.update(&contents);
                zip.write_all(&contents)?;
            } else {
                let mut input = File::open(path)?;
                let mut buffer = [0u8; 64 * 1024];
                loop {
                    let length = input.read(&mut buffer)?;
                    if length == 0 {
                        break;
                    }
                    total = total
                        .checked_add(length as u64)
                        .ok_or_else(|| invalid("Project size overflow"))?;
                    if total > MAX_BYTES {
                        return Err(invalid("Project exceeds the portable ZIP size limit"));
                    }
                    hash.update(&buffer[..length]);
                    zip.write_all(&buffer[..length])?;
                }
            }
            files.push(serde_json::json!({"path":name,"sha256":format!("{:x}",hash.finalize())}));
        }
        let manifest = serde_json::to_vec(
            &serde_json::json!({"format":"guitarocr-project", "version":1,"session":state,"files":files}),
        )?;
        if manifest.len() as u64 > MAX_MANIFEST_BYTES || total + manifest.len() as u64 > MAX_BYTES {
            return Err(invalid(
                "Project manifest or total ZIP content exceeds import limits",
            ));
        }
        zip.start_file("project.json", options)?;
        zip.write_all(&manifest)?;
        zip.finish()?;
        Ok(())
    })?;
    Ok(destination)
}

fn check_session_identity(state: &Value, sid: &str) -> Result<()> {
    if !state.is_object() || state.get("id").and_then(Value::as_str) != Some(sid) {
        return Err(invalid("Session JSON has a mismatched project ID"));
    }
    Ok(())
}
fn existing_session_directory(workspace: &Path, sid: &str) -> Result<PathBuf> {
    if sid.len() != 32
        || !sid
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid(
            "Invalid project ID: expected 32 lowercase hexadecimal digits",
        ));
    }
    let root = fs::canonicalize(workspace)?.join(sid);
    if !fs::symlink_metadata(&root)?.file_type().is_dir() {
        return Err(invalid(
            "Project root must be an existing directory, not a symlink",
        ));
    }
    Ok(root)
}
fn atomic_write(path: &Path, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|e| invalid(format!("Temporary file entropy: {e}")))?;
    let nonce: String = random.iter().map(|b| format!("{b:02x}")).collect();
    let temporary = path.with_file_name(format!(".guitarocr-{nonce}.tmp"));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    let result = (|| {
        write(&mut file)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != io::ErrorKind::NotFound {
                return Err(invalid(format!(
                    "{result:?}; failed to remove temporary file: {error}"
                )));
            }
        }
    }
    result
}
fn machine_checkpoint(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("recognition_context.json"))
}

fn collect_project_files(root: &Path, directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        let relative = path.strip_prefix(root).expect("inside project root");
        if kind.is_symlink()
            || machine_checkpoint(&path)
            || relative.components().any(|c| c.as_os_str() == "tmp")
        {
            continue;
        }
        if kind.is_dir() {
            collect_project_files(root, &path, paths)?;
        } else if kind.is_file()
            && !["session.json", "job.json"]
                .iter()
                .any(|n| path.file_name().is_some_and(|name| name == *n))
            && !["tmp", "zip"]
                .iter()
                .any(|n| path.extension().is_some_and(|ext| ext == *n))
        {
            paths.push(path);
        }
    }
    Ok(())
}
fn encode_paths(value: &mut Value, root: &Path) {
    match value {
        Value::Object(object) => {
            for value in object.values_mut() {
                encode_paths(value, root);
            }
        }
        Value::Array(array) => {
            for value in array {
                encode_paths(value, root);
            }
        }
        Value::String(text) => {
            if let Ok(relative) = Path::new(text).strip_prefix(root) {
                *text = format!(
                    "project://{}",
                    relative.to_string_lossy().replace('\\', "/")
                );
            }
        }
        _ => {}
    }
}
fn encode_json_file(contents: &[u8], suffix: &str, root: &Path) -> Result<Vec<u8>> {
    let records = if suffix == "json" {
        vec![contents]
    } else {
        jsonl_records(contents)
    };
    let mut output = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if suffix == "jsonl" && record.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let mut value: Value = match serde_json::from_slice(record) {
            Ok(value) => value,
            Err(_)
                if suffix == "jsonl" && index == records.len() - 1 && !record.ends_with(b"\n") =>
            {
                break
            }
            Err(error) => return Err(error.into()),
        };
        encode_paths(&mut value, root);
        serde_json::to_writer(&mut output, &value)?;
        output.push(b'\n');
    }
    Ok(output)
}
fn jsonl_records(contents: &[u8]) -> Vec<&[u8]> {
    let mut records = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < contents.len() {
        if contents[index] == b'\r' || contents[index] == b'\n' {
            if contents[index] == b'\r' && contents.get(index + 1) == Some(&b'\n') {
                index += 1;
            }
            records.push(&contents[start..index + 1]);
            start = index + 1;
        }
        index += 1;
    }
    if start < contents.len() {
        records.push(&contents[start..]);
    }
    records
}
