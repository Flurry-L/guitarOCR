//! HTTP projection of persisted stages. Filesystem paths never leave this layer.
use guitarocr_native_core::score::{display_error, display_score_text, parse_measure_target};
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
pub fn directory(root: &Path, sid: &str) -> Result<PathBuf> {
    if sid.len() != 32
        || !sid
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("无效的项目编号".into());
    }
    Ok(root.join(sid))
}
pub fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn stage(path: &Value, kind: &str, root: &Path) -> Result<Value> {
    let path = Path::new(path.as_str().ok_or("无效的阶段路径")?);
    let resolved = path.canonicalize().map_err(|e| e.to_string())?;
    if !resolved.starts_with(root) {
        return Err("项目引用超出目录".into());
    }
    let value = read_json(&resolved)?;
    if value["schema_version"] != "1.0" || value["stage"] != kind {
        return Err("不兼容的项目阶段".into());
    }
    Ok(value)
}
fn annotations(value: &Value) -> Value {
    Value::Array(
        value
            .as_array()
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        pick(
                            row,
                            &[
                                "kind",
                                "candidate_kind",
                                "parsed",
                                "page",
                                "bbox",
                                "part_id",
                            ],
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
    )
}
fn metadata(value: &Value) -> Value {
    let mut value = value.clone();
    for key in ["pitch_instructions", "score_annotations"] {
        if let Some(v) = value.get_mut(key) {
            *v = annotations(v);
        }
    }
    value
}
fn pick(value: &Value, keys: &[&str]) -> Value {
    Value::Object(
        keys.iter()
            .filter_map(|&key| value.get(key).map(|v| (key.to_owned(), v.clone())))
            .collect(),
    )
}
fn asset(root: &Path, sid: &str, value: &Value) -> Result<String> {
    let path = Path::new(value.as_str().ok_or("无效的资源路径")?)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let relative = path.strip_prefix(root).map_err(|_| "项目资源超出目录")?;
    let encoded = relative
        .components()
        .map(|p| {
            utf8_percent_encode(&p.as_os_str().to_string_lossy(), NON_ALPHANUMERIC).to_string()
        })
        .collect::<Vec<_>>()
        .join("/");
    Ok(format!("/api/sessions/{sid}/files/{encoded}"))
}
pub fn project(root: &Path, sid: &str) -> Result<Value> {
    let root = directory(root, sid)?
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let saved = read_json(&root.join("session.json"))?;
    let mut state = pick(&saved, &["id", "revision", "mode", "boxes", "input_names"]);
    if state["input_names"].is_null() {
        state["input_names"] = Value::Array(
            saved["inputs"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|v| v.as_str())
                .map(|p| {
                    json!(Path::new(p)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy())
                })
                .collect(),
        );
    }
    state["mode_setting"] = saved.get("mode_setting").unwrap_or(&saved["mode"]).clone();
    for key in ["layout", "info", "recognition", "export"] {
        state[key] = json!(saved[key].as_str().is_some_and(|s| !s.is_empty()));
    }
    if saved.get("ocr_task").is_some_and(|v| !v.is_null()) {
        state["ocr_task"] = json!(true);
    }
    let mut pages = Vec::new();
    for page in saved["pages"].as_array().ok_or("项目缺少页面")? {
        let mut page = page.clone();
        page["url"] = json!(asset(&root, sid, &page["image"])?);
        let obj = page.as_object_mut().ok_or("无效页面")?;
        obj.remove("image");
        obj.remove("source_pdf");
        pages.push(page);
    }
    state["pages"] = json!(pages);
    state["metadata"] = Value::Null;
    if state["info"] == true {
        let info = stage(&saved["info"], "document_info", &root)?;
        let mut meta = pick(
            &info,
            &[
                "title",
                "artist",
                "instrument",
                "midi_program",
                "tuning_used",
                "capo",
                "transpose",
                "document_metadata",
            ],
        );
        meta["document_metadata"] = metadata(&meta["document_metadata"]);
        if let Some(parts) = info["parts"].as_array() {
            meta["parts"] = json!(parts
                .iter()
                .map(|part| {
                    let mut p = pick(
                        part,
                        &[
                            "id",
                            "name",
                            "instrument",
                            "midi_program",
                            "tuning_used",
                            "capo",
                            "transpose",
                            "document_metadata",
                        ],
                    );
                    p["document_metadata"] = metadata(&p["document_metadata"]);
                    p
                })
                .collect::<Vec<_>>());
        }
        state["metadata"] = meta;
    }
    state["measures"] = json!([]);
    state["review_measures"] = json!([]);
    if state["recognition"] == true {
        let data = stage(&saved["recognition"], "measure_ocr", &root)?;
        if !state["metadata"].is_null() {
            state["metadata"]["tuning_used"] = data["tuning_used"].clone();
            state["metadata"]["document_metadata"] = metadata(&data["document_metadata"]);
        }
        state["review_measures"] = data.get("review_measures").cloned().unwrap_or(json!([]));
        state["score_text_url"] = json!(format!("/api/sessions/{sid}/score.txt"));
        state["score_document_url"] = json!(asset(&root, sid, &data["score_document"])?);
        if data["musicxml"].is_string() {
            state["musicxml_url"] = json!(asset(&root, sid, &data["musicxml"])?);
        }
        let rows = data["records"].as_array().ok_or("识别结果缺少records")?;
        if let Some(parts) = state["metadata"]["parts"].as_array_mut() {
            for part in parts {
                if let Some(resolved) = data["parts"]
                    .as_array()
                    .and_then(|parts| parts.iter().find(|p| p["id"] == part["id"]))
                {
                    part["document_metadata"] = metadata(&resolved["document_metadata"]);
                }
                if let Some(row) = rows.iter().find(|r| r["part_id"] == part["id"]) {
                    if let Some(tuning) = row.get("tuning") {
                        part["tuning_used"] = tuning.clone();
                    }
                }
            }
        }
        let mut measures = Vec::new();
        for row in rows {
            let mut measure = row.clone();
            let object = measure.as_object_mut().ok_or("无效的小节记录")?;
            for key in ["image", "source_page", "source_pdf"] {
                object.remove(key);
            }
            let target = row["target"].as_str().ok_or("小节缺少target")?;
            measure["parsed"] = parse_measure_target(target).map_err(|e| e.to_string())?;
            measure["score_text"] = json!(display_score_text(target));
            measure["context_text"] = row
                .get("previous_context")
                .filter(|v| v.is_string())
                .cloned()
                .unwrap_or(json!(""));
            let reasons = row["fallback_reason"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            measure["annotation_review"] = json!(
                !reasons.is_empty()
                    && reasons.iter().all(|r| r
                        .as_str()
                        .is_some_and(|r| r.starts_with("Uncertain chord symbol")
                            || r.starts_with("Annotation review failed")))
                    && [
                        "timing_errors",
                        "fingering_errors",
                        "export_errors",
                        "pitch_needs_review",
                        "state_needs_review"
                    ]
                    .iter()
                    .all(|key| match &row[*key] {
                        Value::Null | Value::Bool(false) => true,
                        Value::Array(v) => v.is_empty(),
                        _ => false,
                    })
            );
            measure["fallback_reason"] = json!(reasons
                .iter()
                .map(|r| display_error(r.as_str().unwrap_or_default()))
                .collect::<Vec<_>>());
            measure["url"] = json!(asset(&root, sid, &row["image"])?);
            measure["chord_annotations"] = annotations(&row["chord_annotations"]);
            measures.push(measure);
        }
        state["measures"] = json!(measures);
    }
    if state["export"] == true {
        let exported = stage(&saved["export"], "gp5_export", &root)?;
        for (key, url) in [
            ("gp5", "gp5_url"),
            ("encoding_report", "encoding_url"),
            ("musicxml", "musicxml_url"),
        ] {
            if exported[key].is_string() {
                state[url] = json!(asset(&root, sid, &exported[key])?);
            }
        }
        if let Some(path) = exported["gp5"].as_str() {
            state["gp5_name"] = json!(Path::new(path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy());
        }
    }
    state["job"] = Value::Null;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_demo_projects_through_native_api_without_paths_or_python() {
        let workspace = tempfile::tempdir().unwrap();
        let demo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../webapp/static/examples/Harbor-Light-synthetic-project.zip");
        let imported = guitarocr_native_core::project::import_project(
            demo,
            workspace.path(),
            &Default::default(),
        )
        .unwrap();
        let view = project(workspace.path(), &imported.session_id).unwrap();
        assert_eq!(view["measures"].as_array().unwrap().len(), 16);
        assert_eq!(view["metadata"]["parts"].as_array().unwrap().len(), 2);
        assert!(view["measures"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["parsed"]["voices"].is_array()));
        assert!(!view
            .to_string()
            .contains(&workspace.path().to_string_lossy().to_string()));
        assert!(view["pages"][0]["url"]
            .as_str()
            .unwrap()
            .starts_with("/api/sessions/"));
    }
}
