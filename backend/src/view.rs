//! HTTP projection of persisted stages. Filesystem paths never leave this layer.
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use scorelib::score::record_measure;
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
                        let mut projected = pick(
                            row,
                            &[
                                "kind",
                                "candidate_kind",
                                "parsed",
                                "chord_review",
                                "page",
                                "bbox",
                                "part_id",
                                "scope",
                                "measure_number",
                            ],
                        );
                        if row["chord_evidence"].is_object() {
                            projected["chord_evidence"] = pick(
                                &row["chord_evidence"],
                                &[
                                    "schema",
                                    "reader",
                                    "status",
                                    "kind",
                                    "kind_probabilities",
                                    "strings",
                                    "base_fret",
                                    "relative_frets",
                                    "relative_barres",
                                    "diagram",
                                    "confidence",
                                    "accepted",
                                    "review_required",
                                    "name_needs_review",
                                    "original_diagram",
                                    "proposed_diagram",
                                    "rejected_diagram_candidate",
                                    "conflict_fields",
                                    "proposed_changed_fields",
                                    "changed_fields",
                                    "agreement",
                                    "secondary_diagram",
                                    "secondary_confidence",
                                ],
                            );
                        }
                        projected
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
        let mut predictions = Vec::new();
        for source in std::iter::once(&meta).chain(meta["parts"].as_array().into_iter().flatten()) {
            for key in ["score_annotations", "pitch_instructions"] {
                predictions.extend(
                    source["document_metadata"][key]
                        .as_array()
                        .into_iter()
                        .flatten(),
                );
            }
        }
        if let Some(boxes) = state["boxes"].as_array_mut() {
            for b in boxes.iter_mut().filter(|b| {
                matches!(b["kind"].as_str(), Some("annotation" | "transposition"))
                    && b["annotation_type_source"] != "manual"
            }) {
                if let Some(p) = predictions
                    .iter()
                    .find(|p| p["page"] == b["page"] && p["bbox"] == b["bbox"])
                {
                    b["annotation_type"] = p["parsed"]["kind"].clone();
                    b["annotation_text"] = p["parsed"]["text"].clone();
                    b["annotation_type_source"] = json!("information_model");
                }
            }
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
        if data["score_text"].is_string() {
            state["score_text_url"] = json!(format!("/api/sessions/{sid}/score.txt"));
        }
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
            measure["parsed"] = record_measure(row)?;
            if measure["parsed"]["incomplete"] == true {
                // An empty editing voice has no beat, note, or rest.
                measure["parsed"]["voices"] = json!([{"voice":0,"events":[]}]);
            }
            measure["score_text"] = json!(row["target"].as_str().unwrap_or(""));
            measure["url"] = json!(asset(&root, sid, &row["image"])?);
            measure["chord_annotations"] = annotations(&row["chord_annotations"]);
            measure["technique_annotations"] = annotations(&row["technique_annotations"]);
            measures.push(measure);
        }
        state["measures"] = json!(measures);
    }
    if state["export"] == true {
        let exported = stage(&saved["export"], "gp5_export", &root)?;
        state["export_warnings"] = exported.get("warnings").cloned().unwrap_or(json!([]));
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
