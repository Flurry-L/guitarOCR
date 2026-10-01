//! Revision-controlled native writes. Accepted results are replaced only after
//! every required projection has been written successfully.
use guitarocr_native_core::{
    project::{load_session, write_session_atomic},
    score::{
        display_score_text, format_measure_target, model_score_text, score_document,
        validate_measure_target,
    },
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
pub fn revision(saved: &Value, expected: Option<&str>) -> std::result::Result<(), (u16, String)> {
    let expected = expected.ok_or((428, "项目版本缺失，请重新打开项目".into()))?;
    if expected.trim_matches('"').parse::<u64>().ok() != saved["revision"].as_u64() {
        return Err((409, "项目已经更新，请重新加载后再保存".into()));
    }
    Ok(())
}
pub fn json_file(path: &Path, value: &Value) -> Result<()> {
    let mut contents = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    contents.push(b'\n');
    fs::write(path, contents).map_err(|e| e.to_string())
}
pub fn save_recognition(output: &Path, source: &Value) -> Result<PathBuf> {
    let mut result = source.clone();
    let rows = result["records"].as_array().ok_or("识别结果缺少小节")?;
    let text = rows
        .iter()
        .map(|r| {
            r["target"]
                .as_str()
                .ok_or_else(|| "无效的小节target".to_string())
        })
        .collect::<Result<Vec<_>>>()?
        .join("\n")
        + "\n";
    let review: Vec<_> = rows
        .iter()
        .filter(|r| r["needs_review"] == true)
        .map(|r| r["measure_number"].clone())
        .collect();
    result["measures"] = json!(rows.len());
    result["status"] = json!(if review.is_empty() {
        "complete"
    } else {
        "needs_review"
    });
    result["review_measures"] = json!(review);
    fs::write(output.join("prediction.m2"), &text).map_err(|e| e.to_string())?;
    fs::write(output.join("score.txt"), display_score_text(&text)).map_err(|e| e.to_string())?;
    result["m2"] = json!(output.join("prediction.m2"));
    result["score_text"] = json!(output.join("score.txt"));
    let document = score_document(&result)?;
    json_file(&output.join("score.json"), &document)?;
    result["score_document"] = json!(output.join("score.json"));
    let obj = result.as_object_mut().ok_or("无效的识别结果")?;
    obj.remove("musicxml");
    obj.remove("musicxml_error");
    // MusicXML is an optional projection; an unrepresentable score must retain
    // its canonical records and remain editable, matching the research pipeline.
    match guitarocr_native_core::music_exports::write_musicxml(
        &document,
        &output.join("score.musicxml"),
    ) {
        Ok(_) => result["musicxml"] = json!(output.join("score.musicxml")),
        Err(error) => {
            result["musicxml_error"] = json!(error.to_string());
            let _ = fs::remove_file(output.join("score.musicxml"));
        }
    }
    result["schema_version"] = json!("1.0");
    result["stage"] = json!("measure_ocr");
    let manifest = output.join("manifest.json");
    json_file(&manifest, &result)?;
    Ok(manifest)
}
pub fn correct(
    root: &Path,
    sid: &str,
    number: usize,
    body: &Value,
    expected: Option<&str>,
) -> std::result::Result<(), (u16, String)> {
    let mut state = load_session(root, sid).map_err(|e| (400, e.to_string()))?;
    revision(&state, expected)?;
    let path = state["recognition"]
        .as_str()
        .ok_or((409, "请先识别小节".into()))?;
    let mut source: Value =
        serde_json::from_slice(&fs::read(path).map_err(|e| (400, e.to_string()))?)
            .map_err(|e| (400, e.to_string()))?;
    correct_record(&mut source, number, body).map_err(|e| (400, e))?;
    let directory = root.join(sid);
    let output = tempfile::Builder::new()
        .prefix("correction_")
        .tempdir_in(&directory)
        .map_err(|e| (400, e.to_string()))?;
    let manifest = save_recognition(output.path(), &source).map_err(|e| (400, e))?;
    state["recognition"] = json!(manifest);
    state["export"] = Value::Null;
    state["revision"] = json!(
        state["revision"]
            .as_u64()
            .ok_or((400, "无效的项目版本".into()))?
            + 1
    );
    state.as_object_mut().unwrap().remove("ocr_task");
    write_session_atomic(root, sid, &state).map_err(|e| (400, e.to_string()))?;
    let _ = output.keep();
    let _ = fs::remove_file(directory.join("job.json"));
    Ok(())
}
fn correct_record(source: &mut Value, number: usize, body: &Value) -> Result<()> {
    let source_mode = source["mode"].as_str().unwrap_or("tab").to_owned();
    let source_tuning = source["tuning_used"].clone();
    let source_instrument = source["instrument"].as_str().unwrap_or("guitar").to_owned();
    let rows = source["records"].as_array_mut().ok_or("无效的识别结果")?;
    if number == 0 || number > rows.len() {
        return Err("无效的小节编号".into());
    }
    let row = &mut rows[number - 1];
    let mode = row["mode"].as_str().unwrap_or(&source_mode);
    let reviewed = body["reviewed"].as_bool().unwrap_or(false);
    let old = row["target"].as_str().ok_or("小节缺少target")?.to_owned();
    let target = if let Some(measure) = body.get("measure").filter(|v| !v.is_null()) {
        format_measure_target(measure, mode, true)?
    } else if let Some(text) = body["target"].as_str() {
        text.to_owned()
    } else if reviewed {
        old.clone()
    } else {
        return Err("请输入小节内容或核对标记".into());
    };
    if target.len() > 50000 || target.contains('\n') {
        return Err("请输入单个小节的内容".into());
    }
    let target = model_score_text(&target);
    let tuning: Vec<_> = row
        .get("tuning")
        .unwrap_or(&source_tuning)
        .as_array()
        .ok_or("无效的调弦")?
        .iter()
        .map(|v| v.as_i64().ok_or("无效的调弦".to_string()))
        .collect::<Result<_>>()?;
    let (parsed, errors) = validate_measure_target(&target, mode, Some(&tuning), None);
    if !errors.is_empty() {
        return Err(errors.join("；"));
    }
    let timing = guitarocr_native_core::gp5_constraints::timing_errors(&target)?;
    if !timing.is_empty() {
        return Err(timing.join("；"));
    }
    let instrument = row["instrument"].as_str().unwrap_or(&source_instrument);
    if mode == "notation"
        && matches!(instrument, "guitar" | "bass")
        && row["tuning_explicit"] == true
    {
        let errors = guitarocr_native_core::gp5_constraints::notation_fingering_errors(
            &parsed.ok_or("无效的小节")?,
            &tuning,
        )?;
        if !errors.is_empty() {
            return Err(errors.join("；"));
        }
    }
    let changed = target != old;
    row["target"] = json!(target);
    row["manually_edited"] = json!(changed || row["manually_edited"] == true);
    for key in ["timing_errors", "fingering_errors", "export_errors"] {
        row[key] = json!([]);
    }
    if reviewed {
        row["needs_review"] = json!(false);
        row["reviewed"] = json!(true);
        row["pitch_needs_review"] = json!(false);
    } else if changed {
        row["reviewed"] = json!(false);
        row["needs_review"] = json!(true);
    }
    Ok(())
}

fn score_filename(title: &str, fallback: &str) -> String {
    let title = title.trim();
    let title = if title.is_empty()
        || matches!(title.to_lowercase().as_str(), "untitled" | "未命名乐谱")
    {
        Path::new(fallback)
            .file_stem()
            .and_then(|p| p.to_str())
            .unwrap_or("乐谱")
    } else {
        title
    };
    let mut name: String = title
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\"/\\|?*".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    name = name.trim_matches([' ', '.']).to_owned();
    if name.to_lowercase().ends_with(".gp5") {
        name.truncate(name.len() - 4);
    }
    while name.len() > 160 {
        name.pop();
    }
    name = name.trim_end_matches([' ', '.']).to_owned();
    if name.is_empty() {
        name = "乐谱".into();
    }
    let reserved = name.split('.').next().unwrap_or("").to_uppercase();
    if matches!(reserved.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ["COM", "LPT"].iter().any(|p| {
            reserved
                .strip_prefix(p)
                .is_some_and(|s| s.len() == 1 && matches!(s.as_bytes()[0], b'1'..=b'9'))
        })
    {
        name.insert(0, '_');
    }
    name + ".gp5"
}
pub fn export(
    root: &Path,
    sid: &str,
    expected: Option<&str>,
) -> std::result::Result<(), (u16, String)> {
    let mut state = load_session(root, sid).map_err(|e| (400, e.to_string()))?;
    revision(&state, expected)?;
    let recognition = state["recognition"]
        .as_str()
        .ok_or((409, "请先识别并校对小节".into()))?
        .to_owned();
    let source: Value =
        serde_json::from_slice(&fs::read(&recognition).map_err(|e| (400, e.to_string()))?)
            .map_err(|e| (400, e.to_string()))?;
    if source["review_measures"]
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        return Err((
            409,
            format!("请先校对这些小节再导出：{}", source["review_measures"]),
        ));
    }
    let directory = root.join(sid);
    let output = tempfile::Builder::new()
        .prefix("export_")
        .tempdir_in(&directory)
        .map_err(|e| (400, e.to_string()))?;
    let fallback = state["input_names"][0].as_str().unwrap_or("乐谱");
    let filename = score_filename(source["title"].as_str().unwrap_or(""), fallback);
    let gp5 = output.path().join(filename);
    let report =
        guitarocr_native_core::music_exports::write_gp5(&source, &gp5).map_err(|e| (400, e))?;
    let mut exported = json!({"schema_version":"1.0","stage":"gp5_export","recognition":recognition,"gp5":gp5,"encoding_report":report["report_path"]});
    let musicxml = gp5.with_extension("musicxml");
    match guitarocr_native_core::music_exports::write_musicxml(&source, &musicxml) {
        Ok(_) => exported["musicxml"] = json!(musicxml),
        Err(error) => {
            exported["musicxml_error"] = json!(error);
            let _ = fs::remove_file(&musicxml);
        }
    }
    let manifest = output.path().join("manifest.json");
    json_file(&manifest, &exported).map_err(|e| (400, e))?;
    state["export"] = json!(manifest);
    state["revision"] = json!(
        state["revision"]
            .as_u64()
            .ok_or((400, "无效的项目版本".into()))?
            + 1
    );
    write_session_atomic(root, sid, &state).map_err(|e| (400, e.to_string()))?;
    let _ = output.keep();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_review_keeps_target_and_conflicts_do_not_replace_accepted_result() {
        let workspace = tempfile::tempdir().unwrap();
        let demo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../webapp/static/examples/Harbor-Light-synthetic-project.zip");
        let imported = guitarocr_native_core::project::import_project(
            demo,
            workspace.path(),
            &Default::default(),
        )
        .unwrap();
        let before = load_session(workspace.path(), &imported.session_id).unwrap();
        let old: Value =
            serde_json::from_slice(&fs::read(before["recognition"].as_str().unwrap()).unwrap())
                .unwrap();
        correct(
            workspace.path(),
            &imported.session_id,
            1,
            &json!({"reviewed":true}),
            Some("0"),
        )
        .unwrap();
        let saved = load_session(workspace.path(), &imported.session_id).unwrap();
        assert_eq!(saved["revision"], 1);
        let result: Value =
            serde_json::from_slice(&fs::read(saved["recognition"].as_str().unwrap()).unwrap())
                .unwrap();
        assert_eq!(result["records"][0]["target"], old["records"][0]["target"]);
        assert_eq!(result["records"][0]["reviewed"], true);
        assert!(Path::new(result["score_document"].as_str().unwrap()).is_file());
        assert_eq!(
            correct(
                workspace.path(),
                &imported.session_id,
                1,
                &json!({"reviewed":true}),
                Some("0")
            )
            .unwrap_err()
            .0,
            409
        );
        assert_eq!(
            load_session(workspace.path(), &imported.session_id).unwrap(),
            saved
        );
        let mut measure = guitarocr_native_core::score::parse_measure_target(
            result["records"][0]["target"].as_str().unwrap(),
        )
        .unwrap();
        measure["voices"][0]["events"][0]["notes"][0]["velocity"] = json!(72);
        correct(
            workspace.path(),
            &imported.session_id,
            1,
            &json!({"measure":measure}),
            Some("1"),
        )
        .unwrap();
        let saved = load_session(workspace.path(), &imported.session_id).unwrap();
        let result: Value =
            serde_json::from_slice(&fs::read(saved["recognition"].as_str().unwrap()).unwrap())
                .unwrap();
        assert_eq!(result["records"][0]["needs_review"], true);
        assert_eq!(result["records"][0]["reviewed"], false);
        assert_eq!(result["records"][0]["manually_edited"], true);
        assert_eq!(saved["export"], Value::Null);
        assert_eq!(saved["revision"], 2);
    }
    #[test]
    fn filenames_remain_portable() {
        assert_eq!(score_filename("CON", "score.pdf"), "_CON.gp5");
        assert_eq!(score_filename("untitled", "input.pdf"), "input.gp5");
        assert_eq!(score_filename(" ../score? ", "x"), "_score_.gp5");
    }
}

fn edited_single(layout: &Value, previous: &Value, values: &Value) -> Result<Value> {
    if !values.is_object() {
        return Err("谱面信息必须是对象".into());
    }
    for key in ["title", "artist", "instrument"] {
        if values.get(key).is_some_and(|v| !v.is_string()) {
            return Err(format!("{key}必须为文字"));
        }
    }
    for key in ["part_id", "part_name"] {
        if values
            .get(key)
            .is_some_and(|v| !v.is_null() && !v.is_string())
        {
            return Err(format!("{key}必须为文字"));
        }
    }

    let instrument = values["instrument"].as_str().unwrap_or("guitar");
    if !matches!(instrument, "guitar" | "bass" | "pitched" | "drums") {
        return Err("请选择乐器类型".into());
    }
    let fretted = matches!(instrument, "guitar" | "bass");
    let tuning = if fretted {
        values
            .get("tuning_used")
            .cloned()
            .unwrap_or(json!([64, 59, 55, 50, 45, 40]))
    } else {
        json!([])
    };
    let strings = tuning.as_array().ok_or("无效调弦")?;
    if (fretted && !(1..=12).contains(&strings.len()))
        || strings
            .iter()
            .any(|p| !p.as_i64().is_some_and(|n| (0..=127).contains(&n)))
    {
        return Err("调弦须包含1至12个0至127的MIDI音高".into());
    }
    let integer = |key: &str, default: i64, min: i64, max: i64| -> Result<i64> {
        values.get(key).map_or(Ok(default), |v| {
            v.as_i64()
                .filter(|n| (min..=max).contains(n))
                .ok_or(format!("无效的{key}"))
        })
    };
    let capo = integer("capo", 0, 0, 24)?;
    let tempo = integer("tempo_quarter", 120, 20, 400)?;
    let transpose = values
        .get("transpose")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_i64()
                .filter(|n| (-36..=36).contains(n))
                .ok_or("记谱移调须为-36至36".to_string())
        })
        .transpose()?;
    let program = values
        .get("midi_program")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_i64()
                .filter(|n| (0..=127).contains(n))
                .ok_or("MIDI乐器须为0至127".to_string())
        })
        .transpose()?
        .or_else(|| {
            if previous["instrument"] == instrument {
                previous["midi_program"].as_i64()
            } else {
                None
            }
        })
        .unwrap_or(match instrument {
            "guitar" => 25,
            "bass" => 33,
            _ => 0,
        });
    let mut metadata = previous
        .get("document_metadata")
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or(json!({}));
    metadata["tempo_quarter"] = json!(tempo);
    let contexts = if previous["instrument"].as_str().unwrap_or("guitar") == instrument
        && previous["transpose"] == json!(transpose)
    {
        previous
            .get("measure_pitch_contexts")
            .cloned()
            .unwrap_or(json!([]))
    } else if previous["measure_pitch_contexts"]
        .as_array()
        .is_some_and(|r| !r.is_empty())
        || transpose.is_some()
    {
        let records = layout["records"]
            .as_array()
            .ok_or("区域缺少小节")?
            .iter()
            .map(|r| {
                let mut r = r.clone();
                r["instrument"] = json!(instrument);
                r
            })
            .collect::<Vec<_>>();
        let output = guitarocr_native_core::recognition::apply_pitch_regions(
            &records,
            metadata["pitch_instructions"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            instrument,
            transpose,
            None,
        )?;
        json!(output
            .into_iter()
            .map(|row| {
                let mut value = serde_json::Map::new();
                for key in [
                    "measure_number",
                    "pitch_context",
                    "pitch_reference",
                    "pitch_needs_review",
                ] {
                    if let Some(v) = row.get(key) {
                        value.insert(key.into(), v.clone());
                    }
                }
                Value::Object(value)
            })
            .collect::<Vec<_>>())
    } else {
        json!([])
    };
    let title = values["title"].as_str().unwrap_or("未命名乐谱");
    let artist = values["artist"].as_str().unwrap_or("");
    if title.chars().count() > 500 || artist.chars().count() > 500 {
        return Err("曲名或作者最多500个字符".into());
    }
    Ok(
        json!({"document_metadata":metadata,"predictions":null,"title":title,"artist":artist,"tuning_used":tuning,"tuning_source":"manual","tuning_candidates":[tuning],"capo":capo,"instrument":instrument,"midi_program":program,"transpose":transpose,"measure_pitch_contexts":contexts}),
    )
}
fn edited_information(layout: &Value, previous: &Value, values: &Value) -> Result<Value> {
    let mut layout = layout.clone();
    if previous["resolved_records"].is_array() {
        layout["records"] = previous["resolved_records"].clone();
    }
    let Some(parts) = previous["parts"].as_array().filter(|p| !p.is_empty()) else {
        return edited_single(&layout, previous, values);
    };
    let selected = values["part_id"]
        .as_str()
        .unwrap_or_else(|| parts[0]["id"].as_str().unwrap_or(""));
    let index = parts
        .iter()
        .position(|p| p["id"] == selected)
        .ok_or("无效的音轨")?;
    let profiles = previous["measure_profiles"]
        .as_array()
        .ok_or("音轨缺少小节配置")?
        .iter()
        .filter(|p| p["part_id"] == selected)
        .cloned()
        .collect::<Vec<_>>();
    let local_records = layout["records"]
        .as_array()
        .ok_or("区域缺少小节")?
        .iter()
        .filter_map(|r| {
            profiles
                .iter()
                .find(|p| p["measure_number"] == r["measure_number"])
                .map(|p| {
                    let mut r = r.clone();
                    r.as_object_mut()
                        .unwrap()
                        .extend(p.as_object().unwrap().clone());
                    r
                })
        })
        .collect::<Vec<_>>();
    let mut local = layout.clone();
    local["records"] = json!(local_records);
    let edited = edited_single(&local, &parts[index], values)?;
    let mut result = previous.clone();
    result["parts"][index]
        .as_object_mut()
        .ok_or("无效音轨")?
        .extend(edited.as_object().unwrap().clone());
    if let Some(name) = values["part_name"].as_str() {
        let name = name.trim();
        if name.is_empty() || name.chars().count() > 160 {
            return Err("请填写160字内的音轨名称".into());
        }
        result["parts"][index]["name"] = json!(name);
    }
    let name = result["parts"][index]["name"].clone();
    for profile in result["measure_profiles"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .filter(|p| p["part_id"] == selected)
    {
        for (to, from) in [
            ("instrument", "instrument"),
            ("tuning", "tuning_used"),
            ("capo", "capo"),
            ("midi_program", "midi_program"),
        ] {
            profile[to] = edited[from].clone();
        }
        profile["tuning_explicit"] = json!(true);
        profile["tuning_source"] = json!("manual");
        profile["fingering_tunings"] = json!([edited["tuning_used"]]);
        profile["part_name"] = name.clone();
    }
    result["measure_pitch_contexts"] = json!(result["parts"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|p| p["measure_pitch_contexts"]
            .as_array()
            .into_iter()
            .flatten()
            .cloned())
        .collect::<Vec<_>>());
    result["title"] = edited["title"].clone();
    result["artist"] = edited["artist"].clone();
    if index == 0 {
        for (key, value) in edited.as_object().unwrap() {
            if key != "measure_pitch_contexts" {
                result[key] = value.clone();
            }
        }
    }
    result["document_metadata"]["tempo_quarter"] =
        edited["document_metadata"]["tempo_quarter"].clone();
    for part in result["parts"].as_array_mut().unwrap() {
        part["title"] = edited["title"].clone();
        part["artist"] = edited["artist"].clone();
        part["document_metadata"]["tempo_quarter"] =
            edited["document_metadata"]["tempo_quarter"].clone();
    }
    Ok(result)
}
pub fn metadata(
    root: &Path,
    sid: &str,
    values: &Value,
    expected: Option<&str>,
) -> std::result::Result<(), (u16, String)> {
    let mut state = load_session(root, sid).map_err(|e| (400, e.to_string()))?;
    revision(&state, expected)?;
    let layout = crate::view::read_json(Path::new(
        state["layout"]
            .as_str()
            .ok_or((409, "请先保存小节框".into()))?,
    ))
    .map_err(|e| (400, e))?;
    let mut previous = if let Some(path) = state["info"].as_str() {
        crate::view::read_json(Path::new(path)).map_err(|e| (400, e))?
    } else {
        json!({})
    };
    let recognized = if let Some(path) = state["recognition"].as_str() {
        Some(crate::view::read_json(Path::new(path)).map_err(|e| (400, e))?)
    } else {
        None
    };
    if let Some(source) = &recognized {
        previous["tuning_used"] = source["tuning_used"].clone();
        let rows = source["records"]
            .as_array()
            .ok_or((400, "无效识别结果".into()))?;
        for field in ["parts", "measure_profiles"] {
            if let Some(items) = previous[field].as_array_mut() {
                for part in items {
                    let key = if field == "parts" { "id" } else { "part_id" };
                    if let Some(row) = rows.iter().find(|r| r["part_id"] == part[key]) {
                        part[if field == "parts" {
                            "tuning_used"
                        } else {
                            "tuning"
                        }] = row
                            .get("tuning")
                            .cloned()
                            .unwrap_or(source["tuning_used"].clone());
                    }
                }
            }
        }
    }
    let mut information = edited_information(&layout, &previous, values).map_err(|e| (400, e))?;
    information["schema_version"] = json!("1.0");
    information["stage"] = json!("document_info");
    information["layout"] = state["layout"].clone();
    let output = tempfile::Builder::new()
        .prefix("metadata_")
        .tempdir_in(root.join(sid))
        .map_err(|e| (400, e.to_string()))?;
    let manifest = output.path().join("info.json");
    json_file(&manifest, &information).map_err(|e| (400, e))?;
    state["info"] = json!(manifest);
    if let Some(source) = recognized {
        if let Some(mut source) =
            guitarocr_native_core::recognition::update_information(&source, &information)
                .map_err(|e| (400, e))?
        {
            source["info"] = json!(manifest);
            let accepted = save_recognition(output.path(), &source).map_err(|e| (400, e))?;
            state["recognition"] = json!(accepted);
        } else {
            state["recognition"] = Value::Null;
        }
    }
    state["export"] = Value::Null;
    state.as_object_mut().unwrap().remove("ocr_task");
    state["revision"] = json!(state["revision"].as_u64().ok_or((400, "无效版本".into()))? + 1);
    write_session_atomic(root, sid, &state).map_err(|e| (400, e.to_string()))?;
    let _ = output.keep();
    Ok(())
}
