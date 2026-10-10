//! Revision-controlled native writes. Accepted results are replaced only after
//! every required projection has been written successfully.
pub mod chord_library;
use crate::project::{load_session, write_session_atomic};
use scorelib::score::{format_measure_target, score_document, validate_measure_target};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
fn record_bars(document: &Value) -> std::collections::HashMap<String, usize> {
    document["parts"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|part| part["staves"].as_array().into_iter().flatten())
        .flat_map(|staff| staff["measures"].as_array().into_iter().flatten())
        .map(|measure| {
            (
                measure["source_record"].to_string(),
                measure["index"].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

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
    let document = score_document(&result)?;
    let bars = record_bars(&document);
    let timeline = &document["timeline"];
    for row in result["records"]
        .as_array_mut()
        .ok_or("识别结果缺少小节")?
        .iter_mut()
    {
        let bar = bars[&row["measure_number"].to_string()];
        let conflicts = &timeline[bar]["conflicts"];
        if conflicts.as_object().is_some_and(|v| !v.is_empty()) {
            row["global_timing_conflicts"] = conflicts.clone();
            row["needs_review"] = json!(true);
            row["reviewed"] = json!(false);
        } else {
            row.as_object_mut()
                .unwrap()
                .remove("global_timing_conflicts");
        }
    }
    write_recognition(output, result)
}

#[cfg(test)]
mod observation_edit_tests {
    use super::*;

    #[test]
    fn partial_metadata_edit_preserves_existing_values_and_unknown_tempo() {
        let previous = json!({"title":"Before","artist":"Composer","instrument":"guitar",
            "tuning_used":[64,59,55,50,45,40],"tuning_source":"manual","capo":3,
            "transpose":-12,"document_metadata":{"tempo_quarter":null},
            "measure_pitch_contexts":[{"measure_number":1,"pitch_context":{"capo":3}}]});
        let layout = json!({"records":[]});
        let edited = edited_single(&layout, &previous, &json!({"title":"After"})).unwrap();
        assert_eq!(edited["title"], "After");
        for key in [
            "artist",
            "instrument",
            "tuning_used",
            "capo",
            "transpose",
            "measure_pitch_contexts",
        ] {
            assert_eq!(edited[key], previous[key], "{key}");
        }
        assert!(edited["document_metadata"]["tempo_quarter"].is_null());
        let mut previous = previous.clone();
        previous["document_metadata"]["tempo_quarter"] = json!(96);
        let edited = edited_single(&layout, &previous, &json!({"title":"After"})).unwrap();
        assert_eq!(edited["document_metadata"]["tempo_quarter"], 96);
        let edited = edited_single(&layout, &previous, &json!({"tempo_quarter":null})).unwrap();
        assert!(edited["document_metadata"]["tempo_quarter"].is_null());
        assert!(edited_single(&layout, &json!({}), &json!({"title":"Untitled"})).is_err());
        assert!(edited_single(&layout, &json!({}), &json!({"instrument":"guitar"})).is_err());
    }

    #[test]
    fn incomplete_measure_survives_save_and_becomes_exportable_after_edit() {
        let output = tempfile::tempdir().unwrap();
        let mut source = json!({"mode":"tab","instrument":"guitar","tuning_used":[64,59,55,50,45,40],
            "records":[{"source_id":"score","part_id":"guitar","staff_id":"tab","bar_index":0,
                "measure_number":1,"mode":"tab","instrument":"guitar","capo":0,"tuning":[64,59,55,50,45,40],
                "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
                    "voices":[{"voice":0,"events":[{"start":0,"duration":null,"status":"normal",
                        "notes":[{"string":1,"fret":5}]}]}],"relations":[]}}]});
        crate::recognition::reassemble(&mut source).unwrap();
        let manifest = save_recognition(output.path(), &source).unwrap();
        let mut saved: Value = serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
        assert!(saved["records"][0]["target"].is_null());
        assert!(saved["score_text"].is_null());
        assert!(saved["musicxml"].is_null());
        let document = score_document(&saved).unwrap();
        let measure = &document["parts"][0]["staves"][0]["measures"][0];
        assert_eq!(measure["incomplete"], true);
        assert_eq!(measure["voices"], json!([]));
        assert_eq!(
            measure["observation"]["voices"][0]["events"][0]["notes"][0]["fret"],
            5
        );
        assert!(scorelib::music_exports::score_document(&document).is_err());
        assert!(correct_record(&mut saved, 1, &json!({"reviewed":true})).is_err());

        let mut observed = saved["records"][0]["observation"].clone();
        observed["voices"][0]["events"][0]["duration"] = json!("w");
        correct_record(&mut saved, 1, &json!({"observation":observed})).unwrap();
        crate::recognition::reassemble(&mut saved).unwrap();
        assert!(saved["records"][0]["target"]
            .as_str()
            .unwrap()
            .contains("@0:w:s1f5"));
        correct_record(&mut saved, 1, &json!({"reviewed":true})).unwrap();
        crate::recognition::reassemble(&mut saved).unwrap();
        assert_eq!(saved["records"][0]["reviewed"], true);
        assert_eq!(saved["records"][0]["needs_review"], false);
        let manifest = save_recognition(output.path(), &saved).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
        assert!(saved["musicxml"].is_string());
        assert!(saved["score_text"].is_string());
    }
}

/// Persist projections without changing musical records for annotation-only edits.
fn write_recognition(output: &Path, mut result: Value) -> Result<PathBuf> {
    let rows = result["records"].as_array().ok_or("识别结果缺少小节")?;
    let text = rows
        .iter()
        .map(|r| r["target"].as_str())
        .collect::<Option<Vec<_>>>()
        .map(|lines| lines.join("\n") + "\n");
    let review: Vec<_> = rows
        .iter()
        .filter(|r| r["needs_review"] == true || r["target"].is_null())
        .map(|r| r["measure_number"].clone())
        .collect();
    result["measures"] = json!(rows.len());
    result["status"] = json!(if review.is_empty() {
        "complete"
    } else {
        "needs_review"
    });
    result["review_measures"] = json!(review);
    if let Some(text) = text {
        fs::write(output.join("score.txt"), &text).map_err(|e| e.to_string())?;
        result["score_text"] = json!(output.join("score.txt"));
    } else {
        result["score_text"] = Value::Null;
        let _ = fs::remove_file(output.join("score.txt"));
    }
    let document = score_document(&result)?;
    json_file(&output.join("score.json"), &document)?;
    result["score_document"] = json!(output.join("score.json"));
    let obj = result.as_object_mut().ok_or("无效的识别结果")?;
    obj.remove("musicxml");
    obj.remove("musicxml_error");
    // MusicXML is an optional projection; an unrepresentable score must retain
    // its canonical records and remain editable, matching the research pipeline.
    match scorelib::music_exports::write_musicxml(&document, &output.join("score.musicxml")) {
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
    crate::recognition::reassemble(&mut source).map_err(|e| (400, e))?;
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
    let document = score_document(source)?;
    let bars = record_bars(&document);
    let source_mode = source["mode"].as_str().unwrap_or("tab").to_owned();
    let source_tuning = source["tuning_used"].clone();
    let source_instrument = source["instrument"].as_str().map(str::to_owned);
    let rows = source["records"].as_array_mut().ok_or("无效的识别结果")?;
    if number == 0 || number > rows.len() {
        return Err("无效的小节编号".into());
    }
    let row = &mut rows[number - 1];
    if let Some(observation) = body.get("observation") {
        row["observation"] = scorelib::observation::parse_observation(observation)?;
        row["observation_manually_edited"] = json!(true);
        row["manually_edited"] = json!(false);
        row["reviewed"] = json!(false);
        row["needs_review"] = json!(true);
        return Ok(());
    }
    let mode = row["mode"].as_str().unwrap_or(&source_mode);
    let reviewed = body["reviewed"].as_bool().unwrap_or(false);
    let old = row["target"].as_str().map(str::to_owned);
    let target = if let Some(measure) = body.get("measure").filter(|v| !v.is_null()) {
        format_measure_target(measure, mode, true)?
    } else if let Some(text) = body["target"].as_str() {
        text.to_owned()
    } else if reviewed || body.get("visual_techniques").is_some() {
        old.clone()
            .ok_or("该小节还有未识别的音符或节奏，请先补全内容")?
    } else {
        return Err("请输入小节内容或核对标记".into());
    };
    if target.len() > 50000 || target.contains('\n') {
        return Err("请输入单个小节的内容".into());
    }
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
    let parsed = parsed.ok_or("无效的小节")?;
    let previous = scorelib::score::record_measure(row)?;
    let global_changes = ["time_signature", "tempo_quarter"]
        .into_iter()
        .filter_map(|field| {
            (parsed[field] != previous[field]
                || (reviewed
                    && !parsed[field].is_null()
                    && row["global_timing_conflicts"][field].is_array()))
            .then(|| (field, parsed[field].clone()))
        })
        .collect::<Vec<_>>();
    let bar = bars[&row["measure_number"].to_string()];
    let inherit_time = body["measure"].is_object()
        && !previous["time_signature"].is_null()
        && parsed["time_signature"].is_null();
    let previous_time = if bar > 0 {
        document["timeline"][bar - 1]["time_signature"].clone()
    } else {
        Value::Null
    };
    let timing = scorelib::gp5_constraints::timing_errors(&parsed)?;
    if !timing.is_empty() {
        return Err(timing.join("；"));
    }
    let instrument = row["instrument"]
        .as_str()
        .or(source_instrument.as_deref())
        .ok_or("请先选择乐器类型")?;
    if mode == "notation"
        && matches!(instrument, "guitar" | "bass")
        && row["tuning_explicit"] == true
    {
        let errors = scorelib::gp5_constraints::notation_fingering_errors(&parsed, &tuning)?;
        if !errors.is_empty() {
            return Err(errors.join("；"));
        }
    }
    let changed = Some(target.as_str()) != old.as_deref();
    let previous_overlay = row.get("visual_techniques").cloned().unwrap_or(Value::Null);
    let overlay = if let Some(value) = body.get("visual_techniques") {
        if !value.is_null() {
            scorelib::visual_techniques::validate_overlay(&parsed, value)?;
        }
        value.clone()
    } else if !previous_overlay.is_null() && changed {
        scorelib::visual_techniques::retain_unchanged(&previous, &parsed, &previous_overlay)?
    } else {
        previous_overlay.clone()
    };
    let changed = changed || overlay != previous_overlay;
    if overlay.is_null() {
        row.as_object_mut().unwrap().remove("visual_techniques");
    } else {
        row["visual_techniques"] = overlay;
    }
    row["target"] = json!(target);
    row["manually_edited"] = json!(changed || row["manually_edited"] == true);
    if changed || reviewed {
        crate::recognition::signature_state::observe_manual_metadata(row, &target)?;
    }
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
    // Time and tempo apply to all staves at this score position. Propagate
    // only these explicitly edited fields; peers keep their notes, effects,
    // key signatures and independent review decisions.
    if !global_changes.is_empty() {
        for peer in rows.iter_mut() {
            if bars[&peer["measure_number"].to_string()] != bar {
                continue;
            }
            if peer["target"].is_null() {
                // A shared meter edit is useful evidence even before this
                // staff's notes can be assembled. It does not create notes.
                if let Some((_, value)) = global_changes
                    .iter()
                    .find(|(field, _)| *field == "time_signature")
                {
                    if !peer["score_state"].is_object() {
                        peer["score_state"] = json!({});
                    }
                    if !peer["score_state_evidence"].is_object() {
                        peer["score_state_evidence"] = json!({});
                    }
                    if !peer["manual_state"].is_object() {
                        peer["manual_state"] = json!({});
                    }
                    peer["manual_state"]["time"] = json!({"status":if value.is_null() {"absent"} else {"observed"},"value":value,"source":"manual"});
                    peer["score_state"]["time"] = value.clone();
                    peer["score_state_evidence"]["time"] =
                        json!({"status":"observed","value":value,"source":"manual"});
                }
                continue;
            }
            let mut measure =
                scorelib::score::parse_measure_target(peer["target"].as_str().unwrap())?;
            for (field, value) in &global_changes {
                measure[*field] = value.clone();
                if *field == "time_signature" {
                    measure["print_time_signature"] = json!(!value.is_null());
                }
            }
            let target = format_measure_target(
                &measure,
                peer["mode"].as_str().unwrap_or(&source_mode),
                true,
            )?;
            peer["target"] = json!(target);
            if let Some(written) = peer["written_target"].as_str() {
                let mut written = scorelib::score::parse_measure_target(written)?;
                for (field, value) in &global_changes {
                    written[*field] = value.clone();
                    if *field == "time_signature" {
                        written["print_time_signature"] = json!(!value.is_null());
                    }
                }
                peer["written_target"] = json!(format_measure_target(
                    &written,
                    peer["mode"].as_str().unwrap_or(&source_mode),
                    true
                )?);
            }
            if global_changes
                .iter()
                .any(|(field, _)| *field == "time_signature")
            {
                if inherit_time {
                    if !peer["score_state"].is_object() {
                        peer["score_state"] = json!({});
                    }
                    if !peer["score_state_evidence"].is_object() {
                        peer["score_state_evidence"] = json!({});
                    }
                    peer["score_state"]["time"] = previous_time.clone();
                    peer["score_state_evidence"]["time"] = json!({
                        "status": if previous_time.is_null() { "unknown" } else { "inherited" },
                        "value": previous_time, "source":"manual",
                        "observation":{"status":"absent","value":null,"source":"manual"}
                    });
                    if !peer["manual_state"].is_object() {
                        peer["manual_state"] = json!({});
                    }
                    peer["manual_state"]["time"] =
                        json!({"status":"absent","value":null,"source":"manual"});
                }
                // Do not accept a peer's unrelated key evidence as a side
                // effect of a shared meter edit.
                measure["key_signature"] = Value::Null;
                measure["print_key_signature"] = json!(false);
                let time_only = format_measure_target(
                    &measure,
                    peer["mode"].as_str().unwrap_or(&source_mode),
                    true,
                )?;
                crate::recognition::signature_state::observe_manual_metadata(peer, &time_only)?;
            }
            peer["timing_errors"] = json!(scorelib::gp5_constraints::timing_errors(&measure)?);
            if peer["timing_errors"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
            {
                peer["needs_review"] = json!(true);
                peer["reviewed"] = json!(false);
            }
        }
    }
    if bars.values().copied().min() == Some(bar) {
        if let Some((_, tempo)) = global_changes
            .iter()
            .find(|(field, _)| *field == "tempo_quarter")
        {
            if !source["document_metadata"].is_object() {
                source["document_metadata"] = json!({});
            }
            source["document_metadata"]["tempo_quarter"] = tempo.clone();
            source["document_metadata"]["tempo_source"] = json!("manual");
            if let Some(parts) = source["parts"].as_array_mut() {
                for part in parts {
                    if !part["document_metadata"].is_object() {
                        part["document_metadata"] = json!({});
                    }
                    part["document_metadata"]["tempo_quarter"] = tempo.clone();
                    part["document_metadata"]["tempo_source"] = json!("manual");
                }
            }
        }
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
    if std::iter::once(&source)
        .chain(source["parts"].as_array().into_iter().flatten())
        .any(|part| {
            part["document_metadata"]["instrument_needs_review"] == true
                || part["document_metadata"]["tuning_needs_review"] == true
        })
    {
        return Err((409, "请先在谱面信息中确认乐器和调弦再导出。".into()));
    }
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
    let gp5 = output.path().join(&filename);
    let mut exported = json!({"schema_version":"1.0","stage":"gp5_export",
        "recognition":recognition,"gp5":null,"musicxml":null,
        "gp5_error":null,"musicxml_error":null,"warnings":[]});
    match scorelib::music_exports::write_gp5(&source, &gp5) {
        Ok(report) => {
            exported["gp5"] = json!(gp5);
            exported["encoding_report"] = report["report_path"].clone();
            exported["warnings"] = json!(report["report"]
                .as_object()
                .into_iter()
                .flat_map(|report| report.values())
                .filter_map(|field| field["warning"].as_str())
                .collect::<Vec<_>>());
        }
        Err(error) => {
            exported["gp5_error"] = json!(error);
            let _ = fs::remove_file(&gp5);
            let _ = fs::remove_file(output.path().join(format!("{filename}.encoding.json")));
        }
    }
    let musicxml = gp5.with_extension("musicxml");
    match scorelib::music_exports::write_musicxml(&source, &musicxml) {
        Ok(_) => exported["musicxml"] = json!(musicxml),
        Err(error) => {
            exported["musicxml_error"] = json!(error);
            let _ = fs::remove_file(&musicxml);
        }
    }
    if exported["gp5"].is_null() && exported["musicxml"].is_null() {
        return Err((
            400,
            format!(
                "GP5：{}；MusicXML：{}",
                exported["gp5_error"].as_str().unwrap(),
                exported["musicxml_error"].as_str().unwrap()
            ),
        ));
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

    let instrument = values["instrument"]
        .as_str()
        .or_else(|| previous["instrument"].as_str())
        .ok_or("请先选择乐器类型")?;
    if !matches!(instrument, "guitar" | "bass" | "pitched" | "drums") {
        return Err("请选择乐器类型".into());
    }
    let fretted = matches!(instrument, "guitar" | "bass");
    let tuning = if fretted {
        values
            .get("tuning_used")
            .or_else(|| previous.get("tuning_used"))
            .cloned()
            .ok_or("请填写调弦")?
    } else {
        json!([])
    };
    let strings = tuning.as_array().ok_or("无效调弦")?;
    for key in ["confirm_instrument", "confirm_tuning"] {
        if values.get(key).is_some_and(|value| !value.is_boolean()) {
            return Err(format!("{key}必须为布尔值"));
        }
    }
    let confirm_instrument = values["confirm_instrument"] == true
        || (values.get("instrument").is_some() && previous["instrument"] != instrument);
    let confirm_tuning = values["confirm_tuning"] == true
        || (values.get("tuning_used").is_some() && previous["tuning_used"] != tuning);
    if (fretted && !(1..=12).contains(&strings.len()))
        || strings
            .iter()
            .any(|p| !p.as_i64().is_some_and(|n| (0..=127).contains(&n)))
    {
        return Err("调弦须包含1至12个0至127的MIDI音高".into());
    }
    let integer = |key: &str, previous_value: &Value, min: i64, max: i64| -> Result<Option<i64>> {
        let value = values.get(key).unwrap_or(previous_value);
        if value.is_null() {
            return Ok(None);
        }
        value
            .as_i64()
            .filter(|n| (min..=max).contains(n))
            .map(Some)
            .ok_or(format!("无效的{key}"))
    };
    let capo = integer("capo", &previous["capo"], 0, 24)?.unwrap_or(0);
    let tempo = integer(
        "tempo_quarter",
        &previous["document_metadata"]["tempo_quarter"],
        20,
        400,
    )?;
    let transpose = integer("transpose", &previous["transpose"], -36, 36)?;
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
    for (key, limit) in [("subtitle", 500), ("header_notes", 2000)] {
        if let Some(value) = values.get(key) {
            let value = value.as_str().ok_or("谱头信息必须是文字")?;
            if value.chars().count() > limit {
                return Err("谱头信息过长".into());
            }
            metadata[key] = json!(value.trim());
        }
    }
    metadata["tempo_quarter"] = json!(tempo);
    if confirm_instrument {
        metadata["instrument"] = json!(instrument);
        metadata["instrument_source"] = json!("manual");
        metadata["instrument_needs_review"] = json!(false);
        metadata.as_object_mut().unwrap().remove("instrument_issue");
        if !metadata["instrument_evidence"].is_object() {
            metadata["instrument_evidence"] = json!({"candidates":{}});
        }
        let evidence = &mut metadata["instrument_evidence"];
        evidence["value"] = json!(instrument);
        evidence["source"] = json!("manual");
        evidence["status"] = json!("resolved");
        evidence["needs_review"] = json!(false);
        evidence["provisional"] = json!(false);
        if !evidence["candidates"].is_object() {
            evidence["candidates"] = json!({});
        }
        evidence["candidates"]["manual"] = json!(instrument);
    }
    if confirm_tuning {
        metadata.as_object_mut().unwrap().remove("tuning_issue");
        metadata["tuning_needs_review"] = json!(false);
        if !metadata["tuning_evidence"].is_object() {
            metadata["tuning_evidence"] = json!({});
        }
        metadata["tuning_evidence"]["value"] = tuning.clone();
        metadata["tuning_evidence"]["source"] = json!("manual");
        metadata["tuning_evidence"]["needs_review"] = json!(false);
        metadata["string_count"] = if fretted {
            json!(strings.len())
        } else {
            Value::Null
        };
        metadata["string_count_source"] = json!("manual");
        if !metadata["string_count_evidence"].is_object() {
            metadata["string_count_evidence"] = json!({"candidates":{}});
        }
        let count = metadata["string_count"].clone();
        let evidence = &mut metadata["string_count_evidence"];
        evidence["value"] = count.clone();
        evidence["source"] = json!("manual");
        evidence["needs_review"] = json!(false);
        if !evidence["candidates"].is_object() {
            evidence["candidates"] = json!({});
        }
        evidence["candidates"]["manual"] = count;
        metadata["string_count_needs_review"] = json!(false);
    }
    let identity_and_tuning_resolved =
        metadata["instrument_needs_review"] != true && metadata["tuning_needs_review"] != true;
    if let Some(warnings) = metadata["warnings"].as_array_mut() {
        warnings.retain(|warning| {
            !warning.as_str().is_some_and(|text| {
                (confirm_tuning && text.starts_with("TAB 弦数候选不一致或未知，请核对调弦；"))
                    || (confirm_instrument
                        && text.starts_with("乐器身份候选不一致或未知，请确认乐器；"))
                    || (identity_and_tuning_resolved
                        && text.starts_with("乐器身份候选不一致，当前声部与调弦仅临时保留；"))
            })
        });
    }
    let contexts = if !confirm_instrument
        && !confirm_tuning
        && previous["instrument"] == instrument
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
                if (r["instrument_needs_review"] == true || r["tuning_needs_review"] == true)
                    && r["structure_needs_review"] != true
                    && r["alignment_needs_review"] != true
                {
                    r["pitch_needs_review"] = json!(false);
                }
                r["instrument_needs_review"] = metadata["instrument_needs_review"].clone();
                r["tuning_needs_review"] = metadata["tuning_needs_review"].clone();
                r
            })
            .collect::<Vec<_>>();
        let output = crate::recognition::apply_pitch_regions(
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
                value.insert(
                    "instrument_needs_review".into(),
                    metadata["instrument_needs_review"].clone(),
                );
                value.insert(
                    "tuning_needs_review".into(),
                    metadata["tuning_needs_review"].clone(),
                );
                value.entry("pitch_needs_review").or_insert(json!(false));
                if metadata["instrument_needs_review"] == true
                    || metadata["tuning_needs_review"] == true
                {
                    value.insert("pitch_needs_review".into(), json!(true));
                }
                Value::Object(value)
            })
            .collect::<Vec<_>>())
    } else {
        json!([])
    };
    let title = values["title"]
        .as_str()
        .or_else(|| previous["title"].as_str())
        .unwrap_or("")
        .trim();
    let artist = values["artist"]
        .as_str()
        .or(previous["artist"].as_str())
        .unwrap_or("");
    if title.chars().count() > 500 || artist.chars().count() > 500 {
        return Err("曲名或作者最多500个字符".into());
    }
    if title != previous["title"].as_str().unwrap_or("").trim()
        || metadata["title_source"] == "manual"
    {
        crate::recognition::headers::mark_manual(&mut metadata, title);
    }
    let tuning_source = if confirm_tuning {
        "manual"
    } else {
        previous["tuning_source"].as_str().unwrap_or("default")
    };
    Ok(
        json!({"document_metadata":metadata,"predictions":previous["predictions"],"title":title,"artist":artist,"tuning_used":tuning,"tuning_source":tuning_source,"capo":capo,"instrument":instrument,"midi_program":program,"transpose":transpose,"measure_pitch_contexts":contexts}),
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
    // Native information stores contexts at document level. Editing a title
    // must retain the selected part's contexts and every other part's music.
    let contexts = previous["measure_pitch_contexts"]
        .as_array()
        .cloned()
        .unwrap_or_else(|| {
            parts
                .iter()
                .flat_map(|p| {
                    p["measure_pitch_contexts"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .cloned()
                })
                .collect()
        });
    let belongs = |context: &Value| {
        profiles
            .iter()
            .any(|p| p["measure_number"] == context["measure_number"])
    };
    let mut previous_part = parts[index].clone();
    previous_part["measure_pitch_contexts"] =
        json!(contexts.iter().filter(|c| belongs(c)).collect::<Vec<_>>());
    let edited = edited_single(&local, &previous_part, values)?;
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
        profile["tuning_explicit"] = json!(matches!(
            edited["tuning_source"].as_str(),
            Some("manual" | "printed")
        ));
        profile["tuning_source"] = edited["tuning_source"].clone();
        for key in [
            "instrument_evidence",
            "instrument_needs_review",
            "tuning_needs_review",
        ] {
            profile[key] = edited["document_metadata"][key].clone();
        }
        profile["string_count"] = edited["document_metadata"]["string_count"].clone();
        profile["part_name"] = name.clone();
    }
    result["measure_pitch_contexts"] = json!(contexts
        .iter()
        .filter(|c| !belongs(c))
        .chain(
            edited["measure_pitch_contexts"]
                .as_array()
                .into_iter()
                .flatten()
        )
        .collect::<Vec<_>>());
    result["title"] = edited["title"].clone();
    result["artist"] = edited["artist"].clone();
    crate::recognition::headers::copy_title_provenance(
        &mut result["document_metadata"],
        &edited["document_metadata"],
    );
    if index == 0 {
        for (key, value) in edited.as_object().unwrap() {
            if key != "measure_pitch_contexts" {
                result[key] = value.clone();
            }
        }
    }
    for key in ["subtitle", "header_notes"] {
        result["document_metadata"][key] = edited["document_metadata"][key].clone();
    }
    result["document_metadata"]["tempo_quarter"] =
        edited["document_metadata"]["tempo_quarter"].clone();
    for part in result["parts"].as_array_mut().unwrap() {
        for key in ["subtitle", "header_notes"] {
            part["document_metadata"][key] = edited["document_metadata"][key].clone();
        }
        part["title"] = edited["title"].clone();
        part["artist"] = edited["artist"].clone();
        crate::recognition::headers::copy_title_provenance(
            &mut part["document_metadata"],
            &edited["document_metadata"],
        );
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
    if information["document_metadata"]["title_source"] == "manual" {
        state["manual_title"] = information["title"].clone();
    } else {
        state.as_object_mut().unwrap().remove("manual_title");
    }
    if let Some(source) = recognized {
        if let Some(mut source) =
            crate::recognition::update_information(&source, &information).map_err(|e| (400, e))?
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
