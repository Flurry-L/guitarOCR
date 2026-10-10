//! Explicit review of printed chord libraries, independent of musical beats.
use super::*;

fn annotation_mut<'a>(source: &'a mut Value, part_id: &str, index: usize) -> Result<&'a mut Value> {
    let part = if source["parts"]
        .as_array()
        .is_some_and(|parts| !parts.is_empty())
    {
        source["parts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|part| part["id"] == part_id)
            .ok_or("无效的音轨")?
    } else {
        if source["id"].as_str().unwrap_or("part-1") != part_id {
            return Err("无效的音轨".into());
        }
        source
    };
    part["document_metadata"]["score_annotations"]
        .as_array_mut()
        .and_then(|rows| rows.get_mut(index))
        .filter(|annotation| annotation.is_object())
        .ok_or_else(|| "无效的和弦库条目".into())
}

fn reviewed(annotation: &Value, body: &Value) -> Result<Value> {
    let prior = &annotation["chord_review"];
    let manual = prior["source"] == "manual";
    if !manual
        && !matches!(
            annotation["parsed"]["kind"].as_str(),
            Some("chord" | "chord_diagram")
        )
        && !annotation["chord_evidence"].is_object()
    {
        return Err("该条目不是和弦指法候选".into());
    }
    let mut result = annotation.clone();
    let decision = body["decision"].as_str().ok_or("缺少和弦库操作")?;
    if decision == "reset" {
        if !manual || prior.get("original_parsed").is_none() {
            return Err("该条目没有可恢复的原始识别结果".into());
        }
        result["parsed"] = prior["original_parsed"].clone();
        result.as_object_mut().unwrap().remove("chord_review");
        return Ok(result);
    }
    let status = match decision {
        "confirm" => {
            let name = body["name"].as_str().unwrap_or("").trim();
            if name.is_empty() || name.chars().count() > 160 || name.chars().any(char::is_control) {
                return Err("请输入有效的和弦名称".into());
            }
            let diagram = scorelib::chords::normalize_diagram(&body["diagram"]);
            if diagram.is_null() {
                return Err("无效的和弦指法，请检查弦数、品位、手指与横按".into());
            }
            if !result["parsed"].is_object() {
                result["parsed"] = json!({});
            }
            result["parsed"]["kind"] = json!("chord_diagram");
            result["parsed"]["text"] = json!(name);
            result["parsed"]["diagram"] = diagram;
            "confirmed"
        }
        "reject" => {
            if !result["parsed"].is_object() {
                result["parsed"] = json!({});
            }
            result["parsed"]["kind"] = json!("other");
            result["parsed"]["diagram"] = Value::Null;
            "rejected"
        }
        _ => return Err("无效的和弦库操作".into()),
    };
    result["chord_review"] = json!({"source":"manual","status":status,
        "original_parsed":if manual {&prior["original_parsed"]} else {&annotation["parsed"]}});
    Ok(result)
}

pub fn edit(
    root: &Path,
    sid: &str,
    index: usize,
    body: &Value,
    expected: Option<&str>,
) -> std::result::Result<(), (u16, String)> {
    let mut state = load_session(root, sid).map_err(|e| (400, e.to_string()))?;
    revision(&state, expected)?;
    let part_id = body["part_id"].as_str().ok_or((400, "请选择音轨".into()))?;
    let info_path = state["info"]
        .as_str()
        .ok_or((409, "请先识别谱面信息".into()))?;
    let mut info = crate::view::read_json(Path::new(info_path)).map_err(|e| (400, e))?;
    let mut recognized = state["recognition"]
        .as_str()
        .map(|path| crate::view::read_json(Path::new(path)))
        .transpose()
        .map_err(|e| (400, e))?;
    let info_annotation = annotation_mut(&mut info, part_id, index).map_err(|e| (400, e))?;
    let original = match &mut recognized {
        Some(source) => {
            let annotation = annotation_mut(source, part_id, index).map_err(|e| (400, e))?;
            if annotation["page"] != info_annotation["page"]
                || annotation["bbox"] != info_annotation["bbox"]
            {
                return Err((409, "谱头库与识别结果不一致，请重新打开项目".into()));
            }
            annotation.clone()
        }
        None => info_annotation.clone(),
    };
    let edited = reviewed(&original, body).map_err(|e| (400, e))?;
    *info_annotation = edited.clone();
    let directory = root.join(sid);
    let output = tempfile::Builder::new()
        .prefix("chord_library_")
        .tempdir_in(&directory)
        .map_err(|e| (400, e.to_string()))?;
    let info_path = output.path().join("info.json");
    json_file(&info_path, &info).map_err(|e| (400, e))?;
    state["info"] = json!(info_path);
    if let Some(mut source) = recognized {
        *annotation_mut(&mut source, part_id, index).map_err(|e| (400, e))? = edited;
        source["info"] = json!(info_path);
        state["recognition"] =
            json!(write_recognition(output.path(), source).map_err(|e| (400, e))?);
    }
    state["export"] = Value::Null;
    state.as_object_mut().unwrap().remove("ocr_task");
    state["revision"] = json!(state["revision"].as_u64().ok_or((400, "无效版本".into()))? + 1);
    write_session_atomic(root, sid, &state).map_err(|e| (400, e.to_string()))?;
    let _ = output.keep();
    let _ = fs::remove_file(directory.join("job.json"));
    Ok(())
}
