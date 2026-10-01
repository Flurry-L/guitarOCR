//! App-level layout stages: validate boxes, write crops, then switch the session.
use crate::auxiliary_onnx::{initialize_runtime, LayoutDetector};
use crate::edit;
use crate::image_boundary as images;
use crate::layout_postprocess as geometry;
use crate::project::{load_session, write_session_atomic};
use crate::tasks::Progress;
use serde_json::{json, Value};
use std::path::Path;
type Result<T> = std::result::Result<T, String>;
fn mode(value: &str) -> Result<()> {
    if matches!(value, "auto" | "tab" | "notation" | "both") {
        Ok(())
    } else {
        Err("请选择TAB、五线谱或混合谱".into())
    }
}
fn xywh(value: &Value) -> Result<[f64; 4]> {
    let array = value.as_array().ok_or("框坐标必须有四个有限值")?;
    if array.len() != 4 {
        return Err("框坐标必须有四个有限值".into());
    }
    let mut out = [0.; 4];
    for (i, v) in array.iter().enumerate() {
        out[i] = v.as_f64().filter(|v| v.is_finite()).ok_or("无效框坐标")?;
    }
    Ok(out)
}
fn rectangle(image: &image::RgbImage, b: [f64; 4], padding: f64) -> Result<image::RgbImage> {
    let [x, y, w, h] = b;
    let x0 = images::round_even(x - padding).clamp(0., image.width() as f64) as u32;
    let y0 = images::round_even(y - padding).clamp(0., image.height() as f64) as u32;
    let x1 = images::round_even(x + w + padding).clamp(0., image.width() as f64) as u32;
    let y1 = images::round_even(y + h + padding).clamp(0., image.height() as f64) as u32;
    if x1 <= x0 || y1 <= y0 {
        return Err("无效的区域框".into());
    }
    Ok(image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image())
}
fn materialize(
    state: &Value,
    boxes: &[Value],
    requested: &str,
    output: &Path,
    automatic: bool,
    cancelled: &dyn Fn() -> bool,
) -> Result<Value> {
    mode(requested)?;
    if boxes.len() > 5000 {
        return Err("框数量过多".into());
    }
    let mut pages = state["pages"].as_array().ok_or("项目缺少页面")?.clone();
    let mut ordered = boxes.to_vec();
    ordered.sort_by_key(|b| b["page"].as_u64().unwrap_or(0));
    let mut records = Vec::new();
    let mut regions = Vec::new();
    let mut header = 0;
    let mut cached_page: Option<(std::path::PathBuf, image::RgbImage)> = None;
    for item in ordered {
        if cancelled() {
            return Err("区域处理已取消".into());
        }
        let page_number = item["page"]
            .as_u64()
            .filter(|n| *n >= 1 && *n <= pages.len() as u64)
            .ok_or("无效页码")? as usize;
        let page = &pages[page_number - 1];
        let kind = item["kind"].as_str().ok_or("无效区域类型")?;
        if !matches!(
            kind,
            "measure"
                | "header"
                | "tempo"
                | "clef"
                | "transposition"
                | "annotation"
                | "title"
                | "subtitle"
                | "credit"
                | "tuning"
                | "header_text"
        ) {
            return Err("无效区域类型".into());
        }
        let b = xywh(&item["bbox"])?;
        let [x, y, w, h] = b;
        let source = Path::new(page["image"].as_str().ok_or("页面图片缺失")?);
        if cached_page.as_ref().is_none_or(|(path, _)| path != source) {
            cached_page = Some((source.to_owned(), images::open_rgb(source)?));
        }
        let image = &cached_page.as_ref().unwrap().1;
        if !automatic
            && (x < 0.
                || y < 0.
                || w < 2.
                || h < 2.
                || x + w > image.width() as f64 + 0.01
                || y + h > image.height() as f64 + 0.01)
        {
            return Err("框必须位于页面内，宽高至少2像素".into());
        }
        if kind == "measure" {
            let notation = if requested != "auto" {
                requested
            } else {
                item["mode"]
                    .as_str()
                    .or(page["notation_mode"].as_str())
                    .ok_or("请先确定小节谱面类型")?
            };
            if !matches!(notation, "tab" | "notation" | "both") {
                return Err("无效的小节谱面类型".into());
            }
            let path = output.join(format!("m{:04}.png", records.len() + 1));
            images::grayscale_rgb(&images::crop_measure(&image, b)?)
                .save(&path)
                .map_err(|e| e.to_string())?;
            let mut row = item.clone();
            let obj = row.as_object_mut().ok_or("无效区域")?;
            obj.remove("kind");
            row["measure_number"] = json!(records.len() + 1);
            row["mode"] = json!(notation);
            row["mode_source"] = json!(if requested != "auto" {
                "manual"
            } else if automatic {
                "model"
            } else {
                "edited"
            });
            row["image"] = json!(path);
            row["source_page"] = json!(source);
            row["source_pdf"] = page["source_pdf"].clone();
            row["pdf_page"] = page["pdf_page"].clone();
            if row["system_index"].is_null() {
                row["system_index"] = json!(0);
            }
            if row["system_measure_index"].is_null() {
                row["system_measure_index"] = json!(records.len());
            }
            if row["geometry_source"].is_null() {
                row["geometry_source"] = json!(if automatic { "onnx" } else { "manual" });
            }
            records.push(row);
        } else {
            if kind == "header" {
                header += 1;
                if header > 1 {
                    return Err("请只保留一个标题信息框".into());
                }
            }
            let path = output.join(format!("region_{}.png", regions.len()));
            let crop = rectangle(
                &image,
                b,
                if automatic && kind != "header" {
                    6.
                } else {
                    0.
                },
            )?;
            crop.save(&path).map_err(|e| e.to_string())?;
            let mut region = item.clone();
            region.as_object_mut().unwrap().remove("annotation_type");
            region.as_object_mut().unwrap().remove("annotation_text");
            if matches!(kind, "annotation" | "transposition")
                && crate::pixel_refinement::diagram_strings(&crop).is_some()
            {
                region["annotation_type"] = json!("chord_diagram");
            }
            region["image"] = json!(path);
            regions.push(region);
        }
    }
    if records.is_empty() && !automatic {
        return Err("请至少添加一个小节框".into());
    }
    if !automatic {
        geometry::assign_manual_rows(&mut records)?;
    }
    for (index, page) in pages.iter_mut().enumerate() {
        let local = records
            .iter()
            .filter(|r| r["page"] == index + 1)
            .cloned()
            .collect::<Vec<_>>();
        let vote = geometry::mode_vote(&local);
        if !vote["mode"].is_null() {
            page["notation_mode"] = vote["mode"].clone();
            page["notation_mode_source"] = json!(if automatic {
                "model"
            } else if requested == "auto" {
                "edited"
            } else {
                "manual"
            });
        }
    }
    let selected = if requested == "auto" {
        geometry::mode_vote(&records)["mode"]
            .as_str()
            .unwrap_or("auto")
            .to_owned()
    } else {
        requested.to_owned()
    };
    Ok(
        json!({"schema_version":"1.0","stage":"layout","mode":selected,"inputs":state["inputs"],"pages":pages,"records":records,"regions":regions,"info_source":"image"}),
    )
}
fn publish(
    root: &Path,
    sid: &str,
    mut state: Value,
    stage: Value,
    output: tempfile::TempDir,
    requested: &str,
) -> Result<()> {
    let manifest = output.path().join("manifest.json");
    edit::json_file(&manifest, &stage)?;
    state["layout"] = json!(manifest);
    state["mode"] = stage["mode"].clone();
    state["mode_setting"] = json!(requested);
    state["pages"] = stage["pages"].clone();
    state["boxes"] = json!(stage["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| json!({"kind":"measure","page":r["page"],"bbox":r["bbox"],"mode":r["mode"]}))
        .chain(
            stage["regions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| json!({"kind":r["kind"],"page":r["page"],"bbox":r["bbox"],"annotation_type":r["annotation_type"]}))
        )
        .collect::<Vec<_>>());
    for key in ["info", "recognition", "export"] {
        state[key] = Value::Null;
    }
    state.as_object_mut().unwrap().remove("ocr_task");
    state["revision"] = json!(state["revision"].as_u64().ok_or("无效项目版本")? + 1);
    write_session_atomic(root, sid, &state).map_err(|e| e.to_string())?;
    let _ = output.keep();
    Ok(())
}
pub fn save_boxes(
    root: &Path,
    sid: &str,
    body: &Value,
    expected: Option<&str>,
) -> std::result::Result<(), (u16, String)> {
    let state = load_session(root, sid).map_err(|e| (400, e.to_string()))?;
    edit::revision(&state, expected)?;
    let requested = body["mode"].as_str().unwrap_or("auto");
    let boxes = body["boxes"].as_array().ok_or((400, "缺少区域框".into()))?;
    let output = tempfile::Builder::new()
        .prefix("layout_")
        .tempdir_in(root.join(sid))
        .map_err(|e| (400, e.to_string()))?;
    let stage = materialize(&state, boxes, requested, output.path(), false, &|| false)
        .map_err(|e| (400, e))?;
    publish(root, sid, state, stage, output, requested).map_err(|e| (400, e))
}
pub fn detect(
    root: &Path,
    sid: &str,
    ort: &Path,
    model: &Path,
    requested: &str,
    progress: &Progress,
) -> Result<()> {
    mode(requested)?;
    initialize_runtime(ort)?;
    let mut detector = LayoutDetector::new(model)?;
    let state = load_session(root, sid).map_err(|e| e.to_string())?;
    let pages = state["pages"].as_array().ok_or("项目缺少页面")?;
    let mut boxes = Vec::new();
    for (index, page) in pages.iter().enumerate() {
        progress.checkpoint()?;
        let image = images::open_rgb(Path::new(page["image"].as_str().ok_or("图片缺失")?))?;
        let mut found =
            detector.detect_page_with_cancel(&image, 0.25, true, &|| progress.cancelled())?;
        geometry::prepare_page_prediction(&image, &mut found, requested, true)?;
        for item in found["measures"].as_array().into_iter().flatten() {
            let mut item = item.clone();
            item["kind"] = json!("measure");
            item["page"] = json!(index + 1);
            boxes.push(item);
        }
        for item in ["tempo_regions", "pitch_regions", "header_regions"]
            .iter()
            .flat_map(|k| found[*k].as_array().into_iter().flatten())
        {
            let c = xywh(&item["coordinate"])?;
            let label = item["label"].as_str().unwrap_or("");
            let kind = match label {
                "clef_region" => "clef",
                "annotation_region" | "transposition_region" => "annotation",
                "tempo_region" => "tempo",
                _ => label.trim_end_matches("_region"),
            };
            boxes.push(json!({"kind":kind,"page":index+1,"bbox":[c[0],c[1],c[2]-c[0],c[3]-c[1]],"score":item["score"]}));
        }
        progress.update(
            json!({"done":index+1,"total":pages.len(),"message":"正在检测小节与谱面信息"}),
        )?;
    }
    let output = tempfile::Builder::new()
        .prefix("layout_")
        .tempdir_in(root.join(sid))
        .map_err(|e| e.to_string())?;
    let stage = materialize(&state, &boxes, requested, output.path(), true, &|| {
        progress.cancelled()
    })?;
    progress.checkpoint()?;
    publish(root, sid, state, stage, output, requested)
}
