//! Opening text proposals and preservation of manually supplied titles.
use super::*;
use crate::text_regions::TextRegionReader;

/// Find additional text above each part’s first detected staff.
pub(super) fn opening_annotations(
    records: &[Value],
    regions: &[Value],
    reader: &mut TextRegionReader,
) -> Result<Vec<Value>> {
    let mut first = BTreeMap::new();
    for row in records {
        first.entry(text(row, "part_id", "part-1")).or_insert(row);
    }
    let mut result: Vec<Value> = vec![];
    for (part, row) in first {
        let Some(path) = row["source_page"].as_str() else {
            continue;
        };
        let image = image_boundary::open_rgb(Path::new(path))?;
        let bottom = bbox(row)?[1].trunc().min(f64::from(image.height()));
        let top = records
            .iter()
            .filter(|r| r["page"] == row["page"])
            .filter_map(|r| bbox(r).ok())
            .map(|b| b[1] + b[3])
            .filter(|b| *b <= bottom)
            .fold(0., f64::max)
            .trunc()
            .max(0.);
        if bottom <= top {
            continue;
        }
        let crop = direct_crop(&image, [0., top, f64::from(image.width()), bottom - top])?;
        for proposal in reader.predict(&crop)? {
            let mut b = bbox(&proposal)?;
            b[1] += top;
            let covered = regions
                .iter()
                .chain(&result)
                .filter(|r| {
                    r["page"] == row["page"]
                        && matches!(
                            text(r, "kind", ""),
                            "annotation"
                                | "transposition"
                                | "tempo"
                                | "title"
                                | "subtitle"
                                | "credit"
                                | "tuning"
                                | "header_text"
                        )
                })
                .filter_map(|r| bbox(r).ok())
                .any(|a| {
                    let overlap = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.)
                        * ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.);
                    overlap / (b[2] * b[3]) > 0.7
                });
            if !covered {
                result.push(json!({"kind":"annotation", "page":row["page"], "bbox":b,
                    "part_id":part, "source_page":path, "source":"text_region_network",
                    "text_detection_score":proposal["score"]}));
            }
        }
    }
    Ok(result)
}

pub(super) fn resolve_title(metadata: &mut Value, predictions: &[Value], filename: &str) {
    let title = cleaned(&metadata["title"])
        .as_str()
        .unwrap_or("")
        .to_owned();
    let detected = predictions
        .iter()
        .any(|p| p["kind"] == "title" || p["kind"] == "header" || p["parsed"]["kind"] == "title");
    metadata["title"] = json!(title);
    metadata["title_source"] = json!(if title.is_empty() { "missing" } else { "ocr" });
    metadata["title_fallback"] = json!(filename);
    metadata["title_needs_review"] = json!(title.is_empty());
    metadata["title_warning"] = if title.is_empty() {
        json!(if detected {
            "已检测到曲名区域，但未能读出曲名。请对照原谱填写；文件名仅作为项目显示名称。"
        } else {
            "尚未读取到曲名，请对照原谱填写；文件名仅作为项目显示名称。"
        })
    } else {
        Value::Null
    };
}

/// A changed title is manual, including clearing a previously populated title.
pub(crate) fn mark_manual(metadata: &mut Value, title: &str) {
    let previous_warning = metadata["title_warning"].clone();
    if let Some(warnings) = metadata["warnings"].as_array_mut() {
        warnings.retain(|warning| warning != &previous_warning);
    }
    metadata["title"] = json!(title);
    metadata["title_source"] = json!("manual");
    metadata["title_needs_review"] = json!(false);
    metadata["title_warning"] = Value::Null;
}

pub(crate) fn copy_title_provenance(metadata: &mut Value, source: &Value) {
    let old_warning = metadata["title_warning"].clone();
    if let Some(warnings) = metadata["warnings"].as_array_mut() {
        warnings.retain(|warning| warning != &old_warning);
    }
    for key in [
        "title",
        "title_source",
        "title_fallback",
        "title_needs_review",
        "title_warning",
    ] {
        if let Some(value) = source.get(key) {
            metadata[key] = value.clone();
        }
    }
    if let Some(warning) = source["title_warning"].as_str() {
        if !metadata["warnings"].is_array() {
            metadata["warnings"] = json!([]);
        }
        let warnings = metadata["warnings"].as_array_mut().unwrap();
        if !warnings.iter().any(|w| w == warning) {
            warnings.push(json!(warning));
        }
    }
}

/// Re-reading OCR may refresh other metadata, but never replaces a user's title.
pub(crate) fn preserve_manual_title(previous: &Value, information: &mut Value) {
    if previous["document_metadata"]["title_source"] != "manual" {
        return;
    }
    let title = text(previous, "title", "");
    information["title"] = json!(title);
    mark_manual(&mut information["document_metadata"], title);
    if let Some(parts) = information["parts"].as_array_mut() {
        for part in parts {
            part["title"] = json!(title);
            mark_manual(&mut part["document_metadata"], title);
        }
    }
}
