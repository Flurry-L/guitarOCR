//! Port of research/inference/layout/postprocess.py. ONNX's per-class NMS is not sufficient:
//! orders and deduplicates model proposals. Missing boxes remain editable.
use crate::image_boundary::{check_dimensions, Result};
use image::RgbImage;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone)]
struct Detection {
    value: Value,
    xy: [f64; 4],
    score: f64,
    label: String,
    mode: Option<String>,
    system: usize,
}
impl Detection {
    fn parse(value: &Value) -> Result<Self> {
        let a = value["coordinate"]
            .as_array()
            .ok_or("Layout coordinate must be an array")?;
        if a.len() != 4 {
            return Err("Layout coordinate must contain four numbers".into());
        }
        let mut xy = [0.; 4];
        for (i, v) in a.iter().enumerate() {
            xy[i] = v
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or("Layout coordinate must be finite")?;
        }
        let score = value
            .get("score")
            .map(|v| {
                v.as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or("Layout score must be finite")
            })
            .transpose()?
            .unwrap_or(0.);
        if let Some(value) = value.get("fusion_score") {
            if value.as_f64().is_none_or(|n| !n.is_finite()) {
                return Err("Layout fusion score must be finite".into());
            }
        }
        let label = value["label"]
            .as_str()
            .ok_or("Layout label must be a string")?
            .to_owned();
        let mode = mode_for_label(&label)
            .map(str::to_owned)
            .or_else(|| value["mode"].as_str().map(str::to_owned));
        let system = value["system_index"].as_u64().unwrap_or(0) as usize;
        Ok(Self {
            value: value.clone(),
            xy,
            score,
            label,
            mode,
            system,
        })
    }
    fn width(&self) -> f64 {
        self.xy[2] - self.xy[0]
    }
    fn rank_score(&self) -> f64 {
        self.value["fusion_score"].as_f64().unwrap_or(self.score)
    }
    fn height(&self) -> f64 {
        self.xy[3] - self.xy[1]
    }
    fn center(&self) -> f64 {
        if let Some([top, bottom]) = staff_band(&self.value) {
            return (top + bottom) / 2.;
        }
        self.envelope_center()
    }
    fn envelope_center(&self) -> f64 {
        (self.xy[1] + self.xy[3]) / 2.
    }
    fn emit(mut self, index: usize) -> Value {
        self.value["coordinate"] = json!(self.xy);
        self.value["bbox"] = json!([self.xy[0], self.xy[1], self.width(), self.height()]);
        self.value["system_index"] = json!(self.system);
        self.value["system_measure_index"] = json!(index);
        if let Some(mode) = self.mode {
            self.value["mode"] = json!(mode);
        }
        self.value
    }
}

pub(crate) fn staff_band(value: &Value) -> Option<[f64; 2]> {
    let values = value["staff_band"].as_array().filter(|a| a.len() == 2)?;
    let pair = [values[0].as_f64()?, values[1].as_f64()?];
    (pair.iter().all(|v| v.is_finite()) && pair[1] > pair[0]).then_some(pair)
}

fn same_row(b: &Detection, row: &[Detection]) -> bool {
    if let Some([top, bottom]) = staff_band(&b.value) {
        let observed: Vec<_> = row.iter().filter_map(|m| staff_band(&m.value)).collect();
        if !observed.is_empty() {
            return observed.iter().any(|[a, c]| {
                (top - a).abs().max((bottom - c).abs()) <= 0.15 * (bottom - top).min(c - a)
            });
        }
    }
    let overlap = b.xy[3].min(median(row.iter().map(|m| m.xy[3]).collect()))
        - b.xy[1].max(median(row.iter().map(|m| m.xy[1]).collect()));
    overlap >= 0.65 * b.height().min(median(row.iter().map(|m| m.height()).collect()))
        // A tall envelope must not bridge two physically separate rows.
        && row.iter().all(|m| {
            b.xy[3].min(m.xy[3]) - b.xy[1].max(m.xy[1])
                >= 0.65 * b.height().min(m.height())
        })
}
fn mode_for_label(label: &str) -> Option<&str> {
    match label {
        "measure_tab" => Some("tab"),
        "measure_notation" => Some("notation"),
        "measure_both" => Some("both"),
        _ => None,
    }
}
fn pitch(label: &str) -> bool {
    matches!(
        label,
        "clef_region" | "annotation_region" | "transposition_region"
    )
}
fn header_text(label: &str) -> bool {
    crate::recognition::header_field(label.trim_end_matches("_region")).is_some()
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}
fn intersection(a: &Detection, b: &Detection) -> f64 {
    (a.xy[2].min(b.xy[2]) - a.xy[0].max(b.xy[0])).max(0.)
        * (a.xy[3].min(b.xy[3]) - a.xy[1].max(b.xy[1])).max(0.)
}
fn parse_boxes(boxes: &[Value]) -> Result<Vec<Detection>> {
    boxes.iter().map(Detection::parse).collect()
}

pub fn deduplicate_pitch_boxes(boxes: &[Value]) -> Result<Vec<Value>> {
    let mut ordered = parse_boxes(boxes)?;
    ordered.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = vec![];
    for row in ordered {
        if !(pitch(&row.label) || header_text(&row.label) || row.label == "tempo_region")
            || !kept.iter().any(|other| {
                let shared_text = (header_text(&row.label)
                    && (header_text(&other.label) || other.label == "annotation_region"))
                    || (header_text(&other.label) && row.label == "annotation_region");
                if other.label != row.label && !shared_text {
                    return false;
                }
                let hit = intersection(&row, other);
                let union = row.width().max(0.) * row.height().max(0.)
                    + other.width().max(0.) * other.height().max(0.)
                    - hit;
                union > 0. && hit / union > 0.8
            })
        {
            kept.push(row);
        }
    }
    Ok(kept.into_iter().map(|b| b.value).collect())
}

fn separate_interiors(first: &Detection, second: &Detection) -> bool {
    first.xy[2] <= second.xy[0]
        || ((first.xy[0] + first.xy[2]) / 2. < second.xy[0]
            && first.xy[2] < (second.xy[0] + second.xy[2]) / 2.)
}

fn select_nonoverlapping(mut row: Vec<Detection>) -> Vec<Detection> {
    row.sort_by(|a, b| {
        a.xy[2]
            .total_cmp(&b.xy[2])
            .then(a.xy[0].total_cmp(&b.xy[0]))
    });
    if row.is_empty() {
        return vec![];
    }
    let mut scores = Vec::new();
    let mut parents: Vec<Option<usize>> = Vec::new();
    for (i, b) in row.iter().enumerate() {
        let mut previous = None;
        for j in 0..i {
            if separate_interiors(&row[j], b) && previous.is_none_or(|old| scores[j] > scores[old])
            {
                previous = Some(j);
            }
        }
        scores.push(b.rank_score().powi(2) + previous.map_or(0., |j| scores[j]));
        parents.push(previous);
    }
    let mut best = 0;
    for i in 1..row.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    let mut next = Some(best);
    let mut selected = vec![];
    while let Some(i) = next {
        selected.push(row[i].clone());
        next = parents[i];
    }
    selected.reverse();
    let originals: Vec<Value> = selected
        .iter()
        .map(|b| {
            b.value
                .get("original_coordinate")
                .cloned()
                .unwrap_or_else(|| json!(b.xy))
        })
        .collect();
    for i in 1..selected.len() {
        if selected[i - 1].xy[2] > selected[i].xy[0] {
            let boundary = (selected[i - 1].xy[2] + selected[i].xy[0]) / 2.;
            for (index, other, side) in [(i - 1, i, "right"), (i, i - 1, "left")] {
                let own_score = selected[index].score;
                let other_score = selected[other].score;
                selected[index].value["original_coordinate"] = originals[index].clone();
                if !selected[index].value["boundary_review"].is_array() {
                    selected[index].value["boundary_review"] = json!([]);
                }
                selected[index].value["boundary_review"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({
                        "reason":"overlapping_model_boundaries","side":side,"boundary":boundary,
                        "own_coordinate":originals[index],"other_coordinate":originals[other],
                        "own_score":own_score,"other_score":other_score,
                    }));
            }
            selected[i - 1].xy[2] = boundary;
            selected[i].xy[0] = boundary;
        }
    }
    selected
}

pub fn order_measure_boxes(boxes: &[Value], minimum_score: f64) -> Result<Vec<Value>> {
    if !minimum_score.is_finite() {
        return Err("Measure threshold must be finite".into());
    }
    let mut candidates: Vec<_> = parse_boxes(boxes)?
        .into_iter()
        .filter(|b| {
            (b.label == "measure" || mode_for_label(&b.label).is_some())
                && b.score >= minimum_score
                && b.width() > 0.
                && b.height() > 0.
        })
        .collect();
    candidates.sort_by(|a, b| {
        a.center()
            .total_cmp(&b.center())
            .then(a.xy[0].total_cmp(&b.xy[0]))
    });
    let mut rows: Vec<Vec<Detection>> = vec![];
    for b in candidates {
        let same = rows.iter().position(|r| same_row(&b, r));
        if let Some(i) = same {
            rows[i].push(b);
        } else {
            rows.push(vec![b]);
        }
    }
    rows.sort_by(|a, b| {
        median(a.iter().map(|m| m.center()).collect())
            .total_cmp(&median(b.iter().map(|m| m.center()).collect()))
    });
    Ok(rows
        .into_iter()
        .enumerate()
        .flat_map(|(system, row)| {
            select_nonoverlapping(row)
                .into_iter()
                .enumerate()
                .map(move |(i, mut b)| {
                    b.system = system;
                    b.emit(i)
                })
        })
        .collect())
}

pub fn mode_vote(boxes: &[Value]) -> Value {
    let modes = ["tab", "notation", "both"];
    let mut votes = [0.; 3];
    for b in boxes {
        let mode = b["label"]
            .as_str()
            .and_then(mode_for_label)
            .or_else(|| b["mode"].as_str());
        if let Some(i) = modes.iter().position(|m| Some(*m) == mode) {
            let score = b.get("score").and_then(Value::as_f64).unwrap_or(1.);
            if score.is_finite() {
                votes[i] += score.max(0.);
            }
        }
    }
    let total: f64 = votes.iter().sum();
    let mut best = 0;
    for i in 1..3 {
        if votes[i] > votes[best] {
            best = i;
        }
    }
    json!({"mode":if total>0.{Some(modes[best])}else{None},"mode_vote_fraction":if total>0.{votes[best]/total}else{0.},"mode_votes":{"tab":votes[0],"notation":votes[1],"both":votes[2]}})
}

pub fn process_page(page: &RgbImage, boxes: &[Value], threshold: f64) -> Result<Value> {
    check_dimensions(page.width(), page.height())?;
    let boxes = deduplicate_pitch_boxes(boxes)?;
    let measures = order_measure_boxes(&boxes, threshold)?;
    let mut result = mode_vote(&measures);
    result["measures"] = json!(measures);
    result["tempo_regions"] = json!(boxes
        .iter()
        .filter(|b| b["label"] == "tempo_region" && b["score"].as_f64().unwrap_or(0.) >= threshold)
        .collect::<Vec<_>>());
    result["pitch_regions"] = json!(boxes
        .iter()
        .filter(|b| pitch(b["label"].as_str().unwrap_or(""))
            && b["score"].as_f64().unwrap_or(0.) >= threshold)
        .collect::<Vec<_>>());
    let mut headers: Vec<_> = boxes
        .iter()
        .filter(|b| {
            header_text(b["label"].as_str().unwrap_or(""))
                && b["score"].as_f64().unwrap_or(0.) >= threshold
        })
        .collect();
    headers.sort_by(|a, b| {
        a["coordinate"][1]
            .as_f64()
            .unwrap_or(0.)
            .total_cmp(&b["coordinate"][1].as_f64().unwrap_or(0.))
            .then(
                a["coordinate"][0]
                    .as_f64()
                    .unwrap_or(0.)
                    .total_cmp(&b["coordinate"][0].as_f64().unwrap_or(0.)),
            )
    });
    result["header_regions"] = json!(headers);
    Ok(result)
}

/// Editor records stay in the user's reading order. Only physical row identity
/// is assigned; there is deliberately no automatic span selection or dedup.
pub fn assign_manual_rows(records: &mut [Value]) -> Result<()> {
    let mut pages: BTreeMap<u64, Vec<(usize, Detection)>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        let page = record["page"]
            .as_u64()
            .filter(|&p| p > 0)
            .ok_or("Manual measure needs a positive page number")?;
        let values = record["bbox"]
            .as_array()
            .filter(|a| a.len() == 4)
            .ok_or("Manual bbox requires four numbers")?;
        let mut b = [0.; 4];
        for (i, v) in values.iter().enumerate() {
            b[i] = v
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or("Manual bbox must be finite")?;
        }
        if b[2] <= 0. || b[3] <= 0. {
            return Err("Manual bbox width/height must be positive".into());
        }
        pages.entry(page).or_default().push((
            index,
            Detection {
                value: record.clone(),
                xy: [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                score: 0.,
                label: "measure".into(),
                mode: None,
                system: 0,
            },
        ));
    }
    for (_, mut items) in pages {
        items.sort_by(|a, b| {
            a.1.center()
                .total_cmp(&b.1.center())
                .then(a.1.xy[0].total_cmp(&b.1.xy[0]))
        });
        let mut rows: Vec<Vec<Detection>> = vec![];
        let mut assignments = vec![];
        for (index, b) in items {
            let group = rows
                .iter()
                .position(|r| same_row(&b, r))
                .unwrap_or(rows.len());
            if group == rows.len() {
                rows.push(vec![]);
            }
            rows[group].push(b);
            assignments.push((index, group));
        }
        let mut physical_order: Vec<_> = (0..rows.len()).collect();
        physical_order.sort_by(|&a, &b| {
            median(rows[a].iter().map(|m| m.center()).collect())
                .total_cmp(&median(rows[b].iter().map(|m| m.center()).collect()))
        });
        let mut physical_ids = vec![0; rows.len()];
        for (physical_id, group) in physical_order.into_iter().enumerate() {
            physical_ids[group] = physical_id;
        }
        for (index, group) in assignments {
            records[index]["row_index"] = json!(physical_ids[group]);
            records[index]["system_index"] = json!(physical_ids[group]);
        }
    }
    Ok(())
}

/// Complete the per-page crop contract from research/inference/layout/crops.py. Typed ONNX votes
/// win; missing detections stay empty for the editor instead of invoking pixels.
pub fn prepare_page_prediction(
    page: &RgbImage,
    prediction: &mut Value,
    requested_mode: &str,
    allow_empty: bool,
) -> Result<()> {
    check_dimensions(page.width(), page.height())?;
    if !matches!(requested_mode, "auto" | "tab" | "notation" | "both") {
        return Err("Unknown notation mode".into());
    }
    let mut measures = prediction["measures"]
        .as_array()
        .cloned()
        .ok_or("Layout prediction needs measures")?;
    let vote = mode_vote(&measures);
    let mut source = "manual";
    let mode = if requested_mode == "auto" {
        source = "pp_doclayout";
        vote["mode"].as_str().map(str::to_owned)
    } else {
        Some(requested_mode.to_owned())
    };
    if mode.is_none() && !measures.is_empty() {
        return Err("Cannot determine notation type; select a mode explicitly".into());
    }
    if measures.is_empty() && !allow_empty {
        return Err(
            "No complete measures detected on this page; review the image or add measure boxes"
                .into(),
        );
    }
    for row in &mut measures {
        if row["label"].as_str().and_then(mode_for_label).is_some() {
            row["detected_mode"] = row["mode"].clone();
        }
        if row["geometry_source"].is_null() {
            row["geometry_source"] = json!("pp_doclayout");
        }
        if requested_mode != "auto" || row["mode"].is_null() {
            row["mode"] = json!(mode);
        }
        row["mode_source"] = json!(source);
    }
    prediction["notation_mode"] = json!(mode);
    prediction["notation_mode_source"] = json!(source);
    if !vote["mode"].is_null() {
        prediction["model_mode"] = vote["mode"].clone();
        prediction["mode_vote_fraction"] = vote["mode_vote_fraction"].clone();
    }
    prediction["measures"] = json!(measures);
    Ok(())
}
