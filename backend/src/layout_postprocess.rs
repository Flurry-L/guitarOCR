//! Port of research/inference/layout/postprocess.py. ONNX's per-class NMS is not sufficient:
//! this stage orders measures, resolves competing spans, reconciles paired
//! notation/TAB rows and recovers only gaps supported by printed barlines.
use crate::image_boundary::{check_dimensions, round_even, Result};
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
    fn height(&self) -> f64 {
        self.xy[3] - self.xy[1]
    }
    fn center(&self) -> f64 {
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

fn select_nonoverlapping(mut row: Vec<Detection>) -> Vec<Detection> {
    row.sort_by(|a, b| {
        a.xy[2]
            .total_cmp(&b.xy[2])
            .then(a.xy[0].total_cmp(&b.xy[0]))
    });
    let mut scores = vec![0.];
    let mut choices: Vec<Vec<usize>> = vec![vec![]];
    for (i, b) in row.iter().enumerate() {
        let prev = (0..i)
            .rev()
            .find(|&j| row[j].xy[2] <= b.xy[0] + 6f64.max(0.08 * b.width().min(row[j].width())))
            .map(|j| j + 1)
            .unwrap_or(0);
        let selected = scores[prev] + b.score * b.score;
        if selected > *scores.last().unwrap() {
            scores.push(selected);
            let mut selection = choices[prev].clone();
            selection.push(i);
            choices.push(selection);
        } else {
            scores.push(*scores.last().unwrap());
            choices.push(choices.last().unwrap().clone());
        }
    }
    let mut selected: Vec<Detection> = choices
        .last()
        .unwrap()
        .iter()
        .map(|&i| row[i].clone())
        .collect();
    selected.sort_by(|a, b| a.xy[0].total_cmp(&b.xy[0]));
    for i in 1..selected.len() {
        if selected[i - 1].xy[2] > selected[i].xy[0] {
            let boundary = (selected[i - 1].xy[2] + selected[i].xy[0]) / 2.;
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
        let same = rows.iter().position(|r| {
            (b.center() - median(r.iter().map(|m| m.center()).collect())).abs()
                <= 0.35
                    * b.height()
                        .min(median(r.iter().map(|m| m.height()).collect()))
        });
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

fn deduplicate_pairs(boxes: Vec<Detection>) -> Vec<Detection> {
    let pairs: Vec<_> = boxes
        .iter()
        .filter(|b| b.mode.as_deref() == Some("both"))
        .cloned()
        .collect();
    boxes
        .into_iter()
        .filter(|b| {
            !matches!(b.mode.as_deref(), Some("tab" | "notation"))
                || !pairs.iter().any(|pair| {
                    b.width() > 0.
                        && b.height() > 0.
                        && intersection(b, pair) >= 0.9 * b.width() * b.height()
                        && (0.8..=1.2).contains(&(b.width() / pair.width().max(1.)))
                        && pair.height() >= 1.5 * b.height()
                        && pair.score >= b.score
                })
        })
        .collect()
}
fn root(parents: &BTreeMap<usize, usize>, mut i: usize) -> usize {
    while parents[&i] != i {
        i = parents[&i];
    }
    i
}
fn reconcile_pairs(boxes: Vec<Detection>) -> Vec<Detection> {
    let mut parents: BTreeMap<_, _> = boxes.iter().map(|b| (b.system, b.system)).collect();
    for b in &boxes {
        for other in &boxes {
            let a = root(&parents, b.system);
            let c = root(&parents, other.system);
            if a == c {
                continue;
            }
            let tolerance = 0.15 * b.height().min(other.height());
            let pair = |mode: &str| {
                (b.mode.as_deref() == Some("both") && other.mode.as_deref() == Some(mode))
                    || (other.mode.as_deref() == Some("both") && b.mode.as_deref() == Some(mode))
            };
            if (pair("notation") && (b.xy[1] - other.xy[1]).abs() <= tolerance)
                || (pair("tab") && (b.xy[3] - other.xy[3]).abs() <= tolerance)
            {
                parents.insert(a.max(c), a.min(c));
            }
        }
    }
    if parents.iter().all(|(a, b)| a == b) {
        return boxes;
    }
    let mut rows: BTreeMap<usize, Vec<Detection>> = BTreeMap::new();
    for mut b in boxes {
        b.system = root(&parents, b.system);
        rows.entry(b.system).or_default().push(b);
    }
    rows.into_values().flat_map(select_nonoverlapping).collect()
}
fn luma(page: &RgbImage) -> Vec<u8> {
    page.pixels()
        .map(|p| {
            ((19595u32 * p[0] as u32 + 38470u32 * p[1] as u32 + 7471u32 * p[2] as u32 + 32768)
                >> 16) as u8
        })
        .collect()
}
struct Pixels {
    values: Vec<u8>,
    width: usize,
    height: usize,
}
impl Pixels {
    fn ink(&self, x: usize, y: usize) -> bool {
        self.values[y * self.width + x] < 150
    }
    fn barline(&self, x: f64, top: f64, bottom: f64) -> bool {
        let h = bottom - top;
        if h < 35. {
            return false;
        }
        let y0 = ((top + 0.04 * h) as i64).max(0).min(self.height as i64) as usize;
        let y1 = ((bottom - 0.04 * h) as i64).max(0).min(self.height as i64) as usize;
        let x0 = (x as i64 - 18).max(0).min(self.width as i64) as usize;
        let x1 = (x as i64 + 19).max(0).min(self.width as i64) as usize;
        if y1 <= y0 || x1 <= x0 {
            return false;
        }
        let mut longest = 0;
        for x in x0..x1 {
            let mut run = 0;
            for y in y0..y1 {
                if self.ink(x, y) {
                    run += 1;
                    longest = longest.max(run);
                } else {
                    run = 0;
                }
            }
        }
        longest as f64 >= 35f64.max(60f64.min(0.34 * h))
    }
    fn fragment(&self, b: &Detection, boxes: &[Detection]) -> bool {
        if b.score >= 0.5 {
            return false;
        }
        let [left, top, right, bottom] = b.xy;
        let w = b.width();
        let h = b.height();
        let x0 = round_even(left + w * 0.1).max(0.).min(self.width as f64) as usize;
        let x1 = round_even(right - w * 0.1).max(0.).min(self.width as f64) as usize;
        let y0 = round_even(top).max(0.).min(self.height as f64) as usize;
        let y1 = round_even(bottom).max(0.).min(self.height as f64) as usize;
        if x1 <= x0 || y1 <= y0 {
            return false;
        }
        let mut lines = 0;
        let mut previous = None;
        for y in y0..y1 {
            if (x0..x1).filter(|&x| self.ink(x, y)).count() as f64 / (x1 - x0) as f64 > 0.65 {
                if previous.map(|p| y - p > 2).unwrap_or(true) {
                    lines += 1;
                }
                previous = Some(y);
            }
        }
        if !(1..=3).contains(&lines) {
            return false;
        }
        let (mut above, mut below, mut overlaps) = (false, false, false);
        for other in boxes {
            if other.score < 0.7 {
                continue;
            }
            let [x0, y0, x1, y1] = other.xy;
            let overlap = bottom.min(y1) - top.max(y0);
            if right.min(x1) - left.max(x0) < 0.7 * w.min(x1 - x0) || overlap < -0.15 * h {
                continue;
            }
            above |= (y0 + y1) / 2. < top;
            below |= (y0 + y1) / 2. > bottom;
            overlaps |= overlap >= 0.15 * h;
        }
        above && below && overlaps
    }
}

pub fn refine_measure_boxes(page: &RgbImage, boxes: &[Value]) -> Result<Vec<Value>> {
    check_dimensions(page.width(), page.height())?;
    let boxes = reconcile_pairs(deduplicate_pairs(parse_boxes(boxes)?));
    let pixels = Pixels {
        values: luma(page),
        width: page.width() as usize,
        height: page.height() as usize,
    };
    let mut rows: BTreeMap<usize, Vec<Detection>> = BTreeMap::new();
    for b in &boxes {
        if !pixels.fragment(b, &boxes) {
            rows.entry(b.system).or_default().push(b.clone());
        }
    }
    let sizes: Vec<f64> = rows.values().map(|r| r.len() as f64).collect();
    let typical = if sizes.is_empty() {
        0.
    } else {
        round_even(median(if sizes.len() > 2 {
            sizes[1..sizes.len() - 1].to_vec()
        } else {
            sizes
        }))
    };
    let mut result = vec![];
    for (_, mut row) in rows {
        row.sort_by(|a, b| a.xy[0].total_cmp(&b.xy[0]));
        let repair = row.len() >= 3 && (row.len() as f64) < typical;
        let usual = median(row.iter().map(|b| b.width()).collect());
        let mut merged: Vec<Detection> = vec![];
        for current in row {
            if repair {
                if let Some(previous) = merged.last() {
                    if previous.mode == current.mode {
                        let [_, top, right, bottom] = previous.xy;
                        let [next_left, next_top, _, next_bottom] = current.xy;
                        let gap = next_left - right;
                        let ct = top.max(next_top);
                        let cb = bottom.min(next_bottom);
                        if cb > ct && gap >= 0.35 * usual && gap <= 1.5 * usual {
                            let middle = ((right + next_left) / 2.) as i64;
                            let x = middle.clamp(0, pixels.width as i64 - 1) as usize;
                            let y0 = ((ct + 0.2 * (cb - ct)) as i64).clamp(0, pixels.height as i64)
                                as usize;
                            let y1 = ((cb - 0.2 * (cb - ct)) as i64).clamp(0, pixels.height as i64)
                                as usize;
                            let ink = (y0..y1).filter(|&y| pixels.ink(x, y)).count();
                            if ink >= 3
                                && pixels.barline(right, ct, cb)
                                && pixels.barline(next_left, ct, cb)
                            {
                                let mut recovered = current.clone();
                                recovered.xy = [right, ct, next_left, cb];
                                recovered.score = previous.score.min(current.score);
                                recovered.value["score"] = json!(recovered.score);
                                recovered.value["geometry_source"] = json!("barline_gap_recovery");
                                merged.push(recovered);
                            }
                        }
                    }
                }
            }
            merged.push(current);
        }
        result.extend(merged.into_iter().enumerate().map(|(i, b)| b.emit(i)));
    }
    Ok(result)
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
    let boxes = deduplicate_pitch_boxes(boxes)?;
    let measures = refine_measure_boxes(page, &order_measure_boxes(&boxes, threshold)?)?;
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
    let mut pages: BTreeMap<u64, Vec<(usize, [f64; 4])>> = BTreeMap::new();
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
        pages.entry(page).or_default().push((index, b));
    }
    for (_, mut items) in pages {
        items.sort_by(|a, b| (a.1[1] + a.1[3] / 2.).total_cmp(&(b.1[1] + b.1[3] / 2.)));
        let mut rows: Vec<Vec<[f64; 4]>> = vec![];
        for (index, b) in items {
            let group = rows
                .iter()
                .position(|r| {
                    let center = median(r.iter().map(|b| b[1] + b[3] / 2.).collect());
                    let height = median(r.iter().map(|b| b[3]).collect());
                    (b[1] + b[3] / 2. - center).abs() <= 0.35 * b[3].min(height)
                })
                .unwrap_or(rows.len());
            if group == rows.len() {
                rows.push(vec![]);
            }
            rows[group].push(b);
            records[index]["row_index"] = json!(group);
            records[index]["system_index"] = json!(group);
        }
    }
    Ok(())
}

/// Complete the per-page crop contract from research/inference/layout/crops.py. Typed ONNX votes
/// win; staff geometry is used only when mode/measure evidence is missing.
pub fn prepare_page_prediction(
    page: &RgbImage,
    prediction: &mut Value,
    requested_mode: &str,
    allow_empty: bool,
) -> Result<()> {
    if !matches!(requested_mode, "auto" | "tab" | "notation" | "both") {
        return Err("Unknown notation mode".into());
    }
    let mut measures = prediction["measures"]
        .as_array()
        .cloned()
        .ok_or("Layout prediction needs measures")?;
    let vote = mode_vote(&measures);
    let mut source = "manual";
    let mut mode = if requested_mode == "auto" {
        source = "pp_doclayout";
        vote["mode"].as_str().map(str::to_owned)
    } else {
        Some(requested_mode.to_owned())
    };
    if requested_mode == "auto" && mode.is_none() {
        let classification = crate::staff_classifier::classify_notation_layout(page)?;
        mode = match classification["layout"].as_str() {
            Some("tab_only") => Some("tab".into()),
            Some("score_only") => Some("notation".into()),
            Some("score_tab") => Some("both".into()),
            _ => None,
        };
        source = "staff_geometry";
        prediction["staff_classification"] = classification;
    }
    if mode.is_none() && !measures.is_empty() {
        return Err("Cannot determine notation type; select a mode explicitly".into());
    }
    if measures.is_empty() {
        if let Some(mode) = &mode {
            measures = crate::staff_geometry::measure_boxes(page, mode)?;
        }
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
