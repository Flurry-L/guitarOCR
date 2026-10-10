//! Learned localization quality for existing layout proposals, before NMS.
use crate::image_boundary::{check_dimensions, Result};
use crate::image_transforms::resize_rgb_cubic;
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

pub const WIDTH: usize = 768;
pub const HEIGHT: usize = 256;
const LABELS: [&str; 3] = ["measure_tab", "measure_notation", "measure_both"];
const MAX_BATCH: usize = 32;
// Smaller batches reduce desktop peak memory and measured CPU latency.
const RUNTIME_BATCH: usize = 8;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn validate_contract(contract: &Value) -> Result<f64> {
    let threshold = contract["selection"]["minimum_quality"]
        .as_f64()
        .filter(|v| v.is_finite() && (0.0..1.0).contains(v))
        .ok_or("Invalid layout quality threshold")?;
    let mut columns = vec!["raw_detector_score", LABELS[0], LABELS[1], LABELS[2]];
    columns.extend(FEATURE_NAMES);
    if contract["schema"] != "layout_candidate_quality"
        || contract["view_features"] != agreement_contract()
        || contract["preprocessing"]
            != json!({"context_fraction":0.2,
            "resize":"contain_portable_cubic","rounding":"half_up",
            "rgb_padding":255,"mask":"half_open_binary"})
        || contract["image_input"]["name"] != "image"
        || contract["image_input"]["dtype"] != "float32"
        || contract["image_input"]["shape"] != json!(["N", 4, HEIGHT, WIDTH])
        || contract["image_input"]["channels"] != json!(["R", "G", "B", "candidate_mask"])
        || contract["image_input"]["range"] != json!([0, 1])
        || contract["proposal_input"]["name"] != "proposal"
        || contract["proposal_input"]["dtype"] != "float32"
        || contract["proposal_input"]["shape"] != json!(["N", columns.len()])
        || contract["proposal_input"]["columns"] != json!(columns)
        || contract["output"]["name"] != "quality"
        || contract["output"]["dtype"] != "float32"
        || contract["output"]["shape"] != json!(["N"])
        || contract["selection"]["score"] != "rank_product"
        || contract["maximum_batch"] != MAX_BATCH
    {
        return Err("Unsupported layout quality model contract".into());
    }
    Ok(threshold)
}

fn is_measure(box_: &Value) -> bool {
    box_["label"].as_str().is_some_and(|s| LABELS.contains(&s))
}

fn apply_quality(box_: &mut Value, quality: f32) {
    let raw = box_["score"].as_f64().unwrap();
    box_["detector_score"] = box_["score"].clone();
    box_["quality_score"] = json!(quality);
    box_["fusion_score"] = json!(raw * quality as f64);
}

pub fn prepare_crop(page: &RgbImage, coordinate: [f64; 4]) -> Result<Vec<f32>> {
    check_dimensions(page.width(), page.height())?;
    if coordinate.iter().any(|v| !v.is_finite() || v.fract() != 0.) {
        return Err("Layout quality requires finite integer candidate coordinates".into());
    }
    let a = coordinate[0].clamp(0., page.width() as f64) as u32;
    let b = coordinate[1].clamp(0., page.height() as f64) as u32;
    let c = coordinate[2].clamp(0., page.width() as f64) as u32;
    let d = coordinate[3].clamp(0., page.height() as f64) as u32;
    if c <= a || d <= b {
        return Err("Layout quality candidate is empty".into());
    }
    let px = (((c - a) as f64 * 0.2).ceil() as u32).max(1);
    let py = (((d - b) as f64 * 0.2).ceil() as u32).max(1);
    let x = a.saturating_sub(px);
    let y = b.saturating_sub(py);
    let r = (c + px).min(page.width());
    let s = (d + py).min(page.height());
    let scale = (WIDTH as f64 / (r - x) as f64).min(HEIGHT as f64 / (s - y) as f64);
    let nw = (((r - x) as f64 * scale + 0.5).floor() as u32).max(1);
    let nh = (((s - y) as f64 * scale + 0.5).floor() as u32).max(1);
    let crop = image::imageops::crop_imm(page, x, y, r - x, s - y).to_image();
    let resized = resize_rgb_cubic(&crop, nw, nh);
    let left = (WIDTH - nw as usize) / 2;
    let top = (HEIGHT - nh as usize) / 2;
    let area = WIDTH * HEIGHT;
    let mut pixels = vec![1f32; 4 * area];
    pixels[3 * area..].fill(0.);
    for (u, v, p) in resized.enumerate_pixels() {
        let offset = (top + v as usize) * WIDTH + left + u as usize;
        for channel in 0..3 {
            pixels[channel * area + offset] = p[channel] as f32 / 255.;
        }
    }
    let mx = |v: u32| left + (((v - x) as f64 * nw as f64 / (r - x) as f64 + 0.5).floor() as usize);
    let my = |v: u32| top + (((v - y) as f64 * nh as f64 / (s - y) as f64 + 0.5).floor() as usize);
    for v in my(b)..my(d) {
        pixels[3 * area + v * WIDTH + mx(a)..3 * area + v * WIDTH + mx(c)].fill(1.);
    }
    Ok(pixels)
}

pub struct LayoutQualityReader {
    session: Session,
    threshold: f64,
    proposal_width: usize,
}
impl LayoutQualityReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let encoded = session
            .metadata()
            .map_err(err)?
            .custom("guitarocr.layout_quality")
            .map_err(err)?
            .ok_or("Missing layout quality model contract")?;
        let contract: Value = serde_json::from_str(&encoded).map_err(err)?;
        let threshold = validate_contract(&contract)?;
        let proposal_width = contract["proposal_input"]["shape"][1].as_u64().unwrap() as usize;
        for (name, dimensions) in [
            ("image", vec![-1, 4, HEIGHT as i64, WIDTH as i64]),
            ("proposal", vec![-1, proposal_width as i64]),
        ] {
            let input = session
                .inputs
                .iter()
                .find(|v| v.name == name)
                .ok_or("Missing layout quality input")?;
            match &input.input_type {
                ort::value::ValueType::Tensor { shape, .. }
                    if shape.len() == dimensions.len()
                        && shape[0] < 0
                        && shape[1..] == dimensions[1..] => {}
                _ => return Err("Unsupported layout quality input shape".into()),
            }
        }
        if !session.outputs.iter().any(|v| v.name == "quality") {
            return Err("Missing layout quality output".into());
        }
        Ok(Self {
            session,
            threshold,
            proposal_width,
        })
    }

    /// Bound peak allocations independently of page size and number of views.
    pub fn rank(
        &mut self,
        page: &RgbImage,
        boxes: &[Value],
        views: &[View],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Value>> {
        let indices: Vec<_> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| is_measure(b))
            .map(|(i, _)| i)
            .collect();
        let agreements = agreement_features(
            &indices
                .iter()
                .map(|&i| boxes[i].clone())
                .collect::<Vec<_>>(),
            boxes,
            page.dimensions(),
            views,
            cancelled,
        )?;
        let mut scored = boxes.to_vec();
        for (batch_index, indices) in indices.chunks(RUNTIME_BATCH).enumerate() {
            if cancelled() {
                return Err("Layout recognition cancelled".into());
            }
            let mut images = Vec::with_capacity(indices.len() * 4 * HEIGHT * WIDTH);
            let mut proposals = Vec::with_capacity(indices.len() * self.proposal_width);
            for (local, &index) in indices.iter().enumerate() {
                let b = &boxes[index];
                let coordinates = b["coordinate"]
                    .as_array()
                    .filter(|c| c.len() == 4)
                    .ok_or("Invalid layout candidate coordinates")?;
                let mut coordinate = [0.; 4];
                for (i, v) in coordinates.iter().enumerate() {
                    coordinate[i] = v.as_f64().ok_or("Invalid layout candidate coordinate")?;
                }
                images.extend(prepare_crop(page, coordinate)?);
                let score = b["score"]
                    .as_f64()
                    .filter(|s| s.is_finite() && (0.0..=1.0).contains(s))
                    .ok_or("Invalid layout detector score")?;
                proposals.push(score as f32);
                proposals.extend(
                    LABELS
                        .iter()
                        .map(|s| if b["label"] == *s { 1. } else { 0. }),
                );
                proposals.extend(agreements[batch_index * RUNTIME_BATCH + local]);
            }
            let images =
                Tensor::from_array(([indices.len(), 4, HEIGHT, WIDTH], images)).map_err(err)?;
            let proposals = Tensor::from_array(([indices.len(), self.proposal_width], proposals))
                .map_err(err)?;
            let output = self
                .session
                .run(ort::inputs!["image"=>images,"proposal"=>proposals])
                .map_err(err)?;
            let (shape, values) = output["quality"].try_extract_tensor::<f32>().map_err(err)?;
            if shape.as_ref() != [indices.len() as i64]
                || values
                    .iter()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err("Invalid layout quality predictions".into());
            }
            for (&index, &quality) in indices.iter().zip(values) {
                apply_quality(&mut scored[index], quality);
            }
        }
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        Ok(scored
            .into_iter()
            .filter(|b| !is_measure(b) || b["quality_score"].as_f64().unwrap() > self.threshold)
            .collect())
    }
}

// Relationships across executed model views are inputs to this quality head.
pub type View = [i64; 5];
const FEATURE_NAMES: [&str; 8] = [
    "is_full_page_view",
    "other_coverable_view_fraction",
    "same_label_mean_iou",
    "same_label_mean_score_iou",
    "same_label_max_iou",
    "same_label_max_score_iou",
    "any_label_mean_iou",
    "any_label_max_iou",
];

fn agreement_contract() -> Value {
    json!({
        "version":"view_agreement_v1", "features":FEATURE_NAMES,
        "edge_guard_fraction":0.05, "minimum_reference_score":0.25,
        "reference_eligibility":"typed_measure; raw_score_strictly_greater_than_0.25; existing_edge_guard_accepted",
        "views":"deduplicated_actual_executed_measure_views_including_full_page_and_empty_views; annotations_excluded",
        "view_geometry":"[scale,x,y,width,height]; full_page=[-1,0,0,page_width,page_height]",
        "box_geometry":"page_xyxy_half_open_clipped_to_page; IoU=intersection/union; no_plus_one",
        "visibility":"existing_internal_edge_guard",
        "query_origin":"exclude_entire_origin_view; absent_view_means_full_page",
        "per_view":"independent_max_iou_and_max_score_times_iou",
        "cross_view":"mean_and_max_over_other_coverable_actual_views_including_empty_views",
        "coverable_fraction":"number_other_coverable_views/max(1,number_all_other_actual_views)",
        "empty_evidence":0, "arithmetic":"float64_then_cast_float32",
        "candidate_scope":"All raw eligible measure candidates before quality scoring or batching."
    })
}

fn coordinate(box_: &Value, (width, height): (u32, u32)) -> Result<[f64; 4]> {
    let values = box_["coordinate"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("Invalid candidate geometry")?;
    let mut result = [0.; 4];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("Non-finite candidate geometry")?;
    }
    result[0] = result[0].max(0.);
    result[1] = result[1].max(0.);
    result[2] = result[2].min(width as f64);
    result[3] = result[3].min(height as f64);
    if result[2] <= result[0] || result[3] <= result[1] {
        return Err("Empty candidate geometry".into());
    }
    Ok(result)
}

fn origin(box_: &Value, full: View) -> Result<View> {
    match box_.get("view").filter(|v| !v.is_null()) {
        None => Ok(full),
        Some(value) => {
            let values = value
                .as_array()
                .filter(|v| v.len() == 5)
                .ok_or("Invalid candidate source view")?;
            let mut view = [0; 5];
            for (index, value) in values.iter().enumerate() {
                view[index] = value.as_i64().ok_or("Invalid candidate source view")?;
            }
            Ok(view)
        }
    }
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    intersection
        / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection).max(1e-9)
}

/// Each other view contributes once, including evaluated views without boxes.
/// References are the complete raw eligible candidate list, never a batch.
fn agreement_features(
    queries: &[Value],
    references: &[Value],
    size: (u32, u32),
    views: &[View],
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<[f32; 8]>> {
    let full = [-1, 0, 0, size.0 as i64, size.1 as i64];
    let mut actual = Vec::new();
    for &view in views {
        let [_, x, y, width, height] = view;
        if x < 0
            || y < 0
            || width <= 0
            || height <= 0
            || width > size.0 as i64
            || height > size.1 as i64
            || x > size.0 as i64 - width
            || y > size.1 as i64 - height
        {
            return Err("Invalid actual layout view".into());
        }
        if !actual.contains(&view) {
            actual.push(view);
        }
    }
    if actual.is_empty() {
        actual.push(full);
    }
    let mut grouped = vec![vec![]; actual.len()];
    for box_ in references {
        if !matches!(
            box_["label"].as_str(),
            Some("measure_tab" | "measure_notation" | "measure_both")
        ) {
            continue;
        }
        let score = box_["score"]
            .as_f64()
            .filter(|v| v.is_finite() && *v <= 1.)
            .ok_or("Invalid reference detector score")?;
        if score <= 0.25 {
            continue;
        }
        let source = origin(box_, full)?;
        let index = actual
            .iter()
            .position(|v| *v == source)
            .ok_or("Reference source view was not evaluated")?;
        grouped[index].push((
            coordinate(box_, size)?,
            box_["label"].as_str().unwrap(),
            score,
        ));
    }
    let mut result = Vec::with_capacity(queries.len());
    for box_ in queries {
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        let source = origin(box_, full)?;
        if !actual.contains(&source) {
            return Err("Query source view was not evaluated".into());
        }
        let coord = coordinate(box_, size)?;
        let [a, b, c, d] = coord;
        let mut sums = [0.; 3];
        let mut maxima = [0f64; 3];
        let mut visible = 0;
        for (view, references) in actual.iter().zip(&grouped) {
            if *view == source {
                continue;
            }
            let [_, x, y, w, h] = view.map(|v| v as f64);
            let guard = w.min(h) * 0.05;
            if (x > 0. && a - x <= guard)
                || (y > 0. && b - y <= guard)
                || (x + w < size.0 as f64 && c - x >= w - guard)
                || (y + h < size.1 as f64 && d - y >= h - guard)
            {
                continue;
            }
            visible += 1;
            let mut per_view = [0f64; 3];
            for &(other, label, score) in references {
                let overlap = iou(coord, other);
                per_view[2] = per_view[2].max(overlap);
                if box_["label"] == label {
                    per_view[0] = per_view[0].max(overlap);
                    per_view[1] = per_view[1].max(overlap * score);
                }
            }
            for i in 0..3 {
                sums[i] += per_view[i];
                maxima[i] = maxima[i].max(per_view[i]);
            }
        }
        let mean = sums.map(|v| v / visible.max(1) as f64);
        result.push([
            if source[0] == -1 { 1. } else { 0. },
            (visible as f64 / actual.len().saturating_sub(1).max(1) as f64) as f32,
            mean[0] as f32,
            mean[1] as f32,
            maxima[0] as f32,
            maxima[1] as f32,
            mean[2] as f32,
            maxima[2] as f32,
        ]);
    }
    Ok(result)
}
