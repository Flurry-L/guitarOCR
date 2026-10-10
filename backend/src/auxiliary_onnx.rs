//! Native inference for the exported PP-DocLayout and signature networks.
//! The layout graph decodes selected boxes and maps them to original pixels.
use crate::image_boundary::{check_dimensions, round_even, Result};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

pub const LAYOUT_LABELS: [&str; 11] = [
    "measure_tab",
    "measure_notation",
    "measure_both",
    "tempo_region",
    "clef_region",
    "annotation_region",
    "title_region",
    "subtitle_region",
    "credit_region",
    "tuning_region",
    "header_text_region",
];
pub use guitarocr_engine::onnx::initialize_runtime;
use guitarocr_engine::onnx::session;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub struct LayoutDetector {
    session: Session,
    input_size: u32,
    quality: crate::layout_quality::LayoutQualityReader,
}

/// Aspect-preserving dimensions shared by tensor preparation and box decoding.
pub fn layout_dimensions(width: u32, height: u32, size: u32) -> Result<(u32, u32)> {
    check_dimensions(width, height)?;
    validate_layout_size(size)?;
    let ratio = size as f64 / width.max(height) as f64;
    Ok((
        (width as f64 * ratio + 0.5).floor().max(1.) as u32,
        (height as f64 * ratio + 0.5).floor().max(1.) as u32,
    ))
}

fn validate_layout_size(size: u32) -> Result<()> {
    if !(32..=2048).contains(&size) || size % 32 != 0 {
        return Err(format!("Unsupported layout input size: {size}"));
    }
    Ok(())
}
impl LayoutDetector {
    pub fn detect_page(
        &mut self,
        page: &RgbImage,
        threshold: f32,
        include_tempo: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Value> {
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        let full = self.predict(page, threshold)?;
        let (candidates, views) =
            crate::layout_tiling::collect_model_tiles(page, full, cancelled, |tile| {
                self.predict(tile, threshold)
            })?;
        let ranked = self.quality.rank(page, &candidates, &views, cancelled)?;
        let full = crate::layout_tiling::fuse_measures(&ranked)?;
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        let mut boxes = crate::layout_postprocess::deduplicate_pitch_boxes(&full)?;
        if include_tempo && (page.width() > 2400 || page.height() > 3400) {
            let (width, height) = page.dimensions();
            let tw = (width as f64 * 0.55).ceil() as u32;
            let th = (height as f64 * 0.55).ceil() as u32;
            let is_region = |b: &Value| {
                matches!(
                    b["label"].as_str(),
                    Some(
                        "clef_region"
                            | "annotation_region"
                            | "transposition_region"
                            | "tempo_region"
                            | "title_region"
                            | "subtitle_region"
                            | "credit_region"
                            | "tuning_region"
                            | "header_text_region"
                    )
                )
            };
            let mut candidates: Vec<Value> =
                boxes.iter().filter(|b| is_region(b)).cloned().collect();
            for y in [0, height - th] {
                for x in [0, width - tw] {
                    if cancelled() {
                        return Err("Layout recognition cancelled".into());
                    }
                    let tile = image::imageops::crop_imm(page, x, y, tw, th).to_image();
                    for mut b in self.predict(&tile, threshold)? {
                        if !is_region(&b) {
                            continue;
                        }
                        let a = b["coordinate"][0].as_f64().unwrap();
                        let top = b["coordinate"][1].as_f64().unwrap();
                        let c = b["coordinate"][2].as_f64().unwrap();
                        let bottom = b["coordinate"][3].as_f64().unwrap();
                        if (x > 0 && a < 3.)
                            || (y > 0 && top < 3.)
                            || (x + tw < width && c > tw as f64 - 3.)
                            || (y + th < height && bottom > th as f64 - 3.)
                        {
                            continue;
                        }
                        b["coordinate"] = json!([
                            a + x as f64,
                            top + y as f64,
                            c + x as f64,
                            bottom + y as f64
                        ]);
                        b["geometry_source"] = json!("detail_crop");
                        candidates.push(b);
                    }
                }
            }
            candidates.sort_by(|a, b| {
                b["score"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&a["score"].as_f64().unwrap())
            });
            let mut kept: Vec<Value> = vec![];
            for b in candidates {
                let coords = |v: &Value| -> [f64; 4] {
                    std::array::from_fn(|i| v["coordinate"][i].as_f64().unwrap())
                };
                let [x, y, r, bottom] = coords(&b);
                let area = (r - x) * (bottom - y);
                if !kept.iter().any(|other| {
                    if other["label"] != b["label"] {
                        return false;
                    }
                    let [a, t, c, d] = coords(other);
                    let hit = (r.min(c) - x.max(a)).max(0.) * (bottom.min(d) - y.max(t)).max(0.);
                    hit / (area + (c - a) * (d - t) - hit).max(1.) > 0.4
                }) {
                    kept.push(b);
                }
            }
            boxes.retain(|b| !is_region(b));
            boxes.extend(kept);
            boxes = crate::layout_postprocess::deduplicate_pitch_boxes(&boxes)?;
        }
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        let mut output = crate::layout_postprocess::process_page(page, &boxes, threshold as f64)?;
        if !include_tempo {
            output["tempo_regions"] = json!([]);
            output["pitch_regions"] = json!([]);
            output["header_regions"] = json!([]);
        }
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        Ok(output)
    }

    pub fn new(model: &Path, quality_model: &Path) -> Result<Self> {
        let session = session(model)?;
        for name in ["image", "im_shape", "scale_factor"] {
            if !session.inputs.iter().any(|v| v.name == name) {
                return Err(format!("Layout model missing input {name}"));
            }
        }
        if !session.outputs.iter().any(|v| v.name == "fetch_name_0") {
            return Err("Layout model missing fetch_name_0 output".into());
        }
        let contract = session
            .metadata()
            .map_err(err)?
            .custom("guitarocr.layout_preprocess")
            .map_err(err)?;
        if contract.as_deref() != Some("letterbox_v1") {
            return Err(
                "Production layout requires explicit letterbox_v1 preprocessing metadata".into(),
            );
        }
        let input_size = session
            .metadata()
            .map_err(err)?
            .custom("guitarocr.layout_input_size")
            .map_err(err)?
            .as_deref()
            .ok_or("Missing layout input size metadata")?
            .parse::<u32>()
            .map_err(err)?;
        if input_size != 1280 {
            return Err("Layout quality requires the trained 1280 letterbox detector".into());
        }
        let image_input = session
            .inputs
            .iter()
            .find(|value| value.name == "image")
            .unwrap();
        match &image_input.input_type {
            ort::value::ValueType::Tensor { shape, .. }
                if shape.len() == 4
                    && shape[2..]
                        .iter()
                        .all(|&dimension| dimension < 0 || dimension == input_size as i64) => {}
            _ => return Err("Layout preprocessing size does not match the graph input".into()),
        }
        Ok(Self {
            session,
            input_size,
            quality: crate::layout_quality::LayoutQualityReader::new(quality_model)?,
        })
    }

    pub fn predict(&mut self, page: &RgbImage, threshold: f32) -> Result<Vec<Value>> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("Invalid layout threshold".into());
        }
        let input = layout_input(page, self.input_size)?;
        let (resized_width, resized_height) = layout_dimensions(
            page.width(),
            page.height(),
            self.input_size,
        )?;
        let size = self.input_size as usize;
        let tensor = Tensor::from_array(([1usize, 3, size, size], input)).map_err(err)?;
        let shape =
            Tensor::from_array(([1usize, 2], vec![self.input_size as f32; 2])).map_err(err)?;
        let scale = Tensor::from_array((
            [1usize, 2],
            vec![
                (resized_height as f64 / page.height() as f64) as f32,
                (resized_width as f64 / page.width() as f64) as f32,
            ],
        ))
        .map_err(err)?;
        let outputs = self
            .session
            .run(ort::inputs!["image"=>tensor, "im_shape"=>shape, "scale_factor"=>scale])
            .map_err(err)?;
        let (shape, rows) = outputs["fetch_name_0"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        if shape.len() != 2 || shape[1] < 7 {
            return Err(format!("Layout output must be [N,7+], got {shape:?}"));
        }
        decode_layout(
            rows,
            shape[1] as usize,
            page.width(),
            page.height(),
            threshold,
        )
    }
}

/// Letterbox with the same OpenCV cubic resize used in training.
pub fn layout_input(page: &RgbImage, input_size: u32) -> Result<Vec<f32>> {
    let (width, height) =
        layout_dimensions(page.width(), page.height(), input_size)?;
    let resized = crate::image_transforms::resize_rgb_cubic(page, width, height);
    // Allocate only the bounded network canvas, even for very wide score strips.
    let size = input_size as usize;
    let mut result = vec![1f32; 3 * size * size];
    for y in 0..height as usize {
        for x in 0..width as usize {
            for c in 0..3 {
                result[c * size * size + y * size + x] =
                    resized.get_pixel(x as u32, y as u32).0[c] as f32 / 255.;
            }
        }
    }
    Ok(result)
}

pub fn decode_layout(
    values: &[f32],
    columns: usize,
    width: u32,
    height: u32,
    threshold: f32,
) -> Result<Vec<Value>> {
    if columns < 7
        || values.len() % columns != 0
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
    {
        return Err("Invalid layout output shape or threshold".into());
    }
    let mut rows = Vec::new();
    for row in values.chunks_exact(columns) {
        if !row[..7].iter().all(|x| x.is_finite()) {
            return Err("Non-finite layout output".into());
        }
        if row[0] < 0. || row[1] <= threshold {
            continue;
        }
        if row[0].fract() != 0. || row[0] as usize >= LAYOUT_LABELS.len() {
            return Err(format!("Unknown layout class {}", row[0]));
        }
        let cls = row[0] as usize;
        let coord = [
            round_even(row[2] as f64).max(0.) as u32,
            round_even(row[3] as f64).max(0.) as u32,
            round_even(row[4] as f64).max(0.).min(width as f64) as u32,
            round_even(row[5] as f64).max(0.).min(height as f64) as u32,
        ];
        if coord[2] > coord[0] && coord[3] > coord[1] {
            rows.push((
                row[6],
                json!({"cls_id":cls,"label":LAYOUT_LABELS[cls],"score":row[1],"coordinate":coord}),
            ));
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(rows.into_iter().map(|(_, v)| v).collect())
}

pub struct SignatureReader {
    session: Session,
    input_size: (u32, u32),
    calibration: Value,
}
impl SignatureReader {
    pub fn new(model: &Path) -> Result<Self> {
        let session = session(model)?;
        let input = session
            .inputs
            .iter()
            .find(|v| v.name == "images")
            .ok_or("Signature model missing images input")?;
        for name in ["key", "numerator", "denominator"] {
            if !session.outputs.iter().any(|v| v.name == name) {
                return Err(format!("Signature model missing output {name}"));
            }
        }
        let read_metadata = |name: &str| -> Result<Value> {
            session
                .metadata()
                .map_err(err)?
                .custom(name)
                .map_err(err)?
                .map(|value| serde_json::from_str(&value).map_err(err))
                .transpose()
                .map(|v| v.unwrap_or(Value::Null))
        };
        let config = read_metadata("guitarocr.signature_config")?;
        let calibration = read_metadata("guitarocr.signature_calibration")?;
        let input_size = signature_input_size(&config)?;
        let dimensions = vec![3, i64::from(input_size.1), i64::from(input_size.0)];
        match &input.input_type {
            ort::value::ValueType::Tensor { shape, .. }
                if shape.len() == dimensions.len() + 1 && shape[1..] == dimensions => {}
            _ => return Err("Signature preprocessing does not match graph input".into()),
        }
        Ok(Self {
            session,
            input_size,
            calibration,
        })
    }
    /// One bounded batch; callers retain cancellation control between batches.
    pub fn predict(&mut self, pages: &[RgbImage]) -> Result<Vec<Value>> {
        if pages.is_empty() {
            return Ok(vec![]);
        }
        if pages.len() > 64 {
            return Err("Signature batch exceeds 64 images".into());
        }
        let (width, height) = self.input_size;
        let mut data = Vec::new();
        for page in pages {
            data.extend(full_signature_input_at_size(page, width, height)?);
        }
        let shape = vec![pages.len(), 3, height as usize, width as usize];
        let tensor = Tensor::from_array((shape, data)).map_err(err)?;
        let output = self
            .session
            .run(ort::inputs!["images"=>tensor])
            .map_err(err)?;
        let mut logits = Vec::new();
        for (name, classes) in [("key", 16), ("numerator", 33), ("denominator", 8)] {
            let (shape, values) = output[name].try_extract_tensor::<f32>().map_err(err)?;
            if shape.as_ref() != [pages.len() as i64, classes as i64] {
                return Err(format!("Unexpected signature {name} shape: {shape:?}"));
            }
            logits.push(values.to_vec());
        }
        signature_evidence(&logits, &self.calibration)
    }
}

pub fn full_signature_input_at_size(
    page: &RgbImage,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<f32>> {
    check_dimensions(page.width(), page.height())?;
    check_dimensions(target_width, target_height)?;
    let ratio = f64::from(page.width()) / f64::from(page.height());
    let (width, height) = if ratio > f64::from(target_width) / f64::from(target_height) {
        (
            target_width,
            round_even(f64::from(target_width) / ratio).max(1.) as u32,
        )
    } else {
        (
            round_even(f64::from(target_height) * ratio).max(1.) as u32,
            target_height,
        )
    };
    let resized = crate::image_transforms::resize_rgb_lanczos(page, width, height);
    let plane = target_width as usize * target_height as usize;
    let mut data = vec![1f32; 3 * plane];
    let (left, top) = ((target_width - width) / 2, (target_height - height) / 2);
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                data[c * plane
                    + (y + top) as usize * target_width as usize
                    + (x + left) as usize] =
                    (f32::from(resized.get_pixel(x, y).0[c]) / 255. - 0.5) / 0.5;
            }
        }
    }
    Ok(data)
}

pub fn signature_evidence(logits: &[Vec<f32>], calibration: &Value) -> Result<Vec<Value>> {
    if logits.len() != 3 {
        return Err("Signature requires three classifier heads".into());
    }
    let classes = [16, 33, 8];
    let mut labels = Vec::new();
    let mut confidence = Vec::new();
    let batch = logits[0].len() / 16;
    for (head, values) in logits.iter().enumerate() {
        if values.len() != batch * classes[head] {
            return Err("Signature classifier batch mismatch".into());
        }
        let winners = argmax_rows(values, classes[head])?;
        let temperature = calibration["temperatures"][head].as_f64().unwrap_or(1.);
        if !temperature.is_finite() || temperature <= 0. {
            return Err("Invalid signature calibration temperature".into());
        }
        let probabilities: Vec<f64> = values
            .chunks_exact(classes[head])
            .zip(&winners)
            .map(|(row, &winner)| {
                let max = f64::from(row[winner]);
                1. / row
                    .iter()
                    .map(|value| ((f64::from(*value) - max) / temperature).exp())
                    .sum::<f64>()
            })
            .collect();
        labels.push(winners);
        confidence.push(probabilities);
    }
    let mut result = Vec::new();
    for i in 0..batch {
        let raw = signature_value(labels[0][i], labels[1][i], labels[2][i]);
        let mut evidence = json!({});
        for (field, conf) in [
            ("key", confidence[0][i]),
            ("time", confidence[1][i] * confidence[2][i]),
        ] {
            let value = &raw[field];
            let kind = if value.is_null() { "absent" } else { "present" };
            let threshold = calibration["thresholds"][format!("{field}_{kind}")]
                .as_f64()
                .unwrap_or(1.01);
            let calibrated = calibration["calibrated"].as_bool().unwrap_or(false);
            let inconsistent = field == "time" && (labels[1][i] > 0) != (labels[2][i] > 0);
            let known = calibrated && conf >= threshold && !inconsistent;
            evidence[field] = json!({"status":if known { if value.is_null() { "absent" } else { "observed" } } else { "unknown" },
                "value":if known { value.clone() } else { Value::Null },"candidate":value,"confidence":conf,
                "source":"visual_classifier","calibrated":calibrated});
        }
        result.push(json!({"key":evidence["key"]["value"],"time":evidence["time"]["value"],"evidence":evidence,"source":"visual_classifier"}));
    }
    Ok(result)
}

fn argmax_rows(values: &[f32], classes: usize) -> Result<Vec<usize>> {
    if values.len() % classes != 0 || !values.iter().all(|v| v.is_finite()) {
        return Err("Malformed/non-finite signature logits".into());
    }
    Ok(values
        .chunks_exact(classes)
        .map(|row| {
            let mut best = 0;
            for i in 1..row.len() {
                if row[i] > row[best] {
                    best = i;
                }
            }
            best
        })
        .collect())
}
fn signature_value(key: usize, numerator: usize, denominator: usize) -> Value {
    let denominators = [0, 1, 2, 4, 8, 16, 32, 64];
    json!({"key":if key<15 { Some(key as i32-7) } else { None },
        "time":if numerator>0 && denominator>0 {Some(format!("{numerator}/{}",denominators[denominator]))}else{None}})
}

fn signature_input_size(config: &Value) -> Result<(u32, u32)> {
    if config["version"] != 2
        || config["architecture"] != "mobilenet_v3_small_spatial_v1"
        || config["input"]["views"] != 1
        || config["input"]["resize"] != "contain"
        || config["input"]["normalization"] != "minus_one_to_one"
        || config["evidence_version"] != 1
    {
        return Err(
            "Unsupported signature contract: a version 2 full-image model is required".into(),
        );
    }
    let width = config["input"]["width"]
        .as_u64()
        .ok_or("Missing signature width")?;
    let height = config["input"]["height"]
        .as_u64()
        .ok_or("Missing signature height")?;
    if width < 32
        || height < 32
        || width > 1536
        || height > 1024
        || width % 32 != 0
        || height % 32 != 0
    {
        return Err("Invalid spatial signature input size".into());
    }
    Ok((width as u32, height as u32))
}
