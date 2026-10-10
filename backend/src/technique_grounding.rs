//! Learned event pixels and visible technique relations; no note-owner inference.
use crate::image_boundary::{check_dimensions, Result};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

pub const WIDTH: usize = 960;
pub const HEIGHT: usize = 256;
pub const MAX_EVENTS: usize = 32;
pub const FEATURE_DIM: usize = 32;
pub const LABELS: [&str; 5] = ["none", "pm", "let", "both", "unknown"];
fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[derive(Clone, Copy, Debug)]
pub struct ImageTransform {
    pub original_width: u32,
    pub original_height: u32,
    pub width: u32,
    pub height: u32,
    pub offset_x: u32,
    pub offset_y: u32,
}
impl ImageTransform {
    pub fn to_pixels(self, point: [f32; 2]) -> [f64; 2] {
        [
            (point[0] as f64 * WIDTH as f64 - self.offset_x as f64) * self.original_width as f64
                / self.width as f64,
            (point[1] as f64 * HEIGHT as f64 - self.offset_y as f64) * self.original_height as f64
                / self.height as f64,
        ]
    }
}

/// Pillow-compatible contain/Lanczos/RGB [-1,1], plus an empty query plane.
pub fn prepare_image(image: &RgbImage) -> Result<(Vec<f32>, ImageTransform)> {
    check_dimensions(image.width(), image.height())?;
    let scale = (WIDTH as f64 / image.width() as f64).min(HEIGHT as f64 / image.height() as f64);
    let width = (image.width() as f64 * scale).round_ties_even().max(1.) as u32;
    let height = (image.height() as f64 * scale).round_ties_even().max(1.) as u32;
    let transform = ImageTransform {
        original_width: image.width(),
        original_height: image.height(),
        width,
        height,
        offset_x: (WIDTH as u32 - width) / 2,
        offset_y: (HEIGHT as u32 - height) / 2,
    };
    let resized = crate::image_transforms::resize_rgb_lanczos(image, width, height);
    let area = WIDTH * HEIGHT;
    let mut tensor = vec![1.; 4 * area];
    tensor[3 * area..].fill(0.);
    for (x, y, pixel) in resized.enumerate_pixels() {
        let index = (y + transform.offset_y) as usize * WIDTH + (x + transform.offset_x) as usize;
        for channel in 0..3 {
            tensor[channel * area + index] = pixel[channel] as f32 / 127.5 - 1.;
        }
    }
    Ok((tensor, transform))
}

pub fn event_features(measure: &Value, mode: &str) -> Result<(Vec<f32>, Vec<[i64; 2]>)> {
    if !["tab", "notation", "both"].contains(&mode) {
        return Err("Unsupported technique display mode".into());
    }
    let mut all = BTreeMap::new();
    let mut end = 0f64;
    for voice in measure["voices"].as_array().ok_or("Missing event voices")? {
        let index = voice["voice"].as_i64().ok_or("Invalid event voice")?;
        for event in voice["events"].as_array().ok_or("Missing events")? {
            let onset = event["start"].as_i64().ok_or("Invalid event onset")?;
            let ticks = scorelib::score::duration_ticks(&event["duration"])?;
            let duration = ticks.numerator as f64 / ticks.denominator as f64;
            end = end.max(onset as f64 + duration);
            if !event["notes"]
                .as_array()
                .ok_or("Missing event notes")?
                .is_empty()
                && all.insert([index, onset], (event, duration)).is_some()
            {
                return Err("Duplicate sounding event identity".into());
            }
        }
    }
    if all.len() > MAX_EVENTS {
        return Err("Too many event queries; refusing to truncate identities".into());
    }
    let mut features = vec![0.; all.len() * FEATURE_DIM];
    let mut anchors = Vec::new();
    for (index, (&anchor, &(event, duration))) in all.iter().enumerate() {
        let notes = event["notes"].as_array().unwrap();
        let pitches: Vec<_> = notes
            .iter()
            .map(|note| {
                if mode != "tab" {
                    note["pitch"].as_f64().unwrap_or(60.)
                } else {
                    60.
                }
            })
            .collect();
        let mut values = vec![
            anchor[0] as f64 / 4.,
            anchor[1] as f64 / end,
            duration / end,
            index as f64 / all.len().saturating_sub(1).max(1) as f64,
            all.len() as f64 / 32.,
            notes.len() as f64 / 6.,
            pitches.iter().copied().fold(f64::INFINITY, f64::min) / 127.,
            pitches.iter().copied().fold(f64::NEG_INFINITY, f64::max) / 127.,
            pitches.iter().sum::<f64>() / pitches.len() as f64 / 127.,
        ];
        values.extend(["tab", "notation", "both"].map(|name| if mode == name { 1. } else { 0. }));
        for string in 1..=6 {
            let note = notes
                .iter()
                .find(|note| mode != "notation" && note["string"].as_i64() == Some(string));
            values.push(if note.is_some() { 1. } else { 0. });
            values.push(
                note.map(|note| note["fret"].as_f64().unwrap_or(-1.) / 24.)
                    .unwrap_or(0.),
            );
        }
        for (column, value) in values.into_iter().enumerate() {
            features[index * FEATURE_DIM + column] = value as f32;
        }
        anchors.push(anchor);
    }
    Ok((features, anchors))
}

fn metadata(session: &Session, key: &str) -> Result<Value> {
    let encoded = session
        .metadata()
        .map_err(err)?
        .custom(key)
        .map_err(err)?
        .ok_or("Missing technique model contract")?;
    let value: Value = serde_json::from_str(&encoded).map_err(err)?;
    if value["version"] != 1
        || value["width"] != WIDTH
        || value["height"] != HEIGHT
        || value["resize"] != "contain_lanczos"
    {
        return Err("Unsupported technique model image contract".into());
    }
    Ok(value)
}

pub(crate) fn probabilities(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("Invalid technique model logits".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<_> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let total: f32 = values.iter().sum();
    for value in &mut values {
        *value /= total;
    }
    Ok(values)
}

pub struct EventGroundingReader {
    session: Session,
}
impl EventGroundingReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let contract = metadata(&session, "guitarocr.event_grounding")?;
        if contract["feature_dim"] != FEATURE_DIM
            || contract["max_events"] != MAX_EVENTS
            || contract["mode_order"] != json!(["tab", "notation", "both"])
            || contract["output_coordinates"]
                != "xy normalized to the entire contained 960x256 canvas"
            || !["images", "events", "padding"]
                .iter()
                .all(|name| session.inputs.iter().any(|input| input.name == *name))
            || !["positions", "scores"]
                .iter()
                .all(|name| session.outputs.iter().any(|output| output.name == *name))
        {
            return Err("Unsupported event grounding contract".into());
        }
        Ok(Self { session })
    }
    pub fn predict(&mut self, image: &RgbImage, measure: &Value, mode: &str) -> Result<Vec<Value>> {
        let (features, anchors) = event_features(measure, mode)?;
        if anchors.is_empty() {
            return Ok(vec![]);
        }
        let (pixels, transform) = prepare_image(image)?;
        let mut queries = vec![0.; MAX_EVENTS * FEATURE_DIM];
        queries[..features.len()].copy_from_slice(&features);
        let input = Tensor::from_array(([1usize, 4, HEIGHT, WIDTH], pixels)).map_err(err)?;
        let events =
            Tensor::from_array(([1usize, MAX_EVENTS, FEATURE_DIM], queries)).map_err(err)?;
        let padding = Tensor::from_array((
            [1usize, MAX_EVENTS],
            (0..MAX_EVENTS)
                .map(|index| index >= anchors.len())
                .collect::<Vec<_>>(),
        ))
        .map_err(err)?;
        let output = self
            .session
            .run(ort::inputs!["images" => input, "events" => events, "padding" => padding])
            .map_err(err)?;
        let (position_shape, positions) = output["positions"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (score_shape, scores) = output["scores"].try_extract_tensor::<f32>().map_err(err)?;
        if position_shape.as_ref() != [1, MAX_EVENTS as i64, 2]
            || score_shape.len() != 3
            || score_shape[0] != 1
            || score_shape[1] != MAX_EVENTS as i64
            || score_shape[2] <= 0
            || positions.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid event grounding output".into());
        }
        let spatial = score_shape[2] as usize;
        let mut result = Vec::new();
        for (index, anchor) in anchors.iter().enumerate() {
            let normalized = [positions[index * 2], positions[index * 2 + 1]];
            let point = transform.to_pixels(normalized);
            let probability = probabilities(&scores[index * spatial..(index + 1) * spatial])?;
            let entropy = -probability
                .iter()
                .map(|value| value * value.max(1e-30).ln())
                .sum::<f32>();
            result.push(json!({"voice":anchor[0],"onset":anchor[1],"pixel":point,"normalized_canvas_xy":normalized,
                "attention_entropy":entropy,"inside_image":point[0]>=0. && point[0]<image.width() as f64 && point[1]>=0. && point[1]<image.height() as f64,
                "status":"uncalibrated_pixel_evidence"}));
        }
        Ok(result)
    }
}
