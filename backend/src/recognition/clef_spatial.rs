//! Read the clef, its staff position, and its small octave glyph together.
use crate::image_boundary::{check_dimensions, grayscale_rgb, Result};
use image::{Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

const SIZE: usize = 256;

fn labels() -> Vec<Value> {
    let mut result = vec![];
    for clef in ["G2", "F4"] {
        for octave in [-24, -12, 0, 12, 24] {
            result.push(json!({"clef":clef,"clef_octave":octave}));
        }
    }
    for clef in ["C3", "C4", "percussion", "tab"] {
        result.push(json!({"clef":clef,"clef_octave":0}));
    }
    result
}

pub fn image_tensor(image: &RgbImage) -> Result<Vec<f32>> {
    check_dimensions(image.width(), image.height())?;
    let scale =
        (SIZE as f64 / f64::from(image.width())).min(SIZE as f64 / f64::from(image.height()));
    let width = (f64::from(image.width()) * scale).round_ties_even().max(1.) as u32;
    let height = (f64::from(image.height()) * scale)
        .round_ties_even()
        .max(1.) as u32;
    let resized = crate::image_transforms::resize_rgb_lanczos(&grayscale_rgb(image), width, height);
    let mut canvas = RgbImage::from_pixel(SIZE as u32, SIZE as u32, Rgb([255; 3]));
    image::imageops::replace(
        &mut canvas,
        &resized,
        (SIZE as i64 - i64::from(width)) / 2,
        (SIZE as i64 - i64::from(height)) / 2,
    );
    Ok(canvas.pixels().map(|p| f32::from(p[0]) / 255.).collect())
}

pub struct ClefSpatialReader {
    session: Session,
    threshold: f64,
}

impl ClefSpatialReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let encoded = session
            .metadata()
            .map_err(|e| e.to_string())?
            .custom("guitarocr.clef_spatial")
            .map_err(|e| e.to_string())?
            .ok_or("Missing spatial clef model contract")?;
        let config: Value = serde_json::from_str(&encoded).map_err(|e| e.to_string())?;
        let threshold = config["threshold"]
            .as_f64()
            .filter(|v| *v >= 0.5 && *v <= 1.)
            .ok_or("Invalid spatial clef threshold")?;
        if config["version"] != 1
            || config["image_size"] != json!([SIZE, SIZE])
            || config["labels"] != json!(labels())
            || config["grayscale"] != "pil_l"
            || config["resize"] != "contain_lanczos_white_center"
            || config["normalization"] != "zero_one"
        {
            return Err("Unsupported spatial clef model contract".into());
        }
        Ok(Self { session, threshold })
    }

    pub fn predict(&mut self, image: &RgbImage) -> Result<Value> {
        let tensor = Tensor::from_array(([1, 1, SIZE, SIZE], image_tensor(image)?))
            .map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => tensor])
            .map_err(|e| e.to_string())?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let labels = labels();
        if shape.as_ref() != [1, labels.len() as i64] || logits.iter().any(|v| !v.is_finite()) {
            return Err("Invalid spatial clef output".into());
        }
        let (index, probabilities) = guitarocr_engine::onnx::classify(logits);
        let confidence = probabilities[index];
        let accepted = confidence >= self.threshold;
        let label = if accepted {
            labels[index].clone()
        } else {
            json!({"clef":null,"clef_octave":null})
        };
        Ok(
            json!({"source":"clef_spatial_network", "clef":label["clef"],
            "clef_octave":label["clef_octave"], "accepted":accepted, "confidence":confidence,
            "threshold":self.threshold, "probabilities":probabilities,
            "status":if accepted {"recognized"} else {"uncertain"}}),
        )
    }
}

pub fn apply_evidence(parsed: &Value, evidence: &Value) -> (Value, Value) {
    let mut recorded = evidence.clone();
    recorded["language_prediction"] = parsed.clone();
    let result = if evidence["accepted"] == true {
        json!({"clef":evidence["clef"],"clef_octave":evidence["clef_octave"]})
    } else {
        json!({"clef":null,"clef_octave":null})
    };
    (result, recorded)
}
