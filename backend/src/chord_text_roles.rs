//! Text ownership uses complete diagram context and a category-free query mask.
use crate::image_boundary::Result;
use image::{Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

pub struct ChordTextRoleReader {
    session: Session,
    threshold: f64,
}
impl ChordTextRoleReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let metadata = session.metadata().map_err(|e| e.to_string())?;
        let raw = metadata
            .custom("guitarocr.chord_text_roles")
            .map_err(|e| e.to_string())?
            .ok_or("Missing chord text role contract")?;
        let config: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        drop(metadata);
        if config["version"] != 1
            || config["input_size"] != 256
            || config["resize"] != "contain_lanczos_center_white"
            || config["normalization"] != "rgb_minus1_plus1_query_mask"
            || config["labels"] != json!(["other", "chord_name"])
        {
            return Err("Unsupported chord text role contract".into());
        }
        let threshold = config["name_threshold"]
            .as_f64()
            .filter(|v| *v > 0. && *v <= 1.)
            .ok_or("Invalid chord name threshold")?;
        Ok(Self { session, threshold })
    }
    pub fn predict(&mut self, image: &RgbImage, regions: &[Value]) -> Result<Vec<Value>> {
        if regions.is_empty() {
            return Ok(vec![]);
        }
        let mut result = Vec::with_capacity(regions.len());
        for batch in regions.chunks(16) {
            result.extend(self.predict_batch(image, batch)?);
        }
        Ok(result)
    }
    fn predict_batch(&mut self, image: &RgbImage, regions: &[Value]) -> Result<Vec<Value>> {
        let mut input = Vec::with_capacity(regions.len() * 4 * 256 * 256);
        for region in regions {
            let values = region["bbox"]
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or("Invalid text query")?;
            let mut bounds = [0.; 4];
            for (i, v) in values.iter().enumerate() {
                bounds[i] = v.as_f64().ok_or("Invalid text coordinate")?;
            }
            input.extend(image_tensor(image, bounds)?);
        }
        let tensor = Tensor::from_array(([regions.len(), 4, 256, 256], input.into_boxed_slice()))
            .map_err(|e| e.to_string())?;
        let result = self
            .session
            .run(ort::inputs!["images" => tensor])
            .map_err(|e| e.to_string())?;
        let (shape, logits) = result["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if shape.as_ref() != [regions.len() as i64, 2] || logits.iter().any(|v| !v.is_finite()) {
            return Err("Invalid chord text role probabilities".into());
        }
        Ok(logits
            .chunks_exact(2)
            .map(|pair| {
                let probability = 1. / (1. + ((pair[0] - pair[1]) as f64).exp());
                json!({"source":"chord_text_role_network", "name_probability":probability,
                "is_name":probability >= self.threshold})
            })
            .collect())
    }
}

pub fn image_tensor(image: &RgbImage, bounds: [f64; 4]) -> Result<Vec<f32>> {
    crate::image_boundary::check_dimensions(image.width(), image.height())?;
    if bounds.iter().any(|v| !v.is_finite()) || bounds[2] <= 0. || bounds[3] <= 0. {
        return Err("Invalid text query dimensions".into());
    }
    let scale = 256. / image.width().max(image.height()) as f64;
    let width = (image.width() as f64 * scale).round_ties_even().max(1.) as u32;
    let height = (image.height() as f64 * scale).round_ties_even().max(1.) as u32;
    let (ox, oy) = ((256 - width) / 2, (256 - height) / 2);
    let resized = crate::image_transforms::resize_rgb_lanczos(image, width, height);
    let mut canvas = RgbImage::from_pixel(256, 256, Rgb([255; 3]));
    image::imageops::replace(&mut canvas, &resized, ox.into(), oy.into());
    let mut input = vec![0.; 4 * 256 * 256];
    for (i, pixel) in canvas.pixels().enumerate() {
        for c in 0..3 {
            input[c * 256 * 256 + i] = pixel[c] as f32 / 127.5 - 1.;
        }
    }
    let x0 = (ox as f64 + bounds[0] * width as f64 / image.width() as f64)
        .trunc()
        .max(0.) as usize;
    let y0 = (oy as f64 + bounds[1] * height as f64 / image.height() as f64)
        .trunc()
        .max(0.) as usize;
    let x1 = (ox as f64 + (bounds[0] + bounds[2]) * width as f64 / image.width() as f64)
        .trunc()
        .clamp(0., 255.) as usize;
    let y1 = (oy as f64 + (bounds[1] + bounds[3]) * height as f64 / image.height() as f64)
        .trunc()
        .clamp(0., 255.) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            input[3 * 256 * 256 + y * 256 + x] = 1.;
        }
    }
    Ok(input)
}
