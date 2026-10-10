//! Neural text proposals. Source pixels never decide text or musical semantics.
use crate::image_boundary::Result;
use image::{Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

pub struct TextRegionReader {
    session: Session,
    config: Value,
}

impl TextRegionReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let metadata = session.metadata().map_err(|e| e.to_string())?;
        let raw = metadata
            .custom("guitarocr.text_regions")
            .map_err(|e| e.to_string())?
            .ok_or("Missing text region model contract")?;
        let config: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        drop(metadata);
        if config["version"] != 1
            || config["resize"] != "contain_lanczos_stride32_white"
            || config["normalization"] != "bgr_imagenet"
            || config["decoder"] != "probability_components_rect_v1"
            || config["maximum_side"] != 960
        {
            return Err("Unsupported text region model contract".into());
        }
        Ok(Self { session, config })
    }

    pub fn predict(&mut self, image: &RgbImage) -> Result<Vec<Value>> {
        let (pixels, padded, resized) = image_tensor(image)?;
        let input = Tensor::from_array((
            [1usize, 3, padded.1 as usize, padded.0 as usize],
            pixels.into_boxed_slice(),
        ))
        .map_err(|e| e.to_string())?;
        let output = self
            .session
            .run(ort::inputs!["image" => input])
            .map_err(|e| e.to_string())?;
        let (shape, probability) = output["probability"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if shape.as_ref() != [1, 1, padded.1 as i64, padded.0 as i64]
            || probability.iter().any(|v| !v.is_finite())
        {
            return Err("Invalid text region probability map".into());
        }
        decode_regions(
            probability,
            padded,
            resized,
            image.dimensions(),
            &self.config,
        )
    }
}

pub fn image_tensor(image: &RgbImage) -> Result<(Vec<f32>, (u32, u32), (u32, u32))> {
    crate::image_boundary::check_dimensions(image.width(), image.height())?;
    let ratio = (960. / image.width().max(image.height()) as f64).min(1.);
    let width = (image.width() as f64 * ratio).round_ties_even().max(1.) as u32;
    let height = (image.height() as f64 * ratio).round_ties_even().max(1.) as u32;
    let padded = (width.div_ceil(32) * 32, height.div_ceil(32) * 32);
    let resized = crate::image_transforms::resize_rgb_lanczos(image, width, height);
    let mut canvas = RgbImage::from_pixel(padded.0, padded.1, Rgb([255; 3]));
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    let area = (padded.0 * padded.1) as usize;
    let mut tensor = vec![0.; 3 * area];
    for (i, pixel) in canvas.pixels().enumerate() {
        for c in 0..3 {
            tensor[c * area + i] =
                (pixel[2 - c] as f32 / 255. - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
        }
    }
    Ok((tensor, padded, (width, height)))
}

fn decode_regions(
    probability: &[f32],
    padded: (u32, u32),
    resized: (u32, u32),
    original: (u32, u32),
    config: &Value,
) -> Result<Vec<Value>> {
    let width = padded.0 as usize;
    let height = padded.1 as usize;
    let threshold = config["pixel_threshold"]
        .as_f64()
        .ok_or("Missing text threshold")? as f32;
    let box_threshold = config["box_threshold"]
        .as_f64()
        .ok_or("Missing text box threshold")?;
    let minimum = config["minimum_component_side"]
        .as_u64()
        .ok_or("Missing text component size")? as usize;
    let maximum = config["max_candidates"]
        .as_u64()
        .ok_or("Missing text candidate limit")? as usize;
    let unclip = config["unclip_ratio"]
        .as_f64()
        .ok_or("Missing text unclip ratio")?;
    let mut visited = vec![false; probability.len()];
    let mut boxes = vec![];
    let mut candidates = 0;
    for start in 0..probability.len() {
        if visited[start] || probability[start] <= threshold {
            continue;
        }
        candidates += 1;
        if candidates > maximum {
            break;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let (mut left, mut right, mut top, mut bottom) =
            (start % width, start % width, start / width, start / width);
        let (mut sum, mut count) = (0., 0usize);
        while let Some(index) = stack.pop() {
            let (x, y) = (index % width, index / width);
            left = left.min(x);
            right = right.max(x);
            top = top.min(y);
            bottom = bottom.max(y);
            sum += probability[index] as f64;
            count += 1;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbor = ny * width + nx;
                    if !visited[neighbor] && probability[neighbor] > threshold {
                        visited[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
        }
        let (w, h) = (right - left + 1, bottom - top + 1);
        let score = sum / count as f64;
        if score < box_threshold || w.min(h) < minimum {
            continue;
        }
        let pad = (w * h) as f64 * unclip / (2 * (w + h)) as f64;
        let (sx, sy) = (
            original.0 as f64 / resized.0 as f64,
            original.1 as f64 / resized.1 as f64,
        );
        let x = ((left as f64 - pad) * sx).max(0.);
        let y = ((top as f64 - pad) * sy).max(0.);
        let right = (((right + 1) as f64 + pad) * sx).min(original.0 as f64);
        let bottom = (((bottom + 1) as f64 + pad) * sy).min(original.1 as f64);
        if right > x && bottom > y {
            boxes.push(
                json!({"bbox":[x,y,right-x,bottom-y],"score":score,"source":"text_region_network"}),
            );
        }
    }
    boxes.sort_by(|a, b| {
        a["bbox"][1]
            .as_f64()
            .unwrap()
            .total_cmp(&b["bbox"][1].as_f64().unwrap())
            .then(
                a["bbox"][0]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["bbox"][0].as_f64().unwrap()),
            )
    });
    Ok(boxes)
}
