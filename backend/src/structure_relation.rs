//! Learned adjacent-row relations. This model proposes groups; it never rewrites rows.
use image::{imageops, RgbImage};
use ort::{session::Session, value::Tensor};
use serde_json::Value;
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
pub struct RelationPrediction {
    pub systems: Vec<usize>,
    pub same_system_probabilities: Vec<f32>,
    pub needs_review: bool,
    pub review_confidence: f32,
    pub same_part_probabilities: Option<Vec<f32>>,
    pub part_review_confidence: f32,
}

pub struct StructureRelation {
    session: Session,
    width: u32,
    height: u32,
    temperature: f32,
    review_confidence: f32,
    part_temperature: f32,
    part_review_confidence: f32,
}

impl StructureRelation {
    pub fn new(model: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(model)?;
        let text = session
            .metadata()
            .map_err(|e| e.to_string())?
            .custom("guitarocr.structure_relation")
            .map_err(|e| e.to_string())?
            .ok_or("Structure relation model has no preprocessing contract")?;
        let contract: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if contract["same_part"] != true || contract["channels"] != 1
            || contract["geometry_features"] != 12
            || contract["grayscale"] != "ITU-R-601"
            || contract["resize"] != "contain_lanczos_white_center"
            || contract["crop"] != "full_width_between_adjacent_row_centers"
        {
            return Err("Unsupported structure relation preprocessing contract".into());
        }
        let width = contract["width"]
            .as_u64()
            .filter(|&v| v == 768)
            .ok_or("Invalid structure relation width")? as u32;
        let height = contract["height"]
            .as_u64()
            .filter(|&v| v == 192)
            .ok_or("Invalid structure relation height")? as u32;
        let temperature = contract["temperature"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or("Invalid relation temperature")? as f32;
        let review_confidence = contract["review_confidence"]
            .as_f64()
            .filter(|v| v.is_finite() && (0.5..=1.).contains(v))
            .ok_or("Invalid relation confidence threshold")? as f32;
        let part_temperature = contract["part_temperature"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or("Invalid part relation temperature")? as f32;
        let part_review_confidence = contract["part_review_confidence"]
            .as_f64()
            .filter(|v| v.is_finite() && (0.5..=1.).contains(v))
            .ok_or("Invalid part relation confidence threshold")? as f32;
        for name in ["pixels", "geometry"] {
            if !session.inputs.iter().any(|v| v.name == name) {
                return Err(format!("Structure relation model missing input {name}"));
            }
        }
        if !session
            .outputs
            .iter()
            .any(|v| v.name == "same_system_logits")
        {
            return Err("Structure relation model missing logits".into());
        }
        if !session.outputs.iter().any(|v| v.name == "same_part_logits") {
            return Err("Structure relation model missing same-part logits".into());
        }
        Ok(Self {
            session,
            width,
            height,
            temperature,
            review_confidence,
            part_temperature,
            part_review_confidence,
        })
    }

    /// `centers` are ordered row centers in the supplied page's pixel coordinates.
    pub fn predict(
        &mut self,
        page: &RgbImage,
        centers: &[f32],
        modes: &[String],
    ) -> Result<RelationPrediction> {
        if centers.is_empty()
            || centers.len() != modes.len()
            || page.width() == 0
            || page.height() == 0
            || centers
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v >= page.height() as f32)
            || centers.windows(2).any(|v| v[1] <= v[0])
        {
            return Err(
                "Structure relation needs ordered in-page row centers and matching modes".into(),
            );
        }
        if centers.len() == 1 {
            return Ok(RelationPrediction {
                systems: vec![0],
                same_system_probabilities: vec![],
                needs_review: false,
                review_confidence: self.review_confidence,
                same_part_probabilities: Some(Vec::new()),
                part_review_confidence: self.part_review_confidence,
            });
        }
        let pairs = centers.len() - 1;
        let (pixels, geometry) = prepare_inputs(page, centers, modes)?;
        let pixels = Tensor::from_array((
            [pairs, 1, self.height as usize, self.width as usize],
            pixels,
        ))
        .map_err(|e| e.to_string())?;
        let geometry = Tensor::from_array(([pairs, 12], geometry)).map_err(|e| e.to_string())?;
        let output = self
            .session
            .run(ort::inputs!["pixels" => pixels, "geometry" => geometry])
            .map_err(|e| e.to_string())?;
        let (_, logits) = output["same_system_logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if logits.len() != pairs || logits.iter().any(|v| !v.is_finite()) {
            return Err("Structure relation returned invalid logits".into());
        }
        let probabilities: Vec<f32> = logits
            .iter()
            .map(|v| 1. / (1. + (-v / self.temperature).exp()))
            .collect();
        let mut systems = vec![0];
        for &probability in &probabilities {
            systems.push(systems.last().copied().unwrap() + usize::from(probability < 0.5));
        }
        let (_, part_logits) = output["same_part_logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if part_logits.len() != pairs || part_logits.iter().any(|v| !v.is_finite()) {
            return Err("Structure relation returned invalid same-part logits".into());
        }
        let part_probabilities: Vec<f32> = part_logits
            .iter()
            .map(|v| 1. / (1. + (-v / self.part_temperature).exp()))
            .collect();
        let needs_review = probabilities
            .iter()
            .any(|p| p.max(1. - p) < self.review_confidence)
            || part_probabilities
                .iter()
                .zip(systems.windows(2))
                .any(|(p, s)| s[0] == s[1] && p.max(1. - p) < self.part_review_confidence);
        Ok(RelationPrediction {
            systems,
            same_system_probabilities: probabilities,
            needs_review,
            review_confidence: self.review_confidence,
            same_part_probabilities: Some(part_probabilities),
            part_review_confidence: self.part_review_confidence,
        })
    }
}

/// Deterministic Pillow-compatible grayscale/contain preprocessing, exposed for parity checks.
pub fn prepare_inputs(
    page: &RgbImage,
    centers: &[f32],
    modes: &[String],
) -> Result<(Vec<f32>, Vec<f32>)> {
    if centers.is_empty()
        || centers.len() != modes.len()
        || page.width() == 0
        || page.height() == 0
        || centers
            .iter()
            .any(|v| !v.is_finite() || *v < 0. || *v >= page.height() as f32)
        || centers.windows(2).any(|v| v[1] <= v[0])
    {
        return Err(
            "Structure relation needs ordered in-page row centers and matching modes".into(),
        );
    }
    let gray = crate::image_boundary::grayscale_rgb(page);
    let pairs = centers.len() - 1;
    let mut pixels = Vec::with_capacity(pairs * 768 as usize * 192 as usize);
    let mut geometry = Vec::with_capacity(pairs * 12);
    for (index, pair) in centers.windows(2).enumerate() {
        let top = pair[0] as u32;
        let bottom = (pair[1] as u32).max(top + 1).min(page.height());
        let crop = imageops::crop_imm(&gray, 0, top, page.width(), bottom - top).to_image();
        let scale = (768 as f64 / crop.width() as f64).min(192 as f64 / crop.height() as f64);
        let width = (crop.width() as f64 * scale).round_ties_even().max(1.) as u32;
        let height = (crop.height() as f64 * scale).round_ties_even().max(1.) as u32;
        let resized = crate::image_transforms::resize_rgb_lanczos(&crop, width, height);
        let mut canvas = RgbImage::from_pixel(768, 192, image::Rgb([255; 3]));
        imageops::replace(
            &mut canvas,
            &resized,
            ((768 - width) / 2) as i64,
            ((192 - height) / 2) as i64,
        );
        pixels.extend(canvas.pixels().map(|p| f32::from(p.0[0]) / 255.));
        geometry.extend([
            pair[0] / page.height() as f32,
            pair[1] / page.height() as f32,
            (pair[1] - pair[0]) / page.width() as f32,
            page.width() as f32 / page.height() as f32,
        ]);
        for mode in &modes[index..index + 2] {
            let ordinal = match mode.as_str() {
                "notation" => 1,
                "tab" => 2,
                "both" => 3,
                _ => 0,
            };
            geometry.extend((0..4).map(|value| if value == ordinal { 1. } else { 0. }));
        }
    }
    Ok((pixels, geometry))
}
