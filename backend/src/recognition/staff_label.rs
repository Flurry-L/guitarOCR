//! Learned visible staff-name location and reading direction on a complete row.
use crate::image_boundary::{check_dimensions, Result};
use image::{Rgb, RgbImage};
use ort::{
    session::Session,
    tensor::TensorElementType,
    value::{Tensor, ValueType},
};
use serde_json::{json, Value};
use std::path::Path;

const WIDTH: usize = 768;
const HEIGHT: usize = 256;
const STRIDE: usize = 4;
const CELLS: usize = WIDTH * HEIGHT / STRIDE / STRIDE;
const ANGLES: [u16; 4] = [0, 90, 180, 270];

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn validate_contract(config: &Value) -> Result<f64> {
    let threshold = config["label_threshold"]
        .as_f64()
        .filter(|v| v.is_finite() && *v > 0. && *v <= 1.)
        .ok_or("Invalid staff label threshold")?;
    let expected = json!({
            "version":1,"input_size":[WIDTH,HEIGHT],"stride":STRIDE,
            "resize":"contain_lanczos_center_white","rounding":"positive_round_ties_even",
            "normalization":"rgb_minus1_plus1","angles_ccw":ANGLES,
            "box_encoding":"normalized_ltrb_from_integer_grid_origin",
            "label_threshold":threshold,"scope":"one_visible_name_in_complete_staff",
            "score_selection":"first_row_major_maximum_logit",
            "crop_rounding":"floor_left_top_ceil_right_bottom"
        });
    if expected
        .as_object()
        .unwrap()
        .iter()
        .any(|(key, value)| config[key] != *value)
    {
        return Err("Unsupported staff label model contract".into());
    }
    Ok(threshold)
}

#[derive(Clone, Copy)]
struct Transform {
    sx: f64,
    sy: f64,
    left: f64,
    top: f64,
}

fn prepare(image: &RgbImage) -> Result<(Vec<f32>, Transform)> {
    check_dimensions(image.width(), image.height())?;
    let scale =
        (WIDTH as f64 / f64::from(image.width())).min(HEIGHT as f64 / f64::from(image.height()));
    let width = (f64::from(image.width()) * scale).round_ties_even().max(1.) as u32;
    let height = (f64::from(image.height()) * scale)
        .round_ties_even()
        .max(1.) as u32;
    let left = (WIDTH as u32 - width) / 2;
    let top = (HEIGHT as u32 - height) / 2;
    let resized = crate::image_transforms::resize_rgb_lanczos(image, width, height);
    let mut canvas = RgbImage::from_pixel(WIDTH as u32, HEIGHT as u32, Rgb([255; 3]));
    image::imageops::replace(&mut canvas, &resized, i64::from(left), i64::from(top));
    let mut values = vec![0.; 3 * WIDTH * HEIGHT];
    for (i, pixel) in canvas.pixels().enumerate() {
        for channel in 0..3 {
            values[channel * WIDTH * HEIGHT + i] = f32::from(pixel[channel]) / 127.5 - 1.;
        }
    }
    Ok((
        values,
        Transform {
            sx: f64::from(width) / f64::from(image.width()),
            sy: f64::from(height) / f64::from(image.height()),
            left: f64::from(left),
            top: f64::from(top),
        },
    ))
}

pub struct StaffLabelReader {
    session: Session,
    threshold: f64,
}

impl StaffLabelReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let metadata = session.metadata().map_err(err)?;
        let encoded = metadata
            .custom("guitarocr.staff_label")
            .map_err(err)?
            .ok_or("Missing staff label model contract")?;
        drop(metadata);
        let config: Value = serde_json::from_str(&encoded).map_err(err)?;
        let threshold = validate_contract(&config)?;
        if session.inputs.len() != 1 || session.outputs.len() != 3 {
            return Err("Unsupported staff label model inputs or outputs".into());
        }
        let valid_shape = |value: &ValueType, tail: &[i64]| {
            matches!(value,
            ValueType::Tensor { ty: TensorElementType::Float32, shape, .. }
                if shape.len() == tail.len() + 1 && (shape[0] < 0 || shape[0] == 1) && &shape[1..] == tail)
        };
        if session.inputs[0].name != "images"
            || !valid_shape(
                &session.inputs[0].input_type,
                &[3, HEIGHT as i64, WIDTH as i64],
            )
        {
            return Err("Unsupported staff label image input".into());
        }
        for (name, channels) in [
            ("label_logits", 1),
            ("bounds", 4),
            ("orientation_logits", 4),
        ] {
            let output = session
                .outputs
                .iter()
                .find(|value| value.name == name)
                .ok_or("Missing staff label output")?;
            if !valid_shape(
                &output.output_type,
                &[channels, (HEIGHT / STRIDE) as i64, (WIDTH / STRIDE) as i64],
            ) {
                return Err("Unsupported staff label output shape".into());
            }
        }
        Ok(Self {
            session,
            threshold,
        })
    }

    pub fn predict(&mut self, image: &RgbImage) -> Result<Value> {
        let (pixels, transform) = prepare(image)?;
        let input = Tensor::from_array(([1usize, 3, HEIGHT, WIDTH], pixels.into_boxed_slice()))
            .map_err(err)?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => input])
            .map_err(err)?;
        let mut tensors = Vec::new();
        for (name, channels) in [
            ("label_logits", 1),
            ("bounds", 4),
            ("orientation_logits", 4),
        ] {
            let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(err)?;
            if &shape[..]
                != [
                    1,
                    channels,
                    (HEIGHT / STRIDE) as i64,
                    (WIDTH / STRIDE) as i64,
                ]
                || values.iter().any(|v| !v.is_finite())
            {
                return Err("Invalid staff label model output".into());
            }
            tensors.push(values);
        }
        decode(
            tensors[0],
            tensors[1],
            tensors[2],
            image.dimensions(),
            transform,
            self.threshold,
        )
    }
}

fn decode(
    scores: &[f32],
    bounds: &[f32],
    direction: &[f32],
    size: (u32, u32),
    transform: Transform,
    threshold: f64,
) -> Result<Value> {
    if scores.len() != CELLS
        || bounds.len() != 4 * CELLS
        || direction.len() != 4 * CELLS
        || scores
            .iter()
            .chain(bounds)
            .chain(direction)
            .any(|v| !v.is_finite())
    {
        return Err("Invalid staff label model output".into());
    }
    // Strictly greater preserves NumPy's first maximum in row-major order.
    let mut index = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[index] {
            index = i;
        }
    }
    let probability = 1.0f32 / (1.0 + (-scores[index].clamp(-80., 80.)).exp());
    let distances: [f64; 4] = std::array::from_fn(|i| f64::from(bounds[i * CELLS + index]));
    if distances.iter().any(|v| !(0. ..=1.).contains(v)) {
        return Err("Invalid normalized staff label bounds".into());
    }
    let x = (index % (WIDTH / STRIDE) * STRIDE) as f64;
    let y = (index / (WIDTH / STRIDE) * STRIDE) as f64;
    let x1 = ((x - distances[0] * WIDTH as f64 - transform.left) / transform.sx)
        .clamp(0., f64::from(size.0));
    let y1 = ((y - distances[1] * HEIGHT as f64 - transform.top) / transform.sy)
        .clamp(0., f64::from(size.1));
    let x2 = ((x + distances[2] * WIDTH as f64 - transform.left) / transform.sx)
        .clamp(0., f64::from(size.0));
    let y2 = ((y + distances[3] * HEIGHT as f64 - transform.top) / transform.sy)
        .clamp(0., f64::from(size.1));
    let mut angles: [f32; 4] = std::array::from_fn(|i| direction[i * CELLS + index]);
    let mut best = 0;
    for i in 1..4 {
        if angles[i] > angles[best] {
            best = i;
        }
    }
    let maximum = angles[best];
    for value in &mut angles {
        *value = (*value - maximum).exp();
    }
    let sum: f32 = angles.iter().sum();
    Ok(
        json!({"source":"staff_label_network","visible":f64::from(probability) >= threshold && x2 > x1 && y2 > y1,
        "label_probability":probability,"label_threshold":threshold,"bbox":[x1,y1,x2-x1,y2-y1],
        "orientation_ccw":ANGLES[best],"orientation_probability":angles[best]/sum}),
    )
}

pub fn label_crop(image: &RgbImage, prediction: &Value) -> Result<Option<RgbImage>> {
    if prediction["visible"] == false {
        return Ok(None);
    }
    if prediction["visible"] != true {
        return Err("Invalid staff label visibility".into());
    }
    let bbox = prediction["bbox"]
        .as_array()
        .filter(|b| b.len() == 4)
        .ok_or("Invalid staff label box")?;
    let mut b = [0.; 4];
    for (i, value) in bbox.iter().enumerate() {
        b[i] = value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("Invalid staff label box")?;
    }
    let [x, y, w, h] = b;
    if x < 0.
        || y < 0.
        || w <= 0.
        || h <= 0.
        || x + w > f64::from(image.width())
        || y + h > f64::from(image.height())
    {
        return Err("Staff label crop lies outside the image".into());
    }
    let (left, top, right, bottom) = (
        x.floor() as u32,
        y.floor() as u32,
        (x + w).ceil() as u32,
        (y + h).ceil() as u32,
    );
    let crop = image::imageops::crop_imm(image, left, top, right - left, bottom - top).to_image();
    Ok(Some(match prediction["orientation_ccw"].as_u64() {
        Some(0) => crop,
        Some(90) => image::imageops::rotate270(&crop),
        Some(180) => image::imageops::rotate180(&crop),
        Some(270) => image::imageops::rotate90(&crop),
        _ => return Err("Invalid staff label reading direction".into()),
    }))
}
