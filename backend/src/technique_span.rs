//! One image encoder pass for label/span pointers and event coverage.
//! Geometry is diagnostic evidence; only the learned coverage decides membership.
use crate::image_boundary::Result;
use crate::technique_grounding::{
    prepare_image, probabilities, ImageTransform, HEIGHT, LABELS, MAX_EVENTS, WIDTH,
};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub fn prepare_pair(
    current: &RgbImage,
    previous: Option<&RgbImage>,
) -> Result<(Vec<f32>, ImageTransform)> {
    let (current, transform) = prepare_image(current)?;
    let area = WIDTH * HEIGHT;
    let mut pair = vec![1.; area * 8];
    pair[area * 6..].fill(0.);
    let previous = previous.map(prepare_image).transpose()?;
    for channel in 0..3 {
        if let Some((previous, _)) = &previous {
            pair[channel * area * 2..channel * area * 2 + area]
                .copy_from_slice(&previous[channel * area..(channel + 1) * area]);
        }
        pair[channel * area * 2 + area..(channel + 1) * area * 2]
            .copy_from_slice(&current[channel * area..(channel + 1) * area]);
    }
    Ok((pair, transform))
}

pub struct EventSpanRelationReader {
    session: Session,
}
impl EventSpanRelationReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let encoded = session
            .metadata()
            .map_err(err)?
            .custom("guitarocr.event_span_relation")
            .map_err(err)?
            .ok_or("Missing span relation model contract")?;
        let contract: Value = serde_json::from_str(&encoded).map_err(err)?;
        if contract["version"] != 1
            || contract["width"] != WIDTH
            || contract["height"] != HEIGHT * 2
            || contract["resize"] != "contain_lanczos"
            || contract["max_events"] != MAX_EVENTS
            || contract["system_height"] != HEIGHT
            || contract["pair_order"] != json!(["previous", "current"])
            || contract["normalization"] != "RGB/127.5-1; channel4=0"
            || contract["event_coordinates"]
                != "normalized pair canvas; current system in lower half"
            || contract["labels"] != json!(LABELS)
            || !["images", "events", "padding"]
                .iter()
                .all(|name| session.inputs.iter().any(|input| input.name == *name))
            || ![
                "log_probabilities",
                "span_geometry",
                "presence_logits",
                "coverage_logits",
            ]
            .iter()
            .all(|name| session.outputs.iter().any(|output| output.name == *name))
        {
            return Err("Unsupported span relation model contract".into());
        }
        Ok(Self { session })
    }
    pub fn predict(
        &mut self,
        current: &RgbImage,
        previous: Option<&RgbImage>,
        events: &[Value],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Value>> {
        if events.len() > MAX_EVENTS {
            return Err("Too many event relations; refusing to truncate".into());
        }
        if events.is_empty() {
            return Ok(vec![]);
        }
        if cancelled() {
            return Err("Technique recognition cancelled".into());
        }
        let (pixels, transform) = prepare_pair(current, previous)?;
        let mut points = vec![0f32; MAX_EVENTS * 2];
        for (index, event) in events.iter().enumerate() {
            let x = event["pixel"][0]
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or("Invalid event pixel")?;
            let y = event["pixel"][1]
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or("Invalid event pixel")?;
            points[index * 2] = ((x * transform.width as f64 / transform.original_width as f64
                + transform.offset_x as f64)
                / WIDTH as f64) as f32;
            points[index * 2 + 1] = ((y * transform.height as f64
                / transform.original_height as f64
                + transform.offset_y as f64
                + HEIGHT as f64)
                / (HEIGHT * 2) as f64) as f32;
        }
        let input = Tensor::from_array(([1usize, 4, HEIGHT * 2, WIDTH], pixels)).map_err(err)?;
        let points = Tensor::from_array(([1usize, MAX_EVENTS, 2], points)).map_err(err)?;
        let padding = Tensor::from_array((
            [1usize, MAX_EVENTS],
            (0..MAX_EVENTS)
                .map(|i| i >= events.len())
                .collect::<Vec<_>>(),
        ))
        .map_err(err)?;
        let output = self
            .session
            .run(ort::inputs!["images"=>input,"events"=>points,"padding"=>padding])
            .map_err(err)?;
        let (shape, logits) = output["log_probabilities"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (geometry_shape, geometry) = output["span_geometry"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (presence_shape, presence) = output["presence_logits"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (coverage_shape, coverage) = output["coverage_logits"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        if shape.as_ref() != [1, MAX_EVENTS as i64, 5]
            || geometry_shape.as_ref() != [1, MAX_EVENTS as i64, 2, 8]
            || presence_shape.as_ref() != [1, MAX_EVENTS as i64, 2]
            || coverage_shape.as_ref() != [1, MAX_EVENTS as i64, 2]
            || geometry
                .iter()
                .chain(presence)
                .chain(coverage)
                .any(|value| !value.is_finite())
        {
            return Err("Invalid span relation output".into());
        }
        let mut result = vec![];
        for (index, event) in events.iter().enumerate() {
            let x = event["pixel"][0].as_f64().unwrap();
            let y = event["pixel"][1].as_f64().unwrap();
            if x < 0.
                || y < 0.
                || x >= f64::from(current.width())
                || y >= f64::from(current.height())
            {
                let mut row = event.clone();
                row["label"] = json!("unknown");
                row["probabilities"] = json!({"none":0.,"pm":0.,"let":0.,"both":0.,"unknown":1.});
                row["reason"] = json!("event_pixel_outside_current_system");
                row["status"] = json!("uncalibrated_event_relation");
                row["scope"] = json!("visible_event_only");
                row.as_object_mut().unwrap().remove("span_evidence");
                result.push(row);
                continue;
            }
            let values = probabilities(&logits[index * 5..index * 5 + 5])?;
            let best = (0..5)
                .max_by(|&a, &b| values[a].total_cmp(&values[b]).then(b.cmp(&a)))
                .unwrap();
            let mut row = event.clone();
            row["label"] = json!(LABELS[best]);
            row["probabilities"] = Value::Object(
                LABELS
                    .iter()
                    .zip(values)
                    .map(|(label, value)| (label.to_string(), json!(value)))
                    .collect(),
            );
            for (kind, name) in ["pm", "let"].iter().enumerate() {
                row["span_evidence"][*name] = json!({
                    "geometry_pair_normalized": &geometry[index * 16 + kind * 8..index * 16 + kind * 8 + 8],
                    "present_probability": 1. / (1. + (-presence[index * 2 + kind]).exp()),
                    "coverage_probability": 1. / (1. + (-coverage[index * 2 + kind]).exp())});
            }
            row["status"] = json!("uncalibrated_event_relation");
            row["scope"] = json!("visible_event_only");
            result.push(row);
        }
        if cancelled() {
            return Err("Technique recognition cancelled".into());
        }
        Ok(result)
    }
}
