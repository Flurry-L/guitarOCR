//! Classify only an observed literal name; missing text never implies an instrument.
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

const LENGTH: usize = 160;
const LABELS: [&str; 5] = ["unknown", "guitar", "bass", "pitched", "drums"];

fn tokens(text: &str) -> Vec<i64> {
    let mut result = vec![0; LENGTH];
    for (target, source) in result.iter_mut().zip(text.bytes()) {
        *target = i64::from(source) + 1;
    }
    result
}

pub struct InstrumentTextReader {
    session: Session,
    threshold: f64,
}

impl InstrumentTextReader {
    pub fn new(path: &Path) -> Result<Self, String> {
        let session = guitarocr_engine::onnx::session(path)?;
        let encoded = session
            .metadata()
            .map_err(|e| e.to_string())?
            .custom("guitarocr.instrument_text")
            .map_err(|e| e.to_string())?
            .ok_or("Missing instrument text model contract")?;
        let config: Value = serde_json::from_str(&encoded).map_err(|e| e.to_string())?;
        let threshold = config["threshold"]
            .as_f64()
            .filter(|v| *v > 0. && *v <= 1.)
            .ok_or("Invalid instrument text threshold")?;
        if config["version"] != 1
            || config["labels"] != json!(LABELS)
            || config["length"] != LENGTH
            || config["encoding"] != "utf8_bytes_plus_one_zero_pad"
            || config["overlong"] != "abstain_without_truncation"
            || config["empty"] != "abstain"
        {
            return Err("Unsupported instrument text model contract".into());
        }
        Ok(Self { session, threshold })
    }

    pub fn predict(&mut self, text: Option<&str>) -> Result<Value, String> {
        let mut result = json!({"source":"instrument_text_network", "value":null,
            "accepted":false, "confidence":null, "threshold":self.threshold,
            "probabilities":null, "status":"no_text"});
        let Some(text) = text.filter(|s| !s.is_empty()) else {
            return Ok(result);
        };
        if text.len() > LENGTH {
            result["status"] = json!("input_too_long");
            return Ok(result);
        }
        let tensor = Tensor::from_array(([1, LENGTH], tokens(text))).map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["tokens" => tensor])
            .map_err(|e| e.to_string())?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if shape.as_ref() != [1, LABELS.len() as i64] || logits.iter().any(|v| !v.is_finite()) {
            return Err("Invalid instrument text probabilities".into());
        }
        let (index, probabilities) = guitarocr_engine::onnx::classify(logits);
        let confidence = probabilities[index];
        let accepted = index > 0 && confidence >= self.threshold;
        result["accepted"] = json!(accepted);
        result["value"] = if accepted {
            json!(LABELS[index])
        } else {
            Value::Null
        };
        result["confidence"] = json!(confidence);
        result["probabilities"] = json!(probabilities);
        result["status"] = json!(if accepted { "recognized" } else { "unknown" });
        Ok(result)
    }
}
