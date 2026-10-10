//! Full-row learned evidence. Profile disagreements remain explicit candidates.
use crate::image_boundary::Result;
use image::{Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

pub struct StaffReader {
    session: Session,
    temperatures: [f64; 2],
    thresholds: [f64; 2],
}
impl StaffReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let metadata = session.metadata().map_err(|e| e.to_string())?;
        let config = metadata
            .custom("guitarocr.staff_config")
            .map_err(|e| e.to_string())?
            .ok_or("Missing staff evidence contract")?;
        drop(metadata);
        let config: Value = serde_json::from_str(&config).map_err(|e| e.to_string())?;
        if config["version"] != 2
            || config["width"] != 768
            || config["height"] != 256
            || config["resize"] != "contain_lanczos_white_center"
            || config["normalization"] != "minus_one_to_one"
            || config["counts"] != json!([0, 4, 5, 6, 7, 8])
            || config["modes"] != json!(["notation", "tab", "both"])
            || config["count_authority"] != "joint_visual"
        {
            return Err("Unsupported staff evidence contract".into());
        }
        let mut temperatures = [0.; 2];
        let mut thresholds = [0.; 2];
        for (key, target) in [
            ("temperatures", &mut temperatures),
            ("thresholds", &mut thresholds),
        ] {
            let values = config[key]
                .as_array()
                .filter(|v| v.len() == 2)
                .ok_or("Invalid staff calibration")?;
            for (i, v) in values.iter().enumerate() {
                target[i] = v
                    .as_f64()
                    .filter(|x| {
                        x.is_finite()
                            && if key == "thresholds" {
                                (0. ..=1.).contains(x)
                            } else {
                                *x > 0.
                            }
                    })
                    .ok_or("Invalid staff calibration")?;
            }
        }
        Ok(Self {
            session,
            temperatures,
            thresholds,
        })
    }
    pub fn predict(&mut self, image: &RgbImage) -> Result<Value> {
        let input = Tensor::from_array((
            [1usize, 3, 256, 768],
            image_tensor(image)?.into_boxed_slice(),
        ))
        .map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => input])
            .map_err(|e| e.to_string())?;
        let (_, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if logits.len() != 9 || logits.iter().any(|v| !v.is_finite()) {
            return Err("Invalid staff model output".into());
        }
        let mut fields = json!({"source":"staff_network_v2","count_authority":"joint_visual"});
        for (i, name, range, labels) in [
            (0, "string_count", 0..6, json!([null, 4, 5, 6, 7, 8])),
            (1, "mode", 6..9, json!(["notation", "tab", "both"])),
        ] {
            let z: Vec<_> = logits[range]
                .iter()
                .map(|v| f64::from(*v) / self.temperatures[i])
                .collect();
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut p: Vec<_> = z.iter().map(|v| (v - max).exp()).collect();
            let sum: f64 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            let index = (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best });
            fields[name] = json!({"value":labels[index],"confidence":p[index],"accepted":p[index]>=self.thresholds[i],"probabilities":p});
        }
        Ok(fields)
    }
}

pub fn image_tensor(image: &RgbImage) -> Result<Vec<f32>> {
    crate::image_boundary::check_dimensions(image.width(), image.height())?;
    let scale = (768. / f64::from(image.width())).min(256. / f64::from(image.height()));
    let width = (f64::from(image.width()) * scale).round_ties_even().max(1.) as u32;
    let height = (f64::from(image.height()) * scale)
        .round_ties_even()
        .max(1.) as u32;
    let resized = crate::image_transforms::resize_rgb_lanczos(image, width, height);
    let mut canvas = RgbImage::from_pixel(768, 256, Rgb([255; 3]));
    image::imageops::replace(
        &mut canvas,
        &resized,
        i64::from((768 - width) / 2),
        i64::from((256 - height) / 2),
    );
    let mut values = vec![0.; 3 * 768 * 256];
    for (i, p) in canvas.pixels().enumerate() {
        for c in 0..3 {
            values[c * 768 * 256 + i] = (f32::from(p[c]) / 255. - 0.5) / 0.5;
        }
    }
    Ok(values)
}

pub fn resolve_string_count(
    has_tab: bool,
    staff: &Value,
    structure: &Value,
    evidence: &Value,
    manual: Option<usize>,
    staff_task_adapted: bool,
) -> Value {
    let mut candidates = json!({});
    for (name, value) in [("staff_model", staff), ("structure_model", structure)] {
        if value.as_u64().is_some_and(|n| (1..=12).contains(&n)) {
            candidates[name] = value.clone();
        }
    }
    // String count is meaningful only after the two heads agree on whether
    // the image contains TAB. Raw evidence remains available for review.
    if evidence["string_count"]["accepted"] == true
        && evidence["mode"]["accepted"] == true
        && (evidence["mode"]["value"] == "notation") == evidence["string_count"]["value"].is_null()
    {
        candidates["staff_network"] = evidence["string_count"]["value"].clone();
    }
    if let Some(count) = manual {
        candidates["manual"] = json!(count);
    }
    if !has_tab {
        return json!({"value":null,"source":"not_applicable","candidates":candidates,"needs_review":false,
            "local_agreement":false,"structure_conflict":false});
    }
    let source = ["manual", "staff_network", "staff_model", "structure_model"]
        .into_iter()
        .find(|name| !candidates[*name].is_null())
        .unwrap_or("unknown");
    let value = candidates[source].clone();
    // Only a task-adapted local staff reading can corroborate the calibrated
    // mode/count network. An independent page count stays visible but cannot
    // by itself block this local agreement. Identity and tuning are separate.
    let local_agreement = staff_task_adapted
        && matches!(evidence["mode"]["value"].as_str(), Some("tab" | "both"))
        && candidates["staff_network"]
            .as_u64()
            .is_some_and(|n| (1..=12).contains(&n))
        && candidates["staff_model"] == candidates["staff_network"];
    let structure_conflict =
        !candidates["structure_model"].is_null() && candidates["structure_model"] != value;
    let visual_authority = matches!(evidence["mode"]["value"].as_str(), Some("tab" | "both"))
        && candidates["staff_network"]
            .as_u64()
            .is_some_and(|n| (1..=12).contains(&n));
    json!({"value":value,"source":source,"candidates":candidates,
        "needs_review":manual.is_none() && !visual_authority,
        "local_agreement":local_agreement,"structure_conflict":structure_conflict,
        "staff_conflict":!candidates["staff_model"].is_null() && candidates["staff_model"] != value,
        "visual_authority":visual_authority,
        "provisional":manual.is_none() && !visual_authority && !value.is_null()})
}

pub fn resolve_instrument_with_label(
    has_tab: bool,
    staff: &Value,
    structure: &Value,
    manual: Option<&str>,
    label: Option<&Value>,
    continuation: Option<&Value>,
) -> Value {
    let mut candidates = json!({});
    for (source, value) in [("staff_model", staff), ("structure_model", structure)] {
        if value
            .as_str()
            .is_some_and(|v| ["guitar", "bass", "pitched", "drums"].contains(&v))
        {
            candidates[source] = value.clone();
        }
    }
    if let Some(label) = label {
        let known = |value: &Value| {
            value
                .as_str()
                .is_some_and(|v| ["guitar", "bass", "pitched", "drums"].contains(&v))
        };
        let original = &label["staff_prediction"]["instrument"];
        if known(original) {
            candidates["staff_model"] = original.clone();
        }
        let observed = &label["identity"]["value"];
        let accepted = label["identity"]["accepted"] == true && known(observed);
        if accepted {
            candidates["label_text"] = observed.clone();
        }
        if let Some(value) = manual {
            candidates["manual"] = json!(value);
            return json!({"value":value,"source":"manual","status":"resolved","candidates":candidates,
                "needs_review":false,"provisional":false,"label_evidence":label});
        }
        if accepted {
            return json!({"value":observed,"source":"label_text","status":"resolved","candidates":candidates,
                "needs_review":false,"provisional":false,"label_evidence":label,
                "structure_conflict":!structure.is_null() && structure != observed,
                "staff_conflict":!original.is_null() && original != observed});
        }
        if let Some(previous) = continuation.filter(|v| {
            label["status"] == "absent"
                && matches!(
                    v["source"].as_str(),
                    Some("label_text" | "confirmed_continuation")
                )
                && v["status"] == "resolved"
                && known(&v["value"])
        }) {
            candidates["confirmed_continuation"] = previous["value"].clone();
            return json!({"value":previous["value"],"source":"confirmed_continuation","status":"resolved",
                "candidates":candidates,"needs_review":false,"provisional":false,"label_evidence":label});
        }
        return json!({"value":null,"source":"unresolved_label","status":"unresolved",
            "candidates":candidates,"needs_review":true,"provisional":false,"label_evidence":label});
    }
    if let Some(instrument) = manual {
        candidates["manual"] = json!(instrument);
        return json!({"value":instrument,"source":"manual","status":"resolved",
            "candidates":candidates,"needs_review":false,"provisional":false,"provisional_source":null});
    }
    let source = ["structure_model", "staff_model"]
        .into_iter()
        .find(|key| {
            candidates[*key]
                .as_str()
                .is_some_and(|instrument| !has_tab || ["guitar", "bass"].contains(&instrument))
        })
        .unwrap_or("unknown");
    let value = candidates[source].clone();
    let needs_review = value.is_null()
        || candidates
            .as_object()
            .unwrap()
            .values()
            .any(|candidate| candidate != &value);
    let source = if !needs_review && candidates.as_object().unwrap().len() > 1 {
        "agreement"
    } else {
        source
    };
    json!({"value":value,"source":if needs_review {"unresolved"} else {source},
        "status":if needs_review {"unresolved"} else {"resolved"},
        "candidates":candidates,"needs_review":needs_review,
        "provisional_source":if needs_review && !value.is_null() {json!(source)} else {Value::Null},
        "provisional":needs_review && !value.is_null()})
}
