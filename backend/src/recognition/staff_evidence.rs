//! Full-row learned evidence. Profile disagreements remain explicit candidates.
use crate::image_boundary::Result;
use image::RgbImage;
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
    let (canvas, _) = crate::image_transforms::contain_rgb(image, 768, 256);
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
    evidence: &Value,
    manual: Option<usize>,
) -> Value {
    let mut candidates = json!({});
    if staff.as_u64().is_some_and(|n| (1..=12).contains(&n)) {
        candidates["staff_model"] = staff.clone();
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
        return json!({"value":null,"source":"not_applicable","candidates":candidates,"needs_review":false});
    }
    let source = ["manual", "staff_network", "staff_model"]
        .into_iter()
        .find(|name| !candidates[*name].is_null())
        .unwrap_or("unknown");
    let value = candidates[source].clone();
    let needs_review = manual.is_none()
        && (value.is_null()
            || candidates
                .as_object()
                .unwrap()
                .values()
                .any(|candidate| !candidate.is_null() && candidate != &value));
    json!({"value":value,"source":source,"candidates":candidates,
        "needs_review":needs_review,
        "staff_conflict":!candidates["staff_model"].is_null() && candidates["staff_model"] != value,
        "provisional":needs_review && !value.is_null()})
}

/// A printed tuning can disambiguate existing visual counts once the
/// instrument is known. It cannot establish a new count on its own.
pub fn resolve_string_count_with_tuning(
    mut count: Value,
    instrument: &Value,
    tuning_name: &str,
) -> Value {
    if count["source"] == "manual"
        || count["needs_review"] != true
        || instrument["status"] != "resolved"
        || instrument["needs_review"] != false
        || tuning_name.trim().is_empty()
    {
        return count;
    }
    let Some(instrument) = instrument["value"].as_str() else {
        return count;
    };
    let supported: std::collections::BTreeSet<_> = count["candidates"]
        .as_object()
        .into_iter()
        .flat_map(|candidates| candidates.values())
        .filter_map(Value::as_u64)
        .filter(|n| (1..=12).contains(n))
        .filter(|n| super::tuning_from_name(tuning_name, instrument, Some(*n as usize)).is_some())
        .collect();
    if supported.len() == 1 {
        let value = json!(supported.first().unwrap());
        count["value"] = value.clone();
        count["source"] = json!("printed_tuning_agreement");
        count["needs_review"] = json!(false);
        count["provisional"] = json!(false);
        count["staff_conflict"] = json!(
            !count["candidates"]["staff_model"].is_null()
                && count["candidates"]["staff_model"] != value
        );
        count["tuning_evidence"] =
            json!({"text":tuning_name,"instrument":instrument,"string_count":value});
    }
    count
}

pub fn resolve_instrument_with_label(
    has_tab: bool,
    staff: &Value,
    manual: Option<&str>,
    label: Option<&Value>,
    continuation: Option<&Value>,
) -> Value {
    let mut candidates = json!({});
    if staff
        .as_str()
        .is_some_and(|v| ["guitar", "bass", "pitched", "drums"].contains(&v))
    {
        candidates["staff_model"] = staff.clone();
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
                "staff_conflict":!original.is_null() && original != observed});
        }
        if let Some(previous) = continuation.filter(|v| {
            matches!(label["status"].as_str(), Some("absent" | "unreadable"))
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
        // Standard notation establishes a melodic or percussion category
        // without naming its instrument. An unlabelled guitar/bass prediction
        // establishes only melodic notation, not strings or octave transposition.
        if !has_tab && label["status"] == "absent" && known(&candidates["staff_model"]) {
            let category = if candidates["staff_model"] == "drums" {
                "drums"
            } else {
                "pitched"
            };
            return json!({"value":category,"source":"staff_model","status":"resolved",
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
    let value = candidates["staff_model"]
        .as_str()
        .filter(|instrument| !has_tab || ["guitar", "bass"].contains(instrument));
    let needs_review = value.is_none();
    json!({"value":value,"source":if needs_review {"unresolved"} else {"staff_model"},
        "status":if needs_review {"unresolved"} else {"resolved"},
        "candidates":candidates,"needs_review":needs_review,
        "provisional":false})
}

/// A visible clef defines notation's category even when Staff misnames it.
/// It does not replace a printed, continued, or manually chosen instrument.
pub fn apply_notation_clefs(
    identity: &mut Value,
    has_tab: bool,
    label: Option<&Value>,
    predictions: &[Value],
) {
    if has_tab
        || label.is_none_or(|label| label["status"] != "absent")
        || matches!(
            identity["source"].as_str(),
            Some(
                "manual"
                    | "label_text"
                    | "confirmed_continuation"
                    | "visible_instrument_annotation"
            )
        )
    {
        return;
    }
    let categories: std::collections::BTreeSet<_> = predictions
        .iter()
        .filter(|prediction| prediction["kind"] == "clef")
        .filter_map(|prediction| match prediction["parsed"]["clef"].as_str() {
            Some("percussion") => Some("drums"),
            Some("G2" | "F4" | "C3" | "C4") => Some("pitched"),
            _ => None,
        })
        .collect();
    if categories.is_empty() {
        return;
    }
    identity["candidates"]["visible_clef"] = json!(categories);
    let category = if categories.len() == 1 {
        categories.first().copied()
    } else {
        None
    };
    identity["value"] = json!(category);
    identity["source"] = json!("visible_clef");
    identity["status"] = json!(if category.is_some() {
        "resolved"
    } else {
        "unresolved"
    });
    identity["needs_review"] = json!(category.is_none());
    identity["provisional"] = json!(false);
}

#[test]
fn unnamed_notation_keeps_its_visual_category_without_guessing_an_instrument() {
    for instrument in ["guitar", "bass", "pitched", "drums"] {
        let mut label = json!({"status":"absent","visible":false,
            "identity":{"value":null,"accepted":false},
            "staff_prediction":{"instrument":instrument,"name":null}});
        let category = resolve_instrument_with_label(false, &Value::Null, None, Some(&label), None);
        assert_eq!(
            category["value"],
            if instrument == "drums" {
                "drums"
            } else {
                "pitched"
            }
        );
        assert_eq!(category["needs_review"], false);
        assert_eq!(category["source"], "staff_model");
        let prior = json!({"value":"guitar","source":"label_text","status":"resolved"});
        let continued =
            resolve_instrument_with_label(false, &Value::Null, None, Some(&label), Some(&prior));
        assert_eq!(continued["value"], "guitar");
        assert_eq!(continued["source"], "confirmed_continuation");
        for status in ["read", "unreadable"] {
            label["status"] = json!(status);
            let unresolved =
                resolve_instrument_with_label(false, &Value::Null, None, Some(&label), None);
            assert!(unresolved["value"].is_null());
        }
    }
    for instrument in ["guitar", "bass", "pitched", "drums"] {
        let label = json!({"status":"absent", "staff_prediction":{"instrument":instrument}});
        let tab = resolve_instrument_with_label(true, &Value::Null, None, Some(&label), None);
        assert!(tab["value"].is_null());
    }
    let label = json!({"status":"absent","staff_prediction":{"instrument":"pitched"}});
    for (clefs, expected) in [
        (vec!["percussion", "percussion"], Some("drums")),
        (vec!["G2", "F4"], Some("pitched")),
        (vec!["G2", "percussion"], None),
    ] {
        let predictions: Vec<_> = clefs
            .iter()
            .map(|clef| json!({"kind":"clef","parsed":{"clef":clef}}))
            .collect();
        let mut identity =
            resolve_instrument_with_label(false, &Value::Null, None, Some(&label), None);
        apply_notation_clefs(&mut identity, false, Some(&label), &predictions);
        assert_eq!(identity["value"], json!(expected));
        for source in [
            "manual",
            "label_text",
            "confirmed_continuation",
            "visible_instrument_annotation",
        ] {
            identity["value"] = json!("bass");
            identity["source"] = json!(source);
            apply_notation_clefs(&mut identity, false, Some(&label), &predictions);
            assert_eq!(identity["value"], "bass");
        }
    }
}
