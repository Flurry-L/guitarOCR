//! Learned diagram evidence. Raster heuristics never accept or erase fields.
//! The ONNX graph owns the output contract and validation-calibrated thresholds.
use crate::image_boundary::{check_dimensions, Result};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

const NAMES: [&str; 6] = ["kind", "strings", "base", "frets", "fingers", "barres"];
const CONTRACT: &str = "guitarocr.chord-evidence/1";
const PREPROCESSING: &str = "rgb_imagenet_letterbox_lanczos_v1";
fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub struct ChordEvidenceReader {
    session: Session,
    contract: Value,
    consensus: bool,
}

impl ChordEvidenceReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let contract: Value = serde_json::from_str(
            &session
                .metadata()
                .map_err(err)?
                .custom("guitarocr.chord_evidence")
                .map_err(err)?
                .ok_or("Chord evidence model is missing its contract")?,
        )
        .map_err(err)?;
        if contract["schema"] != CONTRACT
            || contract["preprocessing"] != PREPROCESSING
            || contract["input_size"] != 256
            || !contract["reader"].is_string()
        {
            return Err("Unsupported chord evidence contract".into());
        }
        for name in NAMES {
            if !session.outputs.iter().any(|output| output.name == name) {
                return Err(format!("Chord evidence model missing {name}"));
            }
            let threshold = contract["thresholds"][name]
                .as_f64()
                .ok_or("Missing chord threshold")?;
            let temperature = contract["temperature"][name]
                .as_f64()
                .ok_or("Missing chord temperature")?;
            if !threshold.is_finite()
                || !(0.5..=1.001).contains(&threshold)
                || !temperature.is_finite()
                || temperature <= 0.
            {
                return Err("Invalid chord evidence calibration".into());
            }
        }
        for name in ["geometry", "full"] {
            if contract["joint_thresholds"][name]
                .as_f64()
                .is_none_or(|n| !n.is_finite() || !(0.5..=1.001).contains(&n))
            {
                return Err("Invalid joint chord evidence calibration".into());
            }
        }
        let consensus = contract["consensus"] == "calibrated_exact_diagram_v1";
        if contract.get("consensus").is_some() && !consensus {
            return Err("Unsupported chord consensus contract".into());
        }
        if consensus {
            let secondary = &contract["secondary_contract"];
            if secondary["schema"] != CONTRACT
                || secondary["preprocessing"] != PREPROCESSING
                || secondary["input_size"] != 256
            {
                return Err("Unsupported secondary chord contract".into());
            }
            for name in NAMES {
                if !session
                    .outputs
                    .iter()
                    .any(|v| v.name == format!("secondary_{name}"))
                {
                    return Err("Missing secondary chord output".into());
                }
                if secondary["thresholds"][name]
                    .as_f64()
                    .is_none_or(|n| !n.is_finite() || !(0.5..=1.001).contains(&n))
                    || secondary["temperature"][name]
                        .as_f64()
                        .is_none_or(|n| !n.is_finite() || n <= 0.)
                {
                    return Err("Invalid secondary chord calibration".into());
                }
            }
            if !session.outputs.iter().any(|v| v.name == "agreement") {
                return Err("Missing chord agreement output".into());
            }
            for name in ["geometry", "full"] {
                if secondary["joint_thresholds"][name]
                    .as_f64()
                    .is_none_or(|n| !n.is_finite() || !(0.5..=1.001).contains(&n))
                {
                    return Err("Invalid secondary joint chord calibration".into());
                }
            }
        }
        if session.outputs.len() != if consensus { 13 } else { 6 } {
            return Err("Unexpected chord output count".into());
        }
        Ok(Self {
            session,
            contract,
            consensus,
        })
    }

    pub fn predict(&mut self, images: &[RgbImage]) -> Result<Vec<Value>> {
        if images.is_empty() {
            return Ok(vec![]);
        }
        if images.len() > 32 {
            return Err("Chord evidence batch exceeds 32 images".into());
        }
        let mut input = Vec::with_capacity(images.len() * 3 * 256 * 256);
        for image in images {
            input.extend(image_tensor(image)?);
        }
        let tensor = Tensor::from_array(([images.len(), 3, 256, 256], input)).map_err(err)?;
        let result = self
            .session
            .run(ort::inputs!["images" => tensor])
            .map_err(err)?;
        let mut outputs = vec![];
        let mut dimensions: Vec<Vec<i64>> = vec![
            vec![3],
            vec![13],
            vec![37],
            vec![12, 9],
            vec![12, 6],
            vec![7, 11],
        ];
        let mut names: Vec<String> = NAMES.iter().map(|name| name.to_string()).collect();
        if self.consensus {
            dimensions.extend(dimensions.clone());
            dimensions.push(vec![2]);
            names.extend(NAMES.iter().map(|name| format!("secondary_{name}")));
            names.push("agreement".into());
        }
        for (name, dimensions) in names.iter().zip(&dimensions) {
            let (shape, values) = result[name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(err)?;
            let mut expected = vec![images.len() as i64];
            expected.extend_from_slice(dimensions);
            if shape.as_ref() != expected.as_slice() || values.iter().any(|v| !v.is_finite()) {
                return Err(format!("Malformed chord evidence {name} output"));
            }
            outputs.push(values.to_vec());
        }
        (0..images.len())
            .map(|index| {
                let sizes: Vec<usize> = dimensions
                    .iter()
                    .map(|shape| shape.iter().product::<i64>() as usize)
                    .collect();
                let values: Vec<&[f32]> = outputs
                    .iter()
                    .zip(sizes)
                    .map(|(values, size)| &values[index * size..(index + 1) * size])
                    .collect();
                let mut evidence = decode(&values[..6], &self.contract)?;
                if self.consensus {
                    let secondary = decode(&values[6..12], &self.contract["secondary_contract"])?;
                    let valid = evidence["diagram"].is_object() && secondary["diagram"].is_object();
                    let geometry = valid && values[12][0] >= 0.5;
                    let full = valid && values[12][1] >= 0.5;
                    evidence["agreement"] = json!({"geometry":geometry,"full":full});
                    evidence["secondary_diagram"] = secondary["diagram"].clone();
                    evidence["secondary_confidence"] = secondary["confidence"].clone();
                    evidence["accepted"]["geometry"] =
                        json!(geometry && evidence["accepted"]["geometry"] == true);
                    evidence["accepted"]["full"] = json!(
                        full && evidence["accepted"]["geometry"] == true
                            && evidence["accepted"]["full"] == true
                    );
                    if evidence["kind"] != secondary["kind"] {
                        evidence["accepted"]["kind"] = json!(false);
                    }
                    evidence["review_required"] = json!(
                        (evidence["kind"] == "diagram" || secondary["kind"] == "diagram")
                            && evidence["accepted"]["full"] != true
                    );
                }
                Ok(evidence)
            })
            .collect()
    }
}

/// Exactly the same contain geometry and Pillow-compatible resampling as training.
pub fn image_tensor(image: &RgbImage) -> Result<Vec<f32>> {
    check_dimensions(image.width(), image.height())?;
    let scale = 256. / image.width().max(image.height()) as f64;
    let width = (image.width() as f64 * scale + 0.5).floor().max(1.) as u32;
    let height = (image.height() as f64 * scale + 0.5).floor().max(1.) as u32;
    let resized = crate::image_transforms::resize_rgb_lanczos(image, width, height);
    let left = (256 - width) / 2;
    let top = (256 - height) / 2;
    let mut values = vec![0.; 3 * 256 * 256];
    for (channel, (mean, std)) in [(0.485f32, 0.229f32), (0.456, 0.224), (0.406, 0.225)]
        .into_iter()
        .enumerate()
    {
        values[channel * 65536..(channel + 1) * 65536].fill((1. - mean) / std);
        for y in 0..height {
            for x in 0..width {
                values[channel * 65536 + ((top + y) * 256 + left + x) as usize] =
                    (resized.get_pixel(x, y)[channel] as f32 / 255. - mean) / std;
            }
        }
    }
    Ok(values)
}

fn probabilities(values: &[f32], temperature: f64) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let values: Vec<_> = values
        .iter()
        .map(|v| ((*v as f64 - maximum) / temperature).exp())
        .collect();
    let sum: f64 = values.iter().sum();
    values.into_iter().map(|v| v / sum).collect()
}
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .reduce(|best, next| if next.1 > best.1 { next } else { best })
        .unwrap()
        .0
}

/// The output grammar represents each barre as connected adjacent-string edges.
/// Combining those learned labels uses no image pixels or chord-name dictionary.
fn decode(outputs: &[&[f32]], contract: &Value) -> Result<Value> {
    let temp = |name: &str| contract["temperature"][name].as_f64().unwrap();
    let kind = probabilities(outputs[0], temp("kind"));
    let strings = probabilities(outputs[1], temp("strings"));
    let base = probabilities(outputs[2], temp("base"));
    let class = argmax(&kind);
    let count = argmax(&strings);
    let base_fret = argmax(&base);
    let frets: Vec<_> = outputs[3]
        .chunks_exact(9)
        .map(|v| probabilities(v, temp("frets")))
        .collect();
    let fingers: Vec<_> = outputs[4]
        .chunks_exact(6)
        .map(|v| probabilities(v, temp("fingers")))
        .collect();
    let relative: Vec<Value> = frets
        .iter()
        .take(count)
        .map(|v| match argmax(v) {
            0 => json!("x"),
            1 => json!(0),
            n => json!(n - 1),
        })
        .collect();
    let mut relative_barres = vec![];
    let mut barre_confidence = 1f64;
    for (fret, row) in outputs[5].chunks_exact(11).enumerate() {
        let mut start = None;
        for index in 0..count {
            let active = if index < count.saturating_sub(1) {
                let probability =
                    1. / (1. + (-(row[index] as f64 / temp("barres")).clamp(-80., 80.)).exp());
                barre_confidence = barre_confidence.min(probability.max(1. - probability));
                probability >= 0.5
            } else {
                false
            };
            if active && start.is_none() {
                start = Some(index);
            } else if !active {
                if let Some(first) = start.take() {
                    relative_barres.push(json!([fret + 1, first, index]));
                }
            }
        }
    }
    let minimum = |rows: &[Vec<f64>]| {
        if count > 0 {
            rows.iter()
                .take(count)
                .map(|row| row[argmax(row)])
                .fold(1f64, f64::min)
        } else {
            0.
        }
    };
    let mut confidence = json!({"kind":kind[class], "strings":strings[count], "base":base[base_fret],
        "frets":minimum(&frets),"fingers":minimum(&fingers),"barres":if count > 1 {barre_confidence} else {0.}});
    let mut accepted: serde_json::Map<String, Value> = NAMES
        .into_iter()
        .map(|name| {
            (
                name.into(),
                json!(
                    confidence[name].as_f64().unwrap()
                        >= contract["thresholds"][name].as_f64().unwrap()
                ),
            )
        })
        .collect();
    for (name, keys) in [
        (
            "geometry",
            &["kind", "strings", "base", "frets", "barres"][..],
        ),
        ("full", &NAMES[..]),
    ] {
        let joint = keys
            .iter()
            .map(|key| confidence[*key].as_f64().unwrap())
            .fold(1f64, f64::min);
        let ok = keys.iter().all(|key| accepted[*key] == true)
            && joint >= contract["joint_thresholds"][name].as_f64().unwrap()
            && (name != "full" || accepted["geometry"] == true);
        confidence[name] = json!(joint);
        accepted.insert(name.into(), json!(ok));
    }
    let absolute: Vec<Value> = relative
        .iter()
        .map(|v| match v.as_u64() {
            Some(n) if n > 0 => json!(base_fret as u64 + n - 1),
            _ => v.clone(),
        })
        .collect();
    let diagram = if class == 0
        && count > 0
        && base_fret > 0
        && absolute.iter().all(|v| v.as_u64().is_none_or(|n| n <= 36))
    {
        json!({"base_fret":base_fret,"frets":absolute,
            "fingers":fingers.iter().take(count).map(|v| match argmax(v) {0=>Value::Null,n=>json!(n-1)}).collect::<Vec<_>>(),
            "barres":relative_barres.iter().map(|b| json!([base_fret as u64+b[0].as_u64().unwrap()-1,b[1],b[2]])).collect::<Vec<_>>()})
    } else {
        Value::Null
    };
    let diagram = super::normalize_diagram(&diagram);
    let reviewed = class == 0 && !diagram.is_null() && accepted.values().all(|v| v == true);
    Ok(
        json!({"schema":CONTRACT,"reader":contract["reader"],"kind":(["diagram","text","other"][class]),
        "kind_probabilities":kind,"strings":count,"base_fret":base_fret,"relative_frets":relative,
        "relative_barres":relative_barres,"diagram":diagram,"confidence":confidence,
        "accepted":accepted,"review_required":class == 0 && !reviewed}),
    )
}

/// Accepted diagram geometry takes precedence over text-model geometry.
/// Text-model differences remain in the audit; unresolved geometry retains the source.
/// The caller persists both returned evidence and the original model response.
pub fn apply_evidence(parsed: &Value, source_evidence: &Value) -> Result<(Value, Value)> {
    let mut parsed = parsed.clone();
    let mut evidence = source_evidence.clone();
    let original = super::normalize_diagram(&parsed["diagram"]);
    evidence["original_diagram"] = original.clone();
    let accepted = evidence["accepted"].clone();
    let mut changes = vec![];
    let mut applied = vec![];
    if evidence["kind"] == "diagram" && accepted["kind"] == true && evidence["diagram"].is_object()
    {
        if ["strings", "base", "frets", "barres"]
            .iter()
            .all(|key| accepted[*key] == true)
            && accepted["geometry"] == true
        {
            let mut diagram = evidence["diagram"].clone();
            if accepted["fingers"] != true {
                let count = evidence["strings"]
                    .as_u64()
                    .ok_or("Missing chord string count")? as usize;
                diagram["fingers"] = if original["frets"]
                    .as_array()
                    .is_some_and(|v| v.len() == count)
                {
                    original["fingers"].clone()
                } else {
                    json!(vec![Value::Null; count])
                };
            }
            for key in ["base_fret", "frets", "fingers", "barres"] {
                if original[key] != diagram[key] {
                    changes.push(key);
                }
            }
            evidence["proposed_diagram"] = diagram.clone();
            if !original.is_null() && !changes.is_empty() {
                evidence["conflict_fields"] = json!(changes);
            }
            if original.is_null() && (accepted["full"] != true || accepted["fingers"] != true) {
                evidence["review_required"] = json!(true);
            } else {
                parsed["kind"] = json!("chord_diagram");
                parsed["diagram"] = diagram;
                applied = changes.clone();
            }
        } else if !original.is_null() {
            parsed["diagram"] = original.clone();
        }
    } else if evidence["kind"] != "diagram" && accepted["kind"] == true && !original.is_null() {
        let valid_name = crate::recognition::is_chord_symbol(parsed["text"].as_str().unwrap_or(""));
        parsed["kind"] = json!(if valid_name { "chord" } else { "other" });
        parsed["diagram"] = Value::Null;
        evidence["rejected_diagram_candidate"] = original.clone();
        evidence["review_required"] = json!(true);
        applied.push("diagram");
    } else if !original.is_null() && accepted["kind"] != true {
        evidence["review_required"] = json!(true);
    }
    if !changes.is_empty() && !original.is_null() && accepted["full"] != true {
        evidence["review_required"] = json!(true);
    }
    evidence["proposed_changed_fields"] = json!(changes);
    evidence["changed_fields"] = json!(applied);
    if matches!(parsed["kind"].as_str(), Some("chord" | "chord_diagram"))
        && parsed["text"].as_str().is_some_and(|name| {
            !name.trim().is_empty() && !crate::recognition::is_chord_symbol(name)
        })
    {
        evidence["name_needs_review"] = json!(true);
    }
    Ok((parsed, evidence))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_diagram_replaces_conflicting_text_model_geometry() {
        let parsed = json!({"kind":"chord_diagram","text":"E","diagram":{
            "base_fret":7,"frets":["x",7,9,9,9,7],
            "fingers":[null,1,3,3,3,null],"barres":[[7,1,5],[9,2,4]]
        }});
        let mut evidence = json!({"kind":"diagram","strings":6,"diagram":{
            "base_fret":7,"frets":["x",7,9,9,9,"x"],
            "fingers":[null,1,3,3,3,null],"barres":[[9,2,4]]
        },"accepted":{"kind":true,"strings":true,"base":true,"frets":true,
            "fingers":true,"barres":true,"geometry":true,"full":true},
            "review_required":false});
        let (resolved, audit) = apply_evidence(&parsed, &evidence).unwrap();
        assert_eq!(resolved["diagram"], evidence["diagram"]);
        assert_eq!(resolved["text"], "E");
        assert_eq!(audit["original_diagram"], parsed["diagram"]);
        assert_eq!(audit["changed_fields"], json!(["frets", "barres"]));
        assert_eq!(audit["conflict_fields"], audit["changed_fields"]);
        assert_eq!(audit["review_required"], false);

        let annotation = json!({"page":1,"part_id":"part-1","bbox":[120,155,55,70],
            "context_measures":[1],"parsed":resolved,"chord_evidence":audit});
        assert!(!scorelib::chords::pending(&annotation));
        let mut records = vec![
            json!({"measure_number":1,"part_id":"part-1","staff_id":"tab",
            "page":1,"bbox":[100,200,160,100],"mode":"tab","tuning":[64,59,55,50,45,40],
            "target":"M time=4/4 | V0{@0:w:s1f0<chord:E>}"}),
        ];
        crate::recognition::attach_chord_annotations(&mut records, &[annotation.clone()]).unwrap();
        let effect = scorelib::chords::diagram_effect(&evidence["diagram"]).unwrap();
        assert!(records[0]["target"].as_str().unwrap().contains(&effect));
        assert_eq!(records[0]["chord_diagram_choices"], json!([]));
        let output = tempfile::tempdir().unwrap();
        let exported = scorelib::music_exports::write_gp5(
            &json!({"records":records,"document_metadata":{"score_annotations":[annotation]}}),
            &output.path().join("accepted.gp5"),
        )
        .unwrap();
        assert_eq!(exported["report"]["readback_verified"], true);
        assert!(exported["report"]["chord_library_omissions"].is_null());

        // Recovering a unique name must not leave a stale geometry review flag.
        let mut unnamed = parsed.clone();
        unnamed["text"] = json!("unreadable chord");
        let (resolved, audit) = apply_evidence(&unnamed, &evidence).unwrap();
        let mut recovered = json!({"parsed":resolved,"chord_evidence":audit});
        assert!(scorelib::chords::pending(&recovered));
        assert_eq!(recovered["chord_evidence"]["review_required"], false);
        recovered["parsed"]["text"] = json!("E");
        recovered["chord_evidence"]["name_needs_review"] = json!(false);
        assert!(!scorelib::chords::pending(&recovered));

        // An accepted geometry does not clear a separate name or reader issue.
        for flag in ["name_needs_review", "review_required"] {
            let mut uncertain = evidence.clone();
            uncertain[flag] = json!(true);
            let (parsed, audit) = apply_evidence(&parsed, &uncertain).unwrap();
            assert!(scorelib::chords::pending(
                &json!({"parsed":parsed,"chord_evidence":audit})
            ));
        }

        evidence["accepted"]["geometry"] = json!(false);
        let (unresolved, audit) = apply_evidence(&parsed, &evidence).unwrap();
        assert_eq!(unresolved["diagram"], parsed["diagram"]);
        assert_eq!(audit["changed_fields"], json!([]));
        let mut annotation = json!({"parsed":unresolved,"chord_evidence":audit,
            "chord_review":{"source":"model","status":"confirmed"}});
        assert!(scorelib::chords::pending(&annotation));
        annotation["chord_review"]["source"] = json!("manual");
        assert!(!scorelib::chords::pending(&annotation));
    }
}
