//! Learned correspondence between complete staff galleries on adjacent pages.
use crate::image_boundary::{check_dimensions, grayscale_rgb, Result};
use image::{Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

const ROW: [usize; 2] = [768, 192];
const CONTEXT: [usize; 2] = [384, 256];
const IDENTITIES: [Option<&str>; 5] = [
    None,
    Some("guitar"),
    Some("bass"),
    Some("pitched"),
    Some("drums"),
];
const MODES: [&str; 4] = ["unknown", "notation", "tab", "both"];
const LABELS: [&str; 3] = ["different", "same", "unknown"];
const MAX_PARTS: usize = 5;

fn contain(image: &RgbImage, size: [usize; 2]) -> RgbImage {
    let scale =
        (size[0] as f64 / f64::from(image.width())).min(size[1] as f64 / f64::from(image.height()));
    let width = (f64::from(image.width()) * scale).round_ties_even().max(1.) as u32;
    let height = (f64::from(image.height()) * scale)
        .round_ties_even()
        .max(1.) as u32;
    let resized = crate::image_transforms::resize_rgb_lanczos(&grayscale_rgb(image), width, height);
    let mut canvas = RgbImage::from_pixel(size[0] as u32, size[1] as u32, Rgb([255; 3]));
    image::imageops::replace(
        &mut canvas,
        &resized,
        (size[0] as i64 - i64::from(width)) / 2,
        (size[1] as i64 - i64::from(height)) / 2,
    );
    canvas
}

pub struct Gallery {
    rows: Vec<RgbImage>,
    contexts: Vec<Vec<f32>>,
}

impl Gallery {
    pub fn new(images: &[RgbImage]) -> Result<Self> {
        if images.is_empty() || images.len() > MAX_PARTS {
            return Err("Unsupported continuation gallery size".into());
        }
        for image in images {
            check_dimensions(image.width(), image.height())?;
        }
        let rows: Vec<_> = images.iter().map(|image| contain(image, ROW)).collect();
        let mut panel = RgbImage::from_pixel(CONTEXT[0] as u32, CONTEXT[1] as u32, Rgb([255; 3]));
        let mut bands = vec![];
        for (index, row) in rows.iter().enumerate() {
            let top =
                (index as f64 * CONTEXT[1] as f64 / rows.len() as f64).round_ties_even() as usize;
            let bottom = ((index + 1) as f64 * CONTEXT[1] as f64 / rows.len() as f64)
                .round_ties_even() as usize;
            let cell = contain(row, [CONTEXT[0], bottom - top]);
            image::imageops::replace(&mut panel, &cell, 0, top as i64);
            bands.push((top, bottom));
        }
        let pixels: Vec<f32> = panel.pixels().map(|p| f32::from(p[0]) / 255.).collect();
        let contexts = bands
            .into_iter()
            .map(|(top, bottom)| {
                let mut values = pixels.clone();
                values.resize(CONTEXT[0] * CONTEXT[1] * 2, 0.);
                values[CONTEXT[0] * (CONTEXT[1] + top)..CONTEXT[0] * (CONTEXT[1] + bottom)]
                    .fill(1.);
                values
            })
            .collect();
        Ok(Self { rows, contexts })
    }
}

pub struct PairInput {
    pub rows: Vec<f32>,
    pub contexts: Vec<f32>,
    pub features: Vec<f32>,
}

pub fn pair_input(
    previous: &Gallery,
    current: &Gallery,
    indices: [usize; 2],
    previous_identity: Option<&str>,
    modes: [&str; 2],
) -> Result<PairInput> {
    let mut rows = Vec::with_capacity(2 * ROW[0] * ROW[1]);
    let mut contexts = Vec::with_capacity(4 * CONTEXT[0] * CONTEXT[1]);
    for (gallery, index) in [previous, current].into_iter().zip(indices) {
        let row = gallery
            .rows
            .get(index)
            .ok_or("Invalid continuation query index")?;
        rows.extend(row.pixels().map(|p| f32::from(p[0]) / 255.));
        contexts.extend_from_slice(&gallery.contexts[index]);
    }
    let mut features: Vec<f32> = IDENTITIES
        .iter()
        .map(|value| f32::from(*value == previous_identity))
        .collect();
    for mode in modes {
        features.extend(MODES.iter().map(|value| f32::from(*value == mode)));
    }
    features.extend([
        previous.rows.len() as f32 / 8.,
        current.rows.len() as f32 / 8.,
        indices[0] as f32 / previous.rows.len().saturating_sub(1).max(1) as f32,
        indices[1] as f32 / current.rows.len().saturating_sub(1).max(1) as f32,
    ]);
    Ok(PairInput {
        rows,
        contexts,
        features,
    })
}

pub struct PartContinuationReader {
    session: Session,
    threshold: f64,
}

impl PartContinuationReader {
    pub fn new(path: &Path) -> Result<Self> {
        let session = guitarocr_engine::onnx::session(path)?;
        let encoded = session
            .metadata()
            .map_err(|e| e.to_string())?
            .custom("guitarocr.part_continuation")
            .map_err(|e| e.to_string())?
            .ok_or("Missing part continuation model contract")?;
        let config: Value = serde_json::from_str(&encoded).map_err(|e| e.to_string())?;
        let threshold = config["threshold"]
            .as_f64()
            .filter(|v| *v >= 0.5 && *v <= 1.)
            .ok_or("Invalid part continuation threshold")?;
        let mut features: Vec<String> = IDENTITIES
            .iter()
            .map(|value| format!("previous_identity_{}", value.unwrap_or("unknown")))
            .collect();
        for side in ["previous", "current"] {
            features.extend(MODES.iter().map(|mode| format!("{side}_mode_{mode}")));
        }
        features.extend(
            [
                "previous_count_div8",
                "current_count_div8",
                "previous_query_index_normalized",
                "current_query_index_normalized",
            ]
            .map(String::from),
        );
        if config["version"] != 1
            || config["row_size"] != json!(ROW)
            || config["context_size"] != json!(CONTEXT)
            || config["labels"] != json!(LABELS)
            || config["identities"] != json!(IDENTITIES)
            || config["modes"] != json!(MODES)
            || config["features"] != json!(features)
            || config["max_parts"] != MAX_PARTS
            || config["grayscale"] != "pil_l"
            || config["resize"] != "contain_lanczos_white_center"
            || config["normalization"] != "zero_one"
            || config["context_channels"]
                != json!([
                    "previous_pixels",
                    "previous_query_mask",
                    "current_pixels",
                    "current_query_mask"
                ])
        {
            return Err("Unsupported part continuation model contract".into());
        }
        Ok(Self { session, threshold })
    }

    pub fn predict(&mut self, input: PairInput) -> Result<Value> {
        let rows =
            Tensor::from_array(([1, 2, ROW[1], ROW[0]], input.rows)).map_err(|e| e.to_string())?;
        let contexts = Tensor::from_array(([1, 4, CONTEXT[1], CONTEXT[0]], input.contexts))
            .map_err(|e| e.to_string())?;
        let features = Tensor::from_array(([1, 17], input.features)).map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["rows" => rows, "contexts" => contexts, "features" => features])
            .map_err(|e| e.to_string())?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if shape.as_ref() != [1, 3] || logits.iter().any(|value| !value.is_finite()) {
            return Err("Invalid part continuation logits".into());
        }
        let (index, probabilities) = guitarocr_engine::onnx::classify(logits);
        let decision = if probabilities[index] >= self.threshold {
            LABELS[index]
        } else {
            "unknown"
        };
        Ok(
            json!({"source":"part_continuation_network", "decision":decision,
            "confidence":probabilities[index], "probabilities":probabilities,
            "threshold":self.threshold, "accepted":decision == "same",
            "status":if decision == "unknown" {"uncertain"} else {"recognized"}}),
        )
    }

    fn unsupported(&self) -> Value {
        json!({"source":"part_continuation_network", "decision":"unknown", "accepted":false,
            "confidence":null,"threshold":self.threshold,"probabilities":null,"status":"unsupported_context"})
    }
}

fn profile_mode(profile: &Value) -> &str {
    let evidence = &profile["staff_prediction"]["staff_evidence"]["mode"];
    if evidence["accepted"] == true {
        evidence["value"].as_str().unwrap_or("unknown")
    } else {
        "unknown"
    }
}

fn profile_image(profile: &Value) -> Result<RgbImage> {
    let prediction = &profile["staff_prediction"];
    let path = prediction["source_page"]
        .as_str()
        .ok_or("Part continuation needs an observed staff image")?;
    let page = crate::image_boundary::open_rgb(Path::new(path))?;
    super::direct_crop(&page, super::bbox(prediction)?)
}

/// Apply only a unique learned match; retain local absence and every pair score.
pub(super) fn propose(
    previous: &[Value],
    current: &mut [Value],
    indices: &[usize],
    reader: &mut PartContinuationReader,
    previous_page: u64,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    if previous.is_empty() || indices.is_empty() {
        return Ok(());
    }
    let galleries = if previous.len().max(indices.len()) <= MAX_PARTS {
        let before: Vec<_> = previous.iter().map(profile_image).collect::<Result<_>>()?;
        let after: Vec<_> = indices
            .iter()
            .map(|&i| profile_image(&current[i]))
            .collect::<Result<_>>()?;
        Some((Gallery::new(&before)?, Gallery::new(&after)?))
    } else {
        None
    };
    let mut proposals = vec![];
    for (query, &local) in indices.iter().enumerate() {
        let profile = &mut current[local];
        if profile["staff_label_evidence"]["status"] != "absent" {
            continue;
        }
        let mut candidates = vec![];
        for prior in 0..previous.len() {
            if previous[prior]["id"].as_str().is_none_or(str::is_empty) {
                continue;
            }
            super::check(cancelled)?;
            let identity = &previous[prior]["instrument_evidence"];
            let confirmed = identity["status"] == "resolved"
                && matches!(
                    identity["source"].as_str(),
                    Some("label_text" | "confirmed_continuation")
                );
            let mut prediction = if let Some((before, after)) = &galleries {
                reader.predict(pair_input(
                    before,
                    after,
                    [prior, query],
                    if confirmed {
                        identity["value"].as_str()
                    } else {
                        None
                    },
                    [profile_mode(&previous[prior]), profile_mode(profile)],
                )?)?
            } else {
                reader.unsupported()
            };
            prediction["part_id"] = previous[prior]["id"].clone();
            prediction["previous_index"] = json!(prior);
            candidates.push(prediction);
        }
        let accepted: Vec<_> = candidates
            .iter()
            .filter(|c| c["accepted"] == true)
            .cloned()
            .collect();
        profile["continuation_evidence"] = json!({"source":"part_continuation_network", "previous_page":previous_page,
            "local_part":local, "status":"unresolved", "matched_part_id":null, "candidates":candidates,
            "review_reason":if accepted.len() > 1 {"ambiguous_part_continuation"} else {"uncertain_part_continuation"}});
        if accepted.len() == 1 {
            proposals.push((local, accepted[0].clone()));
        }
    }
    let mut counts = std::collections::HashMap::new();
    for (_, proposal) in &proposals {
        *counts
            .entry(proposal["part_id"].as_str().unwrap_or(""))
            .or_insert(0) += 1;
    }
    let counts: std::collections::HashMap<String, usize> =
        counts.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
    for (local, proposal) in proposals {
        let profile = &mut current[local];
        if counts[proposal["part_id"].as_str().unwrap_or("")] != 1 {
            profile["continuation_evidence"]["review_reason"] =
                json!("competing_part_continuations");
            continue;
        }
        let prior = &previous[proposal["previous_index"]
            .as_u64()
            .ok_or("Invalid previous part index")? as usize];
        if !crate::score_structure::compatible_part_profiles(prior, profile) {
            profile["continuation_evidence"]["review_reason"] = json!("conflicting_part_profile");
            continue;
        }
        let evidence = &mut profile["continuation_evidence"];
        evidence["status"] = json!("matched");
        evidence["matched_part_id"] = prior["id"].clone();
        evidence["review_reason"] = Value::Null;
        let continuing = &prior["instrument_evidence"];
        if continuing["status"] != "resolved"
            || !matches!(
                continuing["source"].as_str(),
                Some("label_text" | "confirmed_continuation")
            )
        {
            // A shared part ID does not establish an instrument or name.
            continue;
        }
        let mut identity = super::staff_evidence::resolve_instrument_with_label(
            matches!(profile_mode(profile), "tab" | "both"),
            &profile["staff_prediction"]["parsed"]["instrument"],
            &profile["structure_prediction"]["instrument"],
            None,
            Some(&profile["staff_label_evidence"]),
            Some(&prior["instrument_evidence"]),
        );
        identity["continuation_evidence"] = profile["continuation_evidence"].clone();
        profile["name"] = json!(prior["name"].as_str().unwrap_or(""));
        profile["name_source"] = json!("confirmed_continuation");
        profile["instrument"] = identity["value"].clone();
        profile["instrument_evidence"] = identity;
        profile["program"] = prior["program"].clone();
        profile["program_source"] = json!("confirmed_continuation");
    }
    Ok(())
}
