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
    assignment_margin: f64,
}

pub struct Assignment {
    pub current: usize,
    pub previous: usize,
    pub margin: f64,
    pub accepted: bool,
}

/// The best one-to-one explanation of the learned scores may leave rows unmatched.
pub fn match_gallery(
    probabilities: &[Vec<[f64; 3]>],
    threshold: f64,
    margin: f64,
) -> Vec<Assignment> {
    let edges: Vec<Vec<(usize, f64)>> = probabilities
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(previous, values)| {
                    let p = values[1];
                    (p >= threshold && p > values[0] && p >= values[2]).then(|| {
                        let p = p.clamp(1e-8, 1. - 1e-8);
                        (previous, (p / (1. - p)).ln())
                    })
                })
                .collect()
        })
        .collect();
    fn solve(
        edges: &[Vec<(usize, f64)>],
        current: usize,
        used: usize,
        excluded: Option<(usize, usize)>,
    ) -> (f64, Vec<Option<usize>>) {
        if current == edges.len() {
            return (0., vec![]);
        }
        let (score, mut tail) = solve(edges, current + 1, used, excluded);
        tail.insert(0, None);
        let mut best = (score, tail);
        for &(previous, weight) in &edges[current] {
            if used & (1 << previous) != 0 || excluded == Some((current, previous)) {
                continue;
            }
            let (score, mut tail) = solve(edges, current + 1, used | (1 << previous), excluded);
            if score + weight > best.0 {
                tail.insert(0, Some(previous));
                best = (score + weight, tail);
            }
        }
        best
    }
    let (score, assignment) = solve(&edges, 0, 0, None);
    assignment
        .into_iter()
        .enumerate()
        .filter_map(|(current, previous)| {
            previous.map(|previous| {
                let alternative = solve(&edges, 0, 0, Some((current, previous))).0;
                let difference = score - alternative;
                Assignment {
                    current,
                    previous,
                    margin: difference,
                    accepted: difference > margin.max(1e-6),
                }
            })
        })
        .collect()
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
        let assignment_margin = config["assignment_margin"]
            .as_f64()
            .filter(|v| v.is_finite() && *v >= 0.)
            .ok_or("Invalid part assignment margin")?;
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
            || config["assignment"] != "one_to_one_log_odds"
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
        Ok(Self {
            session,
            threshold,
            assignment_margin,
        })
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

/// Printed identities reserve the same part that the structure resolver will use.
fn constrain_printed_names(
    probabilities: &mut [Vec<[f64; 3]>],
    previous: &[Value],
    current: &[Value],
    indices: &[usize],
) {
    use crate::score_structure::{compatible_part_profiles, part_name_key};
    let names: Vec<_> = indices
        .iter()
        .map(|&i| {
            if current[i]["staff_label_evidence"]["status"] == "read" {
                part_name_key(&current[i])
            } else {
                String::new()
            }
        })
        .collect();
    for (query, name) in names.iter().enumerate() {
        if name.is_empty() || names.iter().filter(|other| *other == name).count() != 1 {
            continue;
        }
        let matches: Vec<_> = previous
            .iter()
            .enumerate()
            .filter(|(_, part)| part_name_key(part) == *name)
            .map(|(i, _)| i)
            .collect();
        match matches.as_slice() {
            // A distinct printed name establishes a new part, outside this gallery.
            [] => probabilities[query].fill([0., 0., 1.]),
            [prior] if compatible_part_profiles(&previous[*prior], &current[indices[query]]) => {
                probabilities[query].fill([0., 0., 1.]);
                for row in probabilities.iter_mut() {
                    row[*prior] = [0., 0., 1.];
                }
            }
            _ => {}
        }
    }
}

/// Resolve complete galleries using learned scores and one-to-one ownership.
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
    let mut probabilities = vec![vec![[0., 0., 1.]; previous.len()]; indices.len()];
    for (query, &local) in indices.iter().enumerate() {
        let profile = &mut current[local];
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
            if let Some(values) = prediction["probabilities"].as_array() {
                probabilities[query][prior] = std::array::from_fn(|i| values[i].as_f64().unwrap());
            }
            candidates.push(prediction);
        }
        // A proposed label box with no readable name cannot veto visual continuation.
        if matches!(
            profile["staff_label_evidence"]["status"].as_str(),
            Some("absent" | "unreadable")
        ) {
            profile["continuation_evidence"] = json!({"source":"part_continuation_network", "previous_page":previous_page,
                "local_part":local, "status":"unresolved", "matched_part_id":null, "candidates":candidates,
                "assignment":"one_to_one_log_odds", "assignment_margin_threshold":reader.assignment_margin,
                "review_reason":"uncertain_part_continuation"});
        }
    }
    constrain_printed_names(&mut probabilities, previous, current, indices);
    for assignment in match_gallery(&probabilities, reader.threshold, reader.assignment_margin) {
        let local = indices[assignment.current];
        let profile = &mut current[local];
        if !matches!(
            profile["staff_label_evidence"]["status"].as_str(),
            Some("absent" | "unreadable")
        ) {
            continue;
        }
        profile["continuation_evidence"]["assignment_margin"] = json!(assignment.margin);
        if !assignment.accepted {
            profile["continuation_evidence"]["review_reason"] =
                json!("ambiguous_part_continuation");
            continue;
        }
        let prior = &previous[assignment.previous];
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

#[cfg(test)]
mod tests {
    use super::{constrain_printed_names, match_gallery};
    use serde_json::json;

    #[test]
    fn global_visual_assignment_resolves_competition_and_leaves_unknowns_open() {
        let p = |same: f64| [(1. - same) / 2., same, (1. - same) / 2.];
        let scores = vec![
            vec![p(0.986), p(0.981), p(0.044)],
            vec![p(0.944), p(0.992), p(0.561)],
            vec![p(0.040), p(0.517), p(0.970)],
        ];
        let selected: Vec<_> = match_gallery(&scores, 0.85, 0.5)
            .into_iter()
            .filter(|m| m.accepted)
            .map(|m| (m.current, m.previous))
            .collect();
        assert_eq!(selected, vec![(0, 0), (1, 1), (2, 2)]);
        let ambiguous = vec![vec![p(0.99)], vec![p(0.99)]];
        assert!(!match_gallery(&ambiguous, 0.5, 0.)
            .iter()
            .any(|m| m.accepted));
        let added = vec![vec![p(0.99)], vec![[0.1, 0.1, 0.8]]];
        let selected: Vec<_> = match_gallery(&added, 0.5, 0.5)
            .into_iter()
            .filter(|m| m.accepted)
            .map(|m| (m.current, m.previous))
            .collect();
        assert_eq!(selected, vec![(0, 0)]);

        // The network prefers a swap, but printed Guitar already owns part 0.
        let previous = vec![json!({"name":"Guitar"}), json!({"name":"Bass"})];
        let mut current = vec![
            json!({"name":"GUITAR.","name_source":"label_text","staff_label_evidence":{"status":"read"}}),
            json!({"name":"","name_source":"missing","staff_label_evidence":{"status":"absent"}}),
        ];
        let mut scores = vec![vec![p(0.9), p(0.999)], vec![p(0.999), p(0.9)]];
        constrain_printed_names(&mut scores, &previous, &current, &[0, 1]);
        let selected: Vec<_> = match_gallery(&scores, 0.8, 0.)
            .into_iter()
            .filter(|m| m.accepted)
            .map(|m| (m.current, m.previous))
            .collect();
        assert_eq!(selected, vec![(1, 1)]);

        // A new printed name cannot consume an old part needed by an anonymous row.
        current[0]["name"] = json!("Piano");
        let mut scores = vec![vec![p(0.99), p(0.999)], vec![p(0.9), p(0.8)]];
        constrain_printed_names(&mut scores, &previous, &current, &[0, 1]);
        let selected: Vec<_> = match_gallery(&scores, 0.8, 0.)
            .into_iter()
            .filter(|m| m.accepted)
            .map(|m| (m.current, m.previous))
            .collect();
        assert_eq!(selected, vec![(1, 0)]);

        current[0]["name"] = json!("Guitar");
        current[0]["strings"] = json!(7);
        let mut scores = vec![vec![p(0.9), p(0.99)], vec![p(0.99), p(0.9)]];
        let original = scores.clone();
        constrain_printed_names(&mut scores, &previous, &current, &[0, 1]);
        assert_eq!(scores, original); // Same name with an incompatible profile is not an anchor.
    }
}
