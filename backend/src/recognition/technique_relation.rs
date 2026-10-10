//! Calibrated event coverage in the real OCR pipeline.
use super::{list, text, Result};
use crate::image_boundary;
use crate::technique_grounding::EventGroundingReader;
use crate::technique_span::EventSpanRelationReader;
use image::RgbImage;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub const PROTOCOL: &str = "visible_technique_relation";
pub const CONTEXT_POLICY: &str = "full_system_with_previous";
const CONTEXT_FIELDS: [&str; 5] = [
    "source_page",
    "technique_system_bbox",
    "technique_query_bbox",
    "technique_previous_context",
    "technique_context_error",
];
const EFFECT_POLICY: &str = "independent_visible_effects";

fn context_identity(row: &Value) -> Value {
    Value::Object(
        CONTEXT_FIELDS
            .into_iter()
            .map(|key| (key.into(), row[key].clone()))
            .collect(),
    )
}

fn review_context_matches(row: &Value) -> bool {
    let review = &row["visual_technique_review"];
    review.is_object()
        && review["context_identity"] == context_identity(row)
        && review["context_policy"] == CONTEXT_POLICY
}

/// Retain the previous reader inputs before replacing them with current crops.
pub fn refresh_context(row: &mut Value, prepared: &Value) {
    for key in CONTEXT_FIELDS {
        if let Some(value) = prepared.get(key) {
            row[key] = value.clone();
        } else if let Some(object) = row.as_object_mut() {
            object.remove(key);
        }
    }
}

fn effect_probabilities(probabilities: &Value) -> Value {
    let values: Option<Vec<f64>> = ["none", "pm", "let", "both", "unknown"]
        .iter()
        .map(|key| {
            probabilities[*key]
                .as_f64()
                .filter(|v| v.is_finite() && (0. ..=1.).contains(v))
        })
        .collect();
    let Some(p) = values.filter(|v| (v.iter().sum::<f64>() - 1.).abs() <= 1e-5) else {
        return json!({});
    };
    // Unknown probability contributes to neither positive nor negative truth.
    json!({"pm":{"present":p[1]+p[3],"absent":p[0]+p[2]},
           "let":{"present":p[2]+p[3],"absent":p[0]+p[1]}})
}

fn effect_decisions(probabilities: &Value, thresholds: &Value) -> Value {
    let scores = effect_probabilities(probabilities);
    let mut result = json!({"pm":null,"let":null});
    for effect in ["pm", "let"] {
        let accepted: Vec<_> = ["present", "absent"]
            .into_iter()
            .filter(|state| {
                thresholds[effect][*state]
                    .as_f64()
                    .zip(scores[effect][*state].as_f64())
                    .is_some_and(|(threshold, score)| score >= threshold)
            })
            .collect();
        if accepted.len() == 1 {
            result[effect] = json!(accepted[0] == "present");
        }
    }
    result
}

fn merged_effects(event: &Value, baseline: &BTreeSet<String>) -> Result<Option<BTreeSet<String>>> {
    let fields = &event["effect_decisions"];
    if fields.as_object().is_none_or(|v| {
        v.len() != 2
            || !v.contains_key("pm")
            || !v.contains_key("let")
            || v.values().any(|v| !v.is_null() && !v.is_boolean())
    }) {
        return Err("Invalid independent visible effect decision".into());
    }
    if fields["pm"].is_null() && fields["let"].is_null() {
        return Ok(None);
    }
    Ok(Some(
        ["pm", "let"]
            .into_iter()
            .filter(|effect| fields[*effect] == true || (fields[*effect].is_null() && baseline.contains(*effect)))
            .map(str::to_owned)
            .collect(),
    ))
}

fn group_key(row: &Value) -> String {
    json!([
        row["source_page"],
        row["page"],
        row["system_index"],
        row["part_id"],
        row["staff_id"]
    ])
    .to_string()
}

/// Keep the complete target row; query-conditioned inference sees shared ink.
/// Detector boxes can overlap and are not ink ownership masks. Only trim
/// optional padding where a neighbor is wholly outside the target row.
fn system_context_bounds(
    page_width: u32,
    page_height: u32,
    boxes: &[[f64; 4]],
    neighbors: &[[f64; 4]],
) -> Result<[u32; 4]> {
    if boxes.is_empty()
        || boxes
            .iter()
            .chain(neighbors)
            .any(|b| !b.iter().all(|v| v.is_finite()) || b[2] <= 0. || b[3] <= 0.)
    {
        return Err("System context requires finite, positive row geometry".into());
    }
    let left = boxes.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
    let top = boxes.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
    let right = boxes
        .iter()
        .map(|b| b[0] + b[2])
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = boxes
        .iter()
        .map(|b| b[1] + b[3])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut system = image_boundary::measure_crop_bounds(
        page_width,
        page_height,
        [left, top, right - left, bottom - top],
        2.,
    )?;
    let visible_top = top.max(0.).floor() as u32;
    let visible_bottom = bottom.min(page_height as f64).ceil() as u32;
    for b in neighbors {
        if b[0] + b[2] <= left || b[0] >= right {
            continue;
        }
        let other_bottom = b[1] + b[3];
        if other_bottom <= top {
            system[1] = system[1].max(visible_top.min(other_bottom.ceil().max(0.) as u32));
        } else if b[1] >= bottom {
            system[3] =
                system[3].min(visible_bottom.max(b[1].floor().min(page_height as f64) as u32));
        }
    }
    Ok(system)
}

/// Geometry is detector/structure evidence, never native/oracle renderer data.
/// All records are considered even for a retry of a single selected measure.
pub fn prepare_contexts(records: &mut [Value], cancelled: &dyn Fn() -> bool) -> Result<()> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, row) in records.iter_mut().enumerate() {
        groups.entry(group_key(row)).or_default().push(index);
    }
    let mut page_cache: Option<(String, RgbImage)> = None;
    for indices in groups.values() {
        super::check(cancelled)?;
        let first = &records[indices[0]];
        let result = (|| -> Result<(Vec<[u32; 4]>, [u32; 4])> {
            let source = first["source_page"]
                .as_str()
                .ok_or("Complete system source page is unavailable")?;
            if first["system_index"].as_u64().is_none()
                || indices.iter().any(|&i| {
                    !["tab", "both", "notation"].contains(&text(&records[i], "mode", ""))
                        || records[i]["structure_needs_review"] == true
                        || records[i]["alignment_needs_review"] == true
                })
            {
                return Err("Complete system geometry is unavailable".into());
            }
            if page_cache.as_ref().is_none_or(|(path, _)| path != source) {
                page_cache = Some((source.into(), image_boundary::open_rgb(Path::new(source))?));
            }
            let page = &page_cache.as_ref().unwrap().1;
            let boxes = indices
                .iter()
                .map(|&i| super::bbox(&records[i]))
                .collect::<Result<Vec<_>>>()?;
            let neighbors = records
                .iter()
                .enumerate()
                .filter(|(index, other)| {
                    !indices.contains(index) && other["source_page"] == first["source_page"]
                })
                .map(|(_, row)| super::bbox(row))
                .collect::<Result<Vec<_>>>()?;
            let system = system_context_bounds(page.width(), page.height(), &boxes, &neighbors)?;
            let targets = boxes
                .into_iter()
                .map(|bbox| {
                    image_boundary::measure_crop_bounds(page.width(), page.height(), bbox, 1.5)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((targets, system))
        })();
        for (local, &index) in indices.iter().enumerate() {
            let row = &mut records[index];
            if let Some(object) = row.as_object_mut() {
                object.remove("technique_context_error");
                object.remove("technique_system_bbox");
                object.remove("technique_query_bbox");
                object.remove("technique_previous_context");
            }
            match &result {
                Ok((targets, system)) => {
                    row["technique_system_bbox"] = json!(system);
                    row["technique_query_bbox"] = json!(targets[local]);
                }
                Err(error) => row["technique_context_error"] = json!(error),
            }
        }
    }
    // Musical identity and bar continuity choose the preceding row. A pixel
    // gap or the previous item in detector order cannot establish that link.
    for indices in groups.values() {
        let first = &records[indices[0]];
        if first.get("technique_context_error").is_some() {
            continue;
        }
        let current_bar = indices
            .iter()
            .filter_map(|&i| records[i]["bar_index"].as_i64())
            .min();
        let Some(current_bar) = current_bar else {
            continue;
        };
        let previous = groups
            .values()
            .filter(|other| other[0] != indices[0])
            .filter(|other| {
                let prior = &records[other[0]];
                prior.get("technique_context_error").is_none()
                    && !first["part_id"].is_null()
                    && !first["staff_id"].is_null()
                    && prior["part_id"] == first["part_id"]
                    && prior["staff_id"] == first["staff_id"]
                    && prior["source_pdf"] == first["source_pdf"]
                    && other
                        .iter()
                        .filter_map(|&i| records[i]["bar_index"].as_i64())
                        .max()
                        == Some(current_bar - 1)
            })
            .collect::<Vec<_>>();
        if previous.len() != 1 {
            continue;
        }
        let prior = &records[previous[0][0]];
        let context = json!({"source_page":prior["source_page"],"page":prior["page"],"system_index":prior["system_index"],
            "bbox":prior["technique_system_bbox"]});
        for &index in indices {
            records[index]["technique_previous_context"] = context.clone();
        }
    }
    super::check(cancelled)
}

fn marks(measure: &Value) -> Result<BTreeMap<(i64, i64), BTreeSet<String>>> {
    let mut result = BTreeMap::new();
    for voice in list(measure, "voices") {
        for event in list(voice, "events")
            .iter()
            .filter(|event| !list(event, "notes").is_empty())
        {
            let anchor = (
                voice["voice"].as_i64().ok_or("Invalid voice")?,
                event["start"].as_i64().ok_or("Invalid onset")?,
            );
            let effects = std::iter::once(event)
                .chain(list(event, "notes"))
                .flat_map(|item| list(item, "effects"))
                .filter_map(Value::as_str)
                .filter(|effect| ["pm", "let"].contains(effect))
                .map(str::to_owned)
                .collect();
            if result.insert(anchor, effects).is_some() {
                return Err("Duplicate event identity".into());
            }
        }
    }
    Ok(result)
}

fn crop_context(page: &RgbImage, bounds: &Value) -> Result<([u32; 4], RgbImage)> {
    let values = bounds
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("Missing complete system crop mapping")?;
    let mut b = [0; 4];
    for (index, value) in values.iter().enumerate() {
        b[index] = value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("Invalid system crop mapping")?;
    }
    if b[2] <= b[0] || b[3] <= b[1] || b[2] > page.width() || b[3] > page.height() {
        return Err("System crop outside source page".into());
    }
    Ok((
        b,
        image::imageops::crop_imm(page, b[0], b[1], b[2] - b[0], b[3] - b[1]).to_image(),
    ))
}

fn apply_evidence(row: &mut Value, original: &str, evidence: &[Value]) -> Result<()> {
    let effects = marks(&scorelib::score::parse_measure_target(original)?)?;
    let sounding = marks(&scorelib::score::parse_measure_target(text(
        row, "target", "",
    ))?)?;
    if effects.keys().ne(sounding.keys()) {
        return Err("Written and sounding event identities differ".into());
    }
    let mut decisions = Vec::new();
    let (mut accepted, mut positives, mut additions, mut removals) = (0, 0, 0, 0);
    for event in evidence {
        let anchor = (
            event["voice"].as_i64().ok_or("Invalid evidence voice")?,
            event["onset"].as_i64().ok_or("Invalid evidence onset")?,
        );
        let mut baseline = effects
            .get(&anchor)
            .ok_or("Unknown evidence event")?
            .clone();
        baseline.extend(sounding.get(&anchor).unwrap().iter().cloned());
        let merged_label = if let Some(merged) = merged_effects(event, &baseline)? {
            additions += merged.difference(&baseline).count();
            removals += baseline.difference(&merged).count();
            accepted += 1;
            positives += usize::from(!merged.is_empty());
            if merged.len() == 2 {
                "both"
            } else if merged.contains("pm") {
                "pm"
            } else if merged.contains("let") {
                "let"
            } else {
                "none"
            }
        } else {
            "unknown"
        };
        decisions.push(json!([anchor.0, anchor.1, merged_label]));
    }
    let overlay = scorelib::visual_techniques::from_output(
        text(row, "target", ""),
        &json!({"events":decisions}).to_string(),
    )?;
    scorelib::visual_techniques::validate_overlay(
        &scorelib::score::parse_measure_target(text(row, "target", ""))?,
        &overlay,
    )?;
    row["visual_techniques"] = overlay;
    row["visual_technique_review"] = json!({"protocol":PROTOCOL,"accepted":true,"scope":"visible_event",
        "accepted_events":accepted,"accepted_positive_events":positives,"added_effects":additions,"removed_effects":removals,"source_note_owners_changed":false,"applied":true,"evidence":evidence});
    Ok(())
}

pub struct CalibratedTechniqueReader {
    grounding: EventGroundingReader,
    relation: EventSpanRelationReader,
    calibration: Value,
    page_cache: Option<(PathBuf, RgbImage)>,
}
impl CalibratedTechniqueReader {
    pub fn new(grounding: &Path, relation: &Path, calibration: &Path) -> Result<Self> {
        let config: Value =
            serde_json::from_slice(&std::fs::read(calibration).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if config["protocol"] != PROTOCOL
            || config["application_policy"] != EFFECT_POLICY
            || config["supported_modes"].as_array().is_none_or(|modes| {
                modes.is_empty()
                    || modes
                        .iter()
                        .any(|m| !["tab", "both", "notation"].contains(&m.as_str().unwrap_or("")))
            })
            || config["input"]["require_full_system_context"] != true
            || config["acceptance"]["rewrite_note_effects"] != false
        {
            return Err("Unsupported calibrated technique relation contract".into());
        }
        let fields = &config["effect_thresholds"];
        if fields
            .as_object()
            .is_none_or(|v| v.len() != 2 || !v.contains_key("pm") || !v.contains_key("let"))
        {
            return Err("Invalid independent technique calibration contract".into());
        }
        for effect in ["pm", "let"] {
            if fields[effect].as_object().is_none_or(|v| {
                v.len() != 2
                    || !v.contains_key("present")
                    || !v.contains_key("absent")
                    || v.values().any(|v| {
                        !v.is_null()
                            && v.as_f64()
                                .is_none_or(|n| !n.is_finite() || !(0.5..=1.).contains(&n))
                    })
            }) {
                return Err("Invalid independent technique calibration threshold".into());
            }
        }
        Ok(Self {
            grounding: EventGroundingReader::new(grounding)?,
            relation: EventSpanRelationReader::new(relation)?,
            calibration: config,
            page_cache: None,
        })
    }

    pub fn predict_with_context(
        &mut self,
        target: &RgbImage,
        system: &RgbImage,
        previous: Option<&RgbImage>,
        offset: [f64; 2],
        original: &str,
        mode: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Value>> {
        super::check(cancelled)?;
        let music = scorelib::score::parse_measure_target(original)?;
        if !list(&self.calibration, "supported_modes")
            .iter()
            .any(|m| m == mode)
        {
            return Ok(unknown(&music, "unsupported_mode")?);
        }
        let grounded = self.grounding.predict(target, &music, mode)?;
        super::check(cancelled)?;
        let queries: Vec<_> = grounded
            .iter()
            .map(|event| {
                let mut event = event.clone();
                event["pixel"] = json!([
                    event["pixel"][0].as_f64().unwrap() + offset[0],
                    event["pixel"][1].as_f64().unwrap() + offset[1]
                ]);
                event
            })
            .collect();
        let mut result = self
            .relation
            .predict(system, previous, &queries, cancelled)?;
        for (event, position) in result.iter_mut().zip(&grounded) {
            let decisions = if position["inside_image"] == true {
                effect_decisions(
                    &event["probabilities"],
                    &self.calibration["effect_thresholds"],
                )
            } else {
                json!({"pm":null,"let":null})
            };
            event["proposed_label"] = event["label"].clone();
            event["label"] = json!("unknown");
            event["effect_probabilities"] = effect_probabilities(&event["probabilities"]);
            event["status"] = json!(
                if !decisions["pm"].is_null() || !decisions["let"].is_null() {
                    "accepted_visual_effects"
                } else {
                    "unresolved_visual_evidence"
                }
            );
            event["effect_decisions"] = decisions;
            if position["inside_image"] != true {
                event["reason"] = json!("pixel_outside_target");
            }
        }
        Ok(result)
    }

    /// Called before the completed OCR row is journaled; failures preserve music.
    pub fn recognize(&mut self, row: &mut Value, cancelled: &dyn Fn() -> bool) -> Result<()> {
        super::check(cancelled)?;
        if row["manually_edited"] == true || review_context_matches(row) {
            return Ok(());
        }
        if let Some(object) = row.as_object_mut() {
            object.remove("visual_technique_review");
            object.remove("visual_techniques");
        }
        let sounding = scorelib::score::parse_measure_target(text(row, "target", ""))?;
        if marks(&sounding)?.is_empty() {
            return Ok(());
        }
        let original = text(row, "written_target", text(row, "target", "")).to_owned();
        let music = match scorelib::score::parse_measure_target(&original) {
            Ok(music) if marks(&music)?.keys().eq(marks(&sounding)?.keys()) => music,
            _ => {
                row["visual_technique_review"] = json!({"protocol":self.calibration["protocol"],"accepted":false,
                    "removed_effects":0,"source_note_owners_changed":false,
                    "error":"Written and sounding event identities differ or original music is unavailable"});
                row["visual_technique_review"]["context_policy"] = json!(CONTEXT_POLICY);
                row["visual_technique_review"]["context_identity"] = context_identity(row);
                return Ok(());
            }
        };
        let result = (|| -> Result<Vec<Value>> {
            if let Some(error) = row["technique_context_error"].as_str() {
                return Err(error.into());
            }
            let path = PathBuf::from(
                row["source_page"]
                    .as_str()
                    .ok_or("Missing system source page")?,
            );
            if self
                .page_cache
                .as_ref()
                .is_none_or(|(cached, _)| *cached != path)
            {
                self.page_cache = Some((path.clone(), image_boundary::open_rgb(&path)?));
            }
            let page = &self.page_cache.as_ref().unwrap().1;
            let (query_box, query) = crop_context(page, &row["technique_query_bbox"])?;
            let (system_box, system) = crop_context(page, &row["technique_system_bbox"])?;
            let previous = if row["technique_previous_context"].is_object() {
                let context = &row["technique_previous_context"];
                let previous_path = Path::new(
                    context["source_page"]
                        .as_str()
                        .ok_or("Missing previous system source page")?,
                );
                Some(if previous_path == path {
                    crop_context(page, &context["bbox"])?.1
                } else {
                    crop_context(&image_boundary::open_rgb(previous_path)?, &context["bbox"])?.1
                })
            } else {
                None
            };
            let exact_query = image_boundary::grayscale_rgb(&query);
            let target = if let Some(path) = row["image"].as_str() {
                let saved = image_boundary::open_rgb(Path::new(path))?;
                if image_boundary::grayscale_rgb(&saved) != exact_query {
                    return Err("Saved measure crop does not match page geometry".into());
                }
                saved
            } else {
                exact_query
            };
            self.predict_with_context(
                &target,
                &system,
                previous.as_ref(),
                [
                    query_box[0] as f64 - system_box[0] as f64,
                    query_box[1] as f64 - system_box[1] as f64,
                ],
                &original,
                text(row, "mode", ""),
                cancelled,
            )
        })();
        super::check(cancelled)?;
        match result {
            Ok(evidence) => apply_evidence(row, &original, &evidence)?,
            Err(error) => {
                apply_evidence(row, &original, &unknown(&music, &error)?)?;
                row["visual_technique_review"]["error"] = json!(error);
            }
        }
        row["visual_technique_review"]["context_policy"] = json!(CONTEXT_POLICY);
        row["visual_technique_review"]["context_identity"] = context_identity(row);
        Ok(())
    }
}

fn unknown(music: &Value, reason: &str) -> Result<Vec<Value>> {
    Ok(marks(music)?
        .into_keys()
        .map(
            |(voice, onset)| json!({"voice":voice,"onset":onset,"label":"unknown","effect_decisions":{"pm":null,"let":null},"reason":reason}),
        )
        .collect())
}
