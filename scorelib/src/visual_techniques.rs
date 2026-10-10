//! Printed event coverage is separate from source playback note ownership.
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const PROTOCOL: &str = "visible_events_v1";
type R<T> = Result<T, String>;
type Anchor = (i64, i64);

fn array<'a>(value: &'a Value, key: &str) -> R<&'a Vec<Value>> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("Missing {key} array"))
}

fn events(measure: &Value) -> R<BTreeMap<Anchor, &Value>> {
    let mut result = BTreeMap::new();
    for voice in array(measure, "voices")? {
        let index = voice["voice"].as_i64().ok_or("Invalid voice")?;
        for event in array(voice, "events")? {
            if array(event, "notes")?.is_empty() {
                continue;
            }
            let onset = event["start"].as_i64().ok_or("Invalid onset")?;
            if result.insert((index, onset), event).is_some() {
                return Err("Duplicate sounding event anchor".into());
            }
        }
    }
    Ok(result)
}

fn marks(event: &Value) -> BTreeSet<String> {
    std::iter::once(event)
        .chain(event["notes"].as_array().into_iter().flatten())
        .flat_map(|owner| owner["effects"].as_array().into_iter().flatten())
        .filter_map(Value::as_str)
        .filter(|v| ["pm", "let"].contains(v))
        .map(str::to_owned)
        .collect()
}

/// Model replies require all existing sounding anchors exactly once.
pub fn from_output(original: &str, raw: &str) -> R<Value> {
    let measure = crate::score::parse_measure_target(original)?;
    let expected = events(&measure)?;
    let value: Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    if value
        .as_object()
        .is_none_or(|v| v.len() != 1 || !v.contains_key("events"))
    {
        return Err("Expected a JSON object containing only events".into());
    }
    let mut classified = BTreeMap::new();
    for item in array(&value, "events")? {
        let item = item
            .as_array()
            .filter(|v| v.len() == 3)
            .ok_or("Expected [voice,onset,label]")?;
        let key = (
            item[0].as_i64().ok_or("Invalid voice")?,
            item[1].as_i64().ok_or("Invalid onset")?,
        );
        let event = expected.get(&key).ok_or("Unknown sounding event anchor")?;
        let label = item[2].as_str().ok_or("Invalid technique label")?;
        let effects = match label {
            "none" => BTreeSet::new(),
            "pm" => BTreeSet::from(["pm".to_owned()]),
            "let" => BTreeSet::from(["let".to_owned()]),
            "both" => BTreeSet::from(["pm".to_owned(), "let".to_owned()]),
            "unknown" => marks(event),
            _ => return Err("Invalid technique label".into()),
        };
        if classified
            .insert(
                key,
                json!({"voice":key.0,"onset":key.1,"effects":effects,
            "status":if label=="unknown" {"unresolved"} else {"classified"}}),
            )
            .is_some()
        {
            return Err("Duplicate sounding event anchor".into());
        }
    }
    if classified.len() != expected.len() {
        return Err("Missing sounding event anchors".into());
    }
    Ok(
        json!({"protocol":PROTOCOL,"scope":"visible_event","events":classified.into_values().collect::<Vec<_>>()}),
    )
}

/// Saved/editor layers may cover a subset; omitted or unresolved events keep
/// their original source effects. No chord owner is asserted by this object.
pub fn validate_overlay(measure: &Value, overlay: &Value) -> R<()> {
    if overlay["protocol"] != PROTOCOL || overlay["scope"] != "visible_event" {
        return Err("Unknown visual technique protocol or scope".into());
    }
    let expected = events(measure)?;
    let mut seen = BTreeSet::new();
    for item in array(overlay, "events")? {
        let key = (
            item["voice"].as_i64().ok_or("Invalid visual voice")?,
            item["onset"].as_i64().ok_or("Invalid visual onset")?,
        );
        if !expected.contains_key(&key) || !seen.insert(key) {
            return Err("Unknown or duplicate visual technique anchor".into());
        }
        if !["classified", "unresolved"].contains(&item["status"].as_str().unwrap_or("")) {
            return Err("Invalid visual technique status".into());
        }
        let effects = array(item, "effects")?;
        if effects
            .iter()
            .any(|e| !["pm", "let"].contains(&e.as_str().unwrap_or("")))
            || effects
                .iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>()
                .len()
                != effects.len()
        {
            return Err("Invalid visual technique effects".into());
        }
    }
    Ok(())
}

/// Retain only unchanged events when a client edits M without supplying a
/// replacement layer. Timing, notes and manual technique edits invalidate it.
pub fn retain_unchanged(old: &Value, new: &Value, overlay: &Value) -> R<Value> {
    validate_overlay(old, overlay)?;
    let old_events = events(old)?;
    let new_events = events(new)?;
    let mut result = overlay.clone();
    result["events"] = json!(array(overlay, "events")?
        .iter()
        .filter(|item| {
            let key = (
                item["voice"].as_i64().unwrap(),
                item["onset"].as_i64().unwrap(),
            );
            match (old_events.get(&key), new_events.get(&key)) {
                (Some(a), Some(b)) => {
                    a["duration"] == b["duration"]
                        && a["status"] == b["status"]
                        && a["notes"] == b["notes"]
                        && marks(a) == marks(b)
                }
                _ => false,
            }
        })
        .cloned()
        .collect::<Vec<_>>());
    Ok(result)
}

/// A visible event identifies coverage, not the owner of each chord tone.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Projection {
    pub events: usize,
    pub text_only_events: usize,
}

/// Project into a private codec/rendering copy. Existing note ownership stays
/// local to that note. A newly classified chord mark without an owner becomes
/// beat text; only a single-note event has an unambiguous new note owner.
pub fn project_measure(measure: &Value) -> R<(Value, Projection)> {
    let mut result = measure.clone();
    let Some(overlay) = measure.get("visual_techniques").filter(|v| !v.is_null()) else {
        return Ok((result, Projection::default()));
    };
    validate_overlay(measure, overlay)?;
    let mut summary = Projection::default();
    for item in array(overlay, "events")? {
        if item["status"] != "classified" {
            continue;
        }
        for voice in result["voices"].as_array_mut().unwrap() {
            if voice["voice"] != item["voice"] {
                continue;
            }
            for event in voice["events"].as_array_mut().unwrap() {
                if event["start"] != item["onset"] {
                    continue;
                }
                let desired = array(item, "effects")?;
                let mut event_effects = event["effects"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|e| **e != "pm" && **e != "let")
                    .cloned()
                    .collect::<Vec<_>>();
                let notes = event["notes"].as_array_mut().unwrap();
                let single = notes.len() == 1;
                for note in notes.iter_mut() {
                    let mut effects = note["effects"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|e| (**e != "pm" && **e != "let") || desired.contains(e))
                        .cloned()
                        .collect::<Vec<_>>();
                    if single {
                        for effect in desired {
                            if !effects.contains(effect) {
                                effects.push(effect.clone());
                            }
                        }
                    }
                    note["effects"] = json!(effects);
                }
                let unowned = desired
                    .iter()
                    .filter(|effect| {
                        !notes.iter().any(|note| {
                            note["effects"]
                                .as_array()
                                .is_some_and(|effects| effects.contains(effect))
                        })
                    })
                    .collect::<Vec<_>>();
                if !unowned.is_empty() {
                    summary.text_only_events += 1;
                    let label = unowned
                        .iter()
                        .map(|effect| if **effect == "pm" { "P.M." } else { "let ring" })
                        .collect::<Vec<_>>()
                        .join(" / ");
                    let text = json!(format!("text:{}", label.replace(' ', "%20")));
                    if !event_effects.contains(&text) {
                        event_effects.push(text);
                    }
                }
                event["effects"] = json!(event_effects);
                summary.events += 1;
            }
        }
    }
    Ok((result, summary))
}

pub fn project_score(score: &Value) -> R<(Value, Projection)> {
    let mut projected = score.clone();
    let mut summary = Projection::default();
    for part in projected["parts"]
        .as_array_mut()
        .ok_or("Missing score parts")?
    {
        for staff in part["staves"]
            .as_array_mut()
            .ok_or("Missing score staves")?
        {
            for measure in staff["measures"]
                .as_array_mut()
                .ok_or("Missing score measures")?
            {
                let (value, projection) = project_measure(measure)?;
                *measure = value;
                summary.events += projection.events;
                summary.text_only_events += projection.text_only_events;
            }
        }
    }
    Ok((projected, summary))
}
