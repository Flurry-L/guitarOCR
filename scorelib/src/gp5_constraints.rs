//! GP5 projection restrictions, deliberately separate from legal score IR.
use crate::score::{duration_ticks, parse_measure_target};
use serde_json::Value;
use std::collections::HashSet;

pub fn timing_errors(target: &str) -> Result<Vec<String>, String> {
    let score = parse_measure_target(target)?;
    let mut errors = Vec::new();
    for voice in score["voices"].as_array().ok_or("Missing voices")? {
        let mut end = 0i128;
        for (index, event) in voice["events"]
            .as_array()
            .ok_or("Missing events")?
            .iter()
            .enumerate()
        {
            let start = event["start"].as_i64().ok_or("Invalid event start")? as i128;
            if start < end {
                errors.push(format!(
                    "声部 {} 的第 {} 个事件与前一个重叠",
                    voice["voice"].as_i64().unwrap_or(0) + 1,
                    index + 1
                ));
            }
            let duration = duration_ticks(&event["duration"])?;
            // The legacy GP5 writer explicitly truncates each event duration.
            end = start + duration.numerator / duration.denominator;
        }
    }
    Ok(errors)
}
fn pitches(note: &Value) -> Result<Vec<i64>, String> {
    let base = note["pitch"]
        .as_i64()
        .ok_or("Notation note is missing pitch")?;
    let mut result = vec![base];
    for effect in note["effects"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if effect == "trill" {
            result.push(base + 1);
        } else if effect.starts_with("grace:") || effect.starts_with("trill:") {
            if let Some(position) = effect.split(':').nth(1) {
                if let Some(index) = position.find('p') {
                    result.push(
                        position[index + 1..]
                            .parse()
                            .map_err(|_| "Invalid ornament pitch")?,
                    );
                }
            }
        }
    }
    Ok(result)
}
fn assign(
    index: usize,
    candidates: &[Vec<usize>],
    owners: &mut [Option<usize>],
    visited: &mut HashSet<usize>,
) -> bool {
    for &string in &candidates[index] {
        if !visited.insert(string) {
            continue;
        }
        if owners[string].is_none() || assign(owners[string].unwrap(), candidates, owners, visited)
        {
            owners[string] = Some(index);
            return true;
        }
    }
    false
}
pub fn notation_fingering_errors(score: &Value, tuning: &[i64]) -> Result<Vec<String>, String> {
    let mut errors = Vec::new();
    for voice in score["voices"].as_array().ok_or("Missing voices")? {
        for event in voice["events"].as_array().ok_or("Missing events")? {
            let Some(notes) = event["notes"].as_array() else {
                continue;
            };
            let mut candidates = Vec::new();
            let mut invalid = false;
            for note in notes {
                let required = pitches(note)?;
                let dead = note["effects"]
                    .as_array()
                    .is_some_and(|es| es.iter().any(|e| e == "dead"));
                let strings: Vec<_> = tuning
                    .iter()
                    .enumerate()
                    .filter(|(_, open)| {
                        dead || required.iter().all(|p| (0..=30).contains(&(p - **open)))
                    })
                    .map(|(i, _)| i)
                    .collect();
                if strings.is_empty() {
                    errors.push(format!(
                        "V{} at {}: pitch {} or its ornament cannot be played with tuning {:?}",
                        voice["voice"], event["start"], note["pitch"], tuning
                    ));
                    invalid = true;
                }
                candidates.push(strings);
            }
            if invalid {
                continue;
            }
            let mut owners = vec![None; tuning.len()];
            if !(0..notes.len())
                .all(|index| assign(index, &candidates, &mut owners, &mut HashSet::new()))
            {
                errors.push(format!(
                    "V{} at {}: chord cannot fit distinct strings with tuning {:?}",
                    voice["voice"], event["start"], tuning
                ));
            }
        }
    }
    Ok(errors)
}
