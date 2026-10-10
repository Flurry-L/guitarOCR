//! Preserve visual notes and record tie relations instead of nearest-pitch edits.
use super::*;

#[derive(Clone)]
struct Previous {
    notes: Vec<Value>,
    measure: Value,
    end_numerator: i128,
    denominator: i128,
    at_bar_end: bool,
}

pub fn resolve(records: &mut [Value]) -> Result<()> {
    let mut last: HashMap<String, HashMap<i64, Previous>> = HashMap::new();
    let mut indices = HashMap::new();
    for row in records {
        let lane = staff_key(row);
        let index = bar_index(row);
        if indices
            .get(&lane)
            .is_some_and(|previous| index != previous + 1)
        {
            last.remove(&lane);
        }
        indices.insert(lane.clone(), index);
        let previous_voices = last.entry(lane).or_default();
        let measure = score::parse_measure_target(text(row, "target", ""))?;
        let voices: HashSet<i64> = list(&measure, "voices")
            .iter()
            .filter_map(|v| v["voice"].as_i64())
            .collect();
        previous_voices.retain(|voice, _| voices.contains(voice));
        let mut relations = Vec::new();
        let mut conflicts = Vec::new();
        let state = super::signature_state::trusted_state(row);
        let ticks = state["time"]
            .as_str()
            .map(time_signature)
            .transpose()?
            .map(|(n, d)| 3840 * n / d);
        for voice in list(&measure, "voices") {
            let voice_id = voice["voice"].as_i64().ok_or("Missing voice ID")?;
            let mut previous = previous_voices.get(&voice_id).cloned();
            for event in list(voice, "events") {
                if matches!(text(event, "status", ""), "rest" | "empty") {
                    previous = None;
                    continue;
                }
                let start = event["start"].as_i64().ok_or("Missing event onset")?;
                for note in list(event, "notes") {
                    if flag(row, "manually_edited")
                        || !list(note, "effects").iter().any(|effect| effect == "tie")
                    {
                        continue;
                    }
                    let matches = previous.as_ref().map_or(0, |p| {
                        p.notes
                            .iter()
                            .filter(|n| {
                                let identity = if !note["string"].is_null() {
                                    n["string"] == note["string"]
                                } else {
                                    n["pitch"] == note["pitch"]
                                };
                                identity
                                    && ["pitch", "fret"].iter().all(|field| {
                                        note[*field].is_null()
                                            || n[*field].is_null()
                                            || note[*field] == n[*field]
                                    })
                            })
                            .count()
                    });
                    let contiguous = previous.as_ref().is_some_and(|p| {
                        if p.measure == row["measure_number"] {
                            p.end_numerator == i128::from(start) * p.denominator
                        } else {
                            start == 0 && p.at_bar_end
                        }
                    });
                    let mut relation = json!({"voice":voice_id,"onset":start,"note":note,"source":"visual_ocr",
                        "status":if matches == 1 && contiguous {"confirmed"} else {"conflict"},
                        "previous_measure_number":previous.as_ref().map(|p|p.measure.clone())});
                    if relation["status"] == "conflict" {
                        relation["reason"] =
                            json!("Tie endpoints disagree or are not temporally continuous");
                        relation["previous_notes"] =
                            json!(previous.as_ref().map_or(vec![], |p| p.notes.clone()));
                        conflicts.push(relation.clone());
                    }
                    relations.push(relation);
                }
                let duration = score::duration_ticks(&event["duration"])?;
                let end_numerator = i128::from(start) * duration.denominator + duration.numerator;
                previous = Some(Previous {
                    notes: list(event, "notes").to_vec(),
                    measure: row["measure_number"].clone(),
                    end_numerator,
                    denominator: duration.denominator,
                    at_bar_end: ticks
                        .is_some_and(|t| end_numerator == i128::from(t) * duration.denominator),
                });
            }
            if let Some(previous) = previous {
                previous_voices.insert(voice_id, previous);
            } else {
                previous_voices.remove(&voice_id);
            }
        }
        // A user-confirmed row is authoritative; never re-open its review state.
        if flag(row, "manually_edited") {
            continue;
        }
        row["tie_relations"] = json!(relations);
        row.as_object_mut()
            .ok_or("Invalid measure")?
            .remove("tie_conflicts");
        if !conflicts.is_empty() {
            row["tie_conflicts"] = json!(conflicts);
            row["needs_review"] = json!(true);
            let message = json!(
                "Visual tie relation conflicts with note identity or timing; notes were preserved"
            );
            let mut reasons = list(row, "fallback_reason").to_vec();
            if !reasons.contains(&message) {
                reasons.push(message);
            }
            row["fallback_reason"] = json!(reasons);
        }
    }
    Ok(())
}
