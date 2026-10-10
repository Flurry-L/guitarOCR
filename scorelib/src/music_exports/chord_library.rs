//! GP5 stores diagrams on beats, not in a standalone document dictionary.
//! Report unused printed diagrams; they remain in the project annotation library.
use super::*;
use crate::chords::{self, Diagram};
use std::collections::HashSet;

fn key(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace('♭', "b")
        .replace('♯', "#")
}

pub(super) fn omitted(score: &Value) -> R<Vec<Value>> {
    let parts = arr(score, "parts")?;
    let mut used = HashSet::new();
    for part in parts {
        for staff in arr(part, "staves")? {
            for measure in arr(staff, "measures")? {
                for voice in arr(measure, "voices")? {
                    for event in arr(voice, "events")? {
                        let es = effects(event);
                        let name = es
                            .iter()
                            .find_map(|e| e.strip_prefix("chord:"))
                            .map(percent_decode)
                            .transpose()?
                            .unwrap_or_default();
                        for diagram in es.iter().filter_map(|e| e.strip_prefix("diagram:")) {
                            used.insert((
                                txt(part, "id", "part-1").to_owned(),
                                key(&name),
                                chords::parse_diagram(diagram)?,
                            ));
                        }
                    }
                }
            }
        }
    }
    let mut omitted = vec![];
    for (index, annotation) in score["annotations"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let parsed = &annotation["parsed"];
        let decision = chords::decision(annotation);
        if decision == Some("rejected") {
            continue;
        }
        let is_chord = matches!(txt(parsed, "kind", ""), "chord" | "chord_diagram");
        let part_id = annotation["part_id"]
            .as_str()
            .or_else(|| (parts.len() == 1).then(|| txt(&parts[0], "id", "part-1")));
        let name = txt(parsed, "text", "").to_owned();
        let pending = chords::pending(annotation);
        if !pending && (!is_chord || parsed["diagram"].is_null()) {
            continue;
        }
        let on_beat = part_id.is_some_and(|id| {
            Diagram::from_annotation(&parsed["diagram"])
                .ok()
                .is_some_and(|diagram| used.contains(&(id.to_owned(), key(&name), diagram)))
        });
        if on_beat {
            continue;
        }
        omitted.push(
            json!({"annotation_index":index,"part_id":part_id,"name":name,
            "diagram":parsed["diagram"],"reason":if pending {
                "unreviewed_chord_candidate"
            } else {
                "not_attached_to_beat"
            }}),
        );
    }
    Ok(omitted)
}
