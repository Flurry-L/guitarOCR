//! GP5 stores diagrams on beats, not in a standalone document dictionary.
//! Unused printed diagrams are therefore carried by silent reference tracks;
//! rest beats preserve diagram identity without changing the musical timeline.
use super::*;
use crate::chords::{self, Diagram};
use std::collections::HashSet;

pub(super) struct Library {
    pub part: Value,
    pub diagrams: Vec<(String, Diagram)>,
}
fn key(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace('♭', "b")
        .replace('♯', "#")
}

pub(super) fn unused(score: &Value) -> R<(Vec<Library>, Vec<Value>)> {
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
    let mut libraries: Vec<Library> = vec![];
    let mut pending = vec![];
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
        let confirmed = decision == Some("confirmed");
        let is_chord = matches!(txt(parsed, "kind", ""), "chord" | "chord_diagram");
        let part_id = annotation["part_id"]
            .as_str()
            .or_else(|| (parts.len() == 1).then(|| txt(&parts[0], "id", "part-1")));
        let name = txt(parsed, "text", "").to_owned();
        if confirmed && name.trim().is_empty() {
            return Err("Confirmed chord library entry is missing its name".into());
        }
        if chords::pending(annotation) {
            // original_diagram is an OCR candidate, not a review decision.
            // A user-selected diagram already on a musical beat still exports;
            // only unused, unresolved library candidates are withheld here.
            // Evidence-only and rejected candidates have no parsed diagram, but
            // must still appear in the export review report.
            let on_beat = part_id.is_some_and(|id| {
                Diagram::from_annotation(&parsed["diagram"])
                    .ok()
                    .is_some_and(|diagram| used.contains(&(id.to_owned(), key(&name), diagram)))
            });
            if !on_beat {
                pending.push(
                    json!({"annotation_index":index,"part_id":part_id,"name":name,
                    "reason":"unreviewed_chord_candidate"}),
                );
            }
            continue;
        }
        if !is_chord || parsed["diagram"].is_null() {
            if confirmed {
                return Err("Confirmed chord library entry is missing its name or diagram".into());
            }
            continue;
        }
        let part_id = part_id
            .ok_or("Printed chord diagram is missing its part; assign it before GP5 export")?;
        let part = parts
            .iter()
            .find(|p| txt(p, "id", "part-1") == part_id)
            .ok_or("Printed chord diagram refers to an unknown part")?;
        let diagram = Diagram::from_annotation(&parsed["diagram"])?;
        if !used.insert((part_id.to_owned(), key(&name), diagram.clone())) {
            continue;
        }
        // Separate string counts keep diagrams faithful even when the OCR's
        // inferred instrument/tuning differs from the printed reference grid.
        if let Some(library) = libraries.iter_mut().find(|l| {
            l.part["id"] == part["id"] && l.diagrams[0].1.frets.len() == diagram.frets.len()
        }) {
            library.diagrams.push((name, diagram));
        } else {
            libraries.push(Library {
                part: part.clone(),
                diagrams: vec![(name, diagram)],
            });
        }
    }
    Ok((libraries, pending))
}

pub(super) fn annotation_effects(name: &str, diagram: &Diagram) -> Value {
    // Chord names become GP5 text directly; escape M separators for the common
    // event-effect reader without changing Unicode or chord spelling.
    let encoded = name
        .replace('%', "%25")
        .replace(':', "%3A")
        .replace(',', "%2C");
    json!([format!("chord:{encoded}"), diagram.effect()])
}
