//! Written pitch spelling and visible accidentals belong to the notation codec.
//! MusicXML `alter` controls sound; renderers also need explicit `accidental`.
use super::{arr, effects, num, pitch, tag, R};
use crate::score::duration_ticks;
use serde_json::Value;
use std::collections::HashMap;

const NATURAL: [i64; 7] = [0, 2, 4, 5, 7, 9, 11];
const STEPS: [&str; 7] = ["C", "D", "E", "F", "G", "A", "B"];

#[derive(Clone, Copy)]
pub(super) struct WrittenPitch {
    step: usize,
    alter: i64,
    octave: i64,
}
impl WrittenPitch {
    pub(super) fn xml(self) -> R<String> {
        Ok(format!(
            "<pitch>{}{}{}</pitch>",
            tag("step", STEPS[self.step])?,
            if self.alter == 0 {
                String::new()
            } else {
                tag("alter", self.alter)?
            },
            tag("octave", self.octave)?
        ))
    }
    fn accidental(self) -> &'static str {
        match self.alter {
            -2 => "flat-flat",
            -1 => "flat",
            0 => "natural",
            1 => "sharp",
            2 => "double-sharp",
            _ => unreachable!("spelling is bounded by a double accidental"),
        }
    }
}

fn key_alterations(fifths: i64) -> [i64; 7] {
    let mut alterations = [0; 7];
    let order = if fifths < 0 {
        [6, 2, 5, 1, 4, 0, 3]
    } else {
        [3, 0, 4, 1, 5, 2, 6]
    };
    for i in 0..fifths.unsigned_abs() as usize {
        alterations[order[i % 7]] += fifths.signum();
    }
    alterations
}

fn spell(p: i64, fifths: i64, mode: &str, key: &[i64; 7]) -> WrittenPitch {
    // Prefer the diatonic spelling, including E#/B# and Cb/Fb in remote keys.
    let (step, alter) = (0..7)
        .find_map(|step| {
            ((NATURAL[step] + key[step]).rem_euclid(12) == p.rem_euclid(12))
                .then_some((step, key[step]))
        })
        .or_else(|| {
            // The IR stores MIDI pitches, not author-selected enharmonic spelling.
            // Spell a minor key's raised leading tone on its seventh degree.
            let tonic = (7 * fifths + 9).rem_euclid(12);
            (mode == "minor")
                .then(|| {
                    (0..7).find_map(|step| {
                        ((NATURAL[step] + key[step]).rem_euclid(12) == (tonic - 2).rem_euclid(12)
                            && p.rem_euclid(12) == (tonic - 1).rem_euclid(12))
                        .then_some((step, key[step] + 1))
                    })
                })
                .flatten()
        })
        .unwrap_or_else(|| {
            let sharp = [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (3, 0),
                (3, 1),
                (4, 0),
                (4, 1),
                (5, 0),
                (5, 1),
                (6, 0),
            ];
            let flat = [
                (0, 0),
                (1, -1),
                (1, 0),
                (2, -1),
                (2, 0),
                (3, 0),
                (4, -1),
                (4, 0),
                (5, -1),
                (5, 0),
                (6, -1),
                (6, 0),
            ];
            (if fifths < 0 { flat } else { sharp })[p.rem_euclid(12) as usize]
        });
    WrittenPitch {
        step,
        alter,
        octave: (p - NATURAL[step] - alter).div_euclid(12) - 1,
    }
}

pub(super) struct PlannedNote {
    pub pitch: WrittenPitch,
    pub accidental: Option<&'static str>,
}

#[derive(Clone, Default)]
pub(super) struct StaffPitchWriter {
    // Previous event in each voice, for ties across barlines and key changes.
    previous: HashMap<(i64, i64), WrittenPitch>,
}

impl StaffPitchWriter {
    pub(super) fn measure(
        &mut self,
        row: &Value,
        part: &Value,
        signature: (i64, &str),
        transpose: i64,
        scale: i128,
    ) -> R<HashMap<(i64, usize, usize), PlannedNote>> {
        let (fifths, mode) = signature;
        let key = key_alterations(fifths);
        let mut events = Vec::new();
        for voice in arr(row, "voices")? {
            let vi = num(voice, "voice", 0);
            let mut cursor = 0;
            if arr(voice, "events")?.is_empty() {
                self.previous.retain(|&(voice, _), _| voice != vi);
            }
            for (ei, event) in arr(voice, "events")?.iter().enumerate() {
                let start = num(event, "start", (cursor / scale) as i64) as i128 * scale;
                let duration = duration_ticks(&event["duration"])?;
                cursor = start + duration.numerator * scale / duration.denominator;
                events.push((start, vi, ei, event));
            }
        }
        events.sort_by_key(|&(start, vi, ei, _)| (start, vi, ei));
        let mut result = HashMap::new();
        let mut active = HashMap::<(usize, i64), Option<i64>>::new();
        let mut start = 0;
        while start < events.len() {
            let end = start + events[start..].partition_point(|e| e.0 == events[start].0);
            let mut changes = HashMap::<(usize, i64), Option<i64>>::new();
            for &(_, vi, ei, event) in &events[start..end] {
                let mut previous = Vec::new();
                for (ni, note) in arr(event, "notes")?.iter().enumerate() {
                    let sounding = pitch(note, part)?;
                    let tie_from = effects(note)
                        .contains(&"tie")
                        .then(|| self.previous.get(&(vi, sounding)))
                        .flatten()
                        .copied();
                    let written =
                        tie_from.unwrap_or_else(|| spell(sounding - transpose, fifths, mode, &key));
                    let at = (written.step, written.octave);
                    let expected = active.get(&at).copied().unwrap_or(Some(key[written.step]));
                    let accidental = (tie_from.is_none() && expected != Some(written.alter))
                        .then(|| written.accidental());
                    result.insert(
                        (vi, ei, ni),
                        PlannedNote {
                            pitch: written,
                            accidental,
                        },
                    );
                    previous.push(((vi, sounding), written));
                    // A continuation over a barline doesn't establish a new accidental.
                    if tie_from.is_none() {
                        changes
                            .entry(at)
                            .and_modify(|value| {
                                if *value != Some(written.alter) {
                                    *value = None;
                                }
                            })
                            .or_insert(Some(written.alter));
                    }
                }
                self.previous.retain(|&(voice, _), _| voice != vi);
                self.previous.extend(previous);
            }
            // Simultaneous chromatic unisons must both be explicit. Their following
            // occurrence also needs a sign; neither voice wins by serialization order.
            for &(_, vi, ei, event) in &events[start..end] {
                for ni in 0..arr(event, "notes")?.len() {
                    let note = result.get_mut(&(vi, ei, ni)).unwrap();
                    if changes.get(&(note.pitch.step, note.pitch.octave)) == Some(&None) {
                        note.accidental = Some(note.pitch.accidental());
                    }
                }
            }
            active.extend(changes);
            start = end;
        }
        Ok(result)
    }
}
