//! Rust types for single-measure observations and their musical state.
//! `Observation::from_value` reads the JSON accepted by the shared observation parser.
//! Additional JSON fields are retained in `extra` maps.
//! The assembly and export interfaces continue to use `serde_json::Value`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Whether a musical state was read, omitted, unresolved, or inherited from an earlier measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateStatus {
    /// Printed symbol is explicitly visible in this measure crop.
    Observed,
    /// Symbol is confirmed absent in this crop (inherits from preceding measure).
    Absent,
    /// Cropped or illegible; cannot determine without upstream context or review.
    Unknown,
    /// Populated during cross-measure assembly from an earlier measure.
    Inherited,
}

/// One musical state value, with its source and interpretation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateFact<T> {
    pub status: StateStatus,
    #[serde(default)]
    pub value: Option<T>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

impl<T> StateFact<T> {
    pub fn unknown() -> Self {
        Self {
            status: StateStatus::Unknown,
            value: None,
            source: None,
            reason: None,
            extra: Map::new(),
        }
    }

    pub fn is_known(&self) -> bool {
        matches!(self.status, StateStatus::Observed | StateStatus::Inherited)
            && self.value.is_some()
    }
}

/// Measure-level musical context observed in or inherited by a measure crop.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<StateFact<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<StateFact<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clef: Option<StateFact<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clef_octave: Option<StateFact<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tuning: Option<StateFact<Vec<i64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capo: Option<StateFact<i64>>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// The printed TAB dead-note marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadNote {
    #[serde(rename = "x")]
    X,
}

/// A visible accidental whose shape is unresolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownAccidental {
    #[serde(rename = "unknown")]
    Unknown,
}

/// The side of a crop crossed by a relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CropEdge {
    Left,
    Right,
}

/// Fret value on a TAB staff: integer fret number (`0..=36`) or dead-note `"x"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FretValue {
    Fret(i64),
    DeadNote(DeadNote),
}

/// Printed accidental on a standard notation staff: semitone offset (`-2..=2`) or `"unknown"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AccidentalValue {
    Semitones(i64),
    Unknown(UnknownAccidental),
}

/// Single note head observed on a tablature staff (`mode = "tab"` or `"both"`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabNoteObservation {
    /// 1-indexed string number (`1` = highest-pitched string).
    #[serde(default)]
    pub string: Option<i64>,
    /// Fret number or `"x"` for dead/muted note.
    #[serde(default)]
    pub fret: Option<FretValue>,
    /// Local note-level articulations (e.g. `"vib"`, `"harm:natural"`, `"bend:..."`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// Single note head observed on a standard notation or percussion staff.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotationNoteObservation {
    /// Diatonic step relative to the bottom staff line (`0` = line 1, `1` = space 1, `2` = line 2, etc.).
    #[serde(default)]
    pub staff_position: Option<i64>,
    /// Printed accidental (`-2..=2`, `"unknown"`, or `None` when no accidental is printed).
    #[serde(default)]
    pub accidental: Option<AccidentalValue>,
    /// Visible notehead shape for percussion/harmonic notation (`"normal"`, `"x"`, `"diamond"`, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notehead: Option<String>,
    /// Local note-level articulations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// A rhythmic onset (note, chord, or rest) within a local voice.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventObservation {
    /// Onset position in ticks from the start of the measure (`960` ticks = 1 quarter note).
    #[serde(default)]
    pub start: Option<i64>,
    /// Printed duration token (`"w"`, `"h"`, `"q"`, `"e"`, `"s"`, `"t"`, `"f"`, with optional `"."` or `"[p:q]"`).
    #[serde(default)]
    pub duration: Option<String>,
    /// `"normal"`, `"rest"`, or `None` if unresolved.
    #[serde(default)]
    pub status: Option<String>,
    /// Primary notes array (TAB notes in `"tab"`/`"both"` mode, notation notes in `"notation"` mode).
    #[serde(default)]
    pub notes: Option<Vec<Value>>,
    /// Separate standard-notation notes when `mode == "both"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notation_notes: Option<Vec<NotationNoteObservation>>,
    /// Beat-level effects (`"pick_down"`, `"stroke_up"`, `"dyn:..."`, `"chord:..."`, `"tempo:..."`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// A polyphonic voice stream within a single measure observation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceObservation {
    /// 0-indexed local voice identifier (`0` = primary voice, `1` = secondary voice).
    pub voice: i64,
    pub events: Vec<EventObservation>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// Endpoint of a visual relation (slur, tie, slide, palm mute, let ring, or ottava).
/// Either an intra-measure index path `[voice, event]` / `[voice, event, note]`
/// or a crop boundary edge (`"left"` / `"right"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RelationEndpoint {
    IndexPath(Vec<i64>),
    CropEdge(CropEdge),
}

/// Visual span or note-to-note connection observed in a measure crop.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RelationObservation {
    /// `"tie"`, `"hammer"`, `"sl"`, `"ss"`, `"pm"`, `"let"`, `"ottava"`, or `None`.
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub start: Option<RelationEndpoint>,
    #[serde(default)]
    pub end: Option<RelationEndpoint>,
    /// `"curve"`, `"diagonal"`, or `"extension"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<f64>,
    /// `"tab"` or `"notation"` in dual-staff (`"both"`) mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation: Option<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// Top-level structured output of the single-measure VLM reader before cross-bar assembly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub state: ObservationState,
    pub voices: Vec<VoiceObservation>,
    pub relations: Vec<RelationObservation>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

impl Observation {
    /// Parse and validate an `Observation` from a `serde_json::Value`.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let validated = crate::observation::parse_observation(value)?;
        serde_json::from_value(validated)
            .map_err(|e| format!("Failed to deserialize typed Observation: {e}"))
    }

    /// Convert this typed `Observation` back into a `serde_json::Value` for assembly.
    pub fn to_value(&self) -> Result<Value, String> {
        let value = serde_json::to_value(self)
            .map_err(|e| format!("Failed to serialize typed Observation: {e}"))?;
        crate::observation::parse_observation(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dual_observation_keeps_notes_relations_and_extra_fields() {
        let value = json!({
            "state": {"time": {"status": "observed", "value": "4/4", "confidence": 0.9}},
            "voices": [{"voice": 0, "events": [{
                "start": 0, "duration": "q", "status": "normal",
                "notes": [{"string": 1, "fret": null}],
                "notation_notes": [{"staff_position": 10, "accidental": null, "label": "A"}]
            }]}],
            "relations": [{"kind": "tie", "start": "left", "end": [0, 0, 0], "representation": "notation"}],
            "source_id": "example"
        });
        let observation = Observation::from_value(&value).unwrap();
        assert!(observation.state.time.as_ref().unwrap().is_known());
        assert_eq!(
            observation.relations[0].start,
            Some(RelationEndpoint::CropEdge(CropEdge::Left))
        );
        let serialized = observation.to_value().unwrap();
        assert_eq!(serialized["state"]["time"]["confidence"], 0.9);
        assert_eq!(
            serialized["voices"][0]["events"][0]["notation_notes"][0]["label"],
            "A"
        );
        assert_eq!(serialized["source_id"], "example");
        assert!(serialized["voices"][0]["events"][0]["notes"][0]["fret"].is_null());
        assert_eq!(Observation::from_value(&serialized).unwrap(), observation);
    }

    #[test]
    fn serialization_uses_the_shared_observation_rules() {
        let mut observation = Observation::from_value(&json!({
            "state": {}, "voices": [{"voice": 0, "events": [{
                "start": 0, "duration": "q", "status": "normal",
                "notes": [{"string": 1, "fret": 0}]
            }]}], "relations": []
        }))
        .unwrap();
        observation.voices[0].events[0].effects.push("tie".into());
        assert!(observation.to_value().is_err());
        assert!(serde_json::from_value::<FretValue>(json!("muted")).is_err());
        assert_eq!(
            serde_json::from_value::<FretValue>(json!("x")).unwrap(),
            FretValue::DeadNote(DeadNote::X)
        );
    }
}
