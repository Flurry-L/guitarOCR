//! JSON Schema definitions and regular expression constraints for constrained VLM decoding
//! of single-measure musical observations (`Observation`).

use serde_json::{json, Value};

pub const LAYOUT_FIELDS: [&str; 3] = ["page", "source_page", "system_index"];
pub const NOTE_RELATIONS: [&str; 4] = ["tie", "hammer", "sl", "ss"];
pub(super) const NOTE_EFFECTS: &[&str] = &[
    "accent", "dead", "ghost", "grace", "hammer", "heavy", "sia", "sib", "sl", "sod", "sou", "ss",
    "stacc", "tap", "trem", "trill", "vib",
];
pub(super) const EVENT_EFFECTS: &[&str] = &[
    "fade",
    "pick_down",
    "pick_up",
    "rasg",
    "stroke_down",
    "stroke_up",
];
// Match the IR's musical ranges before decoding, so a generated effect cannot
// invalidate otherwise readable notes at the observation parser.
pub(super) const NOTE_EFFECT_PATTERN: &str = concat!(
    r"^(?:harm:[^\s,{}|#()<>\[\]]+|bend:[^\s:,{}|#()<>\[\]]+:-?[0-9]+|trem:(?:8|16|32)",
    r"|trill:(?:f(?:[0-9]|[12][0-9]|3[0-6])(?:p(?:[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7]))?|p(?:[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7]))(?::(?:16|32|64))?",
    r"|grace:(?:f(?:[0-9]|[12][0-9]|3[0-6])(?:p(?:[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7]))?|p(?:[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7]))(?::(?:8|16|32|64|128))?:(?:none|slide|bend|hammer):(?:0|1|-):(?:0|1)",
    r"|grace:x(?::(?:8|16|32|64|128))?:(?:none|slide|bend|hammer):(?:0|1|-):1)$",
);
pub(super) const EVENT_EFFECT_PATTERN: &str = concat!(
    r"^(?:slap:(?:none|tapping|slapping|popping)|tempo:[1-9][0-9]{0,2}",
    r"|dyn:(?:[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7])|(?:chord|text):[^\s,{}|#()<>\[\]]+)$",
);
pub(super) const NOTEHEADS: &[&str] = &[
    "normal",
    "x",
    "circle-x",
    "circle-slash",
    "diamond",
    "heavy-x",
    "heavy-x-hat",
    "triangle-up",
];
pub(super) const TIME_PATTERN: &str = r"^(?:[1-9]|[12][0-9]|3[0-2])/(?:1|2|4|8|16|32|64)$";

fn object(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object", "properties":properties, "required":required, "additionalProperties":false})
}

pub fn observation_schema(mode: &str, instrument: Option<&str>) -> Result<Value, String> {
    if !matches!(mode, "tab" | "notation" | "both") {
        return Err("Observation mode must be tab, notation or both".into());
    }
    let int = json!({"type":["integer","null"]});
    let effects = |names: &[&str], pattern: &str| {
        // llama.cpp accepts code point ranges, but not \s in a character
        // class. Spell out Unicode White_Space to retain the same constraint.
        let pattern = pattern.replace(
            r"\s",
            r"\x09-\x0D\x20\x85\xA0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000",
        );
        json!({"type":"array","items":{"anyOf":[{"enum":names},{"type":"string","pattern":pattern}]}})
    };
    let fact = |value| {
        object(
            json!({"status":{"enum":["observed","absent","unknown"]},"value":value}),
            &["status"],
        )
    };
    let state = object(
        json!({
            "time":fact(json!({"anyOf":[{"type":"string","pattern":TIME_PATTERN},{"type":"null"}]})),
            "key":fact(json!({"anyOf":[{"type":"integer","minimum":-7,"maximum":7},{"type":"null"}]})),
            "clef":fact(json!({"enum":["G2","F4","C3","C4","percussion","tab",null]})),
            "clef_octave":fact(json!({"enum":[-24,-12,0,12,24,null]}))
        }),
        &[],
    );
    let tab_note = object(
        json!({"string":int,"fret":{"anyOf":[{"type":"integer"},{"enum":["x",null]}]},"effects":effects(NOTE_EFFECTS, NOTE_EFFECT_PATTERN)}),
        &[],
    );
    // llama.cpp emits required fields before optional ones. Keep the required
    // percussion prefix in the same order as the training answers.
    let notation_note = object(
        json!({"staff_position":int,"accidental":{"anyOf":[{"type":"integer"},{"enum":["unknown",null]}]},"notehead":{"anyOf":[{"enum":NOTEHEADS},{"type":"null"}]},"effects":effects(NOTE_EFFECTS, NOTE_EFFECT_PATTERN)}),
        if mode == "notation" && instrument == Some("drums") {
            &["staff_position", "accidental", "notehead"]
        } else {
            &[]
        },
    );
    let note = if mode == "notation" {
        &notation_note
    } else {
        &tab_note
    };
    let notes = json!({"type":"array","items":note});
    let mut fields = json!({"start":int,"duration":{"type":["string","null"]},
        "status":{"type":["string","null"]},"notes":if mode == "both" {notes.clone()} else {json!({"anyOf":[notes,{"type":"null"}]})}});
    let mut required = vec!["start", "duration", "status", "notes"];
    if mode == "both" {
        fields["notation_notes"] = json!({"type":"array","items":notation_note});
        required.push("notation_notes");
    }
    fields["effects"] = effects(EVENT_EFFECTS, EVENT_EFFECT_PATTERN);
    let event = object(fields, &required);
    let voice = object(
        json!({"voice":{"type":"integer"},"events":{"type":"array","items":event}}),
        &["voice", "events"],
    );
    let relation = |note_endpoints: bool| {
        let endpoint = json!({"anyOf":[{"type":"array","items":{"type":"integer"},
            "minItems":2,"maxItems":if note_endpoints {3} else {2}},{"enum":["left","right",null]}]});
        let mut fields = json!({"kind":{"enum":["tie","hammer","sl","ss","pm","let","ottava",null]},
            "start":endpoint,"end":endpoint,"shape":{"enum":["curve","diagonal","extension"]},
            "value":int,"voice":int,"lane":{"type":["number","null"]}});
        if mode == "both" {
            fields["representation"] = json!({"enum":["tab","notation"]});
            if note_endpoints {
                // Keep optional visual features before the required staff
                // reference, matching training and both decoding engines.
                let optional = ["shape", "value", "voice", "lane"];
                let variants: Vec<Value> = (0..1 << optional.len())
                    .map(|mask| {
                        let properties: serde_json::Map<String, Value> = fields
                            .as_object()
                            .unwrap()
                            .iter()
                            .filter(|(name, _)| {
                                optional
                                    .iter()
                                    .position(|key| key == name)
                                    .is_none_or(|i| mask & (1 << i) != 0)
                            })
                            .map(|(name, value)| (name.clone(), value.clone()))
                            .collect();
                        let required: Vec<&str> = properties.keys().map(String::as_str).collect();
                        object(json!(properties), &required)
                    })
                    .collect();
                return json!({"anyOf":variants});
            }
        }
        object(fields, &["kind", "start", "end"])
    };
    let relation = if mode == "both" {
        json!({"anyOf":[relation(false),relation(true)]})
    } else {
        relation(true)
    };
    Ok(object(
        json!({"state":state,"voices":{"type":"array","items":voice},"relations":{"type":"array","items":relation}}),
        &["state", "voices", "relations"],
    ))
}
