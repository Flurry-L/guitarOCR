//! Single-crop visible evidence and conservative assembly across adjacent bars.
//! Written pitch is the common contract; unseen values remain null until an
//! explicit musical relation establishes them. Research and product share this code.
use crate::score;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

pub const LAYOUT_FIELDS: [&str; 3] = ["page", "source_page", "system_index"];
pub const NOTE_RELATIONS: [&str; 4] = ["tie", "hammer", "sl", "ss"];
const NOTE_EFFECTS: &[&str] = &[
    "accent", "dead", "ghost", "grace", "hammer", "heavy", "sia", "sib", "sl", "sod", "sou", "ss",
    "stacc", "tap", "trem", "trill", "vib",
];
const EVENT_EFFECTS: &[&str] = &[
    "fade",
    "pick_down",
    "pick_up",
    "rasg",
    "stroke_down",
    "stroke_up",
];
const NOTE_EFFECT_PATTERN: &str = r"^(?:harm:[^\s,{}|#]+|bend:[^\s:,{}|#]+:-?[0-9]+|trem:(?:8|16|32)|trill:(?:f[0-9]+(?:p[0-9]+)?|p[0-9]+)(?::(?:16|32|64))?|grace:(?:x|f[0-9]+(?:p[0-9]+)?|p[0-9]+)(?::(?:8|16|32|64|128))?:(?:none|slide|bend|hammer):(?:0|1|-):(?:0|1))$";
const EVENT_EFFECT_PATTERN: &str =
    r"^(?:slap:(?:none|tapping|slapping|popping)|(?:tempo|dyn):[0-9]+|(?:chord|text):[^\s,{}|#]+)$";
fn array(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn integer(v: &Value) -> bool {
    v.as_i64().is_some()
}
fn nullable_integer(v: &Value) -> bool {
    v.is_null() || integer(v)
}
fn object(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object", "properties":properties, "required":required, "additionalProperties":false})
}
pub fn range_relation(r: &Value) -> bool {
    matches!(r["kind"].as_str(), Some("pm" | "let" | "ottava"))
        || (r["kind"].is_null()
            && !matches!(r["shape"].as_str(), Some("curve" | "diagonal"))
            && ["start", "end"]
                .iter()
                .all(|k| !r[k].is_array() || array(&r[k]).len() == 2))
}
const NOTEHEADS: &[&str] = &[
    "normal",
    "x",
    "circle-x",
    "circle-slash",
    "diamond",
    "heavy-x",
    "heavy-x-hat",
    "triangle-up",
];

// Conventional drum-kit positions (bottom line = 0). Different printed
// symbols at one position are distinct evidence, e.g. ride versus ride bell.
fn percussion_pitch(position: i64, shape: &str) -> Option<i64> {
    match (position, shape) {
        (0, "normal") => Some(35),
        (1, "normal") => Some(36),
        (5, "normal" | "diamond") => Some(38),
        (5, "x") => Some(37),
        (2, "normal") => Some(43),
        (3, "normal") => Some(45),
        (4, "normal") => Some(47),
        (6, "normal") => Some(48),
        (7, "normal") => Some(50),
        (9, "x") => Some(42),
        (-1, "x") => Some(44),
        (9, "circle-x" | "circle-slash") => Some(46),
        (8, "x") => Some(51),
        (8, "diamond") => Some(53),
        (10, "x") => Some(55),
        (10, "heavy-x") => Some(49),
        (9, "heavy-x") => Some(57),
        (11, "heavy-x-hat") => Some(52),
        (7..=9, "triangle-up") => Some(56),
        (5, "triangle-up") => Some(54),
        _ => None,
    }
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
pub fn parse_observation(value: &Value) -> Result<Value, String> {
    let o = if let Some(text) = value.as_str() {
        serde_json::from_str(text).map_err(|e| format!("Invalid observation JSON: {e}"))?
    } else {
        value.clone()
    };
    if !o.is_object() {
        return Err("Observation must be a JSON object".into());
    }
    for (field, valid, kind) in [
        ("voices", o["voices"].is_array(), "list"),
        ("state", o["state"].is_object(), "dict"),
        ("relations", o["relations"].is_array(), "list"),
    ] {
        if !valid {
            return Err(format!("Observation {field} must be {kind}"));
        }
    }
    static EFFECT_REGEX: OnceLock<[Regex; 2]> = OnceLock::new();
    let regex = EFFECT_REGEX.get_or_init(|| {
        [
            Regex::new(EVENT_EFFECT_PATTERN).unwrap(),
            Regex::new(NOTE_EFFECT_PATTERN).unwrap(),
        ]
    });
    let mut seen = HashSet::new();
    for voice in array(&o["voices"]) {
        if !voice.is_object()
            || !integer(&voice["voice"])
            || !seen.insert(voice["voice"].as_i64().unwrap())
        {
            return Err("Local voices need distinct integer identifiers".into());
        }
        if !voice["events"].is_array() {
            return Err("Voice events must be an array".into());
        }
        for event in array(&voice["events"]) {
            if !event.is_object() {
                return Err("Events must be objects".into());
            }
            if !event["notes"].is_null() && !event["notes"].is_array() {
                return Err("Event notes must be an array".into());
            }
            if event
                .get("notation_notes")
                .is_some_and(|notes| !notes.is_array())
            {
                return Err("Event notation_notes must be an array".into());
            }
            if !nullable_integer(&event["start"]) {
                return Err("Event start must be integer ticks or null".into());
            }
            if !event["duration"].is_null() && !event["duration"].is_string() {
                return Err("Duration must be a printed duration token or null".into());
            }
            for (index, item) in std::iter::once(event)
                .chain(array(&event["notes"]))
                .chain(array(&event["notation_notes"]))
                .enumerate()
            {
                if !item.is_object() || item.get("effects").is_some_and(|v| !v.is_array()) {
                    return Err("Events and notes need an effects array when supplied".into());
                }
                let mut effects = Vec::new();
                for effect in array(&item["effects"]) {
                    let s = effect.as_str().ok_or("Effects must be strings")?;
                    if ["tie", "pm", "let"].contains(&s) || s.starts_with("ottava:") {
                        return Err(
                            "Tie, PM, let ring and ottava require explicit relations".into()
                        );
                    }
                    let names = if index == 0 {
                        EVENT_EFFECTS
                    } else {
                        NOTE_EFFECTS
                    };
                    if !names.contains(&s) && !regex[usize::from(index > 0)].is_match(s) {
                        return Err(if index == 0 {
                            "Unsupported event effects"
                        } else {
                            "Unsupported note effects"
                        }
                        .into());
                    }
                    effects.push(s);
                }
                if !effects.is_empty() {
                    let payload = if index == 0 {
                        format!("s1f0<{}>", effects.join(","))
                    } else {
                        format!("s1f0({})", effects.join(","))
                    };
                    let (_, errors) = score::validate_measure_target(
                        &format!("M | V0{{@0:w:{payload}}}"),
                        "tab",
                        None,
                        None,
                    );
                    if !errors.is_empty() {
                        return Err(format!("Invalid effects: {}", errors.join("; ")));
                    }
                }
            }
            for note in array(&event["notes"])
                .iter()
                .chain(array(&event["notation_notes"]))
            {
                for field in ["string", "staff_position"] {
                    if !nullable_integer(&note[field]) {
                        return Err(format!("Note {field} must be an integer or null"));
                    }
                }
                if !nullable_integer(&note["fret"]) && note["fret"] != "x" {
                    return Err("Fret must be an integer, x or null".into());
                }
                if !note["notehead"].is_null()
                    && !note["notehead"]
                        .as_str()
                        .is_some_and(|s| NOTEHEADS.contains(&s))
                {
                    return Err("Unsupported visible notehead shape".into());
                }
                if !nullable_integer(&note["accidental"]) && note["accidental"] != "unknown" {
                    return Err("Accidental must be integer semitones, unknown or null".into());
                }
            }
        }
    }
    for item in o["state"].as_object().unwrap().values() {
        if !item.is_object()
            || !matches!(
                item["status"].as_str(),
                Some("observed" | "absent" | "unknown")
            )
        {
            return Err("State facts require observed, absent or unknown status".into());
        }
    }
    for r in array(&o["relations"]) {
        if !r.is_object() {
            return Err("Relations must be objects".into());
        }
        if r.get("representation")
            .is_some_and(|v| !matches!(v.as_str(), Some("tab" | "notation")))
        {
            return Err("Relation representation must be tab or notation".into());
        }
        if !r["kind"].is_null()
            && !matches!(
                r["kind"].as_str(),
                Some("tie" | "hammer" | "sl" | "ss" | "pm" | "let" | "ottava")
            )
        {
            return Err("Unsupported relation kind".into());
        }
        if !r["shape"].is_null()
            && !matches!(
                r["shape"].as_str(),
                Some("curve" | "diagonal" | "extension")
            )
        {
            return Err("Unsupported visible relation shape".into());
        }
        for (key, msg) in [
            (
                "voice",
                "Relation voice must be a local integer identifier or null",
            ),
            ("value", "Relation value must be integer semitones or null"),
        ] {
            if !nullable_integer(&r[key]) {
                return Err(msg.into());
            }
        }
        if !r["lane"].is_null() && !r["lane"].as_f64().is_some_and(f64::is_finite) {
            return Err("Relation lane must be a finite staff-relative coordinate".into());
        }
        for key in ["start", "end"] {
            let v = &r[key];
            if !v.is_null()
                && v != "left"
                && v != "right"
                && !(v.is_array()
                    && [2, 3].contains(&array(v).len())
                    && array(v).iter().all(integer))
            {
                return Err(
                    "Relation endpoint must be a local event/note reference or edge".into(),
                );
            }
        }
    }
    Ok(o)
}
fn duration(token: &Value) -> Option<Value> {
    score::parse_duration_token(token.as_str()?).ok()
}
fn known<'a>(state: &'a Value, field: &str) -> &'a Value {
    let item = &state[field];
    if matches!(item["status"].as_str(), Some("observed" | "inherited")) {
        &item["value"]
    } else {
        &Value::Null
    }
}
fn issue(row: &mut Row, reason: &str, fields: Value) {
    let mut v = json!({"reason":reason});
    if let Some(fields) = fields.as_object() {
        for (k, x) in fields {
            v[k] = x.clone();
        }
    }
    let issues = row.value["unresolved"].as_array_mut().unwrap();
    if !issues.contains(&v) {
        issues.push(v);
    }
}
const TIME_PATTERN: &str = r"^(?:[1-9]|[12][0-9]|3[0-2])/(?:1|2|4|8|16|32|64)$";
fn valid_state_value(field: &str, value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    match field {
        "time" => {
            static TIME: OnceLock<Regex> = OnceLock::new();
            value.as_str().is_some_and(|s| {
                TIME.get_or_init(|| Regex::new(TIME_PATTERN).unwrap())
                    .is_match(s)
            })
        }
        "key" => value.as_i64().is_some_and(|n| (-7..=7).contains(&n)),
        "clef" => matches!(
            value.as_str(),
            Some("G2" | "F4" | "C3" | "C4" | "percussion" | "tab")
        ),
        "clef_octave" => value
            .as_i64()
            .is_some_and(|n| [-24, -12, 0, 12, 24].contains(&n)),
        "instrument_transpose" => value.as_i64().is_some_and(|n| (-48..=48).contains(&n)),
        "instrument_transpose_explicit" => value.is_boolean(),
        "capo" => value.as_i64().is_some_and(|n| (0..=36).contains(&n)),
        "tuning" => value.as_array().is_some_and(|a| {
            a.iter()
                .all(|v| v.as_i64().is_some_and(|n| (0..=127).contains(&n)))
        }),
        _ => true,
    }
}
fn state(observation: &Value, auxiliary: &Value, previous: &Value) -> Value {
    let keys: [&Value; 3] = [observation, auxiliary, previous];
    let keys: BTreeSet<&String> = keys
        .iter()
        .filter_map(|v| v.as_object())
        .flat_map(|v| v.keys())
        .collect();
    let mut result = json!({});
    for field in keys {
        let extra = if valid_state_value(field, &auxiliary[field]["value"]) {
            &auxiliary[field]
        } else {
            &Value::Null
        };
        let observed = if valid_state_value(field, &observation[field]["value"]) {
            &observation[field]
        } else {
            &Value::Null
        };
        // Explicit human correction is authoritative for this field, including
        // "absent" when the user removes a mistaken printed change.
        let definitive_classifier =
            matches!(field.as_str(), "time" | "key" | "clef" | "clef_octave")
                && extra["source"] == "visual_classifier"
                && extra["calibrated"] == true
                && matches!(extra["status"].as_str(), Some("observed" | "absent"));
        let mut seen = if extra["source"] == "manual" || definitive_classifier {
            extra
        } else {
            observed
        };
        if extra["status"] == "observed" {
            if seen["status"] == "observed" && seen["value"] != extra["value"] {
                result[field] =
                    json!({"status":"unknown","value":null,"reason":"conflicting_observations"});
                continue;
            }
            seen = extra;
        }
        result[field] = if seen["status"] == "observed" && !seen["value"].is_null() {
            json!({"status":"observed","value":seen["value"]})
        } else if (seen["status"] == "absent"
            || (observation.get(field).is_none() && auxiliary.get(field).is_none()))
            && !known(previous, field).is_null()
        {
            json!({"status":"inherited","value":known(previous,field)})
        } else {
            json!({"status":"unknown","value":null})
        };
    }
    result
}
struct Row {
    value: Value,
    voice_map: Value,
    voice_links: BTreeMap<(usize, i64, i64), Value>,
    relations: Value,
}

pub fn parse_measure_observation(value: &Value, mode: &str) -> Result<Value, String> {
    let observation = parse_observation(value)?;
    if !matches!(mode, "tab" | "notation" | "both") {
        return Err("Observation mode must be tab, notation or both".into());
    }
    for voice in array(&observation["voices"]) {
        for event in array(&voice["events"]) {
            if mode == "both" {
                if !event["notes"].is_array() || !event["notation_notes"].is_array() {
                    return Err(
                        "Both observations require independent notes and notation_notes arrays"
                            .into(),
                    );
                }
                if array(&event["notes"])
                    .iter()
                    .any(|n| n.get("staff_position").is_some() || n.get("accidental").is_some())
                    || array(&event["notation_notes"])
                        .iter()
                        .any(|n| n.get("string").is_some() || n.get("fret").is_some())
                {
                    return Err("Both observations must not pair TAB and notation fields".into());
                }
            } else if event.get("notation_notes").is_some() {
                return Err("notation_notes only applies to both-mode observations".into());
            }
        }
    }
    if mode == "both"
        && array(&observation["relations"]).iter().any(|relation| {
            ["start", "end"]
                .iter()
                .any(|k| array(&relation[*k]).len() == 3)
                && relation["representation"].is_null()
        })
    {
        return Err("Both note relations require their visible representation".into());
    }
    Ok(observation)
}

// Both representations retain their own note indices until score context has
// resolved their pitches. Internal flat indices keep relation assembly shared.
fn independent_dual_notes(row: &mut Row) {
    if row.value["mode"] != "both" {
        return;
    }
    let mut offsets = HashMap::new();
    for voice in row.value["resolved"]["voices"].as_array_mut().unwrap() {
        let local = voice["voice"].as_i64().unwrap();
        for (ei, event) in voice["events"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            let notation = event["notation_notes"].as_array().unwrap().clone();
            let notes = event["notes"].as_array_mut().unwrap();
            let count = notes.len();
            offsets.insert((local, ei), count);
            notes.extend(notation);
            event["tab_note_count"] = json!(count);
            event.as_object_mut().unwrap().remove("notation_notes");
        }
    }
    for relation in row.relations.as_array_mut().unwrap() {
        if let Some(problem) = relation_problem(&row.value["observation"], relation) {
            relation["observation_problem"] = json!(problem);
        }
        if relation["representation"] == "notation" {
            for key in ["start", "end"] {
                let endpoint = &mut relation[key];
                if array(endpoint).len() == 3 {
                    if let (Some(voice), Some(event), Some(note)) = (
                        endpoint[0].as_i64(),
                        endpoint[1].as_u64(),
                        endpoint[2].as_u64(),
                    ) {
                        if let Some(offset) = offsets.get(&(voice, event as usize)) {
                            endpoint[2] = json!(note + *offset as u64);
                        }
                    }
                }
            }
        }
    }
}
#[derive(Clone)]
struct Node {
    record: usize,
    index: usize,
    relation: Value,
    kind: Value,
    value: Value,
    lane: Value,
    problem: Option<String>,
}
fn voice_index(row: &Row, local: &Value) -> Option<usize> {
    array(&row.value["resolved"]["voices"])
        .iter()
        .position(|v| v["voice"] == *local)
}
fn event<'a>(row: &'a Row, r: &Value) -> Option<&'a Value> {
    let a = r.as_array()?;
    if ![2, 3].contains(&a.len()) {
        return None;
    }
    let vi = voice_index(row, &a[0])?;
    row.value["resolved"]["voices"][vi]["events"]
        .as_array()?
        .get(a[1].as_u64()? as usize)
}
fn note<'a>(row: &'a Row, r: &Value) -> Option<&'a Value> {
    let a = r.as_array()?;
    if a.len() != 3 {
        return None;
    }
    event(row, r)?["notes"]
        .as_array()?
        .get(a[2].as_u64()? as usize)
}
fn note_mut<'a>(row: &'a mut Row, r: &Value) -> Option<&'a mut Value> {
    let a = r.as_array()?;
    if a.len() != 3 {
        return None;
    }
    let vi = voice_index(row, &a[0])?;
    row.value["resolved"]["voices"][vi]["events"]
        .as_array_mut()?
        .get_mut(a[1].as_u64()? as usize)?["notes"]
        .as_array_mut()?
        .get_mut(a[2].as_u64()? as usize)
}
fn local_ref(r: &Value) -> Value {
    Value::Array(array(r).iter().skip(1).cloned().collect())
}
fn global_ref(record: usize, r: &Value) -> Value {
    if r.is_array() {
        Value::Array(
            std::iter::once(json!(record))
                .chain(array(r).iter().cloned())
                .collect(),
        )
    } else {
        r.clone()
    }
}
fn voice_id(row: &Row, local: Option<i64>) -> Option<(String, String)> {
    let local = local?;
    if let Some(v) = row.voice_map.get(local.to_string()) {
        return Some((
            "mapped".into(),
            v.as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| v.to_string()),
        ));
    }
    let voices = array(&row.value["resolved"]["voices"]);
    if voices.len() == 1 && voices[0]["voice"] == local {
        Some(("single".into(), String::new()))
    } else {
        None
    }
}
pub fn relation_voice(r: &Value) -> Option<i64> {
    let mut voices = BTreeSet::new();
    for k in ["start", "end"] {
        if let Some(a) = r[k].as_array() {
            if [2, 3].contains(&a.len()) {
                if let Some(v) = a[0].as_i64() {
                    voices.insert(v);
                }
            }
        }
    }
    if let Some(v) = r["voice"].as_i64() {
        voices.insert(v);
    }
    if voices.len() == 1 {
        voices.into_iter().next()
    } else {
        None
    }
}
fn shapes_match(a: &Value, b: &Value) -> bool {
    if !a["representation"].is_null()
        && !b["representation"].is_null()
        && a["representation"] != b["representation"]
    {
        return false;
    }
    a["shape"].is_null()
        || b["shape"].is_null()
        || a["shape"] == b["shape"]
        || ((a["kind"] == "sl" || b["kind"] == "sl")
            && ((a["shape"] == "curve" && b["shape"] == "diagonal")
                || (a["shape"] == "diagonal" && b["shape"] == "curve")))
}
pub fn relation_problem(o: &Value, r: &Value) -> Option<String> {
    let scope = range_relation(r);
    let prefix = if scope {
        "range"
    } else {
        r["kind"].as_str().unwrap_or("connection")
    };
    let refs: Vec<&Value> = [&r["start"], &r["end"]]
        .into_iter()
        .filter(|v| v.is_array())
        .collect();
    for rf in &refs {
        let a = array(rf);
        let events = array(
            &array(&o["voices"])
                .iter()
                .find(|v| v["voice"] == a[0])
                .unwrap_or(&Value::Null)["events"],
        );
        let ev = a
            .get(1)
            .and_then(Value::as_u64)
            .and_then(|i| events.get(i as usize));
        if ev.is_none() || (!scope && a.len() != 3) {
            return Some(format!("{prefix}_endpoint_unresolved"));
        }
        let note_field =
            if r["representation"] == "notation" && ev.unwrap().get("notation_notes").is_some() {
                "notation_notes"
            } else {
                "notes"
            };
        if a.len() == 3
            && !a[2]
                .as_u64()
                .is_some_and(|i| (i as usize) < array(&ev.unwrap()[note_field]).len())
        {
            return Some(format!("{prefix}_endpoint_unresolved"));
        }
    }
    if (!refs.is_empty() || !r["voice"].is_null())
        && !relation_voice(r).is_some_and(|v| array(&o["voices"]).iter().any(|x| x["voice"] == v))
    {
        return Some(format!("{prefix}_voice_unresolved"));
    }
    if scope && refs.len() == 2 && refs[0][1].as_i64() > refs[1][1].as_i64() {
        return Some("range_endpoint_unresolved".into());
    }
    None
}
fn voices_match(start: &Value, end: &Value, rows: &[Row]) -> bool {
    let Some(a) = start[0].as_u64().and_then(|i| rows.get(i as usize)) else {
        return false;
    };
    let Some(b) = end[0].as_u64().and_then(|i| rows.get(i as usize)) else {
        return false;
    };
    let va = voice_id(a, start[1].as_i64());
    let vb = voice_id(b, end[1].as_i64());
    (va.is_some() && va == vb)
        || start[1]
            .as_i64()
            .zip(end[1].as_i64())
            .is_some_and(|(av, bv)| {
                b.voice_links
                    .contains_key(&(start[0].as_u64().unwrap() as usize, av, bv))
            })
}
fn connection_problem(
    start: &Value,
    end: &Value,
    rows: &[Row],
    kind: Option<&str>,
) -> Option<String> {
    let p = kind.unwrap_or("connection");
    let problem = |why: &str| Some(format!("{p}_{why}"));
    if !start.is_array() || !end.is_array() || array(start).len() != 4 || array(end).len() != 4 {
        return problem("endpoint_unresolved");
    }
    let Some(a) = start[0].as_u64().and_then(|i| rows.get(i as usize)) else {
        return problem("endpoint_unresolved");
    };
    let Some(b) = end[0].as_u64().and_then(|i| rows.get(i as usize)) else {
        return problem("endpoint_unresolved");
    };
    let sr = local_ref(start);
    let er = local_ref(end);
    let (Some(first), Some(last), Some(source), Some(target)) =
        (note(a, &sr), note(b, &er), event(a, &sr), event(b, &er))
    else {
        return problem("endpoint_unresolved");
    };
    let Some(ticks) = duration(&source["duration"]).and_then(|d| score::duration_ticks(&d).ok())
    else {
        return problem("timing_unresolved");
    };
    let (Some(ss), Some(ts)) = (source["start"].as_i64(), target["start"].as_i64()) else {
        return problem("timing_unresolved");
    };
    let finish = ss as i128 * ticks.denominator + ticks.numerator;
    let continuous = if start[0] == end[0] {
        finish == ts as i128 * ticks.denominator
            && end[2].as_i64() == start[2].as_i64().and_then(|i| i.checked_add(1))
    } else {
        let time = known(&a.value["resolved"]["state"], "time")
            .as_str()
            .unwrap_or("");
        let parts: Vec<_> = time.split('/').collect();
        let ratio = if parts.len() == 2 {
            parts[0]
                .parse::<i64>()
                .ok()
                .zip(parts[1].parse::<i64>().ok())
                .filter(|(_, d)| *d != 0)
        } else {
            None
        };
        let Some((n, d)) = ratio else {
            return problem("timing_unresolved");
        };
        let vi = voice_index(a, &start[1]).unwrap();
        b.value["bar_index"].as_i64()
            == a.value["bar_index"].as_i64().and_then(|i| i.checked_add(1))
            && finish * d as i128 == 3840 * n as i128 * ticks.denominator
            && ts == 0
            && start[2].as_u64()
                == Some(
                    array(&a.value["resolved"]["voices"][vi]["events"])
                        .len()
                        .saturating_sub(1) as u64,
                )
            && end[2] == 0
    };
    if !continuous {
        return problem("not_continuous");
    }
    let tab = a.value["mode"] == "tab" || first.get("string").is_some();
    let field = if tab { "string" } else { "staff_position" };
    if (kind == Some("tie") || tab) && (first[field].is_null() || first[field] != last[field]) {
        return problem("identity_unresolved");
    }
    if kind == Some("tie") {
        for field in ["fret", "pitch", "written_pitch", "sounding_pitch"] {
            if !first[field].is_null() && !last[field].is_null() && first[field] != last[field] {
                return Some("tie_endpoint_conflict".into());
            }
        }
    }
    None
}
fn boundary_voices(outgoing: &[usize], incoming: &[usize], nodes: &[Node], rows: &mut [Row]) {
    let mut candidates = Vec::new();
    for &left in outgoing {
        for &right in incoming {
            let (a, b) = (&nodes[left], &nodes[right]);
            let (ar, br) = (&a.relation, &b.relation);
            let (Some(av), Some(bv)) = (relation_voice(ar), relation_voice(br)) else {
                continue;
            };
            if range_relation(ar)
                || range_relation(br)
                || (!ar["kind"].is_null() && !br["kind"].is_null() && ar["kind"] != br["kind"])
                || !shapes_match(ar, br)
                || (!ar["value"].is_null() && !br["value"].is_null() && ar["value"] != br["value"])
            {
                continue;
            }
            let (va, vb) = (
                voice_id(&rows[a.record], Some(av)),
                voice_id(&rows[b.record], Some(bv)),
            );
            if va.as_ref().is_some_and(|v| v.0 == "mapped")
                && vb.as_ref().is_some_and(|v| v.0 == "mapped")
                && va != vb
            {
                continue;
            }
            if !ar["start"].is_array() || !br["end"].is_array() {
                continue;
            }
            if connection_problem(
                &global_ref(a.record, &ar["start"]),
                &global_ref(b.record, &br["end"]),
                rows,
                ar["kind"].as_str().or(br["kind"].as_str()),
            )
            .is_none()
            {
                candidates.push((left, right, av, bv));
            }
        }
    }
    let unique: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|p| {
            candidates.iter().filter(|x| x.0 == p.0).count() == 1
                && candidates.iter().filter(|x| x.1 == p.1).count() == 1
        })
        .collect();
    let pairs: BTreeSet<_> = unique.iter().map(|p| (p.2, p.3)).collect();
    for (left, right, av, bv) in unique {
        if pairs.iter().filter(|p| p.0 == av).count() != 1
            || pairs.iter().filter(|p| p.1 == bv).count() != 1
        {
            continue;
        }
        let (a, b) = (&nodes[left], &nodes[right]);
        let (va, vb) = (
            voice_id(&rows[a.record], Some(av)),
            voice_id(&rows[b.record], Some(bv)),
        );
        if va.is_some() && va == vb {
            continue;
        }
        rows[b.record]
            .voice_links
            .entry((a.record, av, bv))
            .or_insert(json!([[a.record, a.index], [b.record, b.index]]));
    }
}
fn pitch_notes(row: &mut Row, incoming: &HashSet<(i64, usize, usize)>) {
    let entering: HashSet<_> = array(&row.relations)
        .iter()
        .filter(|r| r["start"] == "left" && (r["kind"].is_null() || r["kind"] == "tie"))
        .filter_map(|r| {
            Some((
                r["end"][0].as_i64()?,
                r["end"][1].as_u64()? as usize,
                r["end"][2].as_u64()? as usize,
            ))
        })
        .collect();
    let st = &row.value["resolved"]["state"];
    let base = match known(st, "clef").as_str() {
        Some("G2") => Some(30),
        Some("F4") => Some(18),
        Some("C3") => Some(24),
        Some("C4") => Some(22),
        _ => None,
    };
    let fifths = known(st, "key").as_i64().filter(|n| (-7..=7).contains(n));
    let mut alterations = [0; 7];
    if let Some(n) = fifths {
        let order = if n >= 0 {
            [3, 0, 4, 1, 5, 2, 6]
        } else {
            [6, 2, 5, 1, 4, 0, 3]
        };
        for step in order.into_iter().take(n.unsigned_abs() as usize) {
            alterations[step] = if n > 0 { 1 } else { -1 };
        }
    }
    let tuning = known(st, "tuning").clone();
    let percussion = row.value["instrument"] == "drums" || known(st, "clef") == "percussion";
    let tab = row.value["mode"] == "tab";
    let mut accidentals: HashMap<i64, Option<i64>> = HashMap::new();
    let mut events = Vec::new();
    for (vi, voice) in array(&row.value["resolved"]["voices"]).iter().enumerate() {
        for (ei, event) in array(&voice["events"]).iter().enumerate() {
            events.push((
                event["start"].as_i64(),
                voice["voice"].as_i64().unwrap(),
                vi,
                ei,
            ));
        }
    }
    events.sort_by_key(|(s, v, _, _)| (s.is_none(), s.unwrap_or(0), *v));
    for (_, voice, vi, ei) in events {
        let event = &mut row.value["resolved"]["voices"][vi]["events"][ei];
        let Some(notes) = event["notes"].as_array_mut() else {
            continue;
        };
        for (ni, note) in notes.iter_mut().enumerate() {
            note["pitch"] = Value::Null;
            note["written_pitch"] = Value::Null;
            if percussion {
                if let (Some(position), Some(shape)) =
                    (note["staff_position"].as_i64(), note["notehead"].as_str())
                {
                    if let Some(pitch) = percussion_pitch(position, shape) {
                        note["pitch"] = json!(pitch);
                        note["written_pitch"] = json!(pitch);
                        note["sounding_pitch"] = json!(pitch);
                    }
                }
                continue;
            }
            if tab || note.get("string").is_some() {
                if let (Some(string), Some(fret)) = (note["string"].as_u64(), note["fret"].as_i64())
                {
                    if string >= 1 {
                        if let Some(open) = array(&tuning)
                            .get((string - 1) as usize)
                            .and_then(Value::as_i64)
                        {
                            note["sounding_pitch"] = json!(open + fret);
                        }
                    }
                }
                continue;
            }
            let (Some(position), Some(base)) = (note["staff_position"].as_i64(), base) else {
                continue;
            };
            let diatonic = base + position;
            let accidental = note.get("accidental").cloned().unwrap_or(json!("unknown"));
            if let Some(n) = accidental.as_i64() {
                accidentals.insert(diatonic, Some(n));
            } else if accidental == "unknown" {
                accidentals.insert(diatonic, None);
            }
            if incoming.contains(&(voice, ei, ni)) && accidental.is_null() {
                // A tied accidental may carry into the bar. Until its source
                // is resolved, later unmarked notes at this position are not
                // evidence for the key-signature pitch.
                if entering.contains(&(voice, ei, ni)) {
                    accidentals.entry(diatonic).or_insert(None);
                }
                continue;
            }
            let alter = accidentals
                .get(&diatonic)
                .copied()
                .unwrap_or_else(|| fifths.map(|_| alterations[diatonic.rem_euclid(7) as usize]));
            if let Some(alter) = alter {
                let pitch = 12 * (diatonic.div_euclid(7) + 1)
                    + [0, 2, 4, 5, 7, 9, 11][diatonic.rem_euclid(7) as usize]
                    + alter;
                note["written_pitch"] = json!(pitch);
                note["pitch"] = json!(pitch);
            }
        }
    }
}
// TAB pitch depends on the inherited tuning and written/sounding context.
fn dual_shift(state: &Value, event: &Value) -> Option<i64> {
    let base = known(state, "instrument_transpose").as_i64()?;
    let explicit = known(state, "instrument_transpose_explicit").as_bool()?;
    let clef = known(state, "clef_octave").as_i64()?;
    let capo = known(state, "capo").as_i64()?;
    let transpose = if !explicit && base == clef { 0 } else { base };
    let octave = array(&event["effects"])
        .iter()
        .filter_map(Value::as_str)
        .find_map(|e| {
            e.strip_prefix("ottava:")
                .and_then(|v| v.parse::<i64>().ok())
        })
        .unwrap_or(0);
    Some(transpose + clef - capo + octave)
}
fn resolve_dual_tab(row: &mut Row) {
    if row.value["mode"] != "both" {
        return;
    }
    let state = row.value["resolved"]["state"].clone();
    let tuning = known(&state, "tuning");
    for voice in row.value["resolved"]["voices"].as_array_mut().unwrap() {
        for event in voice["events"].as_array_mut().unwrap() {
            let Some(shift) = dual_shift(&state, event) else {
                continue;
            };
            for field in ["notes", "continuation_notes"] {
                let Some(notes) = event.get_mut(field).and_then(Value::as_array_mut) else {
                    continue;
                };
                for note in notes {
                    let Some(string) = note["string"].as_u64().filter(|s| *s > 0) else {
                        continue;
                    };
                    let Some(open) = array(tuning)
                        .get(string as usize - 1)
                        .and_then(Value::as_i64)
                    else {
                        continue;
                    };
                    if let Some(fret) = note["fret"].as_i64() {
                        let pitch = open + fret - shift;
                        if (0..=127).contains(&pitch) {
                            note["pitch"] = json!(pitch);
                            note["written_pitch"] = json!(pitch);
                            note["sounding_pitch"] = json!(open + fret);
                        }
                    } else if note["fret"].is_null() {
                        if let Some(pitch) = note["pitch"].as_i64() {
                            let fret = pitch + shift - open;
                            if (0..=36).contains(&fret) {
                                note["fret"] = json!(fret);
                                note["sounding_pitch"] = json!(open + fret);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Merge representations only after each has resolved its own relations. The
// note order in either printed staff carries no cross-staff identity.
fn merge_dual_notes(row: &mut Row) -> HashMap<(i64, usize, usize), Vec<usize>> {
    let mut mapping = HashMap::new();
    if row.value["mode"] != "both" {
        return mapping;
    }
    let state = row.value["resolved"]["state"].clone();
    let mut issues = Vec::new();
    for voice in row.value["resolved"]["voices"].as_array_mut().unwrap() {
        let local = voice["voice"].as_i64().unwrap();
        for (ei, event) in voice["events"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            let Some(count) = event["tab_note_count"].as_u64().map(|v| v as usize) else {
                continue;
            };
            let observed_tab_count = count;
            let mut original = event["notes"].as_array().unwrap().clone();
            let derived: Vec<_> = array(&event["continuation_notes"])
                .iter()
                .map(|note| {
                    let mut note = note.clone();
                    note.as_object_mut().unwrap().remove("notation_note");
                    note
                })
                .collect();
            let added = derived.len();
            original.splice(count..count, derived);
            let count = count + added;
            let original_index = |i: usize| {
                if i < observed_tab_count {
                    Some(i)
                } else if i >= count {
                    Some(i - added)
                } else {
                    None
                }
            };
            let mut notes = original[..count].to_vec();
            let pairs = dual_note_pairs(&original, count, &state);
            let mut remaining: Vec<usize> = (count..original.len()).collect();
            let mut matched = HashSet::new();
            for (tab, notation) in pairs {
                for &ni in &tab {
                    merge_note_evidence(&mut notes[ni], &original[notation[0]]);
                    matched.insert(ni);
                }
                for j in notation {
                    mapping.insert((local, ei, original_index(j).unwrap()), tab.clone());
                    remaining.retain(|&i| i != j);
                }
            }
            for ni in 0..count {
                if let Some(index) = original_index(ni) {
                    mapping.insert((local, ei, index), vec![ni]);
                }
            }
            if matched.len() != count || !remaining.is_empty() {
                issues.push(json!({"voice":local,"event":ei,"tab_notes":count,
                    "notation_notes":original.len()-count,"matched_notes":matched.len()}));
                // Retain unmatched observed notes. Their absent string/fret
                // prevents a falsely complete combined IR from being emitted.
                for j in remaining {
                    mapping.insert((local, ei, original_index(j).unwrap()), vec![notes.len()]);
                    notes.push(original[j].clone());
                }
            }
            event["notes"] = json!(notes);
            event.as_object_mut().unwrap().remove("tab_note_count");
            event.as_object_mut().unwrap().remove("continuation_notes");
        }
    }
    for fields in issues {
        issue(row, "dual_note_pairing_unresolved", fields);
    }
    resolve_dual_tab(row);
    mapping
}
fn dual_note_pairs(
    notes: &[Value],
    tab_count: usize,
    state: &Value,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut pairs = Vec::new();
    let mut remaining: Vec<usize> = (tab_count..notes.len()).collect();
    let mut matched = HashSet::new();
    let pitches: BTreeSet<i64> = notes[..tab_count]
        .iter()
        .filter_map(|n| n["pitch"].as_i64())
        .collect();
    for pitch in pitches {
        let tab: Vec<_> = (0..tab_count)
            .filter(|&i| notes[i]["pitch"].as_i64() == Some(pitch))
            .collect();
        let notation: Vec<_> = remaining
            .iter()
            .copied()
            .filter(|&j| notes[j]["pitch"].as_i64() == Some(pitch))
            .collect();
        if notation.is_empty()
            || notation.len() > tab.len()
            || !notation.iter().all(|&j| notes[j] == notes[notation[0]])
        {
            continue;
        }
        matched.extend(tab.iter().copied());
        remaining.retain(|i| !notation.contains(i));
        pairs.push((tab, notation));
    }
    let unmatched: Vec<_> = (0..tab_count).filter(|i| !matched.contains(i)).collect();
    if !unmatched.is_empty()
        && !remaining.is_empty()
        && remaining.len() <= unmatched.len()
        && unmatched
            .iter()
            .filter_map(|&i| notes[i]["pitch"].as_i64())
            .collect::<BTreeSet<_>>()
            .len()
            <= 1
        && remaining.iter().all(|&j| notes[j] == notes[remaining[0]])
        && unmatched
            .iter()
            .all(|&i| compatible_notation_pitch(&notes[remaining[0]], &notes[i], state))
    {
        pairs.push((unmatched, remaining));
    }
    pairs
}

// Exchange determined pitches without collapsing the independently observed
// note arrays, so notation ties can carry TAB-derived pitch to the next event.
fn exchange_dual_pitches(row: &mut Row) {
    if row.value["mode"] != "both" {
        return;
    }
    resolve_dual_tab(row);
    let state = row.value["resolved"]["state"].clone();
    for voice in row.value["resolved"]["voices"].as_array_mut().unwrap() {
        for event in voice["events"].as_array_mut().unwrap() {
            let Some(count) = event["tab_note_count"].as_u64() else {
                continue;
            };
            let notes = event["notes"].as_array_mut().unwrap();
            for (tab, notation) in dual_note_pairs(notes, count as usize, &state) {
                let pitches: BTreeSet<_> = tab
                    .iter()
                    .chain(&notation)
                    .filter_map(|&i| notes[i]["pitch"].as_i64())
                    .collect();
                if pitches.len() != 1 {
                    continue;
                }
                let pitch = *pitches.iter().next().unwrap();
                for ni in tab.into_iter().chain(notation) {
                    if notes[ni]["pitch"].is_null() {
                        notes[ni]["pitch"] = json!(pitch);
                        notes[ni]["written_pitch"] = json!(pitch);
                    }
                }
            }
        }
    }
    resolve_dual_tab(row);
}

fn propagate_dual_tie(start: &Value, end: &Value, rows: &mut [Row]) {
    let source = &rows[start[0].as_u64().unwrap() as usize];
    if source.value["mode"] != "both" {
        return;
    }
    let original = event(source, &local_ref(start)).unwrap();
    let Some(count) = original["tab_note_count"].as_u64().map(|v| v as usize) else {
        return;
    };
    let index = start[3].as_u64().unwrap() as usize;
    if index < count {
        return;
    }
    let notes = array(&original["notes"]);
    let strings: HashSet<_> = dual_note_pairs(notes, count, &source.value["resolved"]["state"])
        .into_iter()
        .filter(|(_, notation)| notation.contains(&index))
        .flat_map(|(tab, _)| tab)
        .filter_map(|i| notes[i]["string"].as_i64())
        .chain(
            array(&original["continuation_notes"])
                .iter()
                .filter(|note| note["notation_note"] == index)
                .filter_map(|note| note["string"].as_i64()),
        )
        .collect();
    let destination = &mut rows[end[0].as_u64().unwrap() as usize];
    if destination.value["mode"] != "both" {
        return;
    }
    let Some(pitch) = note(destination, &local_ref(end)).and_then(|n| n["pitch"].as_i64()) else {
        return;
    };
    let vi = voice_index(destination, &end[1]).unwrap();
    let event = &mut destination.value["resolved"]["voices"][vi]["events"]
        [end[2].as_u64().unwrap() as usize];
    let Some(count) = event["tab_note_count"].as_u64().map(|v| v as usize) else {
        return;
    };
    let unique_string = strings.len() == 1;
    if !unique_string && !strings.is_empty() {
        let mut candidates: Vec<_> = strings.iter().copied().collect();
        candidates.sort_unstable();
        let ni = end[3].as_u64().unwrap() as usize;
        event["notes"][ni]["tab_string_candidates"] = json!(candidates);
    }
    for string in strings {
        if let Some(note) = event["notes"].as_array_mut().unwrap()[..count]
            .iter_mut()
            .find(|note| note["string"] == string)
        {
            if note["fret"].is_null() && note["pitch"].is_null() {
                note["pitch"] = json!(pitch);
                note["written_pitch"] = json!(pitch);
            }
        } else if unique_string {
            if !event["continuation_notes"].is_array() {
                event["continuation_notes"] = json!([]);
            }
            let derived = event["continuation_notes"].as_array_mut().unwrap();
            if !derived.iter().any(|note| note["string"] == string) {
                derived.push(
                    json!({"string":string,"fret":null,"pitch":pitch,"written_pitch":pitch,
                    "notation_note":end[3],"effects":["tie"]}),
                );
            }
        }
    }
    exchange_dual_pitches(destination);
}

fn compatible_notation_pitch(notation: &Value, tab: &Value, state: &Value) -> bool {
    let Some(pitch) = tab["pitch"].as_i64() else {
        return true;
    };
    if let Some(written) = notation["pitch"].as_i64() {
        return written == pitch;
    }
    let base = match known(state, "clef").as_str() {
        Some("G2") => 30,
        Some("F4") => 18,
        Some("C3") => 24,
        Some("C4") => 22,
        _ => return true,
    };
    let Some(position) = notation["staff_position"].as_i64() else {
        return true;
    };
    let diatonic = base + position;
    let natural =
        12 * (diatonic.div_euclid(7) + 1) + [0, 2, 4, 5, 7, 9, 11][diatonic.rem_euclid(7) as usize];
    // An unreadable accidental may be double-flat through double-sharp; it
    // cannot move a head to a different octave or unrelated staff position.
    (pitch - natural).abs() <= 2
}
fn merge_note_evidence(tab: &mut Value, notation: &Value) {
    for (key, value) in notation.as_object().unwrap() {
        if key == "effects" {
            for effect in array(value).iter().filter_map(Value::as_str) {
                add_effect(tab, effect);
            }
        } else if tab[key].is_null() {
            tab[key] = value.clone();
        }
    }
}
fn compatible(left: &Node, right: &Node, rows: &[Row]) -> bool {
    let (a, b) = (&left.relation, &right.relation);
    if (!left.kind.is_null() && !right.kind.is_null() && left.kind != right.kind)
        || (!left.value.is_null() && !right.value.is_null() && left.value != right.value)
        || !shapes_match(a, b)
    {
        return false;
    }
    let scope = range_relation(a) && range_relation(b);
    if range_relation(a) != range_relation(b)
        || (scope
            && [a, b]
                .iter()
                .any(|r| r["start"].is_null() || r["end"].is_null()))
    {
        return false;
    }
    if !voices_match(
        &json!([left.record, relation_voice(a)]),
        &json!([right.record, relation_voice(b)]),
        rows,
    ) {
        return false;
    }
    if scope {
        return true;
    }
    let (Some(first), Some(last)) = (
        note(&rows[left.record], &a["start"]),
        note(&rows[right.record], &b["end"]),
    ) else {
        return false;
    };
    if rows[left.record].value["mode"] == "notation"
        && left.kind.as_str().or(right.kind.as_str()) != Some("tie")
    {
        return true;
    }
    let field = if rows[left.record].value["mode"] == "tab" || first.get("string").is_some() {
        "string"
    } else {
        "staff_position"
    };
    !first[field].is_null() && first[field] == last[field]
}
fn add_effect(value: &mut Value, effect: &str) {
    if value.get("effects").is_none() {
        value["effects"] = json!([]);
    }
    let effects = value["effects"].as_array_mut().unwrap();
    let effect = json!(effect);
    if !effects.contains(&effect) {
        effects.push(effect);
    }
}
fn note_technique(chain: &Value, rows: &mut [Row]) -> Option<String> {
    let (start, end, kind) = (
        &chain["start"],
        &chain["end"],
        chain["kind"].as_str().unwrap(),
    );
    if start.is_array() && end == "right" {
        // sl/ss/hammer need a destination. Slide-out marks are the separate
        // sou/sod note effects; an unmatched crop edge cannot establish one.
        return Some(format!("{kind}_endpoint_unresolved"));
    }
    if let Some(problem) = connection_problem(start, end, rows, Some(kind)) {
        return Some(problem);
    }
    let same_voice = if start[0] == end[0] {
        start[1] == end[1]
    } else {
        voices_match(start, end, rows)
    };
    if !same_voice {
        return Some(format!("{kind}_voice_unresolved"));
    }
    for endpoint in [start, end] {
        let row = &rows[endpoint[0].as_u64().unwrap() as usize];
        let note = note(row, &local_ref(endpoint)).unwrap();
        let field = if row.value["mode"] == "tab" || note.get("string").is_some() {
            "fret"
        } else {
            "pitch"
        };
        if note[field].is_null() {
            return Some(format!("{kind}_value_unresolved"));
        }
    }
    add_effect(
        note_mut(
            &mut rows[start[0].as_u64().unwrap() as usize],
            &local_ref(start),
        )
        .unwrap(),
        kind,
    );
    None
}
fn slide_destination(chain: &mut Value, rows: &[Row], following: &HashMap<usize, usize>) {
    let start = &chain["start"];
    if !matches!(chain["kind"].as_str(), Some("sl" | "ss"))
        || !start.is_array()
        || chain["end"] != "right"
    {
        return;
    }
    let record = start[0].as_u64().unwrap() as usize;
    let Some(&next) = following.get(&record) else {
        return;
    };
    if !matches!(rows[record].value["mode"].as_str(), Some("tab" | "both")) {
        return;
    }
    let target = &rows[next].value;
    let mut candidates = Vec::new();
    for voice in array(&target["resolved"]["voices"]) {
        for (ni, n) in array(&voice["events"][0]["notes"]).iter().enumerate() {
            let end = json!([next, voice["voice"], 0, ni]);
            if !integer(&n["fret"])
                || !voices_match(start, &end, rows)
                || connection_problem(start, &end, rows, chain["kind"].as_str()).is_some()
            {
                continue;
            }
            if array(&rows[next].relations)
                .iter()
                .any(|r| r["start"] == "left" && r["end"] == local_ref(&end))
            {
                continue;
            }
            candidates.push(end);
        }
    }
    if candidates.len() == 1 {
        chain["end"] = candidates.remove(0);
        chain["end_evidence"] = json!("unique_adjacent_visible_note");
    }
}
fn tie(chain: &Value, rows: &mut [Row]) -> Option<String> {
    let (start, end) = (&chain["start"], &chain["end"]);
    if let Some(problem) = connection_problem(start, end, rows, Some("tie")) {
        return Some(problem);
    }
    let same_voice = if start[0] == end[0] {
        start[1] == end[1]
    } else {
        voices_match(start, end, rows)
    };
    if !same_voice {
        return Some("tie_voice_unresolved".into());
    }
    let first = note(
        &rows[start[0].as_u64().unwrap() as usize],
        &local_ref(start),
    )
    .unwrap()
    .clone();
    let source = &rows[start[0].as_u64().unwrap() as usize];
    let field = if source.value["mode"] == "tab" || first.get("string").is_some() {
        "fret"
    } else {
        "pitch"
    };
    if first[field].is_null() {
        // Keep the visible connection, but an unknown source cannot establish
        // a sounding tie to a known note later in an editable export.
        return Some("tie_value_unresolved".into());
    }
    let b = &mut rows[end[0].as_u64().unwrap() as usize];
    let mode = b.value["mode"].as_str().unwrap().to_owned();
    let last = note_mut(b, &local_ref(end)).unwrap();
    for field in ["fret", "pitch", "written_pitch", "sounding_pitch"] {
        if last[field].is_null() && !first[field].is_null() {
            last[field] = first[field].clone();
        }
    }
    add_effect(last, "tie");
    exchange_dual_pitches(b);
    resolve_key_notes_after_tie(b, &local_ref(end));
    propagate_dual_tie(start, end, rows);
    let last = note(&rows[end[0].as_u64().unwrap() as usize], &local_ref(end)).unwrap();
    let needed: &[&str] = if mode == "tab" || last.get("string").is_some() {
        &["fret"]
    } else {
        &["pitch"]
    };
    if needed.iter().any(|k| last[k].is_null()) {
        Some("tie_value_unresolved".into())
    } else {
        None
    }
}
fn resolve_key_notes_after_tie(row: &mut Row, endpoint: &Value) {
    let state = &row.value["resolved"]["state"];
    let base = match known(state, "clef").as_str() {
        Some("G2") => 30,
        Some("F4") => 18,
        Some("C3") => 24,
        Some("C4") => 22,
        _ => return,
    };
    let Some(fifths) = known(state, "key").as_i64() else {
        return;
    };
    let tied = note(row, endpoint).unwrap();
    let Some(position) = tied["staff_position"].as_i64() else {
        return;
    };
    let diatonic = base + position;
    let order = if fifths >= 0 {
        [3, 0, 4, 1, 5, 2, 6]
    } else {
        [6, 2, 5, 1, 4, 0, 3]
    };
    let alteration = if order
        .into_iter()
        .take(fifths.unsigned_abs() as usize)
        .any(|step| step == diatonic.rem_euclid(7))
    {
        fifths.signum()
    } else {
        0
    };
    let pitch = 12 * (diatonic.div_euclid(7) + 1)
        + [0, 2, 4, 5, 7, 9, 11][diatonic.rem_euclid(7) as usize]
        + alteration;
    // Both engraving conventions agree when the continued pitch equals the
    // key signature. An altered tie alone cannot decide later unmarked notes.
    if tied["pitch"].as_i64() != Some(pitch) {
        return;
    }
    let Some(start) =
        event(row, &json!(array(endpoint)[..2])).and_then(|event| event["start"].as_i64())
    else {
        return;
    };
    let mut events = Vec::new();
    for (vi, voice) in array(&row.value["resolved"]["voices"]).iter().enumerate() {
        for (ei, event) in array(&voice["events"]).iter().enumerate() {
            if let Some(time) = event["start"].as_i64().filter(|time| *time >= start) {
                events.push((time, vi, ei));
            }
        }
    }
    events.sort_unstable();
    for (_, vi, ei) in events {
        for note in row.value["resolved"]["voices"][vi]["events"][ei]["notes"]
            .as_array_mut()
            .unwrap()
        {
            if note["staff_position"].as_i64() != Some(position) {
                continue;
            }
            if !note.get("accidental").is_some_and(Value::is_null) {
                return;
            }
            if note["pitch"].is_null() {
                note["pitch"] = json!(pitch);
                note["written_pitch"] = json!(pitch);
            }
        }
    }
}
fn apply_range(chain: &mut Value, nodes: &[Node], rows: &mut [Row]) -> bool {
    let mut valid = true;
    for node in nodes {
        let row = &mut rows[node.record];
        let r = &node.relation;
        let local = relation_voice(r);
        let Some(vi) = voice_index(row, &json!(local)) else {
            issue(
                row,
                "range_voice_unresolved",
                json!({"relation":node.index}),
            );
            valid = false;
            continue;
        };
        let (start, end) = (&r["start"], &r["end"]);
        let left = if start == "left" {
            Some(0)
        } else {
            event(row, start)
                .and_then(|_| start[1].as_u64())
                .map(|v| v as usize)
        };
        let mut right = if end == "right" {
            array(&row.value["resolved"]["voices"][vi]["events"])
                .len()
                .checked_sub(1)
        } else {
            event(row, end)
                .and_then(|_| end[1].as_u64())
                .map(|v| v as usize)
        };
        if start.is_array()
            && left.is_some()
            && end.is_null()
            && matches!(chain["kind"].as_str(), Some("pm" | "let"))
        {
            right = left;
            issue(
                row,
                "range_endpoint_unresolved",
                json!({"relation":node.index}),
            );
            valid = false;
        }
        let (Some(left), Some(right)) = (left, right) else {
            issue(
                row,
                "range_endpoint_unresolved",
                json!({"relation":node.index}),
            );
            valid = false;
            continue;
        };
        if right < left {
            issue(
                row,
                "range_endpoint_unresolved",
                json!({"relation":node.index}),
            );
            valid = false;
            continue;
        }
        for ei in left..=right {
            chain["coverage"]
                .as_array_mut()
                .unwrap()
                .push(json!([node.record, local, ei]));
            match chain["kind"].as_str() {
                Some(kind @ ("pm" | "let")) => {
                    row.value["resolved"]["visual_techniques"]
                        .as_array_mut()
                        .unwrap()
                        .push(
                            json!({"voice":local,"event":ei,"effect":kind,"relation":node.index}),
                        );
                    let mut owners =
                        if array(&row.value["resolved"]["voices"][vi]["events"][ei]["notes"]).len()
                            == 1
                        {
                            vec![0]
                        } else {
                            vec![]
                        };
                    let event = &row.value["resolved"]["voices"][vi]["events"][ei];
                    if event["tab_note_count"] == 1 && array(&event["notes"]).len() == 2 {
                        owners = vec![0, 1];
                    }
                    if array(start).len() == 3 && left == right {
                        owners = if note(row, start).is_some() {
                            vec![start[2].as_u64().unwrap() as usize]
                        } else {
                            vec![]
                        };
                    }
                    for ni in owners {
                        add_effect(
                            &mut row.value["resolved"]["voices"][vi]["events"][ei]["notes"][ni],
                            kind,
                        );
                    }
                }
                Some("ottava") => {
                    let shift = chain["value"].as_i64();
                    if !shift.is_some_and(|n| [-24, -12, 12, 24].contains(&n)) {
                        issue(
                            row,
                            "ottava_value_unresolved",
                            json!({"relation":node.index}),
                        );
                        valid = false;
                        continue;
                    }
                    let effect = format!("ottava:{}", shift.unwrap());
                    let event = &mut row.value["resolved"]["voices"][vi]["events"][ei];
                    let prior: Vec<_> = array(&event["effects"])
                        .iter()
                        .filter_map(Value::as_str)
                        .filter(|s| s.starts_with("ottava:"))
                        .collect();
                    if !prior.is_empty() && !prior.contains(&effect.as_str()) {
                        issue(
                            row,
                            "ottava_range_conflict",
                            json!({"voice":local,"event":ei}),
                        );
                        valid = false;
                        continue;
                    }
                    add_effect(event, &effect);
                }
                _ => {
                    issue(
                        row,
                        "relation_kind_unresolved",
                        json!({"relation":node.index}),
                    );
                    valid = false;
                }
            }
        }
    }
    valid
}
fn ir(row: &mut Row) -> Option<String> {
    let resolved = row.value["resolved"].clone();
    let st = &resolved["state"];
    let mode = row.value["mode"].as_str().unwrap().to_owned();
    let mut measure = json!({"voices":[]});
    let time = known(st, "time");
    if !time.is_null() {
        measure["time_signature"] = time.clone();
        measure["print_time_signature"] = json!(st["time"]["status"] == "observed");
    }
    for voice in array(&resolved["voices"]) {
        let mut current = json!({"voice":voice["voice"],"events":[]});
        for (ei, event) in array(&voice["events"]).iter().enumerate() {
            if array(&row.value["unresolved"]).iter().any(|issue| {
                issue["reason"] == "dual_note_pairing_unresolved"
                    && issue["voice"] == voice["voice"]
                    && issue["event"] == ei
            }) {
                return None;
            }
            let dur = duration(&event["duration"]);
            if dur.is_none()
                || event["start"].is_null()
                || !matches!(event["status"].as_str(), Some("normal" | "rest" | "empty"))
            {
                issue(
                    row,
                    "event_unresolved",
                    json!({"voice":voice["voice"],"event":ei}),
                );
                return None;
            }
            let mut notes = Vec::new();
            let needed: &[&str] = match mode.as_str() {
                "tab" => &["string", "fret"],
                "notation" => &["pitch"],
                _ => &["string", "fret", "pitch"],
            };
            for (ni, note) in array(&event["notes"]).iter().enumerate() {
                if needed.iter().any(|k| note[k].is_null()) {
                    issue(
                        row,
                        "note_unresolved",
                        json!({"voice":voice["voice"],"event":ei,"note":ni}),
                    );
                    return None;
                }
                let mut n = json!({});
                for &key in needed.iter().chain(std::iter::once(&"effects")) {
                    if let Some(v) = note.get(key) {
                        n[key] = v.clone();
                    }
                }
                notes.push(n);
            }
            if event["status"] == "normal" && notes.is_empty() {
                issue(
                    row,
                    "event_notes_unresolved",
                    json!({"voice":voice["voice"],"event":ei}),
                );
                return None;
            }
            current["events"].as_array_mut().unwrap().push(json!({"start":event["start"],"duration":dur.unwrap(),"status":event["status"],"notes":notes,"effects":event.get("effects").cloned().unwrap_or(json!([]))}));
        }
        measure["voices"].as_array_mut().unwrap().push(current);
    }
    if array(&measure["voices"]).is_empty() {
        issue(row, "voices_unresolved", json!({}));
        return None;
    }
    match score::format_measure_target(&measure, &mode, true) {
        Ok(target) => {
            let (_, errors) = score::validate_measure_target(&target, &mode, None, None);
            if errors.is_empty() {
                Some(target)
            } else {
                issue(row, "ir_unresolved", json!({"detail":errors.join("; ")}));
                None
            }
        }
        Err(e) => {
            issue(row, "ir_unresolved", json!({"detail":e}));
            None
        }
    }
}

/// Project the known music of an incomplete observation for an editable export.
/// Pitches remain written pitches: the product applies its shared pitch-context
/// conversion before encoding. This never changes the observation or its IR.
/// Rests inserted here occupy unknown spans in the export, not recognized rests.
pub fn export_measure(record: &Value) -> Result<Value, String> {
    let resolved = &record["resolved"];
    let signature = record["score_state"]["time"]
        .as_str()
        .or_else(|| known(&resolved["state"], "time").as_str());
    let length = signature
        .and_then(|time| time.split_once('/'))
        .and_then(|(n, d)| n.parse::<i64>().ok().zip(d.parse::<i64>().ok()))
        .filter(|(n, d)| *n > 0 && *d > 0)
        .map(|(n, d)| 3840 * n / d);
    let mut omitted_notes = 0;
    let mut omitted_events = 0;
    let mut tab_preferred_events = 0;
    let mut voices = Vec::new();
    let complete_position = |note: &Value| {
        note["string"].as_i64().is_some_and(|s| s > 0)
            && (note["fret"].as_i64().is_some_and(|f| f >= 0) || note["fret"] == "x")
    };
    for voice in array(&resolved["voices"]) {
        if array(&voice["events"]).is_empty() {
            voices.push(json!({"voice":voice["voice"],"events":[]}));
            continue;
        }
        let mut events = Vec::new();
        for (index, event) in array(&voice["events"]).iter().enumerate() {
            let (Some(start), Some(rhythm)) = (
                event["start"].as_i64().filter(|s| *s >= 0),
                duration(&event["duration"]),
            ) else {
                omitted_events += 1;
                omitted_notes += array(&event["notes"]).len();
                continue;
            };
            if !matches!(event["status"].as_str(), Some("normal" | "rest" | "empty")) {
                omitted_events += 1;
                omitted_notes += array(&event["notes"]).len();
                continue;
            }
            let conflict = record["mode"] == "both"
                && array(&record["unresolved"]).iter().any(|issue| {
                    issue["reason"] == "dual_note_pairing_unresolved"
                        && issue["voice"] == voice["voice"]
                        && issue["event"] == index
                });
            let prefer_tab = conflict && array(&event["notes"]).iter().any(complete_position);
            if prefer_tab {
                tab_preferred_events += 1;
            }
            let mut notes = Vec::new();
            for note in array(&event["notes"]) {
                let position = complete_position(note);
                // An unmatched notation reading describes the same chord, not
                // additional notes to stack on top of its visible TAB fingering.
                if prefer_tab && note.get("string").is_none() {
                    continue;
                }
                let pitch = note["pitch"].as_i64().filter(|p| (0..=127).contains(p));
                if !position && (pitch.is_none() || prefer_tab) {
                    omitted_notes += 1;
                    continue;
                }
                let mut projected =
                    json!({"effects":note.get("effects").cloned().unwrap_or(json!([]))});
                if position {
                    projected["string"] = note["string"].clone();
                    projected["fret"] = note["fret"].clone();
                }
                if !prefer_tab {
                    if let Some(pitch) = pitch {
                        projected["pitch"] = json!(pitch);
                    }
                }
                for key in ["velocity", "swap_accidentals"] {
                    if let Some(value) = note.get(key) {
                        projected[key] = value.clone();
                    }
                }
                notes.push(projected);
            }
            let status = if event["status"] == "normal" && notes.is_empty() {
                omitted_events += 1;
                json!("rest")
            } else {
                event["status"].clone()
            };
            events.push(json!({"start":start,"duration":rhythm,"status":status,
                "notes":notes,"effects":event.get("effects").cloned().unwrap_or(json!([]))}));
        }
        events.sort_by_key(|event| event["start"].as_i64().unwrap());
        let mut occupied = Vec::new();
        let mut cursor = 0;
        for event in events {
            let start = event["start"].as_i64().unwrap();
            let ticks = score::duration_ticks(&event["duration"])?;
            // Keep original durations here; export_rhythm bounds any overlap
            // at the next attack without moving notes.
            let end = if event["status"] == "empty" {
                start
            } else {
                start + (ticks.numerator / ticks.denominator) as i64
            };
            cursor = cursor.max(end);
            occupied.push(event);
        }
        if let Some(length) = length.filter(|length| *length > cursor) {
            for mut rest in score::rest_events(length - cursor)? {
                rest["start"] = json!(cursor + rest["start"].as_i64().unwrap());
                occupied.push(rest);
            }
        }
        voices.push(json!({"voice":voice["voice"],"events":occupied}));
    }
    if voices.is_empty() {
        voices.push(json!({"voice":0,"events":[]}));
    }
    if voices
        .iter()
        .all(|voice| array(&voice["events"]).is_empty())
        && omitted_events == 0
    {
        omitted_events = 1;
    }
    let mut warnings = Vec::new();
    if omitted_notes > 0 {
        warnings.push(format!(
            "{omitted_notes} 个音符尚未确定，已保留同拍的其他音符"
        ));
    }
    if omitted_events > 0 {
        warnings.push(format!(
            "{omitted_events} 处音符或节奏尚未确定，空缺用占位休止保留时间"
        ));
    }
    if tab_preferred_events > 0 {
        warnings.push(format!(
            "{tab_preferred_events} 处双谱读数不一致，导出采用已识别的 TAB 指法"
        ));
    }
    Ok(
        json!({"measure":{"voices":voices,"time_signature":signature,
        "print_time_signature":resolved["state"]["time"]["status"]=="observed"},
        "omitted_notes":omitted_notes,"omitted_events":omitted_events,
        "tab_preferred_events":tab_preferred_events,"warnings":warnings}),
    )
}

/// Make an explicitly approximate file projection while retaining every known
/// attack. A later attack bounds an overlapping duration in the same voice.
/// Overflow stays in its source bar; valid pickup measures are not padded.
pub fn export_rhythm(measure: &Value, time_signature: &str) -> Result<Value, String> {
    let (n, d) = time_signature
        .split_once('/')
        .ok_or("Missing export meter")?;
    let n = n.parse::<i64>().map_err(|_| "Invalid export meter")?;
    let d = d.parse::<i64>().map_err(|_| "Invalid export meter")?;
    if n <= 0 || d <= 0 {
        return Err("Invalid export meter".into());
    }
    let length = 3840 * n / d;
    let mut projected = measure.clone();
    let voices = projected["voices"]
        .as_array_mut()
        .ok_or("Missing export voices")?;
    let all_empty = voices
        .iter()
        .all(|voice| array(&voice["events"]).is_empty());
    if voices.is_empty() {
        voices.push(json!({"voice":0,"events":[]}));
    }
    let (mut adjusted_events, mut split_events, mut overflow_events) = (0, 0, 0);
    for (vi, voice) in voices.iter_mut().enumerate() {
        let mut events = array(&voice["events"]).to_vec();
        if events.is_empty() {
            if all_empty && vi == 0 {
                voice["events"] = json!(score::rest_events(length)?);
            }
            continue;
        }
        events.sort_by_key(|event| event["start"].as_i64().unwrap_or(0));
        let mut out = Vec::new();
        let mut cursor = 0;
        for (index, event) in events.iter().enumerate() {
            let start = event["start"].as_i64().ok_or("Missing event onset")?;
            if start < cursor {
                return Err("Simultaneous events in one voice need distinct note ownership".into());
            }
            if start > cursor {
                for mut rest in score::rest_events(start - cursor)? {
                    rest["start"] = json!(cursor + rest["start"].as_i64().unwrap());
                    out.push(rest);
                }
            }
            let ticks = score::duration_ticks(&event["duration"])?;
            let next = events.get(index + 1).and_then(|e| e["start"].as_i64());
            let shortened = event["status"] != "empty"
                && next.is_some_and(|next| {
                    i128::from(next - start) * ticks.denominator < ticks.numerator
                });
            let pieces = if shortened {
                let span = next.unwrap() - start;
                if span <= 0 {
                    return Err(
                        "Simultaneous events in one voice need distinct note ownership".into(),
                    );
                }
                adjusted_events += 1;
                let pieces = score::rest_events(span)?;
                if pieces.len() > 1 {
                    split_events += 1;
                }
                pieces
            } else {
                vec![json!({"start":0,"duration":event["duration"]})]
            };
            let mut end = start;
            for (piece_index, piece) in pieces.iter().enumerate() {
                let mut retained = event.clone();
                retained["start"] = json!(start + piece["start"].as_i64().unwrap());
                retained["duration"] = piece["duration"].clone();
                if pieces.len() > 1 {
                    let last = piece_index + 1 == pieces.len();
                    if piece_index > 0 {
                        retained["effects"] = json!(array(&event["effects"])
                            .iter()
                            .filter(|e| e.as_str().is_some_and(|s| s.starts_with("ottava:")))
                            .cloned()
                            .collect::<Vec<_>>());
                        // A dead note is an attack, not a pitch to sustain.
                        // Later duration fragments must not strike it again.
                        let notes = retained["notes"].as_array_mut().ok_or("Missing notes")?;
                        notes.retain(|note| {
                            note["fret"] != "x" && !array(&note["effects"]).contains(&json!("dead"))
                        });
                        if notes.is_empty() {
                            retained["status"] = json!("rest");
                        }
                    }
                    for note in retained["notes"].as_array_mut().ok_or("Missing notes")? {
                        let effects = array(&note["effects"]).to_vec();
                        let outgoing = |effect: &&Value| {
                            matches!(
                                effect.as_str(),
                                Some("sl" | "ss" | "hammer" | "sou" | "sod")
                            )
                        };
                        let mut kept = if piece_index == 0 {
                            effects
                                .iter()
                                .filter(|e| !outgoing(e))
                                .cloned()
                                .collect::<Vec<_>>()
                        } else {
                            effects
                                .iter()
                                .filter(|e| {
                                    e.as_str().is_some_and(|s| {
                                        matches!(s, "pm" | "let" | "vib" | "trem")
                                            || s.starts_with("harm:")
                                            || s.starts_with("trem:")
                                    })
                                })
                                .cloned()
                                .collect::<Vec<_>>()
                        };
                        if piece_index > 0 && !kept.contains(&json!("tie")) {
                            kept.push(json!("tie"));
                        }
                        if last {
                            kept.extend(effects.iter().filter(outgoing).cloned());
                        }
                        note["effects"] = json!(kept);
                    }
                }
                let ticks = score::duration_ticks(&retained["duration"])?;
                end = retained["start"].as_i64().unwrap()
                    + if event["status"] == "empty" {
                        0
                    } else {
                        (ticks.numerator / ticks.denominator) as i64
                    };
                out.push(retained);
            }
            if end > length {
                overflow_events += 1;
            }
            cursor = end;
        }
        voice["events"] = json!(out);
    }
    let mut warnings = Vec::new();
    if adjusted_events > 0 {
        warnings.push(format!(
            "{adjusted_events} 个重叠事件按下一音的起点截短，已保留所有音符起点"
        ));
    }
    if split_events > 0 {
        warnings.push(format!("{split_events} 个时值用分段时值表示"));
    }
    if overflow_events > 0 {
        warnings.push(format!(
            "{overflow_events} 个事件超过拍号长度，已保留在本节原有时间位置"
        ));
    }
    Ok(
        json!({"measure":projected,"adjusted_events":adjusted_events,
        "split_events":split_events,"overflow_events":overflow_events,"warnings":warnings}),
    )
}
// A manual edit may change values while retaining visual relation anchors.
// Structural edits invalidate those anchors; the original observation is still
// retained verbatim, but its old indices must not be applied to different notes.
fn same_anchors(observed: &Value, manual: &Value) -> bool {
    let (a, b) = (array(observed), array(manual));
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            let (ae, be) = (array(&a["events"]), array(&b["events"]));
            a["voice"] == b["voice"]
                && ae.len() == be.len()
                && ae.iter().zip(be).all(|(a, b)| {
                    let (an, bn) = (array(&a["notes"]), array(&b["notes"]));
                    a["start"] == b["start"]
                        // Manual IR has merged notes; independent notation
                        // indices no longer address that array.
                        && array(&a["notation_notes"]).is_empty()
                        && an.len() == bn.len()
                        && an.iter().zip(bn).all(|(a, b)| a["string"] == b["string"])
                })
        })
}
fn validate_manual_voices(voices: &Value) -> Result<(), String> {
    let voices = voices
        .as_array()
        .ok_or("Manual resolved_voices must be an array")?;
    let mut seen = HashSet::new();
    for voice in voices {
        let local = voice["voice"]
            .as_i64()
            .ok_or("Manual voice identifiers must be integers")?;
        if !seen.insert(local) {
            return Err("Manual voice identifiers must be distinct".into());
        }
        let events = voice["events"]
            .as_array()
            .ok_or("Manual events must be an array")?;
        for event in events {
            if !event.is_object()
                || !nullable_integer(&event["start"])
                || duration(&event["duration"]).is_none()
                || !matches!(event["status"].as_str(), Some("normal" | "rest" | "empty"))
            {
                return Err("Manual events require valid rhythm and status".into());
            }
            if !event["notes"].is_array() || array(&event["notes"]).iter().any(|n| !n.is_object()) {
                return Err("Manual notes must be an array of objects".into());
            }
            for item in std::iter::once(event).chain(array(&event["notes"])) {
                if item.get("effects").is_some_and(|effects| {
                    !effects.is_array() || array(effects).iter().any(|e| !e.is_string())
                }) {
                    return Err("Manual effects must be an array of strings".into());
                }
            }
        }
    }
    Ok(())
}
/// Assemble observations in score order without modifying the input evidence.
pub fn assemble(records: &[Value], auxiliary: Option<&[Value]>) -> Result<Value, String> {
    if auxiliary.is_some_and(|a| a.len() != records.len()) {
        return Err("Auxiliary observations must align with records".into());
    }
    let mut rows = Vec::new();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let mut value = json!({});
        for key in ["source_id", "part_id", "staff_id", "bar_index", "mode"] {
            value[key] = record
                .get(key)
                .ok_or_else(|| format!("Record requires {key}"))?
                .clone();
        }
        if !integer(&value["bar_index"]) {
            return Err("Record bar_index must be an integer".into());
        }
        if !matches!(value["mode"].as_str(), Some("tab" | "notation" | "both")) {
            return Err("Record mode must be tab, notation or both".into());
        }
        for key in LAYOUT_FIELDS {
            if let Some(v) = record.get(key) {
                value[key] = v.clone();
            }
        }
        if let Some(instrument) = record.get("instrument") {
            value["instrument"] = instrument.clone();
        }
        value["observation"] =
            parse_measure_observation(&record["observation"], value["mode"].as_str().unwrap())?;
        value["pitch_reference"] = json!("written");
        value["resolved"] = json!({"voices":value["observation"]["voices"],"state":{},"visual_techniques":[],"pitch_reference":"written"});
        value["unresolved"] = json!([]);
        value["ir"] = Value::Null;
        let extra = auxiliary.and_then(|a| a.get(index)).unwrap_or(&Value::Null);
        let lane = json!([
            value["source_id"],
            value["part_id"],
            value["staff_id"],
            value["mode"]
        ])
        .to_string();
        if let Some((_, indices)) = groups.iter_mut().find(|(k, _)| *k == lane) {
            indices.push(index);
        } else {
            groups.push((lane, vec![index]));
        }
        let mut relations = value["observation"]["relations"].clone();
        if let Some(manual) = extra.get("resolved_voices") {
            validate_manual_voices(manual)?;
            if !array(&relations).is_empty()
                && !same_anchors(&value["observation"]["voices"], manual)
            {
                relations = json!([]);
                value["unresolved"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"reason":"manual_relation_anchors_changed"}));
            }
        }
        let mut row = Row {
            value,
            voice_map: extra.get("voice_map").cloned().unwrap_or(json!({})),
            voice_links: BTreeMap::new(),
            relations,
        };
        independent_dual_notes(&mut row);
        rows.push(row);
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut links: BTreeMap<usize, usize> = BTreeMap::new();
    let mut order = HashMap::new();
    let mut following = HashMap::new();
    for (gi, (_, indices)) in groups.iter_mut().enumerate() {
        indices.sort_by_key(|&i| rows[i].value["bar_index"].as_i64().unwrap());
        let mut counts = HashMap::new();
        for &i in indices.iter() {
            *counts
                .entry(rows[i].value["bar_index"].as_i64().unwrap())
                .or_insert(0) += 1;
        }
        let mut previous = json!({});
        let mut last_bar: Option<i64> = None;
        let mut outgoing: Vec<usize> = Vec::new();
        for (position, &index) in indices.iter().enumerate() {
            order.insert(index, (gi, position));
            let bar = rows[index].value["bar_index"].as_i64().unwrap();
            let unique = counts[&bar] == 1;
            let contiguous = unique && last_bar.and_then(|b| b.checked_add(1)) == Some(bar);
            if contiguous && counts[&last_bar.unwrap()] == 1 {
                following.insert(indices[position - 1], index);
            }
            if !contiguous {
                previous = json!({});
                outgoing.clear();
            }
            if !unique {
                issue(&mut rows[index], "duplicate_bar_identity", json!({}));
            }
            let extra = auxiliary.and_then(|a| a.get(index)).unwrap_or(&Value::Null);
            for (origin, values) in [
                (
                    "observation",
                    rows[index].value["observation"]["state"].clone(),
                ),
                ("auxiliary", extra["state"].clone()),
            ] {
                if let Some(values) = values.as_object() {
                    for (field, fact) in values {
                        if !valid_state_value(field, &fact["value"]) {
                            issue(
                                &mut rows[index],
                                "state_value_invalid",
                                json!({"field":field,"source":origin,"value":fact["value"]}),
                            );
                        }
                    }
                }
            }
            rows[index].value["resolved"]["state"] = state(
                &rows[index].value["observation"]["state"],
                &extra["state"],
                &previous,
            );
            // An explicitly identified complete score header with no key
            // symbols establishes zero alterations. A crop starting midway
            // through a score cannot establish that fact.
            let key_fact = &extra["state"]["key"];
            let key_absent = if key_fact["source"] == "manual"
                || (key_fact["source"] == "visual_classifier"
                    && key_fact["calibrated"] == true
                    && matches!(key_fact["status"].as_str(), Some("observed" | "absent")))
            {
                key_fact["status"] == "absent"
            } else {
                rows[index].value["observation"]["state"]["key"]["status"] == "absent"
            };
            if bar == 0
                && extra["staff_start"] == true
                && key_absent
                && known(&rows[index].value["resolved"]["state"], "key").is_null()
            {
                rows[index].value["resolved"]["state"]["key"] =
                    json!({"status":"observed","value":0,"reason":"empty_initial_signature"});
            }
            previous = rows[index].value["resolved"]["state"].clone();
            last_bar = Some(bar);
            let relations = array(&rows[index].relations).to_vec();
            let mut local = Vec::new();
            let mut incoming_notes = HashSet::new();
            for (ri, r) in relations.into_iter().enumerate() {
                let problem = r["observation_problem"]
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| relation_problem(&rows[index].value["resolved"], &r));
                nodes.push(Node {
                    record: index,
                    index: ri,
                    kind: r["kind"].clone(),
                    value: r["value"].clone(),
                    lane: r["lane"].clone(),
                    relation: r.clone(),
                    problem: problem.clone(),
                });
                if let Some(p) = problem {
                    issue(&mut rows[index], &p, json!({"relation":ri}));
                    continue;
                }
                local.push(nodes.len() - 1);
                let end = &r["end"];
                let start = &r["start"];
                if (r["kind"].is_null() || r["kind"] == "tie")
                    && r["shape"] != "diagonal"
                    && array(end).len() == 3
                    && (start == "left"
                        || (array(start).len() == 3
                            && start[0] == end[0]
                            && connection_problem(
                                &global_ref(index, start),
                                &global_ref(index, end),
                                &rows,
                                Some("tie"),
                            )
                            .is_none()))
                {
                    incoming_notes.insert((
                        end[0].as_i64().unwrap(),
                        end[1].as_u64().unwrap() as usize,
                        end[2].as_u64().unwrap() as usize,
                    ));
                }
            }
            pitch_notes(&mut rows[index], &incoming_notes);
            if let Some(manual) = extra.get("resolved_voices") {
                rows[index].value["resolved"]["voices"] = manual.clone();
            }
            let incoming: Vec<_> = local
                .iter()
                .copied()
                .filter(|&n| nodes[n].relation["start"] == "left")
                .collect();
            boundary_voices(&outgoing, &incoming, &nodes, &mut rows);
            let candidates: BTreeMap<usize, Vec<usize>> = incoming
                .iter()
                .map(|&n| {
                    (
                        n,
                        outgoing
                            .iter()
                            .copied()
                            .filter(|&p| compatible(&nodes[p], &nodes[n], &rows))
                            .collect(),
                    )
                })
                .collect();
            let mut matches: BTreeMap<usize, usize> = candidates
                .iter()
                .filter_map(|(&n, prior)| {
                    if prior.len() == 1
                        && candidates
                            .values()
                            .filter(|v| v.contains(&prior[0]))
                            .count()
                            == 1
                    {
                        Some((n, prior[0]))
                    } else {
                        None
                    }
                })
                .collect();
            let mut left: Vec<_> = outgoing
                .iter()
                .copied()
                .filter(|p| {
                    !matches.values().any(|v| v == p) && range_relation(&nodes[*p].relation)
                })
                .collect();
            let mut right: Vec<_> = incoming
                .iter()
                .copied()
                .filter(|n| !matches.contains_key(n) && range_relation(&nodes[*n].relation))
                .collect();
            if left.len() == right.len() && left.len() > 1 {
                let prior = &rows[nodes[left[0]].record].value;
                let row = &rows[index].value;
                let page = ["source_page", "page"]
                    .into_iter()
                    .find(|k| !prior[k].is_null() && !row[k].is_null());
                let same = page.is_some_and(|p| {
                    !row["system_index"].is_null()
                        && prior["system_index"] == row["system_index"]
                        && prior[p] == row[p]
                });
                let distinct = |side: &[usize]| {
                    let vals: Vec<_> = side
                        .iter()
                        .filter_map(|&n| nodes[n].lane.as_f64())
                        .collect();
                    vals.len() == side.len()
                        && vals.iter().enumerate().all(|(i, v)| !vals[..i].contains(v))
                };
                if same && distinct(&left) && distinct(&right) {
                    left.sort_by(|&a, &b| {
                        nodes[a]
                            .lane
                            .as_f64()
                            .unwrap()
                            .total_cmp(&nodes[b].lane.as_f64().unwrap())
                    });
                    right.sort_by(|&a, &b| {
                        nodes[a]
                            .lane
                            .as_f64()
                            .unwrap()
                            .total_cmp(&nodes[b].lane.as_f64().unwrap())
                    });
                    if left
                        .iter()
                        .zip(&right)
                        .all(|(p, n)| candidates[n].contains(p))
                    {
                        for (p, n) in left.into_iter().zip(right) {
                            matches.insert(n, p);
                        }
                    }
                }
            }
            for (n, p) in matches {
                links.insert(p, n);
                if nodes[n].kind.is_null() {
                    nodes[n].kind = nodes[p].kind.clone();
                }
                if nodes[n].value.is_null() {
                    nodes[n].value = nodes[p].value.clone();
                }
            }
            outgoing = local
                .into_iter()
                .filter(|&n| nodes[n].relation["end"] == "right")
                .collect();
            if !unique {
                previous = json!({});
                last_bar = None;
                outgoing.clear();
            }
        }
    }
    let linked: HashSet<_> = links.values().copied().collect();
    let mut chains = Vec::new();
    for root in (0..nodes.len()).filter(|i| !linked.contains(i)) {
        let mut connected = vec![nodes[root].clone()];
        let mut current = root;
        while let Some(&next) = links.get(&current) {
            connected.push(nodes[next].clone());
            current = next;
        }
        let mut kinds = Vec::new();
        let mut values = Vec::new();
        for n in &connected {
            for (list, key) in [(&mut kinds, "kind"), (&mut values, "value")] {
                let v = &n.relation[key];
                if !v.is_null() && !list.contains(v) {
                    list.push(v.clone());
                }
            }
        }
        let first = &connected[0];
        let last = connected.last().unwrap();
        let start = &first.relation["start"];
        let end = &last.relation["end"];
        let segments: Vec<_> = connected
            .iter()
            .map(|n| json!({"record":n.record,"relation":n.index}))
            .collect();
        let mut chain = json!({"kind":if kinds.len()==1{kinds[0].clone()}else{Value::Null},"value":if values.len()==1{values[0].clone()}else{Value::Null},"start":global_ref(first.record,start),"end":global_ref(last.record,end),"segments":segments,"coverage":[],"status":"resolved"});
        if let Some(problem) = &first.problem {
            chain["status"] = json!("unresolved");
            chain["reason"] = json!(problem);
            chains.push(chain);
            continue;
        }
        for pair in connected.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if let (Some(av), Some(bv)) = (relation_voice(&a.relation), relation_voice(&b.relation))
            {
                if let Some(evidence) = rows[b.record].voice_links.get(&(a.record, av, bv)) {
                    if chain.get("voice_links").is_none() {
                        chain["voice_links"] = json!([]);
                    }
                    chain["voice_links"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"from":[a.record,av],"to":[b.record,bv],"evidence":evidence}));
                }
            }
        }
        if range_relation(&first.relation) {
            let valid = apply_range(&mut chain, &connected, &mut rows);
            if chain["kind"].is_null() {
                chain["status"] = json!("unknown");
            } else if !valid {
                chain["status"] = json!("unresolved");
            } else if start == "left" || end == "right" {
                chain["status"] = json!("open");
            }
        } else if matches!(chain["kind"].as_str(), Some("hammer" | "sl" | "ss")) {
            slide_destination(&mut chain, &rows, &following);
        } else if chain["kind"].is_null() {
            chain["status"] = json!("unknown");
            for n in &connected {
                issue(
                    &mut rows[n.record],
                    "relation_kind_unresolved",
                    json!({"relation":n.index}),
                );
            }
        }
        chains.push(chain);
    }
    let mut ties: Vec<_> = (0..chains.len())
        .filter(|&i| chains[i]["kind"] == "tie" && chains[i]["status"] != "unresolved")
        .collect();
    ties.sort_by_key(|&i| {
        let s = &chains[i]["start"];
        if s.is_array() {
            let o = order[&(s[0].as_u64().unwrap() as usize)];
            (
                o.0 as i64,
                o.1 as i64,
                array(s)
                    .iter()
                    .skip(1)
                    .map(|v| v.as_i64().unwrap())
                    .collect::<Vec<_>>(),
            )
        } else {
            (-1, -1, vec![])
        }
    });
    for row in &mut rows {
        exchange_dual_pitches(row);
    }
    for i in ties {
        if let Some(reason) = tie(&chains[i], &mut rows) {
            chains[i]["status"] = json!(if reason == "tie_value_unresolved" {
                "established"
            } else {
                "unresolved"
            });
            chains[i]["reason"] = json!(reason);
            for segment in array(&chains[i]["segments"]) {
                issue(
                    &mut rows[segment["record"].as_u64().unwrap() as usize],
                    &reason,
                    json!({"relation":segment["relation"]}),
                );
            }
        }
    }
    // Ties can supply the source or destination pitch of a visible technique.
    // Apply these effects only after that propagation, while anchors are local.
    for chain in &mut chains {
        if !matches!(chain["kind"].as_str(), Some("hammer" | "sl" | "ss"))
            || chain["status"] == "unresolved"
        {
            continue;
        }
        if let Some(reason) = note_technique(chain, &mut rows) {
            chain["status"] = json!(if chain["end"] == "right" {
                "open"
            } else {
                "unresolved"
            });
            chain["reason"] = json!(reason);
            for segment in array(&chain["segments"]) {
                issue(
                    &mut rows[segment["record"].as_u64().unwrap() as usize],
                    &reason,
                    json!({"relation":segment["relation"]}),
                );
            }
        }
    }
    let note_maps: Vec<_> = rows.iter_mut().map(merge_dual_notes).collect();
    for chain in &mut chains {
        for key in ["start", "end"] {
            let endpoint = chain[key].clone();
            if array(&endpoint).len() == 4 {
                let ri = endpoint[0].as_u64().unwrap() as usize;
                let anchor = (
                    endpoint[1].as_i64().unwrap(),
                    endpoint[2].as_u64().unwrap() as usize,
                    endpoint[3].as_u64().unwrap() as usize,
                );
                if let Some(notes) = note_maps[ri].get(&anchor) {
                    if notes.len() == 1 {
                        chain[key][3] = json!(notes[0]);
                    } else {
                        chain[format!("{key}_notes")] = json!(notes
                            .iter()
                            .map(|&ni| json!([ri, anchor.0, anchor.1, ni]))
                            .collect::<Vec<_>>());
                        chain[key] = Value::Null;
                    }
                }
            }
        }
    }
    for row in &mut rows {
        row.value["ir"] = json!(ir(row));
    }
    Ok(
        json!({"records":rows.into_iter().map(|r|r.value).collect::<Vec<_>>(),"relations":chains,"pitch_reference":"written"}),
    )
}

/// Exact training prompt, shared by product inference and research.
pub fn observation_prompt(mode: &str) -> Result<String, String> {
    if !["tab", "notation", "both"].contains(&mode) {
        return Err(format!("Unsupported observation mode: {mode}"));
    }
    Ok(concat!(
        "Read the visible contents of this {mode} measure image. ",
        "Return one JSON object with state, voices and relations. ",
        "Do not infer information outside the image. ",
        "state contains printed changes: each field has status observed, absent, or unknown and value. ",
        "absent means clearly not printed; unknown means unreadable. Omitted state fields describe no local change; use explicit unknown for an unreadable printed mark. ",
        "State fields are time (N/D), key (signed number of sharps or flats, -7..7), clef (G2, F4, C3, C4, percussion or tab), and clef_octave (0, -12, 12, -24 or 24, from the small printed clef modifier). ",
        "voices contains local voice groups with voice and events. ",
        "Each event has start (ticks, quarter=960), duration (w,h,q,e,s,t,f, optional dots or [3:2]), status (normal or rest), notes and effects. ",
        "Use null for an unreadable start or duration. ",
        "Read string and fret from TAB; use null for an unprinted fret, never x unless a dead-note cross is visible. ",
        "Read the target measure between its barlines; neighbouring music in the crop margin does not belong to it. ",
        "Chord fingering grids, their dots, finger numbers, starting frets and mute/open-string symbols are annotations, never played notes or rhythmic events. ",
        "For a visible chord name aligned with an event in this measure, put chord:NAME in that event's effects; percent-encode reserved characters, for example chord:F%23 or chord:B/F%23. ",
        "Read the printed name, not a name inferred from the played notes or fingering. A grid without a readable name does not justify a chord name. ",
        "Do not attach a neighbouring measure's chord or a page-top chord dictionary to an event. Never add notes or change their frets to match a chord diagram. ",
        "Do not output diagram effects: the separate diagram reader supplies fingering geometry. ",
        "Read staff_position from notation: bottom line=0, one step per line or space upwards. ",
        "In both mode, each event has two independent arrays: notes contains only TAB string/fret/effects, and notation_notes contains only notation staff_position/accidental/notehead/effects. Read each array from top to bottom on its own staff; never pair notes across the two representations, and never create a duplicate voice or event for the second staff. Include both arrays even for rests. State clef and key refer to the notation staff, not the TAB letters. ",
        "The two arrays need not have equal length. Omit a note from a representation when neither its glyph nor a connection endpoint is visible there. Use a null fret only for an otherwise visibly located TAB note. Count a completely overlapping notation head once; do not invent hidden copies. ",
        "accidental is null if not printed, an integer -2..2 if printed, or unknown if unreadable. ",
        "For percussion, retain staff_position and the visible notehead shape: normal, x, circle-x, circle-slash, diamond, heavy-x, heavy-x-hat or triangle-up. Use null for an unreadable shape; a percussion x is an instrument notehead, not a muted guitar note. ",
        "Do not output MIDI pitch or a global voice identity. ",
        "Include only visible effects using IR names. ",
        "A printed H/P curve uses hammer on its origin note. ",
        "A fully visible local H/P connection already recorded as a note effect does not need a duplicate relation. ",
        "A downstroke staple uses event effect pick_down; an upstroke V uses pick_up. These picking signs are not chord names or dynamics. ",
        "A slide between two notes uses sl when a legato arc is printed, or ss for a plain shift slide. ",
        "A short diagonal entering a note from below uses sib, entering from above sia; leaving a note downwards uses sod, leaving upwards sou. ",
        "Place these on the note they touch; an entering or leaving slide need not connect to another note. ",
        "When an H/P or between-note slide crosses the crop edge, describe its visible fragment as a relation with kind hammer, sl or ss instead of assigning an effect to a hidden origin note. ",
        "Keep kind null when the fragment does not identify the technique. ",
        "Use stacc for a staccato dot, accent for an accent mark, heavy for a strong accent and vib for a vibrato mark. ",
        "H/P, slides and sustain ties are different visible marks and can coexist with a technique extension line. ",
        "relations describes visible tie curves, technique labels and extension lines. ",
        "Each has kind (tie, hammer, sl, ss, pm, let, ottava, or null when its label/type is not visible), start and end. ",
        "A note endpoint is [voice,event_index,note_index]; a span endpoint is [voice,event_index]. ",
        "Indices are zero-based and refer to the events and notes you output. ",
        "In both mode, a note connection also has representation tab or notation, identifying the staff where its visible mark is drawn; its note_index addresses notes or notation_notes respectively. Keep marks on each staff independent even when their music appears equivalent. Event-only spans need no representation. ",
        "Rests and tied continuations with hidden frets also count as events; note_index is the position in that event's notes array, not a string number or staff position. ",
        "Include a relation only when its curve, technique label or extension line is visible. ",
        "A PM/let label without a visible extension line identifies its marked start event; its end is null. ",
        "For a labelled extension, its start is the event marked by the label, not the event under the first dash after the text. ",
        "Count each visible line once; separate lines require separate visible evidence. ",
        "Use shape curve, diagonal or extension when visible to distinguish unnamed note connections from scope lines. ",
        "If a label or line has ambiguous voice or endpoint ownership, retain the mark with null ownership; do not choose using an unseen measure. ",
        "List incoming relations first, then local relations in event/note order, then outgoing relations. ",
        "Follow the visible geometry: use left or right where a line continues beyond the image; use an event endpoint where the line ends visibly inside it. ",
        "For an open endpoint, include the local voice and visible lane when known (top staff line=0, bottom line=1; lines above the staff have negative lane). ",
        "ottava value is the printed semitone shift. ",
        "Preserve continuations without guessing their hidden origin or type. ",
        "Do not copy a tied note's missing fret or an earlier technique label. ",
        "Omit properties that do not apply. ",
        "No explanatory text.",
    ).replace("{mode}", mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_export_keeps_known_chord_members_and_later_onsets() {
        let row = json!({"source_id":"score","part_id":0,"staff_id":0,"bar_index":0,"mode":"tab",
            "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
            "relations":[],"voices":[{"voice":0,"events":[
                {"start":0,"duration":"q","status":"normal","notes":[{"string":1,"fret":5},{"string":2,"fret":null}]},
                {"start":960,"duration":null,"status":"normal","notes":[{"string":3,"fret":null}]},
                {"start":1920,"duration":"q","status":"normal","notes":[{"string":2,"fret":7}]}]}]}});
        let assembled = assemble(&[row], None).unwrap();
        let record = &assembled["records"][0];
        let original = record.clone();
        assert!(record["ir"].is_null());
        let projection = export_measure(record).unwrap();
        let rhythm = export_rhythm(&projection["measure"], "4/4").unwrap();
        let events = array(&rhythm["measure"]["voices"][0]["events"]);
        assert_eq!(events.len(), 4);
        assert_eq!(
            events[0]["notes"],
            json!([{"string":1,"fret":5,"effects":[]}])
        );
        assert_eq!(events[1]["start"], 960);
        assert_eq!(events[1]["status"], "rest");
        assert_eq!(events[2]["start"], 1920);
        assert_eq!(events[2]["notes"][0]["fret"], 7);
        assert_eq!(events[3]["start"], 2880);
        assert_eq!(projection["omitted_notes"], 2);
        assert_eq!(projection["omitted_events"], 1);
        assert_eq!(record, &original);
    }

    #[test]
    fn partial_dual_export_uses_tab_without_doubling_the_chord() {
        let record = json!({"mode":"both","score_state":{"time":"4/4"},
            "unresolved":[{"reason":"dual_note_pairing_unresolved","voice":0,"event":0}],
            "resolved":{"voices":[{"voice":0,"events":[
                {"start":0,"duration":"h","status":"normal","notes":[
                    {"string":1,"fret":5,"pitch":81,"effects":["pm"]},
                    {"string":2,"fret":null,"pitch":null},
                    {"staff_position":5,"pitch":79}]},
                {"start":1920,"duration":"h","status":"normal","notes":[{"pitch":76}]}]}]}});
        let projection = export_measure(&record).unwrap();
        let events = &projection["measure"]["voices"][0]["events"];
        assert_eq!(
            events[0]["notes"],
            json!([{"string":1,"fret":5,"effects":["pm"]}])
        );
        assert_eq!(events[1]["notes"], json!([{"pitch":76,"effects":[]}]));
        assert_eq!(projection["omitted_notes"], 1);
        assert_eq!(projection["tab_preferred_events"], 1);
    }

    #[test]
    fn rhythm_export_keeps_spring_attacks_and_reports_only_actual_empty_bars() {
        let measure = json!({"section":"Kept","voices":[{"voice":0,"events":[]},
            {"voice":1,"events":([240,360,480,720,1200,1680].into_iter().map(|start|
                json!({"start":start,"duration":{"value":8,"dotted":true},"status":"normal",
                    "notes":[{"string":2,"fret":17}],"effects":[]})).collect::<Vec<_>>())}]});
        let result = export_rhythm(&measure, "6/8").unwrap();
        assert_eq!(result["measure"]["section"], "Kept");
        assert_eq!(result["adjusted_events"], 5);
        assert_eq!(result["measure"]["voices"][0]["events"], json!([]));
        let notes = array(&result["measure"]["voices"][1]["events"])
            .iter()
            .filter(|event| event["status"] == "normal")
            .collect::<Vec<_>>();
        assert_eq!(
            notes
                .iter()
                .map(|event| event["start"].clone())
                .collect::<Vec<_>>(),
            vec![
                json!(240),
                json!(360),
                json!(480),
                json!(720),
                json!(1200),
                json!(1680)
            ]
        );
        assert_eq!(
            notes
                .iter()
                .map(|event| score::duration_ticks(&event["duration"]).unwrap().numerator)
                .collect::<Vec<_>>(),
            vec![120, 120, 240, 480, 480, 720]
        );
        let record =
            json!({"resolved":{"voices":[{"voice":0,"events":[]},{"voice":1,"events":[]}]}});
        let partial = export_measure(&record).unwrap();
        assert_eq!(partial["omitted_events"], 1);
        let empty = export_rhythm(&partial["measure"], "4/4").unwrap();
        assert!(!array(&empty["measure"]["voices"][0]["events"]).is_empty());
        assert!(array(&empty["measure"]["voices"][1]["events"]).is_empty());
    }

    #[test]
    fn connections_need_known_endpoints_but_slide_out_does_not() {
        let row = json!({"source_id":"score","part_id":0,"staff_id":0,"bar_index":0,"mode":"tab",
            "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
            "relations":[{"kind":"ss","start":[0,0,0],"end":"right"}],
            "voices":[{"voice":0,"events":[{"start":0,"duration":"w","status":"normal",
                "notes":[{"string":1,"fret":5,"effects":["sou"]}]}]}]}});
        let result = assemble(&[row.clone()], None).unwrap();
        let measure =
            score::parse_measure_target(result["records"][0]["ir"].as_str().unwrap()).unwrap();
        assert_eq!(
            measure["voices"][0]["events"][0]["notes"][0]["effects"],
            json!(["sou"])
        );
        assert_eq!(
            result["records"][0]["unresolved"][0]["reason"],
            "ss_endpoint_unresolved"
        );
        let mut next = row.clone();
        next["bar_index"] = json!(1);
        next["observation"]["relations"] = json!([{"kind":"ss","start":"left","end":[0,0,0]}]);
        next["observation"]["voices"][0]["events"][0]["notes"][0] = json!({"string":1,"fret":7});
        let linked = assemble(&[row.clone(), next], None).unwrap();
        let measure =
            score::parse_measure_target(linked["records"][0]["ir"].as_str().unwrap()).unwrap();
        assert!(
            array(&measure["voices"][0]["events"][0]["notes"][0]["effects"]).contains(&json!("ss"))
        );
        assert_eq!(linked["relations"][0]["status"], "resolved");

        // The visible relation remains, but omitting its unknown endpoint must
        // not turn the known note into an orphan tie or an unrelated slide.
        for kind in ["tie", "ss", "hammer"] {
            let mut partial = row.clone();
            partial["observation"]["voices"][0]["events"] = json!([
                {"start":0,"duration":"h","status":"normal","notes":[{"string":1,"fret":if kind=="tie"{Value::Null}else{json!(5)}}]},
                {"start":1920,"duration":"h","status":"normal","notes":[{"string":1,"fret":if kind=="tie"{json!(5)}else{Value::Null}}]}
            ]);
            partial["observation"]["relations"] =
                json!([{"kind":kind,"start":[0,0,0],"end":[0,1,0]}]);
            let result = assemble(&[partial], None).unwrap();
            assert_eq!(
                result["relations"][0]["reason"],
                format!("{kind}_value_unresolved")
            );
            let export = export_measure(&result["records"][0]).unwrap();
            let notes: Vec<_> = array(&export["measure"]["voices"][0]["events"])
                .iter()
                .flat_map(|event| array(&event["notes"]))
                .collect();
            assert_eq!(notes.len(), 1);
            assert_eq!(notes[0]["fret"], 5);
            assert!(!array(&notes[0]["effects"]).contains(&json!(kind)));
        }

        // A resolved incoming tie may establish the next slide's source value.
        let mut continued = row;
        continued["observation"]["voices"][0]["events"] = json!([
            {"start":0,"duration":"q","status":"normal","notes":[{"string":1,"fret":5}]},
            {"start":960,"duration":"q","status":"normal","notes":[{"string":1,"fret":null}]},
            {"start":1920,"duration":"h","status":"normal","notes":[{"string":1,"fret":7}]}
        ]);
        continued["observation"]["relations"] = json!([
            {"kind":"tie","start":[0,0,0],"end":[0,1,0]},
            {"kind":"ss","start":[0,1,0],"end":[0,2,0]}
        ]);
        let result = assemble(&[continued], None).unwrap();
        let target =
            score::parse_measure_target(result["records"][0]["ir"].as_str().unwrap()).unwrap();
        assert_eq!(target["voices"][0]["events"][1]["notes"][0]["fret"], 5);
        assert_eq!(
            target["voices"][0]["events"][1]["notes"][0]["effects"],
            json!(["tie", "ss"])
        );
    }

    #[test]
    fn chord_name_keeps_its_event_without_supplying_diagram_notes() {
        let observation = json!({"state":{"time":{"status":"observed","value":"4/4"}},
            "voices":[{"voice":0,"events":[{"start":0,"duration":"w","status":"normal",
                "notes":[{"string":6,"fret":2}],"effects":["chord:F%23","dyn:79"]}]}],
            "relations":[{"kind":"pm","start":[0,0],"end":[0,0]}]});
        parse_observation(&observation).unwrap();
        let result = assemble(
            &[json!({"source_id":"score","part_id":0,"staff_id":0,
            "bar_index":0,"mode":"tab","observation":observation})],
            None,
        )
        .unwrap();
        let measure =
            score::parse_measure_target(result["records"][0]["ir"].as_str().unwrap()).unwrap();
        let event = &measure["voices"][0]["events"][0];
        assert_eq!(array(&event["notes"]).len(), 1);
        assert_eq!(event["notes"][0]["string"], 6);
        assert_eq!(event["notes"][0]["fret"], 2);
        assert!(array(&event["notes"][0]["effects"]).contains(&json!("pm")));
        assert!(array(&event["effects"]).contains(&json!("chord:F%23")));
        assert!(array(&event["effects"]).contains(&json!("dyn:79")));
        let mut wrong_task = observation;
        wrong_task["voices"][0]["events"][0]["effects"] =
            json!(["diagram:1:2/4/4/3/2/2:1/3/4/2/1/1:-"]);
        assert!(parse_observation(&wrong_task).is_err());
    }

    #[test]
    fn manual_note_values_reassemble_but_changed_anchors_do_not_reuse_visual_indices() {
        let event = |start, fret| json!({"start":start,"duration":"h","status":"normal","notes":[{"string":1,"fret":fret}]});
        let records = vec![
            json!({"source_id":"score","part_id":0,"staff_id":0,"bar_index":0,"mode":"tab",
                "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
                "voices":[{"voice":0,"events":[event(0,Some(3)),event(1920,Some(3))]}],
                "relations":[{"kind":"tie","start":[0,1,0],"end":"right"}]}}),
            json!({"source_id":"score","part_id":0,"staff_id":0,"bar_index":1,"mode":"tab",
                "observation":{"state":{"time":{"status":"absent"}},
                "voices":[{"voice":0,"events":[event(0,None),event(1920,Some(3))]}],
                "relations":[{"kind":"tie","start":"left","end":[0,0,0]}]}}),
        ];
        let original = records.clone();
        let mut bad_clef = records.clone();
        bad_clef[0]["observation"]["state"]["clef"] = json!({"status":"observed","value":"1"});
        let partial_state = assemble(&bad_clef, None).unwrap();
        assert!(partial_state["records"][0]["resolved"]["state"]["clef"]["value"].is_null());
        assert!(partial_state["records"][0]["ir"].is_string());
        assert_eq!(
            partial_state["records"][0]["observation"]["state"]["clef"]["value"],
            "1"
        );
        assert!(array(&partial_state["records"][0]["unresolved"])
            .iter()
            .any(|v| v["reason"] == "state_value_invalid" && v["field"] == "clef"));
        let corrected_meter = assemble(
            &records,
            Some(&[
                json!({"state":{"time":{"status":"observed","value":"3/4","source":"manual"}}}),
                json!({}),
            ]),
        )
        .unwrap();
        assert_eq!(
            corrected_meter["records"][0]["resolved"]["state"]["time"],
            json!({"status":"observed","value":"3/4"})
        );
        assert_eq!(
            corrected_meter["records"][1]["resolved"]["state"]["time"],
            json!({"status":"inherited","value":"3/4"})
        );
        let mut manual = records[0]["observation"]["voices"].clone();
        manual[0]["events"][1]["notes"][0]["fret"] = json!(7);
        let updated = assemble(
            &records,
            Some(&[json!({"resolved_voices":manual}), json!({})]),
        )
        .unwrap();
        assert_eq!(
            updated["records"][1]["resolved"]["voices"][0]["events"][0]["notes"][0]["fret"],
            7
        );
        assert_eq!(updated["relations"][0]["status"], "resolved");
        assert_eq!(
            updated["records"][0]["observation"],
            original[0]["observation"]
        );
        manual[0]["events"].as_array_mut().unwrap().remove(0);
        let changed = assemble(
            &records,
            Some(&[json!({"resolved_voices":manual}), json!({})]),
        )
        .unwrap();
        assert!(
            changed["records"][1]["resolved"]["voices"][0]["events"][0]["notes"][0]["fret"]
                .is_null()
        );
        assert!(array(&changed["records"][0]["unresolved"])
            .iter()
            .any(|i| i["reason"] == "manual_relation_anchors_changed"));
        assert_eq!(
            changed["records"][0]["observation"],
            original[0]["observation"]
        );
        assert_eq!(records, original);

        let mut plain = records[0].clone();
        plain["observation"]["voices"] = json!([]);
        plain["observation"]["relations"] = json!([]);
        let edited = assemble(&[plain], Some(&[json!({"resolved_voices":manual})])).unwrap();
        assert!(edited["records"][0]["ir"].is_string());
        assert_eq!(edited["records"][0]["unresolved"], json!([]));

        let mut dual = records[0].clone();
        dual["mode"] = json!("both");
        dual["observation"]["relations"][0]["representation"] = json!("notation");
        for event in dual["observation"]["voices"][0]["events"]
            .as_array_mut()
            .unwrap()
        {
            event["notation_notes"] = json!([{"staff_position":9,"accidental":null}]);
        }
        let mut manual = records[0]["observation"]["voices"].clone();
        for event in manual[0]["events"].as_array_mut().unwrap() {
            event["notes"][0]["pitch"] = json!(79);
        }
        let edited = assemble(&[dual.clone()], Some(&[json!({"resolved_voices":manual})])).unwrap();
        assert!(edited["records"][0]["ir"].is_string());
        assert_eq!(edited["relations"], json!([]));
        assert_eq!(
            edited["records"][0]["unresolved"],
            json!([{"reason":"manual_relation_anchors_changed"}])
        );
        assert_eq!(edited["records"][0]["observation"], dual["observation"]);
    }

    #[test]
    fn one_unknown_accidental_cannot_describe_different_known_pitches() {
        let values = json!({"time":"4/4","key":0,"clef":"G2","clef_octave":0,
            "tuning":[64,59,55,50,45,40],"instrument_transpose":-12,
            "instrument_transpose_explicit":false,"capo":0});
        let state: serde_json::Map<String, Value> = values
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), json!({"status":"observed","value":v})))
            .collect();
        let mut row = json!({"source_id":"score","part_id":0,"staff_id":0,
            "bar_index":0,"mode":"both","observation":{"state":{},"relations":[],
                "voices":[{"voice":0,"events":[{"start":0,"duration":"w","status":"normal",
                    "notes":[{"string":1,"fret":0},{"string":2,"fret":6}],
                    "notation_notes":[{"staff_position":7,"accidental":"unknown"}]}]}]}});
        let extra = [json!({"state":state})];
        let result = assemble(&[row.clone()], Some(&extra)).unwrap();
        assert!(result["records"][0]["ir"].is_null());
        assert!(array(&result["records"][0]["unresolved"])
            .iter()
            .any(|issue| issue["reason"] == "dual_note_pairing_unresolved"));
        row["observation"]["voices"][0]["events"][0]["notes"][1]["fret"] = json!(5);
        let result = assemble(&[row], Some(&extra)).unwrap();
        assert!(result["records"][0]["ir"].is_string());
        assert_eq!(
            array(&result["records"][0]["resolved"]["voices"][0]["events"][0]["notes"]).len(),
            2
        );
    }
}
