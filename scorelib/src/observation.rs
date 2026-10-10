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
pub fn observation_schema() -> Value {
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
    let note = object(
        json!({"string":int,"fret":{"anyOf":[{"type":"integer"},{"enum":["x",null]}]},"staff_position":int,"accidental":{"anyOf":[{"type":"integer"},{"enum":["unknown",null]}]},"effects":effects(NOTE_EFFECTS, NOTE_EFFECT_PATTERN)}),
        &[],
    );
    let event = object(
        json!({"start":int,"duration":{"type":["string","null"]},"status":{"type":["string","null"]},"notes":{"anyOf":[{"type":"array","items":note},{"type":"null"}]},"effects":effects(EVENT_EFFECTS, EVENT_EFFECT_PATTERN)}),
        &["start", "duration", "status", "notes"],
    );
    let voice = object(
        json!({"voice":{"type":"integer"},"events":{"type":"array","items":event}}),
        &["voice", "events"],
    );
    let endpoint = json!({"anyOf":[{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3},{"enum":["left","right",null]}]});
    let relation = object(
        json!({"kind":{"enum":["tie","hammer","sl","ss","pm","let","ottava",null]},"shape":{"enum":["curve","diagonal","extension"]},"start":endpoint,"end":endpoint,"value":int,"voice":int,"lane":{"type":["number","null"]}}),
        &["kind", "start", "end"],
    );
    object(
        json!({"state":state,"voices":{"type":"array","items":voice},"relations":{"type":"array","items":relation}}),
        &["state", "voices", "relations"],
    )
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
            if !nullable_integer(&event["start"]) {
                return Err("Event start must be integer ticks or null".into());
            }
            if !event["duration"].is_null() && !event["duration"].is_string() {
                return Err("Duration must be a printed duration token or null".into());
            }
            for (index, item) in std::iter::once(event)
                .chain(array(&event["notes"]))
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
            for note in array(&event["notes"]) {
                for field in ["string", "staff_position"] {
                    if !nullable_integer(&note[field]) {
                        return Err(format!("Note {field} must be an integer or null"));
                    }
                }
                if !nullable_integer(&note["fret"]) && note["fret"] != "x" {
                    return Err("Fret must be an integer, x or null".into());
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
        let mut seen = if extra["source"] == "manual" {
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
        } else if seen["status"] == "absent" && !known(previous, field).is_null() {
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
        if a.len() == 3
            && !a[2]
                .as_u64()
                .is_some_and(|i| (i as usize) < array(&ev.unwrap()["notes"]).len())
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
    let tab = matches!(a.value["mode"].as_str(), Some("tab" | "both"));
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
            if tab {
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
    let field = if matches!(
        rows[left.record].value["mode"].as_str(),
        Some("tab" | "both")
    ) {
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
        if let Some(note) = note_mut(
            &mut rows[start[0].as_u64().unwrap() as usize],
            &local_ref(start),
        ) {
            add_effect(note, kind);
        }
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
    let b = &mut rows[end[0].as_u64().unwrap() as usize];
    let mode = b.value["mode"].as_str().unwrap().to_owned();
    let last = note_mut(b, &local_ref(end)).unwrap();
    for field in ["fret", "pitch", "written_pitch", "sounding_pitch"] {
        if last[field].is_null() && !first[field].is_null() {
            last[field] = first[field].clone();
        }
    }
    add_effect(last, "tie");
    let needed: &[&str] = match mode.as_str() {
        "tab" => &["fret"],
        "notation" => &["pitch"],
        _ => &["fret", "pitch"],
    };
    if needed.iter().any(|k| last[k].is_null()) {
        Some("tie_value_unresolved".into())
    } else {
        None
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
        measure["print_time_signature"] =
            json!(row.value["observation"]["state"]["time"]["status"] == "observed");
    }
    for voice in array(&resolved["voices"]) {
        let mut current = json!({"voice":voice["voice"],"events":[]});
        for (ei, event) in array(&voice["events"]).iter().enumerate() {
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
        value["observation"] = parse_observation(&record["observation"])?;
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
            if !same_anchors(&value["observation"]["voices"], manual) {
                relations = json!([]);
                value["unresolved"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"reason":"manual_relation_anchors_changed"}));
            }
        }
        rows.push(Row {
            value,
            voice_map: extra.get("voice_map").cloned().unwrap_or(json!({})),
            voice_links: BTreeMap::new(),
            relations,
        });
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
            previous = rows[index].value["resolved"]["state"].clone();
            last_bar = Some(bar);
            let relations = array(&rows[index].relations).to_vec();
            let mut local = Vec::new();
            let mut incoming_notes = HashSet::new();
            for (ri, r) in relations.into_iter().enumerate() {
                let problem = relation_problem(&rows[index].value["observation"], &r);
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
            if let Some(reason) = note_technique(&chain, &mut rows) {
                chain["status"] = json!(if chain["end"] == "right" {
                    "open"
                } else {
                    "unresolved"
                });
                chain["reason"] = json!(reason);
                for n in &connected {
                    issue(&mut rows[n.record], &reason, json!({"relation":n.index}));
                }
            }
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
        "absent means clearly not printed; unknown means unreadable. ",
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
        "accidental is null if not printed, an integer -2..2 if printed, or unknown if unreadable. ",
        "Do not output MIDI pitch or a global voice identity. ",
        "Include only visible effects using IR names. ",
        "A printed H/P curve uses hammer on its origin note. ",
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
    }
}
