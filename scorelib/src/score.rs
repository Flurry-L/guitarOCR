//! Native M2 score syntax and hard score-IR invariants.
//!
//! The JSON shape matches `scorelib/python/scorelib/m2.py` and the score editor. Parsing never
//! assigns strings to pitched/percussion notes, quantizes durations, or discards
//! effects. Display-mode projection is explicit in the formatter; desktop saves
//! should pass `preserve_playback = true`. Integers outside i64 and arithmetic
//! outside i128 are rejected rather than rounded. No Python runtime is used.

use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

type ScoreResult<T> = Result<T, String>;

/// Reduced, exact tick count. There is deliberately no implicit integer/float
/// conversion: an exporter must explicitly choose its quantization policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    pub numerator: i128,
    pub denominator: i128,
}

impl Rational {
    pub fn new(numerator: i128, denominator: i128) -> ScoreResult<Self> {
        if denominator <= 0 || numerator < 0 {
            return Err(
                "Tick fraction requires a nonnegative numerator and positive denominator".into(),
            );
        }
        let (mut a, mut b) = (numerator, denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Ok(Self {
            numerator: numerator / a,
            denominator: denominator / a,
        })
    }
}

fn duration_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(w|h|q|e|s|t|f|d[0-9]+)(\.\.|\.)?(?:\[([0-9]+):([0-9]+)\])?$").unwrap()
    })
}
fn note_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(?:s([0-9]+)f(x|-?[0-9]+))?(?:p(-?[0-9]+))?(?:\((.*)\))?$").unwrap()
    })
}
fn voice_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^V([0-9]+)\{(.*)\}$").unwrap())
}
fn position_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:f([0-9]+))?(?:p([0-9]+))?$").unwrap())
}
fn integer(text: &str) -> ScoreResult<i64> {
    text.parse::<i64>()
        .map_err(|_| format!("Invalid or out-of-range M2 integer: {text:?}"))
}
fn int_field(value: &Value, key: &str) -> ScoreResult<i64> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("M2 field {key:?} must be an i64 integer"))
}
fn int_default(value: &Value, key: &str, default: i64) -> ScoreResult<i64> {
    if value.get(key).is_none() {
        Ok(default)
    } else {
        int_field(value, key)
    }
}
fn bool_default(value: &Value, key: &str) -> ScoreResult<bool> {
    match value.get(key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(format!("M2 field {key:?} must be a boolean")),
    }
}
fn text_field<'a>(value: &'a Value, key: &str) -> ScoreResult<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("M2 field {key:?} must be text"))
}
fn list_default<'a>(value: &'a Value, key: &str) -> ScoreResult<&'a [Value]> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(items)) => Ok(items),
        _ => Err(format!("M2 field {key:?} must be an array")),
    }
}
fn strings(value: &Value, key: &str) -> ScoreResult<Vec<String>> {
    list_default(value, key)?
        .iter()
        .map(|v| {
            let text = v
                .as_str()
                .ok_or_else(|| format!("M2 field {key:?} must contain only strings"))?;
            if text.is_empty() || split_top_level(text)?.len() != 1 {
                return Err(format!(
                    "M2 {key} item contains an unescaped separator or is empty"
                ));
            }
            Ok(text.to_owned())
        })
        .collect()
}
fn known_fields(value: &Value, names: &[&str], kind: &str) -> ScoreResult<()> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("M2 {kind} must be an object"))?;
    for key in object.keys() {
        if !names.contains(&key.as_str()) {
            return Err(format!("Unsupported M2 {kind} field: {key}"));
        }
    }
    Ok(())
}

/// Split commas outside note effects, beat effects and tuplet brackets.
/// Unlike the legacy reader, malformed/mismatched brackets are not silently
/// repaired, and empty list entries are rejected.
fn split_top_level(text: &str) -> ScoreResult<Vec<&str>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut values = Vec::new();
    let mut start = 0;
    let mut stack = Vec::new();
    for (index, c) in text.char_indices() {
        match c {
            '(' | '<' | '[' => stack.push(c),
            ')' | '>' | ']' => {
                let expected = match c {
                    ')' => '(',
                    '>' => '<',
                    _ => '[',
                };
                if stack.pop() != Some(expected) {
                    return Err(format!("Unbalanced M2 delimiters: {text:?}"));
                }
            }
            ',' if stack.is_empty() => {
                if start == index {
                    return Err(format!("Empty M2 list item: {text:?}"));
                }
                values.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err(format!("Unbalanced M2 delimiters: {text:?}"));
    }
    if start == text.len() {
        return Err(format!("Empty M2 list item: {text:?}"));
    }
    values.push(&text[start..]);
    Ok(values)
}

pub fn parse_duration_token(token: &str) -> ScoreResult<Value> {
    let captures = duration_pattern()
        .captures(token)
        .ok_or_else(|| format!("Invalid M2 duration token: {token:?}"))?;
    let base = &captures[1];
    let value = match base {
        "w" => 1,
        "h" => 2,
        "q" => 4,
        "e" => 8,
        "s" => 16,
        "t" => 32,
        "f" => 64,
        _ => integer(&base[1..])?,
    };
    let dots = captures.get(2).map_or("", |m| m.as_str());
    let enters = captures.get(3).map_or(Ok(1), |m| integer(m.as_str()))?;
    let times = captures.get(4).map_or(Ok(1), |m| integer(m.as_str()))?;
    if value <= 0 || enters <= 0 || times <= 0 {
        return Err(format!(
            "M2 duration and tuplet counts must be positive: {token:?}"
        ));
    }
    Ok(
        json!({"value": value, "dotted": dots == ".", "double_dotted": dots == "..", "tuplet_enters": enters, "tuplet_times": times}),
    )
}

pub fn duration_ticks(duration: &Value) -> ScoreResult<Rational> {
    let value = int_field(duration, "value")?;
    let enters = int_default(duration, "tuplet_enters", 1)?;
    let times = int_default(duration, "tuplet_times", 1)?;
    if value <= 0 || enters <= 0 || times <= 0 {
        return Err("M2 duration and tuplet counts must be positive".into());
    }
    let (dot_num, dot_den) = if bool_default(duration, "double_dotted")? {
        (7, 4)
    } else if bool_default(duration, "dotted")? {
        (3, 2)
    } else {
        (1, 1)
    };
    let numerator = 3840_i128
        .checked_mul(times as i128)
        .and_then(|n| n.checked_mul(dot_num))
        .ok_or("M2 tick numerator overflow")?;
    let denominator = (value as i128)
        .checked_mul(enters as i128)
        .and_then(|n| n.checked_mul(dot_den))
        .ok_or("M2 tick denominator overflow")?;
    Rational::new(numerator, denominator)
}

fn duration_token(duration: &Value) -> ScoreResult<String> {
    known_fields(
        duration,
        &[
            "value",
            "dotted",
            "double_dotted",
            "tuplet_enters",
            "tuplet_times",
        ],
        "duration",
    )?;
    duration_ticks(duration)?;
    let value = int_field(duration, "value")?;
    let mut token = match value {
        1 => "w".into(),
        2 => "h".into(),
        4 => "q".into(),
        8 => "e".into(),
        16 => "s".into(),
        32 => "t".into(),
        64 => "f".into(),
        _ => format!("d{value}"),
    };
    if bool_default(duration, "double_dotted")? {
        token.push_str("..");
    } else if bool_default(duration, "dotted")? {
        token.push('.');
    }
    let enters = int_default(duration, "tuplet_enters", 1)?;
    let times = int_default(duration, "tuplet_times", 1)?;
    if enters != 1 || times != 1 {
        token.push_str(&format!("[{enters}:{times}]"));
    }
    Ok(token)
}

fn parse_note_token(token: &str) -> ScoreResult<Value> {
    let captures = note_pattern()
        .captures(token)
        .ok_or_else(|| format!("Invalid M2 note token: {token:?}"))?;
    if captures.get(1).is_none() && captures.get(3).is_none() {
        return Err(format!("Invalid M2 note token: {token:?}"));
    }
    let mut velocity = 95;
    let mut swap_accidentals = false;
    let mut effects = Vec::new();
    for effect in split_top_level(captures.get(4).map_or("", |m| m.as_str()))? {
        if let Some(number) = effect.strip_prefix("vel:") {
            velocity = integer(number)?;
        } else if effect == "accswap" {
            swap_accidentals = true;
        } else {
            effects.push(effect);
        }
    }
    let mut note =
        json!({"effects": effects, "velocity": velocity, "swap_accidentals": swap_accidentals});
    if let Some(string) = captures.get(1) {
        note["string"] = json!(integer(string.as_str())?);
        note["fret"] = if &captures[2] == "x" {
            json!("x")
        } else {
            json!(integer(&captures[2])?)
        };
    }
    if let Some(pitch) = captures.get(3) {
        note["pitch"] = json!(integer(pitch.as_str())?);
    }
    Ok(note)
}

fn unquote(text: &str) -> ScoreResult<String> {
    let mut bytes = Vec::with_capacity(text.len());
    let source = text.as_bytes();
    let mut i = 0;
    while i < source.len() {
        if source[i] == b'%' && i + 2 < source.len() {
            let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
            if let (Some(a), Some(b)) = (hex(source[i + 1]), hex(source[i + 2])) {
                bytes.push(a * 16 + b);
                i += 3;
                continue;
            }
        }
        bytes.push(source[i]);
        i += 1;
    }
    String::from_utf8(bytes)
        .map_err(|_| "M2 metadata contains invalid percent-encoded UTF-8".into())
}
fn quote(text: &str) -> String {
    const HEX: &[u8] = b"0123456789ABCDEF";
    let mut output = String::new();
    for c in text.bytes() {
        if c.is_ascii_alphanumeric() || b"-_.~".contains(&c) {
            output.push(c as char);
        } else {
            output.push('%');
            output.push(HEX[(c >> 4) as usize] as char);
            output.push(HEX[(c & 15) as usize] as char);
        }
    }
    output
}

/// Parse the entire M2 grammar, retaining all sixteen voice slots, arbitrary
/// extension durations, note/beat effects, velocities and accidental spellings.
/// Semantic constraints are intentionally separate from syntactic parsing.
pub fn parse_measure_target(text: &str) -> ScoreResult<Value> {
    let (prefix, voice_text) = text
        .trim()
        .split_once('|')
        .ok_or("M2 target must start with 'M2' and contain '|'")?;
    let mut tokens = prefix.split_whitespace();
    if tokens.next() != Some("M2") {
        return Err("M2 target must start with 'M2' and contain '|'".into());
    }
    let mut metadata = HashMap::new();
    for token in tokens {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| format!("Invalid M2 metadata token: {token:?}"))?;
        if ![
            "time", "tempo", "key", "feel", "bar", "rep", "alt", "section", "dir", "from",
        ]
        .contains(&key)
        {
            return Err(format!("Unknown M2 metadata: {key}"));
        }
        if metadata.insert(key, value).is_some() {
            return Err(format!("Duplicate M2 metadata: {key}"));
        }
    }
    let meta_text = |key| metadata.get(key).map_or(Value::Null, |s| json!(s));
    let meta_int = |key| {
        metadata
            .get(key)
            .map_or(Ok(Value::Null), |s| integer(s).map(|n| json!(n)))
    };
    let meta_decoded = |key| {
        metadata
            .get(key)
            .map_or(Ok(Value::Null), |s| unquote(s).map(|text| json!(text)))
    };
    let bars: Vec<&str> = metadata
        .get("bar")
        .filter(|s| !s.is_empty())
        .map_or(Vec::new(), |s| s.split(',').collect());
    let mut measure = json!({
        "time_signature": meta_text("time"), "print_time_signature": metadata.contains_key("time"),
        "tempo_quarter": meta_int("tempo")?, "key_signature": meta_text("key"),
        "print_key_signature": metadata.contains_key("key"), "triplet_feel": meta_text("feel"),
        "bars": bars, "repeat_count": meta_int("rep")?, "alternate_endings": meta_int("alt")?,
        "section": meta_decoded("section")?, "direction": meta_decoded("dir")?,
        "from_direction": meta_decoded("from")?, "voices": [],
    });
    let mut voices = Vec::new();
    for part in voice_text
        .split("||")
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let captures = voice_pattern()
            .captures(part)
            .ok_or_else(|| format!("Invalid M2 voice: {part:?}"))?;
        let mut events: Vec<Value> = Vec::new();
        for event in captures[2].split_whitespace() {
            let body = event
                .strip_prefix('@')
                .ok_or_else(|| format!("Invalid M2 event: {event:?}"))?;
            let (start, rest) = body
                .split_once(':')
                .ok_or_else(|| format!("Invalid M2 event start: {event:?}"))?;
            let duration_end = if rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
                // Find a payload separator, skipping only the duration's tuplet
                // bracket, never brackets that belong to an effect payload.
                let mut bracket = false;
                rest.char_indices().find_map(|(i, c)| {
                    match c {
                        '[' => bracket = true,
                        ']' => bracket = false,
                        ':' if !bracket => return Some(i),
                        _ => {}
                    }
                    None
                })
            } else {
                None
            }
            .ok_or_else(|| format!("Invalid M2 event duration: {event:?}"))?;
            let duration = parse_duration_token(&rest[..duration_end])?;
            let start = integer(start)?;
            let mut payload = &rest[duration_end + 1..];
            if payload == "^" {
                let mut previous = events
                    .last()
                    .ok_or("M2 repeated payload '^' cannot be the first event")?
                    .clone();
                previous["start"] = json!(start);
                previous["duration"] = duration;
                events.push(previous);
                continue;
            }
            if payload.is_empty() {
                return Err(format!("Missing M2 event payload: {event:?}"));
            }
            let mut beat_effects = Vec::new();
            if payload.ends_with('>') {
                let (notes, effects) = payload
                    .rsplit_once('<')
                    .ok_or_else(|| format!("Invalid M2 beat effects: {payload:?}"))?;
                beat_effects = split_top_level(&effects[..effects.len() - 1])?;
                payload = notes;
            }
            let (status, notes) = match payload {
                "e" => ("empty", Vec::new()),
                "r" => ("rest", Vec::new()),
                "z" => ("normal", Vec::new()),
                _ => (
                    "normal",
                    split_top_level(payload)?
                        .into_iter()
                        .map(parse_note_token)
                        .collect::<ScoreResult<Vec<_>>>()?,
                ),
            };
            if payload.is_empty() {
                return Err(format!("Missing M2 event payload: {event:?}"));
            }
            events.push(json!({"start": start, "duration": duration, "status": status, "notes": notes, "effects": beat_effects}));
        }
        voices.push(json!({"voice": integer(&captures[1])?, "events": events}));
    }
    measure["voices"] = json!(voices);
    Ok(measure)
}

fn ornament_position(token: &str) -> ScoreResult<(Option<i64>, Option<i64>)> {
    if token == "x" {
        return Ok((None, None));
    }
    let c = position_pattern()
        .captures(token)
        .filter(|_| !token.is_empty())
        .ok_or_else(|| format!("Invalid ornament position: {token}"))?;
    Ok((
        c.get(1).map(|m| integer(m.as_str())).transpose()?,
        c.get(2).map(|m| integer(m.as_str())).transpose()?,
    ))
}

fn visible_effect(effect: &str, mode: &str, preserve_playback: bool) -> ScoreResult<String> {
    let mut parts: Vec<String> = effect.split(':').map(str::to_owned).collect();
    if matches!(parts[0].as_str(), "grace" | "trill") && parts.len() > 1 {
        if parts[0] == "trill" && mode == "notation" && !preserve_playback {
            return Ok("trill".into());
        }
        let (fret, pitch) = ornament_position(&parts[1])?;
        if mode == "notation" && pitch.is_some() {
            parts[1] = format!("p{}", pitch.unwrap());
        } else if mode == "tab" && fret.is_some() {
            parts[1] = format!("f{}", fret.unwrap());
        }
        if !preserve_playback {
            if parts[0] == "trill" {
                parts.truncate(2);
            } else {
                if parts.len() == 6 {
                    parts.remove(2);
                }
                if parts.len() != 5 {
                    return Err(format!("Invalid grace effect: {effect}"));
                }
                if mode == "tab" {
                    parts[3] = "-".into();
                }
                if parts[4] == "1" {
                    parts[1] = "x".into();
                }
            }
        }
    }
    Ok(parts.join(":"))
}

fn note_token(note: &Value, mode: &str, preserve_playback: bool) -> ScoreResult<String> {
    known_fields(
        note,
        &[
            "string",
            "fret",
            "pitch",
            "effects",
            "velocity",
            "swap_accidentals",
        ],
        "note",
    )?;
    let mut token = String::new();
    if mode != "notation" {
        let string = int_field(note, "string")?;
        let fret = if note.get("fret").and_then(Value::as_str) == Some("x") {
            "x".into()
        } else {
            int_field(note, "fret")?.to_string()
        };
        token.push_str(&format!("s{string}f{fret}"));
    }
    if mode != "tab" {
        token.push_str(&format!("p{}", int_field(note, "pitch")?));
    }
    let source_effects = strings(note, "effects")?;
    if source_effects
        .iter()
        .any(|e| e == "accswap" || e.starts_with("vel:"))
    {
        return Err("M2 velocity/spelling belong in note fields, not musical effects".into());
    }
    let mut effects = source_effects
        .iter()
        .map(|e| visible_effect(e, mode, preserve_playback))
        .collect::<ScoreResult<Vec<_>>>()?;
    let velocity = int_default(note, "velocity", 95)?;
    if velocity != 95 {
        effects.push(format!("vel:{velocity}"));
    }
    if mode != "tab" && bool_default(note, "swap_accidentals")? {
        effects.push("accswap".into());
    }
    if !effects.is_empty() {
        token.push('(');
        token.push_str(&effects.join(","));
        token.push(')');
    }
    Ok(token)
}

/// Format a measure in a requested display mode. As in the authoritative M2
/// formatter, notation strips string positions, TAB strips pitches/spelling,
/// and `preserve_playback = false` projects ornaments/dynamics to visible marks.
/// Unknown JSON fields and unrepresentable data are rejected rather than lost.
pub fn format_measure_target(
    measure: &Value,
    mode: &str,
    preserve_playback: bool,
) -> ScoreResult<String> {
    if !["tab", "notation", "both"].contains(&mode) {
        return Err(format!("Unsupported display mode: {mode}"));
    }
    known_fields(
        measure,
        &[
            "time_signature",
            "print_time_signature",
            "tempo_quarter",
            "key_signature",
            "print_key_signature",
            "triplet_feel",
            "bars",
            "repeat_count",
            "alternate_endings",
            "section",
            "direction",
            "from_direction",
            "voices",
            // Source position and annotations belong to the score document;
            // formatting projects its musical fields into the M2 token stream.
            "index",
            "number",
            "targets",
            "source_record",
            "mode",
            "needs_review",
            "chord_annotations",
            "pitch_context",
            "pitch_reference",
            "written",
        ],
        "measure",
    )?;
    let mut metadata = Vec::new();
    if bool_default(measure, "print_time_signature")? {
        metadata.push(format!("time={}", text_field(measure, "time_signature")?));
    }
    if !measure["tempo_quarter"].is_null() {
        let tempo = int_field(measure, "tempo_quarter")?;
        if tempo != 0 {
            metadata.push(format!("tempo={tempo}"));
        }
    }
    if mode != "tab" && bool_default(measure, "print_key_signature")? {
        metadata.push(format!("key={}", text_field(measure, "key_signature")?));
    }
    if !measure["triplet_feel"].is_null() {
        let feel = text_field(measure, "triplet_feel")?;
        if !feel.is_empty() {
            metadata.push(format!("feel={feel}"));
        }
    }
    let bars = strings(measure, "bars")?;
    if !bars.is_empty() {
        metadata.push(format!("bar={}", bars.join(",")));
    }
    for (key, name) in [("repeat_count", "rep"), ("alternate_endings", "alt")] {
        if !measure[key].is_null() {
            let n = int_field(measure, key)?;
            if n != 0 {
                metadata.push(format!("{name}={n}"));
            }
        }
    }
    for (key, name) in [
        ("section", "section"),
        ("direction", "dir"),
        ("from_direction", "from"),
    ] {
        if !measure[key].is_null() {
            let text = text_field(measure, key)?;
            let retain = if key == "section" {
                !text.is_empty()
            } else {
                !["none", "null"].contains(&text.to_lowercase().as_str())
            };
            if retain {
                metadata.push(format!("{name}={}", quote(text)));
            }
        }
    }
    if metadata
        .iter()
        .any(|m| m.chars().any(char::is_whitespace) || m.contains('|'))
    {
        return Err("M2 metadata contains an unescaped separator".into());
    }
    let mut voices = Vec::new();
    for voice in list_default(measure, "voices")? {
        known_fields(voice, &["voice", "events"], "voice")?;
        let mut previous: Option<String> = None;
        let mut events = Vec::new();
        for event in list_default(voice, "events")? {
            known_fields(
                event,
                &["start", "duration", "status", "notes", "effects"],
                "event",
            )?;
            let duration = duration_token(&event["duration"])?;
            let status = event
                .get("status")
                .map_or(Ok("normal"), |_| text_field(event, "status"))?;
            let notes = list_default(event, "notes")?;
            let mut payload = match status {
                "empty" | "rest" => {
                    if !notes.is_empty() {
                        return Err("Rest/empty M2 event cannot contain notes".into());
                    }
                    if status == "empty" {
                        "e".into()
                    } else {
                        "r".into()
                    }
                }
                "normal" => {
                    // Precompute fallible sort keys: no default values may hide
                    // malformed notes during a comparator call.
                    let mut ordered = Vec::new();
                    for (index, note) in notes.iter().enumerate() {
                        let mut effects = strings(note, "effects")?;
                        effects.sort();
                        let primary = if mode == "notation" {
                            -(int_field(note, "pitch")? as i128)
                        } else {
                            int_field(note, "string")? as i128
                        };
                        let secondary = if mode == "notation" {
                            effects
                        } else {
                            vec![note.to_string()]
                        };
                        ordered.push((
                            primary,
                            secondary,
                            int_default(note, "velocity", 95)?,
                            bool_default(note, "swap_accidentals")?,
                            index,
                        ));
                    }
                    ordered.sort();
                    let notes = ordered
                        .iter()
                        .map(|key| note_token(&notes[key.4], mode, preserve_playback))
                        .collect::<ScoreResult<Vec<_>>>()?;
                    if notes.is_empty() {
                        "z".into()
                    } else {
                        notes.join(",")
                    }
                }
                _ => return Err(format!("Unsupported M2 event status: {status}")),
            };
            let effects: Vec<String> = strings(event, "effects")?
                .into_iter()
                .filter(|e| preserve_playback || !e.starts_with("dyn:"))
                .collect();
            if !effects.is_empty() {
                payload.push('<');
                payload.push_str(&effects.join(","));
                payload.push('>');
            }
            if payload.chars().any(char::is_whitespace)
                || payload.contains('|')
                || payload.contains('{')
                || payload.contains('}')
            {
                return Err("M2 payload contains an unescaped separator".into());
            }
            let emitted = if previous.as_ref() == Some(&payload) {
                "^"
            } else {
                &payload
            };
            events.push(format!(
                "@{}:{duration}:{emitted}",
                int_field(event, "start")?
            ));
            previous = Some(payload);
        }
        voices.push(format!(
            "V{}{{{}}}",
            int_field(voice, "voice")?,
            events.join(" ")
        ));
    }
    let prefix = if metadata.is_empty() {
        "M2".into()
    } else {
        format!("M2 {}", metadata.join(" "))
    };
    let output = format!("{prefix} | {}", voices.join(" || "));
    // Catch effect delimiters that cannot be represented by the textual grammar.
    parse_measure_target(&output)?;
    Ok(output)
}

fn valid_note_effect(effect: &str) -> bool {
    if [
        "accent", "dead", "ghost", "grace", "hammer", "heavy", "let", "pm", "sia", "sib", "sl",
        "sod", "sou", "ss", "stacc", "tap", "tie", "trem", "trill", "vib",
    ]
    .contains(&effect)
    {
        return true;
    }
    let parts: Vec<&str> = effect.split(':').collect();
    match parts[0] {
        "harm" | "slide" => effect
            .split_once(':')
            .is_some_and(|(_, tail)| !tail.is_empty()),
        "bend" => parts.len() == 3 && !parts[1].is_empty() && signed_decimal(parts[2]),
        "trem" => parts.len() == 2 && integer(parts[1]).is_ok_and(|n| [8, 16, 32].contains(&n)),
        "grace" | "trill" if parts.len() >= 2 => {
            let Ok((fret, pitch)) = ornament_position(parts[1]) else {
                return false;
            };
            if fret.is_some_and(|f| !(0..=36).contains(&f))
                || pitch.is_some_and(|p| !(0..=127).contains(&p))
            {
                return false;
            }
            if parts[0] == "trill" {
                parts[1] != "x"
                    && (parts.len() == 2
                        || parts.len() == 3
                            && integer(parts[2]).is_ok_and(|n| [16, 32, 64].contains(&n)))
            } else {
                let mut parts = parts;
                if parts.len() == 6 {
                    if !integer(parts[2]).is_ok_and(|n| [8, 16, 32, 64, 128].contains(&n)) {
                        return false;
                    }
                    parts.remove(2);
                }
                parts.len() == 5
                    && ["none", "slide", "bend", "hammer"].contains(&parts[2])
                    && ["0", "1", "-"].contains(&parts[3])
                    && ["0", "1"].contains(&parts[4])
                    && (parts[1] != "x" || parts[4] == "1")
            }
        }
        _ => false,
    }
}
fn signed_decimal(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit()) && integer(text).is_ok()
}
fn valid_diagram(effect: &str) -> bool {
    let p: Vec<&str> = effect.split(':').collect();
    if p.len() != 5 {
        return false;
    }
    let Ok(base) = integer(p[1]) else {
        return false;
    };
    if !(1..=36).contains(&base) {
        return false;
    }
    let frets: Vec<&str> = p[2].split('/').collect();
    if !(1..=12).contains(&frets.len())
        || frets.iter().any(|f| {
            *f != "x" && !integer(f).is_ok_and(|n| (0..=36).contains(&n) && (n == 0 || n >= base))
        })
    {
        return false;
    }
    let fingers: Vec<&str> = p[3].split('/').collect();
    if fingers.len() != frets.len()
        || fingers
            .iter()
            .any(|f| *f != "-" && !integer(f).is_ok_and(|n| (0..=4).contains(&n)))
    {
        return false;
    }
    if p[4] != "-" {
        for barre in p[4].split(';') {
            let Ok(numbers) = barre
                .split('/')
                .map(integer)
                .collect::<ScoreResult<Vec<_>>>()
            else {
                return false;
            };
            if numbers.len() != 3
                || !(base..=36).contains(&numbers[0])
                || numbers[1] < 0
                || numbers[1] >= numbers[2]
                || numbers[2] >= frets.len() as i64
            {
                return false;
            }
        }
    }
    true
}
fn valid_beat_effect(effect: &str) -> bool {
    if [
        "fade",
        "pick_down",
        "pick_up",
        "rasg",
        "stroke_down",
        "stroke_up",
    ]
    .contains(&effect)
    {
        return true;
    }
    let Some((kind, value)) = effect.split_once(':') else {
        return false;
    };
    match kind {
        "ottava" => ["12", "-12", "24", "-24"].contains(&value),
        "diagram" => valid_diagram(effect),
        "slap" => ["none", "tapping", "slapping", "popping"].contains(&value),
        "chord" | "text" => !value.is_empty(),
        "tempo" | "dyn" => {
            !value.is_empty()
                && value.bytes().all(|b| b.is_ascii_digit())
                && integer(value).is_ok_and(|n| {
                    if kind == "tempo" {
                        (1..=999).contains(&n)
                    } else {
                        (0..=127).contains(&n)
                    }
                })
        }
        _ => false,
    }
}

/// Validate syntax plus hard IR invariants, matching `scorelib/python/scorelib/constraints.py`.
/// Pickup/overfull measures, gaps and overlaps remain valid IR: export-specific
/// timing restrictions belong in the exporter, not this validator.
pub fn validate_measure_target(
    target: &str,
    mode: &str,
    tuning: Option<&[i64]>,
    string_count: Option<usize>,
) -> (Option<Value>, Vec<String>) {
    if !["tab", "notation", "both"].contains(&mode) {
        return (None, vec![format!("unsupported_mode:{mode}")]);
    }
    let measure = match parse_measure_target(target) {
        Ok(m) => m,
        Err(e) => return (None, vec![format!("syntax:{e}")]),
    };
    let mut errors = Vec::new();
    const DURATIONS: [i64; 7] = [1, 2, 4, 8, 16, 32, 64];
    if let Some(signature) = measure["time_signature"].as_str().filter(|s| !s.is_empty()) {
        let valid = signature.split_once('/').is_some_and(|(n, d)| {
            !n.is_empty()
                && !d.is_empty()
                && n.bytes().all(|c| c.is_ascii_digit())
                && d.bytes().all(|c| c.is_ascii_digit())
                && integer(n).is_ok_and(|v| (1..=255).contains(&v))
                && integer(d).is_ok_and(|v| DURATIONS.contains(&v))
        });
        if !valid {
            errors.push("invalid_time_signature".into());
        }
    }
    if measure["tempo_quarter"]
        .as_i64()
        .is_some_and(|t| !(1..=999).contains(&t))
    {
        errors.push("invalid_tempo".into());
    }
    let voices = measure["voices"]
        .as_array()
        .expect("parser produces voices");
    if voices.is_empty() {
        errors.push("missing_voice".into());
    }
    let max_string = string_count
        .filter(|n| *n != 0)
        .unwrap_or_else(|| tuning.filter(|t| !t.is_empty()).map_or(8, <[i64]>::len));
    let mut voice_ids = HashSet::new();
    for voice in voices {
        let id = voice["voice"].as_i64().unwrap();
        if !(0..=15).contains(&id) {
            errors.push(format!("invalid_voice:V{id}"));
        }
        if !voice_ids.insert(id) {
            errors.push(format!("duplicate_voice:V{id}"));
        }
        let events = voice["events"].as_array().unwrap();
        if events.is_empty() {
            errors.push(format!("empty_voice:V{id}"));
        }
        let starts: Vec<i64> = events
            .iter()
            .map(|e| e["start"].as_i64().unwrap())
            .collect();
        if starts.iter().any(|s| *s < 0) {
            errors.push(format!("negative_event_start:V{id}"));
        }
        if starts.windows(2).any(|w| w[0] >= w[1]) {
            errors.push(format!("non_increasing_event_starts:V{id}"));
        }
        for (ei, event) in events.iter().enumerate() {
            let duration = &event["duration"];
            if !DURATIONS.contains(&duration["value"].as_i64().unwrap())
                || !(1..=32).contains(&duration["tuplet_enters"].as_i64().unwrap())
                || !(1..=32).contains(&duration["tuplet_times"].as_i64().unwrap())
            {
                errors.push(format!("invalid_duration:V{id}:E{ei}"));
            }
            let effects = event["effects"].as_array().unwrap();
            for effect in effects {
                let effect = effect.as_str().unwrap();
                if !valid_beat_effect(effect) {
                    errors.push(format!("unknown_beat_effect:{effect}"));
                }
            }
            if effects
                .iter()
                .filter(|e| e.as_str().unwrap().starts_with("ottava:"))
                .count()
                > 1
            {
                errors.push(format!("duplicate_ottava:V{id}:E{ei}"));
            }
            let mut strings = HashSet::new();
            let mut duplicate_string = false;
            for (ni, note) in event["notes"].as_array().unwrap().iter().enumerate() {
                let location = format!("V{id}:E{ei}:N{ni}");
                let has_string = note.get("string").is_some() && note.get("fret").is_some();
                let has_pitch = note.get("pitch").is_some();
                if mode == "tab" && (!has_string || has_pitch) {
                    errors.push(format!("tab_note_fields:{location}"));
                } else if mode == "notation" && (!has_pitch || has_string) {
                    errors.push(format!("notation_note_fields:{location}"));
                } else if mode == "both" && (!has_string || !has_pitch) {
                    errors.push(format!("both_note_fields:{location}"));
                }
                if has_string {
                    let string = note["string"].as_i64().unwrap();
                    if !strings.insert(string) {
                        duplicate_string = true;
                    }
                    if string < 1 || string as u64 > max_string as u64 {
                        errors.push(format!("invalid_string:{location}"));
                    }
                    if let Some(fret) = note["fret"].as_i64() {
                        if !(0..=36).contains(&fret) {
                            errors.push(format!("invalid_fret:{location}"));
                        }
                        if mode == "both" && has_pitch {
                            if let Some(tuning) = tuning {
                                if string >= 1
                                    && string as u64 <= tuning.len() as u64
                                    && note["pitch"].as_i64().unwrap() as i128
                                        != tuning[string as usize - 1] as i128 + fret as i128
                                {
                                    errors.push(format!("pitch_string_fret_conflict:{location}"));
                                }
                            }
                        }
                    }
                }
                if has_pitch && !(0..=127).contains(&note["pitch"].as_i64().unwrap()) {
                    errors.push(format!("invalid_pitch:{location}"));
                }
                if !(0..=127).contains(&note["velocity"].as_i64().unwrap()) {
                    errors.push(format!("invalid_velocity:{location}"));
                }
                for effect in note["effects"].as_array().unwrap() {
                    let effect = effect.as_str().unwrap();
                    if !valid_note_effect(effect) {
                        errors.push(format!("unknown_note_effect:{effect}"));
                    }
                }
            }
            if duplicate_string {
                errors.push(format!("duplicate_string_in_chord:V{id}:E{ei}"));
            }
        }
    }
    let mut seen = HashSet::new();
    errors.retain(|e| seen.insert(e.clone()));
    (Some(measure), errors)
}

fn rename_headers(text: &str, to_display: bool) -> String {
    let names = if to_display {
        [("M2", "MEASURE"), ("C2", "CONTEXT")]
    } else {
        [("MEASURE", "M2"), ("CONTEXT", "C2")]
    };
    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let indent = line.len() - line.trim_start_matches([' ', '\t']).len();
        let body = &line[indent..];
        let replacement = names.iter().find(|(from, _)| {
            body.strip_prefix(from)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with([' ', '\t', '|', '\n']))
        });
        if let Some((from, to)) = replacement {
            output.push_str(&line[..indent]);
            output.push_str(to);
            output.push_str(&body[from.len()..]);
        } else {
            output.push_str(line);
        }
    }
    output
}

/// Human-facing headers only; annotations and similar prefixes remain intact.
pub fn display_score_text(text: &str) -> String {
    rename_headers(text, true)
}
/// Accept readable or legacy headers without altering musical content.
pub fn model_score_text(text: &str) -> String {
    rename_headers(text, false)
}
/// Human-facing validation errors; technical logs may retain original text.
pub fn display_error(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(M2|C2)\b").unwrap())
        .replace_all(
            text,
            |c: &regex::Captures<'_>| {
                if &c[1] == "M2" {
                    "MEASURE"
                } else {
                    "CONTEXT"
                }
            },
        )
        .into_owned()
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Number(v) => v
            .as_i64()
            .map_or_else(|| v.as_f64() != Some(0.0), |n| n != 0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
fn default_tuning(instrument: &str) -> ScoreResult<Value> {
    match instrument {
        "guitar" => Ok(json!([64, 59, 55, 50, 45, 40])),
        "bass" => Ok(json!([43, 38, 33, 28])),
        "pitched" | "drums" => Ok(json!([])),
        _ => Err(format!("Unsupported instrument: {instrument}")),
    }
}
fn pitch_reference(instrument: &str) -> &'static str {
    match instrument {
        "drums" => "percussion_key",
        "guitar" | "bass" => "before_capo",
        _ => "sounding",
    }
}
fn record_id(record: &Value, key: &str, default: &str) -> ScoreResult<String> {
    match record.get(key) {
        None => Ok(default.into()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => Ok(n.to_string()),
        _ => Err(format!(
            "Score field {key} must be a string or integer identifier"
        )),
    }
}
fn vote<T: PartialEq>(votes: &mut HashMap<i64, Vec<(T, usize)>>, index: i64, value: T) {
    let entries = votes.entry(index).or_default();
    if let Some((_, count)) = entries.iter_mut().find(|(item, _)| *item == value) {
        *count += 1;
    } else {
        entries.push((value, 1));
    }
}
fn most_common<T>(votes: &[(T, usize)]) -> Option<&T> {
    // Counter.most_common breaks ties by first appearance, not sorted key.
    votes
        .iter()
        .reduce(|best, next| if next.1 > best.1 { next } else { best })
        .map(|(value, _)| value)
}

/// Derive `guitarocr.score/2` from recognition records, matching the authoritative
/// `scorelib/python/scorelib/score_document.py` grouping and continuity rules. Source records are
/// borrowed and never rewritten. Unknown record data stays in the recognition
/// artifact; the derived score contains the documented score-document fields.
pub fn score_document(result: &Value) -> ScoreResult<Value> {
    let records = result
        .get("records")
        .and_then(Value::as_array)
        .ok_or("Score result requires records array")?;
    let mut parts: Vec<Value> = Vec::new();
    let mut signatures: HashMap<i64, Vec<(String, usize)>> = HashMap::new();
    let mut tempos: HashMap<i64, Vec<(i64, usize)>> = HashMap::new();
    let mut last_index = -1;
    let result_instrument = result
        .get("instrument")
        .map_or(Ok("guitar"), |_| text_field(result, "instrument"))?;
    for record in records {
        if !record.is_object() {
            return Err("Score record must be an object".into());
        }
        let part_id = record_id(record, "part_id", "part-1")?;
        let staff_id = record_id(record, "staff_id", "staff-1")?;
        let instrument = if truthy(&record["instrument"]) {
            text_field(record, "instrument")?
        } else {
            result_instrument
        };
        let default_tuning = default_tuning(instrument)?;
        let same_instrument = instrument == result_instrument;
        let tuning = if matches!(instrument, "pitched" | "drums") {
            json!([])
        } else {
            let tuning = record.get("tuning").unwrap_or_else(|| {
                if same_instrument {
                    result.get("tuning_used").unwrap_or(&default_tuning)
                } else {
                    &default_tuning
                }
            });
            let values = tuning.as_array().ok_or("Score tuning must be an array")?;
            if values.iter().any(|n| n.as_i64().is_none()) {
                return Err("Score tuning must contain integer pitches".into());
            }
            tuning.clone()
        };
        let part_index = match parts.iter().position(|p| p["id"] == part_id) {
            Some(index) => index,
            None => {
                let default_program = json!(match instrument {
                    "guitar" => 25,
                    "bass" => 33,
                    _ => 0,
                });
                let program = record.get("midi_program").unwrap_or_else(|| {
                    if same_instrument {
                        result.get("midi_program").unwrap_or(&default_program)
                    } else {
                        &default_program
                    }
                });
                let capo = if record.get("capo").is_some() {
                    int_field(record, "capo")?
                } else if same_instrument {
                    int_default(result, "capo", 0)?
                } else {
                    0
                };
                parts.push(json!({"id": part_id, "name": record.get("part_name").cloned().unwrap_or(json!(instrument)), "instrument": instrument,
                    "pitch_reference": pitch_reference(instrument), "midi_program": program,
                    "tuning": tuning, "capo": capo, "staves": []}));
                parts.len() - 1
            }
        };
        let part = &mut parts[part_index];
        if part["instrument"] != instrument || part["tuning"] != tuning {
            return Err(format!(
                "Conflicting instrument or tuning in part {part_id}"
            ));
        }
        let staves = part["staves"].as_array_mut().unwrap();
        let staff_index = match staves.iter().position(|s| s["id"] == staff_id) {
            Some(index) => index,
            None => {
                staves.push(json!({"id":staff_id, "measures":[]}));
                staves.len() - 1
            }
        };
        let measures = staves[staff_index]["measures"].as_array_mut().unwrap();
        let index = int_default(record, "bar_index", measures.len() as i64)?;
        // Bound allocation for untrusted imported JSON, with an explicit error
        // rather than truncated/missing timeline entries.
        if !(0..1_000_000).contains(&index) {
            return Err(format!(
                "Score bar index must be between 0 and 999999: {index}"
            ));
        }
        if measures.iter().any(|m| m["index"] == index) {
            return Err(format!("Duplicate bar in {part_id}/{staff_id}: {index}"));
        }
        let target = model_score_text(text_field(record, "target")?);
        let mut parsed = parse_measure_target(&target)?;
        if truthy(&record["written_target"]) && !truthy(&record["manually_edited"]) {
            if let Some(target) = record["written_target"].as_str() {
                if let Ok(written) = parse_measure_target(target) {
                    parsed["written"] = written;
                }
            }
        }
        last_index = last_index.max(index);
        let signature = if truthy(&parsed["time_signature"]) {
            &parsed["time_signature"]
        } else {
            &record["score_state"]["time"]
        };
        if truthy(signature) {
            vote(
                &mut signatures,
                index,
                signature
                    .as_str()
                    .ok_or("Score state time signature must be text")?
                    .to_owned(),
            );
        }
        if let Some(tempo) = parsed["tempo_quarter"].as_i64().filter(|t| *t != 0) {
            vote(&mut tempos, index, tempo);
        }
        parsed["index"] = json!(index);
        parsed["source_record"] = record
            .get("measure_number")
            .ok_or("Score record requires measure_number")?
            .clone();
        parsed["mode"] = if truthy(&record["mode"]) {
            record["mode"].clone()
        } else {
            result
                .get("mode")
                .ok_or("Score result requires mode")?
                .clone()
        };
        parsed["needs_review"] = json!(truthy(&record["needs_review"]));
        parsed["chord_annotations"] = record
            .get("chord_annotations")
            .cloned()
            .unwrap_or(json!([]));
        parsed["pitch_reference"] = json!(pitch_reference(instrument));
        if truthy(&record["pitch_context"]) {
            parsed["pitch_context"] = record["pitch_context"].clone();
        }
        measures.push(parsed);
    }
    for part in &mut parts {
        for staff in part["staves"].as_array_mut().unwrap() {
            staff["measures"]
                .as_array_mut()
                .unwrap()
                .sort_by_key(|m| m["index"].as_i64().unwrap());
        }
    }
    let mut timeline = Vec::new();
    let mut signature = "4/4".to_owned();
    for index in 0..=last_index {
        if let Some(value) = signatures.get(&index).and_then(|v| most_common(v)) {
            signature = value.clone();
        }
        let mut timing = json!({"index": index, "time_signature": signature});
        if let Some(tempo) = tempos.get(&index).and_then(|v| most_common(v)) {
            timing["tempo_quarter"] = json!(tempo);
        }
        timeline.push(timing);
    }
    let annotation_parts: Vec<&Value> = match result.get("parts").filter(|p| truthy(p)) {
        Some(Value::Array(parts)) => parts.iter().collect(),
        Some(_) => return Err("Score parts must be an array".into()),
        None => vec![result],
    };
    let mut annotations = Vec::new();
    for part in annotation_parts {
        let source = &part["document_metadata"]["score_annotations"];
        if source.is_null() {
            continue;
        }
        for annotation in source
            .as_array()
            .ok_or("Score annotations must be an array")?
        {
            if !annotation.is_object() {
                return Err("Score annotation must be an object".into());
            }
            let mut annotation = annotation.clone();
            annotation["part_id"] = part.get("id").cloned().unwrap_or(json!("part-1"));
            annotations.push(annotation);
        }
    }
    Ok(
        json!({"schema":"guitarocr.score/2", "ticks_per_quarter":960,
        "title":result.get("title").cloned().unwrap_or(json!("")), "artist":result.get("artist").cloned().unwrap_or(json!("")),
        "timeline":timeline, "parts":parts, "annotations":annotations}),
    )
}
