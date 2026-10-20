//! Visible metadata transcription and its musical interpretation.
pub mod prompts;
use crate::chords::{is_chord_symbol, normalize_diagram};
use crate::instruments::program_from_visible_name;
use prompts::*;
use regex::Regex;
use serde_json::{json, Value};
type Result<T> = std::result::Result<T, String>;

pub fn clean_text(value: &Value) -> Value {
    value
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map_or(Value::Null, |s| json!(s))
}
fn integer_range(value: &Value, low: i64, high: i64) -> Value {
    value
        .as_i64()
        .filter(|n| (low..=high).contains(n))
        .map_or(Value::Null, |n| json!(n))
}
fn json_response(raw: &str) -> Result<Value> {
    let start = raw.find('{').ok_or("Model response has no JSON object")?;
    let end = raw
        .rfind('}')
        .ok_or("Model response has no closing brace")?;
    if end < start {
        return Err("Invalid JSON object delimiters".into());
    }
    let v: Value =
        serde_json::from_str(&raw[start..=end]).map_err(|e| format!("Invalid model JSON: {e}"))?;
    if !v.is_object() {
        return Err("Expected model JSON object".into());
    }
    Ok(v)
}
pub fn conventional_octave(name: &str) -> Option<i64> {
    match name
        .trim()
        .trim_end_matches('.')
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .as_str()
    {
        "piccolo" | "picc" => Some(12),
        "contrabass" | "double bass" | "guitar" | "bass guitar" => Some(-12),
        _ => None,
    }
}
pub fn explicit_transposition(name: &str) -> Option<i64> {
    let name = name
        .trim()
        .to_lowercase()
        .replace('−', "-")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let name = name.trim_end_matches('.');
    if ["concert pitch", "at concert pitch"].contains(&name) {
        return Some(0);
    }
    if let Some(n) = conventional_octave(name) {
        return Some(n);
    }
    if let Some(c) = Regex::new(r"^(?:transpose|written to sounding:)\s*([+-]?\d+)\s+semitones?$")
        .unwrap()
        .captures(name)
    {
        return c[1].parse::<i64>().ok().filter(|n| (-36..=36).contains(n));
    }
    if let Some(c) = Regex::new(r"^sounds\s+(\d+)\s+semitones?\s+(lower|higher)$")
        .unwrap()
        .captures(name)
    {
        return c[1]
            .parse::<i64>()
            .ok()
            .map(|n| if &c[2] == "lower" { -n } else { n })
            .filter(|n| (-36..=36).contains(n));
    }
    let name = Regex::new(r"\b([a-g])[ -]flat\b")
        .unwrap()
        .replace_all(&name.replace('♭', "b"), "${1}b")
        .into_owned();
    for (instrument, key, n) in [
        ("clarinet", "bb", -2),
        ("clarinet", "a", -3),
        ("clarinet", "eb", 3),
        ("bass clarinet", "bb", -14),
        ("trumpet", "bb", -2),
        ("trumpet", "a", -3),
        ("trumpet", "c", 0),
        ("trumpet", "d", 2),
        ("trumpet", "eb", 3),
        ("soprano sax(?:ophone)?", "bb", -2),
        ("alto sax(?:ophone)?", "eb", -9),
        ("tenor sax(?:ophone)?", "bb", -14),
        ("baritone sax(?:ophone)?", "eb", -21),
        ("(?:french )?horn", "f", -7),
        ("(?:english horn|cor anglais)", "f", -7),
    ] {
        if Regex::new(&format!("^(?:{instrument} in {key}|{key} {instrument})$"))
            .unwrap()
            .is_match(&name)
        {
            return Some(n);
        }
    }
    None
}
pub fn parse_response(raw: &str, kind: &str) -> Value {
    let Ok(v) = json_response(raw) else {
        return json!({});
    };
    if let Some(field) = header_field(kind) {
        return json!({field: clean_text(&v["text"])});
    }
    // A text proposal cannot establish fingering geometry.
    if kind == "text_annotation" && v["kind"] == "chord_diagram" {
        return json!({"kind":"other","text":clean_text(&v["text"]),"semitones":null,"capo":null});
    }
    match kind {
        "header" => {
            json!({"title":clean_text(&v["title"]),"artist":clean_text(&v["artist"]),"tuning_name":clean_text(&v["tuning_name"])})
        }
        "staff" => {
            json!({"instrument":v["instrument"].as_str().filter(|s|["guitar","bass","pitched","drums"].contains(s)),"string_count":integer_range(&v["string_count"],1,12),"name":clean_text(&v["name"])})
        }
        "clef" => {
            json!({"clef":v["clef"].as_str().filter(|s|["G2","F4","C3","C4","percussion","tab"].contains(s)),"clef_octave":v["clef_octave"].as_i64().filter(|n|[-24,-12,0,12,24].contains(n))})
        }
        "tempo" => {
            let denominator = match v["beat_unit"].as_str() {
                Some("whole") => 1.,
                Some("half") => 2.,
                Some("quarter") => 4.,
                Some("eighth") => 8.,
                Some("sixteenth") => 16.,
                _ => return json!({}),
            };
            let Some(dotted) = v["dotted"].as_bool() else {
                return json!({});
            };
            let Some(bpm) = integer_range(&v["bpm"], 1, 999).as_i64() else {
                return json!({});
            };
            // GP5 stores an integer quarter-note tempo. OCR only transcribes
            // the printed pulse; musical unit conversion belongs here.
            let quarter =
                (bpm as f64 * 4. / denominator * if dotted { 1.5 } else { 1. }).round() as i64;
            if !(1..=999).contains(&quarter) {
                return json!({});
            }
            json!({"tempo_quarter":quarter,"beat_unit":v["beat_unit"],"dotted":dotted,"bpm":bpm})
        }
        "annotation" | "transposition" | "text_annotation" => {
            let mut kind = v["kind"].as_str().unwrap_or("");
            let words = clean_text(&v["text"]);
            let words_str = words.as_str().unwrap_or("");
            // Harmonic syntax is independent of the region's detection label.
            // Keep a misclassified title/instrument label as visible text.
            if kind == "chord" && !is_chord_symbol(words_str) {
                kind = "other";
            }
            if [
                "chord",
                "chord_diagram",
                "technique",
                "tempo",
                "title",
                "credit",
                "other",
            ]
            .contains(&kind)
            {
                let mut r = json!({"kind":kind,"semitones":null,"capo":null,"text":words});
                if kind == "chord_diagram" {
                    r["diagram"] = normalize_diagram(&v["diagram"]);
                }
                return r;
            }
            let shift = match kind {
                "instrument" => explicit_transposition(words_str).or_else(|| {
                    (program_from_visible_name(words_str).is_some() && v["semitones"] == 0)
                        .then_some(0)
                }),
                "ottava" => octave_mark(words_str),
                _ => None,
            };
            let capo = (kind == "capo").then(|| capo_fret(words_str)).flatten();
            if shift.is_none() && capo.is_none() {
                json!({"kind":null,"semitones":null,"capo":null,"text":if ["instrument","ottava","capo"].contains(&kind){words}else{Value::Null}})
            } else {
                json!({"kind":kind,"semitones":shift,"capo":capo,"text":words})
            }
        }
        _ => json!({}),
    }
}
pub fn prompt(kind: &str) -> Result<&'static str> {
    match kind {
        k if header_field(k).is_some() => Ok(HEADER_TEXT_PROMPT),
        "header" => Ok(HEADER_PROMPT),
        "tempo" => Ok(TEMPO_PROMPT),
        "staff" => Ok(STAFF_PROMPT),
        "clef" => Ok(CLEF_PROMPT),
        "annotation" | "transposition" | "text_annotation" => Ok(ANNOTATION_PROMPT),
        _ => Err(format!("Unsupported metadata region: {kind}")),
    }
}

fn octave_mark(text: &str) -> Option<i64> {
    let mark: String = text
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace() && !['.', '-', '–', '_'].contains(c))
        .collect();
    match mark.as_str() {
        "8va" | "8vaalta" | "ottava" => Some(12),
        "8vb" | "8vabassa" | "8ba" | "ottavabassa" => Some(-12),
        "15ma" | "15maalta" => Some(24),
        "15mb" | "15mabassa" | "15ba" => Some(-24),
        _ => None,
    }
}

fn capo_fret(text: &str) -> Option<i64> {
    use std::sync::OnceLock;
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| Regex::new(
        r"(?i)^\s*(?:capo|capodastro|capotasto|变调夹)\s*[:=.]?\s*(?:(?:at|on)\s+)?(?:fret\s+)?(\d{1,2}|[IVX]{1,5})(?:st|nd|rd|th)?\s*(?:fret|品)?\s*$"
    ).unwrap());
    let captures = pattern.captures(text)?;
    let label = captures[1].to_uppercase();
    if let Ok(fret) = label.parse::<i64>() {
        return (0..=24).contains(&fret).then_some(fret);
    }
    (1..=24).find(|&fret| {
        let roman = format!(
            "{}{}",
            "X".repeat(fret as usize / 10),
            ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"][fret as usize % 10]
        );
        roman == label
    })
}
