//! Prompts and JSON schemas for metadata, header, clef, staff, tempo, and annotation OCR regions.

type Result<T> = std::result::Result<T, String>;
use serde_json::{json, Map, Value};

/// Detector roles determine the destination; OCR only transcribes the crop.
pub fn header_field(kind: &str) -> Option<&'static str> {
    match kind {
        "title" => Some("title"),
        "subtitle" => Some("subtitle"),
        "credit" => Some("artist"),
        "tuning" => Some("tuning_name"),
        "header_text" => Some("header_notes"),
        _ => None,
    }
}

pub fn document_header(kind: &str) -> bool {
    matches!(
        kind,
        "header" | "title" | "subtitle" | "credit" | "header_text"
    )
}

pub const HEADER_TEXT_PROMPT: &str = concat!(
    "Transcribe all visible text in this cropped score header, preserving its original language, ",
    "spelling and punctuation. Return one JSON object with text; use null if unreadable. Do not ",
    "infer missing words or identify the song.",
);
pub const HEADER_PROMPT: &str = concat!(
    "Read only visible song title, artist or author credit (including composer, arranger or ",
    "studio), and tuning label. Preserve the original language and spelling. Return one JSON ",
    "object with title, artist, tuning_name; use null for absent fields. Tempo, playing ",
    "techniques and chord names are not titles or author credits. Instrument and voice names are ",
    "not song titles. An instrument's transposition designation is not a tuning label.",
);

pub const DIAGRAM_NAME_PROMPT: &str = concat!(
    "Transcribe only the visible chord symbol in this cropped heading, preserving accidentals, ",
    "extensions and slash bass. Return JSON with text, or null if unreadable. Do not infer the ",
    "name from a fingering or identify the song.",
);

pub const TEMPO_PROMPT: &str = concat!(
    "Read the visible metronome marking: a beat note, an equals sign and a number. Return JSON ",
    "with beat_unit (whole, half, quarter, eighth, or sixteenth), dotted (true or false), and ",
    "bpm (the printed number). Do not convert the number. Use null for all three fields when no ",
    "complete metronome marking is visible or its beat unit or value is unreadable.",
);
pub const STAFF_PROMPT: &str = concat!(
    "Read the first staff and its instrument label. Return JSON with instrument (guitar, bass, ",
    "pitched, drums, or null if unknown) and string_count (number of TAB lines, or null if TAB ",
    "is absent), and name (the visible instrument label, preserving abbreviations, or null if ",
    "absent). Use the percussion clef for drums. Use pitched for other melodic instruments or ",
    "unlabelled melodic notation; use null for unnamed TAB. Do not infer a string count from the ",
    "five lines of standard notation.",
);
pub const CLEF_PROMPT: &str = concat!(
    "Read the visible clef. Return JSON with clef (G2, F4, C3, C4, percussion, tab, or null) and ",
    "clef_octave (0, -12, 12, -24, 24, or null if unreadable). A small 8 or 15 below the clef ",
    "lowers its pitches; above raises them. Use tab with clef_octave 0 for a TAB symbol.",
);
pub const ANNOTATION_PROMPT: &str = concat!(
    "Classify and read the visible score annotation. Return JSON with kind (instrument, ottava, ",
    "capo, chord, chord_diagram, technique, tempo, title, credit, other, or null if unreadable), ",
    "semitones (sounding minus written pitch, or null), capo (fret number, or null), and text ",
    "(visible words, or null). A chord name such as Bb or F# and a chord fingering diagram are ",
    "NOT transposition instructions. Let ring, P.M., vibrato and their continuation lines are ",
    "techniques, NOT ottava. Use title for a song title and credit for a composer, arranger, ",
    "artist or studio credit; preserve the original language. Use null pitch fields for ",
    "non-pitch annotations. For ottava use 12 for 8va, -12 for 8vb, 24 for 15ma, -24 for 15mb. ",
    "For instrument transposition use only an explicit instruction or an unambiguous instrument ",
    "label. Do not treat a key signature or tuning as an instrument transposition. For ",
    "chord_diagram also return diagram with base_fret, frets (absolute fret numbers, 0 open, \"x\" ",
    "muted, LOW to HIGH string), fingers (visible numbers or null per string), barres ([absolute ",
    "fret, zero-based low string index, high string index]). Preserve chord accidentals and ",
    "slash bass. Read only visible dots and finger numbers; never derive a fingering from the ",
    "chord name. Transcribe the evidence; do not invent missing words.",
);

fn nullable(kind: &str) -> Value {
    json!({"anyOf":[{"type":kind},{"type":"null"}]})
}

pub fn information_schema(kind: &str) -> Result<Value> {
    let mut properties = Map::new();
    match kind {
        k if header_field(k).is_some() => {
            properties.insert("text".into(), nullable("string"));
        }
        "header" => {
            for key in ["title", "artist", "tuning_name"] {
                properties.insert(key.into(), nullable("string"));
            }
        }
        "tempo" => {
            properties.insert(
                "beat_unit".into(),
                json!({"enum":["whole","half","quarter","eighth","sixteenth",null]}),
            );
            properties.insert("dotted".into(), nullable("boolean"));
            properties.insert(
                "bpm".into(),
                json!({"anyOf":[{"type":"integer","minimum":1,"maximum":999},{"type":"null"}]}),
            );
        }
        "staff" => {
            properties.insert(
                "instrument".into(),
                json!({"enum":["guitar","bass","pitched","drums",null]}),
            );
            properties.insert(
                "string_count".into(),
                json!({"anyOf":[{"type":"integer","minimum":1,"maximum":12},{"type":"null"}]}),
            );
            properties.insert("name".into(), nullable("string"));
        }
        "clef" => {
            properties.insert(
                "clef".into(),
                json!({"enum":["G2","F4","C3","C4","percussion","tab",null]}),
            );
            properties.insert("clef_octave".into(), json!({"enum":[0,-12,12,-24,24,null]}));
        }
        "annotation" | "transposition" | "text_annotation" => {
            properties.insert("kind".into(),json!({"enum":["instrument","ottava","capo","chord","chord_diagram","technique","tempo","title","credit","other",null]}));
            properties.insert(
                "semitones".into(),
                json!({"anyOf":[{"type":"integer","minimum":-36,"maximum":36},{"type":"null"}]}),
            );
            properties.insert(
                "capo".into(),
                json!({"anyOf":[{"type":"integer","minimum":0,"maximum":24},{"type":"null"}]}),
            );
            properties.insert("text".into(), nullable("string"));
        }
        _ => return Err(format!("Unsupported metadata region kind: {kind}")),
    }
    let required: Vec<String> = properties.keys().cloned().collect();
    if matches!(kind, "annotation" | "transposition" | "text_annotation") {
        properties.insert("diagram".into(),json!({"anyOf":[{"type":"null"},{"type":"object","additionalProperties":false,"required":["base_fret","frets","fingers","barres"],"properties":{"base_fret":{"type":"integer","minimum":1,"maximum":36},"frets":{"type":"array","minItems":3,"maxItems":12,"items":{"anyOf":[{"type":"integer","minimum":0,"maximum":36},{"const":"x"}]}},"fingers":{"type":"array","minItems":3,"maxItems":12,"items":{"enum":[null,0,1,2,3,4]}},"barres":{"type":"array","items":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"integer","minimum":0,"maximum":36}}}}}]}));
    }
    Ok(
        json!({"type":"object","additionalProperties":false,"required":required,"properties":properties}),
    )
}
