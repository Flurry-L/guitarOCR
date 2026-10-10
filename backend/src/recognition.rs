//! Native OCR stages. This module owns model prompts and score semantics, never
//! session persistence, task state, revision checks, or the model process.
//! Inputs are explicit local crops and schema-1.0 stage values. No Python path.
mod annotations;
pub mod chord_evidence;
pub mod clef_spatial;
pub(crate) mod headers;
pub mod instrument_text;
pub mod part_continuation;
pub(crate) mod signature_state;
pub mod staff_evidence;
pub mod staff_label;
mod staff_profile;
use crate::auxiliary_onnx::SignatureReader;
use crate::image_boundary;
use crate::score_structure::program_from_visible_name;
pub use annotations::attach_chord_annotations;
use guitarocr_engine::llama;
use image::{Rgb, RgbImage};
use regex::Regex;
use scorelib::chords::{diagram_effect, normalize_diagram};
use scorelib::score;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    io::Cursor,
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, String>;
pub const CANCELLED: &str = "Recognition cancelled; completed records can be retained";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrTask {
    Base,
    Staff,
    Clef,
}
impl OcrTask {
    fn adapter_key(self) -> Option<&'static str> {
        match self {
            Self::Base => None,
            Self::Staff => Some("staff"),
            Self::Clef => Some("clef"),
        }
    }

    fn from_region(kind: &str) -> Self {
        match kind {
            "staff" => Self::Staff,
            "clef" => Self::Clef,
            _ => Self::Base,
        }
    }
}

/// Injection point for a native engine, also allowing model-free contract tests.
pub trait Generator {
    fn task_adapter_applied(&self, _task: OcrTask) -> bool {
        false
    }
    fn generate_task(
        &mut self,
        task: OcrTask,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<llama::Generation> {
        let _ = task;
        self.generate(messages, max_tokens, schema, grammar)
    }
    fn generate_task_batch(
        &mut self,
        messages: Vec<(Value, Option<Value>, OcrTask)>,
        max_tokens: usize,
    ) -> Result<Vec<llama::Generation>> {
        messages
            .into_iter()
            .map(|(m, s, task)| self.generate_task(task, m, max_tokens, s, None))
            .collect()
    }
    fn report(&mut self, _message: &str) -> Result<()> {
        Ok(())
    }
    fn generate(
        &mut self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<llama::Generation>;
}
impl Generator for guitarocr_engine::runtime::Session<'_> {
    fn task_adapter_applied(&self, task: OcrTask) -> bool {
        task.adapter_key()
            .is_some_and(|key| self.adapter_enabled_for(key))
    }
    fn generate_task(
        &mut self,
        task: OcrTask,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<llama::Generation> {
        let adapted = self.task_adapter_applied(task);
        self.generate_with_adapter(messages, max_tokens, schema, grammar, adapted)
    }
    fn generate_task_batch(
        &mut self,
        messages: Vec<(Value, Option<Value>, OcrTask)>,
        max_tokens: usize,
    ) -> Result<Vec<llama::Generation>> {
        let messages = messages
            .into_iter()
            .map(|(m, s, t)| (m, s, self.task_adapter_applied(t)))
            .collect();
        self.generate_batch_with_adapters(messages, max_tokens)
    }
    fn report(&mut self, message: &str) -> Result<()> {
        guitarocr_engine::runtime::Session::report(self, message)
    }
    fn generate(
        &mut self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<llama::Generation> {
        guitarocr_engine::runtime::Session::generate(self, messages, max_tokens, schema, grammar)
    }
}
fn check(cancelled: &dyn Fn() -> bool) -> Result<()> {
    if cancelled() {
        Err(CANCELLED.into())
    } else {
        Ok(())
    }
}
fn text<'a>(value: &'a Value, key: &str, default: &'a str) -> &'a str {
    value[key].as_str().unwrap_or(default)
}
fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value[key].as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn flag(value: &Value, key: &str) -> bool {
    value[key].as_bool().unwrap_or(false)
}
fn tuning_needs_review(row: &Value) -> bool {
    flag(row, "tuning_needs_review") || !list(row, "tuning_needs_review").is_empty()
}
fn merge(to: &mut Value, from: &Value) {
    if let (Some(to), Some(from)) = (to.as_object_mut(), from.as_object()) {
        to.extend(from.clone());
    }
}
fn integers(value: &Value) -> Result<Vec<i64>> {
    value
        .as_array()
        .ok_or("Expected integer array")?
        .iter()
        .map(|v| v.as_i64().ok_or("Expected integer").map_err(str::to_owned))
        .collect()
}
fn integer_range(value: &Value, low: i64, high: i64) -> Value {
    value
        .as_i64()
        .filter(|v| (low..=high).contains(v))
        .map_or(Value::Null, |v| json!(v))
}
fn cleaned(value: &Value) -> Value {
    value
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map_or(Value::Null, |s| json!(s))
}
fn require_stage(value: &Value, stage: &str) -> Result<()> {
    if value["schema_version"] != "1.0" || value["stage"] != stage {
        Err(format!("Expected {stage} manifest schema 1.0"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ImagePolicy {
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub max_upscale: f64,
    pub patch_multiple: u32,
    pub page_max_pixels: u64,
}
impl Default for ImagePolicy {
    fn default() -> Self {
        Self {
            min_pixels: 256 * 256,
            max_pixels: 768 * 768,
            max_upscale: 2.5,
            patch_multiple: 28,
            page_max_pixels: 768 * 768,
        }
    }
}
impl ImagePolicy {
    pub fn from_value(v: &Value) -> Result<Self> {
        if v["version"].as_u64() != Some(2) {
            return Err("Unsupported score image policy version".into());
        }
        let mut p = Self::default();
        p.min_pixels = match v.get("min_pixels") {
            Some(value) => value.as_u64().ok_or("Invalid min_pixels")?,
            None => 256 * 256,
        };
        p.max_pixels = v["max_pixels"].as_u64().ok_or("Missing max_pixels")?;
        p.max_upscale = v["max_upscale"].as_f64().ok_or("Missing max_upscale")?;
        p.patch_multiple = v["patch_multiple"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("Invalid patch_multiple")?;
        p.page_max_pixels = match v.get("page_max_pixels") {
            Some(value) => value.as_u64().ok_or("Invalid page_max_pixels")?,
            None => p.max_pixels,
        };
        p.validate()?;
        Ok(p)
    }
    fn validate(&self) -> Result<()> {
        if self.patch_multiple == 0
            || self.patch_multiple > 256
            || self.max_pixels.min(self.page_max_pixels) < u64::from(self.patch_multiple).pow(2)
            || self.max_pixels < 112 * 112
            || self.page_max_pixels < 112 * 112
            || self.max_pixels > image_boundary::MAX_PAGE_PIXELS
            || self.page_max_pixels > image_boundary::MAX_PAGE_PIXELS
            || self.min_pixels < 112 * 112
            || self.min_pixels > self.max_pixels
            || !self.max_upscale.is_finite()
            || self.max_upscale <= 0.
            || self.max_upscale > 16.
        {
            return Err("Invalid image policy bounds".into());
        }
        Ok(())
    }
}
pub fn normalize_score_image(image: &RgbImage, policy: &ImagePolicy) -> Result<RgbImage> {
    image_boundary::check_dimensions(image.width(), image.height())?;
    policy.validate()?;
    // Recognition content cannot determine scale. Printed lines, grids and
    // font weight remain evidence for the model, never preprocessing classes.
    let area = f64::from(image.width()) * f64::from(image.height());
    let mut scale = policy
        .max_upscale
        .min((policy.min_pixels as f64 / area).sqrt().max(1.));
    let budget = if image.width() >= 600 && image.height() >= 900 {
        policy.page_max_pixels
    } else {
        policy.max_pixels
    };
    scale =
        scale.min((budget as f64 / (f64::from(image.width()) * f64::from(image.height()))).sqrt());
    let multiple = policy.patch_multiple;
    let (width, height, pw, ph) = loop {
        let w = (f64::from(image.width()) * scale).round_ties_even().max(1.) as u32;
        let h = (f64::from(image.height()) * scale)
            .round_ties_even()
            .max(1.) as u32;
        let pw = w.div_ceil(multiple) * multiple;
        let ph = h.div_ceil(multiple) * multiple;
        if u64::from(pw) * u64::from(ph) <= budget {
            break (w, h, pw, ph);
        }
        scale *= 0.98;
        if scale < 1e-12 {
            return Err("Image policy cannot fit a padded image".into());
        }
    };
    let mut pw = pw;
    while u64::from(pw) * u64::from(ph) < 112 * 112 {
        pw += multiple;
    }
    if u64::from(pw) * u64::from(ph) > budget {
        return Err("Image policy cannot fit minimum processor area".into());
    }
    image_boundary::check_dimensions(pw, ph)?;
    let resized = if image.dimensions() == (width, height) {
        image.clone()
    } else {
        crate::image_transforms::resize_rgb_lanczos(image, width, height)
    };
    let mut canvas = RgbImage::from_pixel(pw, ph, Rgb([255; 3]));
    image::imageops::replace(
        &mut canvas,
        &resized,
        i64::from((pw - width) / 2),
        i64::from((ph - height) / 2),
    );
    Ok(canvas)
}
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(ALPHABET[(n >> 18) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if c.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
pub fn image_message(image: &RgbImage, policy: Option<&ImagePolicy>) -> Result<Value> {
    let normalized = policy
        .map(|p| normalize_score_image(image, p))
        .transpose()?;
    let image = normalized.as_ref().unwrap_or(image);
    let mut data = Cursor::new(Vec::new());
    image
        .write_to(&mut data, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(
        json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}",base64(data.get_ref()))}}),
    )
}
pub fn image_messages(
    images: &[RgbImage],
    prompt: &str,
    policy: Option<&ImagePolicy>,
) -> Result<Value> {
    let mut content = images
        .iter()
        .map(|im| image_message(im, policy))
        .collect::<Result<Vec<_>>>()?;
    content.push(json!({"type":"text","text":prompt}));
    Ok(json!([{"role":"user","content":content}]))
}
fn bbox(v: &Value) -> Result<[f64; 4]> {
    let a = v["bbox"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("Missing xywh bounding box")?;
    let mut b = [0.; 4];
    for (i, v) in a.iter().enumerate() {
        b[i] = v
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or("Invalid box coordinate")?;
    }
    if b[2] <= 0. || b[3] <= 0. {
        return Err("Empty bounding box".into());
    }
    Ok(b)
}
fn direct_crop(page: &RgbImage, b: [f64; 4]) -> Result<RgbImage> {
    let x0 = b[0].round_ties_even().clamp(0., f64::from(page.width())) as u32;
    let y0 = b[1].round_ties_even().clamp(0., f64::from(page.height())) as u32;
    let x1 = (b[0] + b[2])
        .round_ties_even()
        .clamp(0., f64::from(page.width())) as u32;
    let y1 = (b[1] + b[3])
        .round_ties_even()
        .clamp(0., f64::from(page.height())) as u32;
    if x1 <= x0 || y1 <= y0 {
        return Err("Region is outside source page".into());
    }
    Ok(image::imageops::crop_imm(page, x0, y0, x1 - x0, y1 - y0).to_image())
}
pub fn record_image(row: &Value) -> Result<RgbImage> {
    if let Some(path) = row["image"].as_str().filter(|s| !s.is_empty()) {
        return image_boundary::open_rgb(Path::new(path));
    }
    let page = image_boundary::open_rgb(Path::new(
        row["source_page"]
            .as_str()
            .ok_or("Missing local image/source_page path")?,
    ))?;
    image_boundary::crop_measure(&page, bbox(row)?)
}
fn region_image(row: &Value) -> Result<RgbImage> {
    if let Some(path) = row["image"].as_str().filter(|s| !s.is_empty()) {
        return image_boundary::open_rgb(Path::new(path));
    }
    let page = image_boundary::open_rgb(Path::new(
        row["source_page"]
            .as_str()
            .ok_or("Missing local region image/source_page path")?,
    ))?;
    let mut region = row.clone();
    if let Some(crop) = row.get("crop_bbox") {
        region["bbox"] = crop.clone();
    }
    direct_crop(&page, bbox(&region)?)
}

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
pub const HEADER_TEXT_PROMPT: &str = "Transcribe all visible text in this cropped score header, preserving its original language, spelling and punctuation. Return one JSON object with text; use null if unreadable. Do not infer missing words or identify the song.";
pub const HEADER_PROMPT: &str = "Read only visible song title, artist or author credit (including composer, arranger or studio), and tuning label. Preserve the original language and spelling. Return one JSON object with title, artist, tuning_name; use null for absent fields. Tempo, playing techniques and chord names are not titles or author credits. Instrument and voice names are not song titles. An instrument's transposition designation is not a tuning label.";

pub const DIAGRAM_NAME_PROMPT: &str = "Transcribe only the visible chord symbol in this cropped heading, preserving accidentals, extensions and slash bass. Return JSON with text, or null if unreadable. Do not infer the name from a fingering or identify the song.";

pub const TEMPO_PROMPT: &str = "Read the printed quarter-note tempo. Return one JSON object with tempo_quarter; use null if unreadable.";
pub const STAFF_PROMPT: &str = "Read the first staff and its instrument label. Return JSON with instrument (guitar, bass, pitched, drums, or null if unknown) and string_count (number of TAB lines, or null if TAB is absent), and name (the visible instrument label, preserving abbreviations, or null if absent). Use the percussion clef for drums. Use pitched for other melodic instruments or unlabelled melodic notation; use null for unnamed TAB. Do not infer a string count from the five lines of standard notation.";
pub const CLEF_PROMPT: &str = "Read the visible clef. Return JSON with clef (G2, F4, C3, C4, percussion, tab, or null) and clef_octave (0, -12, 12, -24, 24, or null if unreadable). A small 8 or 15 below the clef lowers its pitches; above raises them. Use tab with clef_octave 0 for a TAB symbol.";
pub const ANNOTATION_PROMPT: &str = "Classify and read the visible score annotation. Return JSON with kind (instrument, ottava, capo, chord, chord_diagram, technique, tempo, title, credit, other, or null if unreadable), semitones (sounding minus written pitch, or null), capo (fret number, or null), and text (visible words, or null). A chord name such as Bb or F# and a chord fingering diagram are NOT transposition instructions. Let ring, P.M., vibrato and their continuation lines are techniques, NOT ottava. Use title for a song title and credit for a composer, arranger, artist or studio credit; preserve the original language. Use null pitch fields for non-pitch annotations. For ottava use 12 for 8va, -12 for 8vb, 24 for 15ma, -24 for 15mb. For instrument transposition use only an explicit instruction or an unambiguous instrument label. Do not treat a key signature or tuning as an instrument transposition. For chord_diagram also return diagram with base_fret, frets (absolute fret numbers, 0 open, \"x\" muted, LOW to HIGH string), fingers (visible numbers or null per string), barres ([absolute fret, zero-based low string index, high string index]). Preserve chord accidentals and slash bass. Read only visible dots and finger numbers; never derive a fingering from the chord name. Transcribe the evidence; do not invent missing words.";
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
                "tempo_quarter".into(),
                json!({"anyOf":[{"type":"integer","minimum":20,"maximum":400},{"type":"null"}]}),
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
pub fn parse_info_response(raw: &str, kind: &str) -> Value {
    let Ok(v) = json_response(raw) else {
        return json!({});
    };
    // A text proposal cannot establish fingering geometry.
    if kind == "text_annotation" && v["kind"] == "chord_diagram" {
        return json!({"kind":"other","text":cleaned(&v["text"]),"semitones":null,"capo":null});
    }
    match kind {
        k if header_field(k).is_some() => {
            let key = header_field(k).unwrap();
            json!({key: cleaned(&v["text"])})
        }
        "header" => {
            json!({"title":cleaned(&v["title"]),"artist":cleaned(&v["artist"]),"tuning_name":cleaned(&v["tuning_name"])})
        }
        "staff" => {
            json!({"instrument":v["instrument"].as_str().filter(|s|["guitar","bass","pitched","drums"].contains(s)),"string_count":integer_range(&v["string_count"],1,12),"name":cleaned(&v["name"])})
        }
        "clef" => {
            json!({"clef":v["clef"].as_str().filter(|s|["G2","F4","C3","C4","percussion","tab"].contains(s)),"clef_octave":v["clef_octave"].as_i64().filter(|n|[-24,-12,0,12,24].contains(n))})
        }
        "tempo" => {
            let n = integer_range(&v["tempo_quarter"], 20, 400);
            if n.is_null() {
                json!({})
            } else {
                json!({"tempo_quarter":n})
            }
        }
        "annotation" | "transposition" | "text_annotation" => {
            let mut kind = text(&v, "kind", "");
            let words = cleaned(&v["text"]);
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
            let mut shift = None;
            let mut capo = None;
            match kind{
 "instrument"=>{shift=explicit_transposition(words_str).or_else(||(program_from_visible_name(words_str).is_some()&&v["semitones"]==0).then_some(0));},
 "ottava"=>{let mark=Regex::new(r"[\s.\-–_]+").unwrap().replace_all(&words_str.to_lowercase(),"").into_owned();shift=match mark.as_str(){"8va"|"8vaalta"|"ottava"=>Some(12),"8vb"|"8vabassa"|"8ba"|"ottavabassa"=>Some(-12),"15ma"|"15maalta"=>Some(24),"15mb"|"15mabassa"|"15ba"=>Some(-24),_=>None};},
 "capo"=>{if let Some(c)=Regex::new(r"(?i)^\s*(?:capo|capodastro|capotasto|变调夹)\s*[:=.]?\s*(?:(?:at|on)\s+)?(?:fret\s+)?(\d{1,2}|[IVX]{1,5})(?:st|nd|rd|th)?\s*(?:fret|品)?\s*$").unwrap().captures(words_str){capo=c[1].parse::<i64>().ok().or_else(||(1..25).find(|&n|format!("{}{}","X".repeat(n as usize/10),["","I","II","III","IV","V","VI","VII","VIII","IX"][n as usize%10])==c[1].to_uppercase())).filter(|n|(0..=24).contains(n));}},_=>{}}
            if shift.is_none() && capo.is_none() {
                json!({"kind":null,"semitones":null,"capo":null,"text":if ["instrument","ottava","capo"].contains(&kind){words}else{Value::Null}})
            } else {
                json!({"kind":kind,"semitones":shift,"capo":capo,"text":words})
            }
        }
        _ => json!({}),
    }
}
pub fn metadata_from_predictions(predictions: &[Value]) -> Value {
    let mut m = json!({"source":"image_score_ocr"});
    let mut fields = json!({});
    for p in predictions {
        let kind = text(p, "kind", "");
        if let Some(key) = header_field(kind) {
            if let Some(value) = cleaned(&p["parsed"][key]).as_str() {
                let old = fields[key].as_str().unwrap_or("");
                if old.is_empty() {
                    fields[key] = json!(value);
                } else if matches!(key, "artist" | "header_notes")
                    && !old.lines().any(|line| line == value)
                {
                    fields[key] = json!(format!("{old}\n{value}"));
                }
            }
        } else if kind == "header" {
            if let Some(fields) = p["parsed"].as_object() {
                for (key, value) in fields {
                    if !cleaned(value).is_null() && cleaned(&m[key]).is_null() {
                        m[key] = value.clone();
                    }
                }
            }
        } else if kind == "staff" || (kind == "tempo" && m["tempo_quarter"].is_null()) {
            merge(&mut m, &p["parsed"]);
        } else if kind == "annotation"
            && p["parsed"]["kind"] == "tempo"
            && m["tempo_quarter"].is_null()
        {
            let tempo = integer_range(&p["parsed"]["tempo_quarter"], 20, 400);
            if !tempo.is_null() {
                m["tempo_quarter"] = tempo;
                continue;
            }
            if let Some(c) = Regex::new(r"♩\s*(?:[=:]\s*)?(\d{2,3})(?:\D|$)")
                .unwrap()
                .captures(text(&p["parsed"], "text", ""))
            {
                if let Ok(n) = c[1].parse::<i64>() {
                    if (20..=400).contains(&n) {
                        m["tempo_quarter"] = json!(n);
                    }
                }
            }
        }
    }
    merge(&mut m, &fields);
    // A small header text crop can recover a credit missed in the full header.
    // Only accept these roles inside the header, never from lyrics or staff labels.
    for p in predictions.iter().filter(|p| flag(p, "inside_header")) {
        let key = match text(&p["parsed"], "kind", "") {
            "title" => "title",
            "credit" => "artist",
            _ => continue,
        };
        if cleaned(&m[key]).is_null() && !cleaned(&p["parsed"]["text"]).is_null() {
            m[key] = p["parsed"]["text"].clone();
        }
    }
    // A tuning label is interpreted only after resolving this part's identity
    // and string count. "Standard" alone does not establish guitar pitches.
    m
}
pub fn standard_tuning(instrument: &str, count: Option<usize>) -> Result<Vec<i64>> {
    let count = count.unwrap_or(if instrument == "bass" { 4 } else { 6 });
    let pitches = match (instrument, count) {
        ("pitched" | "drums", _) => vec![],
        ("guitar", 4) => vec![64, 59, 55, 50],
        ("guitar", 5) => vec![64, 59, 55, 50, 45],
        ("guitar", 6) => vec![64, 59, 55, 50, 45, 40],
        ("guitar", 7) => vec![64, 59, 55, 50, 45, 40, 35],
        ("guitar", 8) => vec![64, 59, 55, 50, 45, 40, 35, 30],
        ("bass", 4) => vec![43, 38, 33, 28],
        ("bass", 5) => vec![43, 38, 33, 28, 23],
        ("bass", 6) => vec![48, 43, 38, 33, 28, 23],
        ("bass", 7) => vec![53, 48, 43, 38, 33, 28, 23],
        _ => {
            return Err(format!(
                "No unambiguous default tuning for {count}-string {instrument}; enter open-string pitches"
            ));
        }
    };
    Ok(pitches)
}
pub fn tuning_from_name(name: &str, instrument: &str, count: Option<usize>) -> Option<Vec<i64>> {
    let label = Regex::new(r"[\s\-–]+")
        .unwrap()
        .replace_all(
            &name
                .trim()
                .to_lowercase()
                .replace('½', "1/2")
                .replace('♭', "b"),
            " ",
        )
        .into_owned();
    // Strip only the explicit instrument matching the selected part. Unknown
    // labels and conflicting instrument names remain unresolved.
    let mut label = label.as_str();
    for prefix in ["guitar", "bass"] {
        if let Some(rest) = label.strip_prefix(&format!("{prefix} ")) {
            if prefix != instrument {
                return None;
            }
            label = rest;
            break;
        }
    }
    let label = label.strip_suffix(" tuning").unwrap_or(label);
    let label = match label {
        "half step down" | "1/2 step down" | "eb standard" | "e flat standard" => {
            "tune down 1/2 step"
        }
        "whole step down" | "full step down" | "one step down" | "d standard" => "tune down 1 step",
        "two steps down" | "c standard" => "tune down 2 step",
        other => other,
    };
    if instrument == "guitar" && (count.is_none() || count == Some(6)) {
        return match label {
            "standard" => Some(vec![64, 59, 55, 50, 45, 40]),
            "dropped d" | "drop d" => Some(vec![64, 59, 55, 50, 45, 38]),
            "dropped c" | "drop c" => Some(vec![62, 57, 53, 48, 43, 36]),
            "dropped b" | "drop b" => Some(vec![61, 56, 52, 47, 42, 35]),
            "dropped d tune down 1/2 step" => Some(vec![63, 58, 54, 49, 44, 37]),
            "tune down 1/2 step" => Some(vec![63, 58, 54, 49, 44, 39]),
            "tune down 1 step" => Some(vec![62, 57, 53, 48, 43, 38]),
            "tune down 2 step" => Some(vec![60, 55, 51, 46, 41, 36]),
            "dadgad" => Some(vec![62, 57, 55, 50, 45, 38]),
            "open d" => Some(vec![62, 57, 54, 50, 45, 38]),
            "open g" => Some(vec![62, 59, 55, 50, 43, 38]),
            "open e" => Some(vec![64, 59, 56, 52, 47, 40]),
            "open a" => Some(vec![64, 61, 57, 52, 45, 40]),
            "double drop d" => Some(vec![62, 59, 55, 50, 45, 38]),
            _ => None,
        };
    }
    if !["guitar", "bass"].contains(&instrument)
        || (instrument == "guitar" && matches!(count, Some(4 | 5)))
    {
        return None;
    }
    let mut tuning = standard_tuning(instrument, count).ok()?;
    let shift = match label {
        "standard" => 0,
        "tune down 1/2 step" => -1,
        "tune down 1 step" => -2,
        "tune down 2 step" => -4,
        _ => {
            if instrument != "bass" || tuning.len() != 4 {
                return None;
            }
            tuning[3] = 26;
            match label {
                "drop d" | "dropped d" => 0,
                "drop c" | "dropped c" => -2,
                "drop b" | "dropped b" => -3,
                "dropped d tune down 1/2 step" => -1,
                _ => return None,
            }
        }
    };
    Some(tuning.iter().map(|p| p + shift).collect())
}

#[derive(Clone, Debug)]
pub struct InformationConfig {
    pub layout_path: PathBuf,
    pub fallback_title: String,
    pub image_policy: Option<ImagePolicy>,
    pub instrument: Option<String>,
    pub midi_program: Option<u8>,
    pub tuning: Option<Vec<i64>>,
    pub capo: Option<u8>,
    pub transpose: Option<i8>,
    pub staff_profile: bool,
    pub pitch_context: bool,
    pub score_structure: bool,
    /// Image evidence models supplied by the model catalog.
    pub chord_evidence_model: Option<PathBuf>,
    pub structure_relation_model: Option<PathBuf>,
    pub staff_evidence_model: Option<PathBuf>,
    pub staff_label_model: Option<PathBuf>,
    pub instrument_text_model: Option<PathBuf>,
    pub part_continuation_model: Option<PathBuf>,
    pub clef_spatial_model: Option<PathBuf>,
    pub text_regions_model: Option<PathBuf>,
    pub chord_text_roles_model: Option<PathBuf>,
    /// Dedicated native stage directory for geometry-produced fused crops.
    pub structure_output: Option<PathBuf>,
}
impl Default for InformationConfig {
    fn default() -> Self {
        Self {
            layout_path: PathBuf::new(),
            fallback_title: String::new(),
            image_policy: None,
            instrument: None,
            midi_program: None,
            tuning: None,
            capo: None,
            transpose: None,
            staff_profile: true,
            pitch_context: true,
            score_structure: true,
            chord_evidence_model: None,
            structure_relation_model: None,
            staff_evidence_model: None,
            staff_label_model: None,
            instrument_text_model: None,
            part_continuation_model: None,
            clef_spatial_model: None,
            text_regions_model: None,
            chord_text_roles_model: None,
            structure_output: None,
        }
    }
}
pub struct InformationOutput {
    pub manifest: Value,
    pub predictions: Vec<Value>,
    pub structure_predictions: Vec<Value>,
}
fn region_prompt(kind: &str) -> Result<&'static str> {
    match kind {
        k if header_field(k).is_some() => Ok(HEADER_TEXT_PROMPT),
        "header" => Ok(HEADER_PROMPT),
        "text_annotation" => Ok(ANNOTATION_PROMPT),
        "tempo" => Ok(TEMPO_PROMPT),
        "staff" => Ok(STAFF_PROMPT),
        "clef" => Ok(CLEF_PROMPT),
        "annotation" | "transposition" => Ok(ANNOTATION_PROMPT),
        _ => Err(format!("Unsupported metadata region: {kind}")),
    }
}
pub fn recognize_regions<G: Generator + ?Sized>(
    engine: &mut G,
    regions: &[Value],
    policy: Option<&ImagePolicy>,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<Value>> {
    recognize_regions_with_evidence(engine, regions, policy, cancelled, None, None)
}

fn recognize_regions_with_evidence<G: Generator + ?Sized>(
    engine: &mut G,
    regions: &[Value],
    policy: Option<&ImagePolicy>,
    cancelled: &dyn Fn() -> bool,
    mut reader: Option<&mut chord_evidence::ChordEvidenceReader>,
    mut clef_reader: Option<&mut clef_spatial::ClefSpatialReader>,
) -> Result<Vec<Value>> {
    let mut result = vec![];
    for (batch_index, batch) in regions.chunks(4).enumerate() {
        check(cancelled)?;
        engine.report(&format!(
            "正在读取谱面文字与和弦：{} / {}",
            batch_index * 4,
            regions.len()
        ))?;
        let images = batch.iter().map(region_image).collect::<Result<Vec<_>>>()?;
        let evidence = reader
            .as_deref_mut()
            .map(|reader| reader.predict(&images))
            .transpose()?;
        let mut messages = Vec::new();
        for (index, (region, source)) in batch.iter().zip(&images).enumerate() {
            let kind = text(region, "kind", "");
            let read_kind = if region["source"] == "text_region_network" {
                "text_annotation"
            } else {
                kind
            };
            let schema = information_schema(read_kind)?;
            let diagram = matches!(kind, "annotation" | "transposition")
                && evidence
                    .as_ref()
                    .is_some_and(|rows| rows[index]["kind"] == "diagram");
            let mut image = source.clone();
            if header_field(kind).is_some() || diagram {
                let height = if header_field(kind).is_some() {
                    144.
                } else {
                    288.
                };
                let scale = (height / image.height() as f64)
                    .min(4.)
                    .min(2048. / image.width() as f64);
                if scale > 1. {
                    image = crate::image_transforms::resize_rgb_lanczos(
                        &image,
                        (image.width() as f64 * scale).round() as u32,
                        (image.height() as f64 * scale).round() as u32,
                    );
                }
            }
            messages.push((
                image_messages(&[image], region_prompt(read_kind)?, policy)?,
                Some(schema),
                OcrTask::from_region(read_kind),
            ));
        }
        let responses = engine.generate_task_batch(messages, 512)?;
        if responses.len() != batch.len() {
            return Err("Incomplete information batch".into());
        }
        for (index, ((region, image), mut response)) in
            batch.iter().zip(&images).zip(responses).enumerate()
        {
            let kind = text(region, "kind", "");
            check(cancelled)?;
            let read_kind = if region["source"] == "text_region_network" {
                "text_annotation"
            } else {
                kind
            };
            let mut parsed = parse_info_response(response.content.trim(), read_kind);
            let mut visual_evidence = Value::Null;
            let mut attempts = vec![
                json!({"raw":response.content,"parsed":parsed,"generated_token_count":response.completion_tokens}),
            ];
            if kind == "title" && cleaned(&parsed["title"]).is_null() {
                // A detected title is required metadata evidence. Try one focused
                // reread before reporting it as unreadable, without inventing text.
                let scale = (216. / image.height() as f64)
                    .min(6.)
                    .min(2048. / image.width() as f64)
                    .max(1.);
                let crop = crate::image_transforms::resize_rgb_lanczos(
                    image,
                    (image.width() as f64 * scale).round() as u32,
                    (image.height() as f64 * scale).round() as u32,
                );
                check(cancelled)?;
                response = engine.generate(
                    image_messages(&[crop], HEADER_TEXT_PROMPT, policy)?,
                    512,
                    Some(information_schema("title")?),
                    None,
                )?;
                parsed = parse_info_response(response.content.trim(), kind);
                attempts.push(json!({"raw":response.content,"parsed":parsed,"generated_token_count":response.completion_tokens}));
            }
            if matches!(kind, "annotation" | "transposition") {
                if let Some(evidence) = evidence.as_ref().filter(|_| read_kind != "text_annotation")
                {
                    (parsed, visual_evidence) =
                        chord_evidence::apply_evidence(&parsed, &evidence[index])?;
                } else if parsed["kind"] == "chord_diagram" {
                    visual_evidence = json!({"status":"unavailable","review_required":true});
                }
                if parsed["kind"] == "tempo" {
                    check(cancelled)?;
                    let tempo = engine.generate(
                        image_messages(&[image.clone()], TEMPO_PROMPT, policy)?,
                        64,
                        Some(information_schema("tempo")?),
                        None,
                    )?;
                    let value = parse_info_response(tempo.content.trim(), "tempo");
                    attempts.push(json!({"purpose":"tempo_value","raw":tempo.content,
                        "parsed":value,"generated_token_count":tempo.completion_tokens}));
                    merge(&mut parsed, &value);
                }
            }
            let confirmed = ["instrument", "ottava", "capo"].contains(&text(&parsed, "kind", ""))
                && (!parsed["semitones"].is_null() || !parsed["capo"].is_null());
            let resolved = if ["annotation", "transposition"].contains(&kind) {
                if confirmed {
                    "transposition"
                } else {
                    "annotation"
                }
            } else {
                kind
            };
            let mut p = region.clone();
            p["task_adapter_applied"] =
                json!(engine.task_adapter_applied(OcrTask::from_region(read_kind)));
            p["inside_header"] = json!(regions
                .iter()
                .any(|header| { header["kind"] == "header" && region_inside(region, header) }));
            merge(
                &mut p,
                &json!({"kind":resolved,"candidate_kind":kind,"raw":response.content.trim(),"parsed":parsed,"generated_token_count":response.completion_tokens,"attempts":attempts}),
            );
            if !visual_evidence.is_null() {
                p["chord_evidence"] = visual_evidence;
            }
            if read_kind == "clef" {
                if let Some(reader) = clef_reader.as_deref_mut() {
                    let evidence = reader.predict(image)?;
                    let (parsed, recorded) = clef_spatial::apply_evidence(&p["parsed"], &evidence);
                    p["parsed"] = parsed;
                    p["clef_spatial_evidence"] = recorded;
                }
            }
            result.push(p);
        }
    }
    engine.report(&format!(
        "谱面文字与和弦读取完成：{} / {}",
        regions.len(),
        regions.len()
    ))?;
    Ok(result)
}

fn region_inside(region: &Value, container: &Value) -> bool {
    if region["page"] != container["page"] {
        return false;
    }
    let (Ok(a), Ok(b)) = (bbox(region), bbox(container)) else {
        return false;
    };
    a[0] >= b[0] && a[1] >= b[1] && a[0] + a[2] <= b[0] + b[2] && a[1] + a[3] <= b[1] + b[3]
}

fn staff_image(rows: &[Value]) -> Result<Option<(Value, RgbImage)>> {
    let Some(first) = rows.first() else {
        return Ok(None);
    };
    let Some(path) = first["source_page"].as_str() else {
        return Ok(None);
    };
    let row_id = first.get("row_index").unwrap_or(&first["system_index"]);
    let same: Vec<&Value> = rows
        .iter()
        .filter(|r| {
            r["page"] == first["page"] && r.get("row_index").unwrap_or(&r["system_index"]) == row_id
        })
        .collect();
    let mut top = f64::INFINITY;
    let mut bottom = 0.0f64;
    for r in same {
        let b = bbox(r)?;
        top = top.min(b[1]);
        bottom = bottom.max(b[1] + b[3]);
    }
    let page = image_boundary::open_rgb(Path::new(path))?;
    let y0 = (top.trunc() - 12.).max(0.);
    let y1 = (bottom.trunc() + 13.).min(f64::from(page.height()));
    let crop = direct_crop(&page, [0., y0, f64::from(page.width()), y1 - y0])?;
    Ok(Some((
        json!({"kind":"staff","page":first["page"],"source_page":path,"bbox":[0.,y0,page.width(),y1-y0],"part_id":first.get("part_id").cloned().unwrap_or(json!("part-1"))}),
        crop,
    )))
}
fn glyph(c: char) -> [u8; 7] {
    match c {
        'R' => [30, 17, 17, 30, 20, 18, 17],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        _ => [0; 7],
    }
}
/// Stable embedded glyphs avoid a desktop-font/runtime dependency.
pub fn structure_image(page: &RgbImage, rows: &[Vec<Value>]) -> Result<RgbImage> {
    let mut marked = page.clone();
    for (i, rows) in rows.iter().enumerate() {
        let mut top = f64::INFINITY;
        let mut bottom = 0.0f64;
        for row in rows {
            let b = bbox(row)?;
            let band = crate::layout_postprocess::staff_band(row).unwrap_or([b[1], b[1] + b[3]]);
            top = top.min(band[0]);
            bottom = bottom.max(band[1]);
        }
        let label = format!("R{}", i + 1);
        let width = label.len() as u32 * 18 + 6;
        let x = marked.width().saturating_sub(width);
        let y = ((top + bottom) / 2. - 14.).round_ties_even().max(0.) as u32;
        for yy in y.saturating_sub(2)..(y + 31).min(marked.height()) {
            for xx in x.saturating_sub(2)..marked.width() {
                marked.put_pixel(xx, yy, Rgb([255; 3]));
            }
        }
        for (j, c) in label.chars().enumerate() {
            for (gy, bits) in glyph(c).iter().enumerate() {
                for gx in 0..5 {
                    if bits & (1 << (4 - gx)) != 0 {
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let (xx, yy) =
                                    (x + j as u32 * 18 + gx * 3 + dx, y + gy as u32 * 3 + dy);
                                if xx < marked.width() && yy < marked.height() {
                                    marked.put_pixel(xx, yy, Rgb([0, 65, 220]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let scale = 1.0f64
        .min(1200. / f64::from(marked.width()))
        .min(1680. / f64::from(marked.height()));
    if scale < 1. {
        marked = crate::image_transforms::resize_rgb_lanczos(
            &marked,
            (f64::from(marked.width()) * scale)
                .round_ties_even()
                .max(1.) as u32,
            (f64::from(marked.height()) * scale)
                .round_ties_even()
                .max(1.) as u32,
        );
    }
    Ok(marked)
}
pub fn structure_prompt(count: usize, modes: &[String]) -> String {
    let mut p="Read the score structure. Each staff row is marked R1, R2, ... in blue. A notation+TAB pair is ONE row of the same part. A piano grand staff has TWO rows of the same part, staff 0 for right hand and staff 1 for left hand. Group simultaneous parts into systems using brackets, barlines and names. Return JSON {\"parts\":[{\"name\":printed instrument name,\"instrument\":guitar|bass|pitched|drums,\"strings\":TAB line count or null,\"program\":zero-based GM program}],\"rows\":[[system,part,staff],...]}. Indices start at 0; rows follow the blue numbers. Keep the same part index in successive systems. Do not infer TAB strings from notation lines. If notation and TAB of one instrument have separate blue row labels, assign both rows the SAME system, part and staff indices. Different printed part names must keep different part indices, including Guitar and Guitar II or multiple parts of the same instrument. A row that already contains notation+TAB cannot pair with another TAB row. ".to_owned();
    p += &format!("There are {count} marked rows.");
    if modes.iter().any(|m| m == "notation") && modes.iter().any(|m| m == "tab") {
        p += &format!(
            " Detected row types: {}.",
            modes
                .iter()
                .enumerate()
                .map(|(i, m)| format!("R{}={m}", i + 1))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    p
}
fn structure_schema(count: usize) -> Value {
    json!({"type":"object","additionalProperties":false,"required":["parts","rows"],"properties":{"parts":{"type":"array","minItems":1,"maxItems":count,"items":{"type":"object","additionalProperties":false,"required":["name","instrument","strings","program"],"properties":{"name":{"type":"string"},"instrument":{"enum":["guitar","bass","pitched","drums"]},"strings":{"anyOf":[{"type":"null"},{"type":"integer","minimum":1,"maximum":12}]},"program":{"type":"integer","minimum":0,"maximum":127}}}},"rows":{"type":"array","minItems":count,"maxItems":count,"items":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"integer","minimum":0}}}}})
}
fn majority_mode(rows: &[Value], fallback: &str) -> String {
    let mut counts: Vec<(String, usize)> = vec![];
    for r in rows {
        let mode = text(r, "mode", fallback);
        if let Some(c) = counts.iter_mut().find(|(s, _)| s == mode) {
            c.1 += 1;
        } else {
            counts.push((mode.into(), 1));
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(std::cmp::Ordering::Greater))
        .map(|x| x.0)
        .unwrap_or_else(|| fallback.into())
}
fn structure_row_centers(
    rows: &[Vec<Value>],
    source_height: u32,
    marked_height: u32,
) -> Result<Vec<f32>> {
    if source_height == 0 || marked_height == 0 {
        return Err("Structure page dimensions must be nonzero".into());
    }
    // The relation image has already been proportionally resized for the VLM.
    // Detector boxes remain in source-page coordinates. Use the actual rounded
    // output height, matching Python's read_relations transformation.
    let scale = f64::from(marked_height) / f64::from(source_height);
    rows.iter()
        .map(|row| {
            let bands =
                row.iter()
                    .map(|record| {
                        let b = bbox(record)?;
                        Ok(crate::layout_postprocess::staff_band(record)
                            .unwrap_or([b[1], b[1] + b[3]]))
                    })
                    .collect::<Result<Vec<_>>>()?;
            let top = bands.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
            let bottom = bands.iter().map(|b| b[1]).fold(0., f64::max);
            Ok(((top + bottom) / 2. * scale) as f32)
        })
        .collect()
}
pub fn recognize_structure<G: Generator + ?Sized>(
    engine: &mut G,
    records: &[Value],
    config: &InformationConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<Value>> {
    let grouped = crate::score_structure::staff_rows(records)?;
    let mut pages: BTreeMap<i64, Vec<Vec<Value>>> = BTreeMap::new();
    for row in grouped {
        let page = row
            .first()
            .and_then(|r| r["page"].as_i64())
            .ok_or("Staff row has no page number")?;
        pages.entry(page).or_default().push(row);
    }
    let mut predictions = vec![];
    let mut relation_reader = config
        .structure_relation_model
        .as_deref()
        .map(crate::structure_relation::StructureRelation::new)
        .transpose()?;
    let mut staff_reader = (config.staff_profile && config.staff_label_model.is_some())
        .then(|| staff_profile::StaffProfileReader::new(config))
        .transpose()?;
    if config.part_continuation_model.is_some() && staff_reader.is_none() {
        return Err("Part continuation requires observed staff label profiles".into());
    }
    let mut continuation = config
        .part_continuation_model
        .as_deref()
        .map(part_continuation::PartContinuationReader::new)
        .transpose()?;
    let mut identity_preview = vec![];
    let mut previous_profiles = vec![];
    let mut previous_page = 0;
    let mut previous_document = String::new();
    for (&page, rows) in &pages {
        check(cancelled)?;
        engine.report(&format!("正在读取第 {page} 页的声部与谱表结构"))?;
        let path = rows[0][0]["source_page"]
            .as_str()
            .ok_or("Structure requires source_page images")?;
        let original = image_boundary::open_rgb(Path::new(path))?;
        let marked = structure_image(&original, rows)?;
        let document = rows[0][0]["source_pdf"].as_str().unwrap_or("");
        let modes: Vec<String> = rows.iter().map(|r| majority_mode(r, "notation")).collect();
        let centers = structure_row_centers(rows, original.height(), marked.height())?;
        let relation = relation_reader
            .as_mut()
            .map(|reader| reader.predict(&marked, &centers, &modes))
            .transpose()?;
        let systems = relation.as_ref().map(|relation| &relation.systems);
        let mut prompt = structure_prompt(rows.len(), &modes);
        if let Some(systems) = systems {
            prompt += &format!(
                " A visual row-relation network proposes system indices {systems:?} for the marked rows. Read the image to verify these groups and assign every visible part and staff."
            );
        }
        let schema = structure_schema(rows.len());
        let initial = image_messages(&[marked], &prompt, config.image_policy.as_ref())?;
        let mut messages = initial.clone();
        let mut parsed = None;
        let mut resolution_audit = Value::Null;
        let mut raw = String::new();
        let mut attempts = vec![];
        for attempt in 1..=3 {
            check(cancelled)?;
            let response = engine.generate(messages.clone(), 2048, Some(schema.clone()), None)?;
            raw = response.content;
            let result = if let Some(relation) = &relation {
                crate::score_structure::parse_structure_with_relation_heads(
                    &raw,
                    rows.len(),
                    Some(&relation.systems),
                    &relation.same_system_probabilities,
                    relation.same_part_probabilities.as_deref(),
                    Some(&modes),
                    relation.review_confidence,
                    relation.part_review_confidence,
                )
            } else {
                crate::score_structure::parse_structure_with_audit_proposal(
                    &raw,
                    rows.len(),
                    None,
                    Some(&modes),
                )
            };
            match result {
                Ok((value, mut audit)) => {
                    audit["system_indices_constrained"] = json!(false);
                    audit["system_grouping_source"] = audit
                        .get("selected_system_source")
                        .cloned()
                        .unwrap_or(json!("structure_model"));
                    if let Some(relation) = &relation {
                        audit["relation_prediction"] = json!({"systems":relation.systems,
                            "same_system_probabilities":relation.same_system_probabilities,
                            "same_part_probabilities":relation.same_part_probabilities,
                            "needs_review":relation.needs_review,"review_confidence":relation.review_confidence,
                            "part_review_confidence":relation.part_review_confidence});
                    }
                    resolution_audit = audit;
                    attempts.push(json!({"attempt":attempt,"raw":raw,"accepted":true}));
                    parsed = Some(value);
                    break;
                }
                Err(error) => {
                    attempts
                        .push(json!({"attempt":attempt,"raw":raw,"accepted":false,"error":error}));
                    messages = initial.clone();
                    messages.as_array_mut().unwrap().extend([json!({"role":"assistant","content":raw}),json!({"role":"user","content":format!("Correct the score structure. {} Return JSON covering every marked row exactly once.",error.chars().take(2000).collect::<String>())})]);
                }
            }
        }
        let mut parsed = parsed.ok_or_else(|| {
            format!("Could not resolve score parts on page {page}; check staff boxes and retry")
        })?;
        if let Some(reader) = staff_reader.as_mut() {
            let assignments = parsed["rows"]
                .as_array()
                .ok_or("Missing structure rows")?
                .clone();
            let profiles = parsed["parts"]
                .as_array_mut()
                .ok_or("Missing structure parts")?;
            for (part, profile) in profiles.iter_mut().enumerate() {
                let Some(first) = assignments
                    .iter()
                    .find(|a| a[1].as_u64() == Some(part as u64))
                else {
                    continue;
                };
                let local_rows: Vec<Value> = rows
                    .iter()
                    .zip(&assignments)
                    .filter(|(_, assignment)| {
                        assignment[1].as_u64() == Some(part as u64)
                            && assignment[0] == first[0]
                            && assignment[2] == first[2]
                    })
                    .flat_map(|(boxes, _)| boxes.iter().cloned())
                    // A notation/TAB pair can occupy separate detector rows.
                    // The learned structure assigns both to this same staff;
                    // read their complete union without altering stored boxes.
                    .map(|mut row| {
                        row["row_index"] = json!(0);
                        row
                    })
                    .collect();
                if let Some(prediction) =
                    reader.read(engine, &local_rows, config.image_policy.as_ref(), cancelled)?
                {
                    let has_tab = local_rows
                        .iter()
                        .any(|r| matches!(text(r, "mode", "notation"), "tab" | "both"));
                    staff_profile::apply(profile, &prediction, has_tab);
                }
            }
        }
        if let Some(reader) = continuation.as_mut() {
            let assignments: Vec<[i64; 3]> =
                serde_json::from_value(parsed["rows"].clone()).map_err(|e| e.to_string())?;
            let batches = crate::score_structure::identity_batches(
                &assignments,
                parsed["anonymous_relation_groups"] == true,
            )?;
            let profiles = parsed["parts"]
                .as_array_mut()
                .ok_or("Missing structure parts")?;
            for (system, assignments) in batches {
                let active: Vec<usize> = assignments
                    .iter()
                    .map(|row| row[1] as usize)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if previous_document == document {
                    part_continuation::propose(
                        &previous_profiles,
                        profiles,
                        &active,
                        reader,
                        previous_page,
                        cancelled,
                    )?;
                    if system.is_some() {
                        for &local in &active {
                            if profiles[local]["continuation_evidence"].is_object() {
                                profiles[local]["continuation_evidence"]["scope"] =
                                    json!(if previous_page == page as u64 {
                                        "adjacent_systems"
                                    } else {
                                        "adjacent_pages"
                                    });
                            }
                        }
                    }
                }
                // Preview and score_grid share the same system-local identity
                // batches, including the existing unique-name/match checks.
                let (mapping, _) = crate::score_structure::resolve_identity_batch(
                    profiles,
                    &assignments,
                    page as u64,
                    system,
                    &mut identity_preview,
                )?;
                previous_profiles = active
                    .into_iter()
                    .map(|local| {
                        let mut profile = profiles[local].clone();
                        profile["id"] = identity_preview[mapping[&local]]["id"].clone();
                        profile
                    })
                    .collect();
                previous_page = page as u64;
                previous_document = document.to_owned();
            }
        }
        predictions.push(json!({"page":page,"source_page":path,"raw":raw,"parsed":parsed,
            "systems_from_image":systems,"attempts":attempts,
            "structure_needs_review":resolution_audit["needs_review"],"resolution_audit":resolution_audit}));
    }
    Ok(predictions)
}

fn distance(record: &Value, region: &Value) -> Result<(f64, f64, f64)> {
    let a = bbox(record)?;
    let b = bbox(region)?;
    let center = b[1] + b[3] / 2.;
    Ok((
        (a[1] - center).max(center - a[1] - a[3]).max(0.),
        (center - a[1] - a[3] / 2.).abs(),
        (a[0] - b[0]).abs(),
    ))
}
fn nearest(records: &[Value], region: &Value) -> Option<usize> {
    records
        .iter()
        .enumerate()
        .filter(|(_, r)| r["page"] == region["page"])
        .filter_map(|(i, r)| distance(r, region).ok().map(|d| (i, d)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

fn staff_key(row: &Value) -> String {
    format!(
        "{}\u{0}{}\u{0}{}",
        text(row, "source_id", ""),
        text(row, "part_id", "part-1"),
        text(row, "staff_id", "staff-1")
    )
}
fn bar_index(row: &Value) -> i64 {
    row["bar_index"]
        .as_i64()
        .or_else(|| row["measure_index"].as_i64())
        .unwrap_or_else(|| row["measure_number"].as_i64().unwrap_or(1) - 1)
}

fn apply_clef_changes(
    state: &mut Value,
    pending: &mut HashSet<String>,
    changes: &[&Value],
    measure_number: &Value,
) {
    let clefs: Vec<_> = changes
        .iter()
        .filter(|p| p["kind"] == "clef" && p["parsed"]["clef"] != "tab")
        .collect();
    if clefs.is_empty() {
        return;
    }
    // A single staff/measure context cannot encode competing clefs or an
    // event-level change inside the bar. Preserve the evidence, not whichever
    // candidate the detector happened to return last.
    let mut unresolved = Vec::new();
    for field in ["clef", "clef_octave"] {
        let mut values = Vec::new();
        for p in &clefs {
            let value = &p["parsed"][field];
            if !value.is_null() && !values.contains(value) {
                values.push(value.clone());
            }
        }
        let unknown = clefs.iter().any(|p| {
            p["parsed"][field].is_null()
                && !clefs.iter().any(|other| {
                    !other["parsed"][field].is_null()
                        && p["page"] == other["page"]
                        && p["bbox"] == other["bbox"]
                })
        });
        if values.len() == 1 && !unknown {
            state[field] = values.pop().unwrap();
            pending.remove(field);
        } else {
            state[field] = Value::Null;
            pending.insert(field.into());
            unresolved.push(field);
        }
    }
    if unresolved.is_empty() {
        state.as_object_mut().unwrap().remove("clef_evidence");
    } else {
        let mut candidates: Vec<_> = clefs
            .iter()
            .map(|p| {
                let mut candidate = json!({});
                for field in ["page", "bbox", "parsed", "clef_spatial_evidence"] {
                    if let Some(value) = p.get(field) {
                        candidate[field] = value.clone();
                    }
                }
                candidate
            })
            .collect();
        candidates.sort_by_key(Value::to_string);
        state["clef_evidence"] = json!({"status":"unresolved","scope":"staff_measure",
            "measure_number":measure_number,"fields":unresolved,"candidates":candidates});
    }
}

/// Assign changes to their actual staff; ottava spans retain horizontal scope.
pub fn apply_pitch_regions(
    records: &[Value],
    predictions: &[Value],
    instrument: &str,
    transpose: Option<i64>,
    default_transpose: Option<i64>,
) -> Result<Vec<Value>> {
    let mut records = records.to_vec();
    let mut changes: HashMap<usize, Vec<&Value>> = HashMap::new();
    let mut spans: HashMap<usize, Vec<Value>> = HashMap::new();
    let mut unresolved = HashSet::new();
    for p in predictions {
        let kind = text(p, "kind", "");
        if !["clef", "transposition"].contains(&kind) || p["bbox"].is_null() {
            continue;
        }
        let pk = text(&p["parsed"], "kind", "");
        if kind == "transposition" && !["instrument", "ottava", "capo"].contains(&pk) {
            continue;
        }
        let Some(mut anchor) = nearest(&records, p) else {
            continue;
        };
        let ab = bbox(&records[anchor])?;
        let pb = bbox(p)?;
        let same: Vec<usize> = records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let b = bbox(r).ok()?;
                (r["page"] == p["page"]
                    && r["system_index"] == records[anchor]["system_index"]
                    && staff_key(r) == staff_key(&records[anchor])
                    && (b[1] + b[3] / 2. - ab[1] - ab[3] / 2.).abs() < b[3].min(ab[3]) / 2.)
                    .then_some(i)
            })
            .collect();
        if kind == "transposition" && pk == "ottava" {
            for i in same {
                let b = bbox(&records[i])?;
                let left = b[0].max(pb[0]);
                let right = (b[0] + b[2]).min(pb[0] + pb[2]);
                if right - left <= 2.0f64.max(b[2] * 0.015) {
                    continue;
                }
                let mut region = json!({});
                for key in ["page", "bbox", "image"] {
                    if let Some(v) = p.get(key) {
                        region[key] = v.clone();
                    }
                }
                spans.entry(i).or_default().push(json!({"semitones":p["parsed"]["semitones"],"start":((left-b[0])/b[2]*10000.).round_ties_even()/10000.,"end":((right-b[0])/b[2]*10000.).round_ties_even()/10000.,"region":region}));
                if p["parsed"]["semitones"].is_null() {
                    unresolved.insert(i);
                }
            }
        } else {
            let point = if kind == "clef" {
                pb[0] + pb[2] / 2.
            } else {
                pb[0]
            };
            if let Some(&i) = same
                .iter()
                .find(|&&i| bbox(&records[i]).is_ok_and(|b| b[0] <= point && point < b[0] + b[2]))
            {
                anchor = i;
            } else if kind != "clef" {
                if let Some(&i) = same.iter().min() {
                    anchor = i;
                }
            }
            changes.entry(anchor).or_default().push(p);
        }
    }
    let mut states: HashMap<String, Value> = HashMap::new();
    let mut uncertain: HashMap<String, HashSet<String>> = HashMap::new();
    let mut explicit = HashSet::new();
    for (i, r) in records.iter_mut().enumerate() {
        let instrument = text(r, "instrument", instrument).to_owned();
        let key = staff_key(r);
        let state=states.entry(key.clone()).or_insert_with(||json!({"instrument_transpose":transpose.or(default_transpose).unwrap_or(if ["guitar","bass"].contains(&instrument.as_str()){-12}else{0}),"clef":null,"clef_octave":0}));
        let pending = uncertain.entry(key.clone()).or_default();
        apply_clef_changes(
            state,
            pending,
            changes.get(&i).map(Vec::as_slice).unwrap_or(&[]),
            &r["measure_number"],
        );
        for p in changes.get(&i).into_iter().flatten() {
            let parsed = &p["parsed"];
            if p["kind"] == "clef" {
                continue;
            } else {
                match text(parsed, "kind", "") {
                    "instrument" if transpose.is_none() => {
                        if parsed["semitones"].is_null() {
                            pending.insert("instrument_transpose".into());
                        } else {
                            state["instrument_transpose"] = parsed["semitones"].clone();
                            if conventional_octave(text(parsed, "text", "")).is_none() {
                                explicit.insert(key.clone());
                            }
                            pending.remove("instrument_transpose");
                        }
                    }
                    "capo" => {
                        if parsed["capo"].is_null() {
                            pending.insert("capo".into());
                        } else {
                            state["capo"] = parsed["capo"].clone();
                            pending.remove("capo");
                        }
                    }
                    "instrument" => {}
                    _ => {
                        unresolved.insert(i);
                    }
                }
            }
        }
        let mut local_clef = json!({});
        if changes.get(&i).is_some_and(|changes| {
            changes
                .iter()
                .any(|p| p["kind"] == "clef" && p["parsed"]["clef"] != "tab")
        }) {
            for field in ["clef", "clef_octave"] {
                local_clef[field] = json!({"status":if state[field].is_null() || pending.contains(field) {"unknown"}else{"observed"},
                    "value":state[field],"source":"information_ocr"});
            }
        }
        r["clef_observation"] = local_clef;
        let mut effective = state.clone();
        effective["instrument_transpose_base"] = state["instrument_transpose"].clone();
        effective["instrument_transpose_explicit"] =
            json!(transpose.is_some() || explicit.contains(&key));
        if transpose.is_none()
            && !explicit.contains(&key)
            && state["clef_octave"].as_i64().is_some_and(|v| v != 0)
            && state["clef_octave"] == state["instrument_transpose"]
        {
            effective["instrument_transpose"] = json!(0);
        }
        effective["octave_spans"] = json!(spans.get(&i).cloned().unwrap_or_default());
        r["pitch_context"] = effective;
        r["pitch_reference"] = json!(if instrument == "drums" {
            "percussion_key"
        } else if ["guitar", "bass"].contains(&instrument.as_str()) {
            "before_capo"
        } else {
            "sounding"
        });
        if unresolved.contains(&i) || !pending.is_empty() {
            r["pitch_needs_review"] = json!(true);
        }
    }
    Ok(records)
}

pub fn recognize_information<G: Generator + ?Sized>(
    engine: &mut G,
    layout: &Value,
    config: &InformationConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<InformationOutput> {
    require_stage(layout, "layout")?;
    if config.staff_label_model.is_some() != config.instrument_text_model.is_some() {
        return Err("staff_label_model and instrument_text_model must be declared together".into());
    }
    if config.score_structure && config.structure_output.is_none() {
        return Err("Structure stage requires an explicit output directory".into());
    }
    if config.transpose.is_some_and(|n| !(-36..=36).contains(&n))
        || config.capo.is_some_and(|n| n > 24)
    {
        return Err("Invalid transpose or capo override".into());
    }
    if layout["info_source"] == "pdf" || layout["info_source"] == "geometry" {
        return Err(
            "Native metadata currently requires image regions; rerun layout using image mode"
                .into(),
        );
    }
    let mut records = list(layout, "records").to_vec();
    if records.is_empty() {
        return Err("Add at least one measure before recognizing document information".into());
    }
    let mut structure_predictions = if config.score_structure {
        recognize_structure(engine, &records, config, cancelled)?
    } else {
        vec![]
    };
    let mut parts = if config.score_structure {
        let output = config
            .structure_output
            .as_deref()
            .ok_or("Structure stage requires an explicit output directory")?;
        let (p, r) =
            crate::score_grid::resolve_document(&records, &mut structure_predictions, output)?;
        records = r;
        p
    } else {
        list(layout, "parts").to_vec()
    };
    if parts.is_empty() {
        let mut seen = HashSet::new();
        for row in &records {
            let id = text(row, "part_id", "part-1");
            if seen.insert(id.to_owned()) {
                parts.push(json!({"id":id,"name":row.get("part_name").cloned().unwrap_or(json!("Guitar")),"instrument":row.get("instrument").cloned().unwrap_or(json!("guitar")),"program":row.get("midi_program").cloned().unwrap_or(json!(25)),"strings":row.get("string_count").cloned().unwrap_or(Value::Null)}));
            }
        }
    }
    for r in &mut records {
        if r["part_id"].is_null() {
            r["part_id"] = json!("part-1");
        }
        if r["staff_id"].is_null() {
            r["staff_id"] = json!("staff-1");
        }
        if r["bar_index"].is_null() {
            r["bar_index"] = json!(bar_index(r));
        }
        if r["row_index"].is_null() {
            r["row_index"] = r["system_index"].clone();
        }
    }
    let mut regions = list(layout, "regions").to_vec();
    let mut text_reader = config
        .text_regions_model
        .as_deref()
        .map(crate::text_regions::TextRegionReader::new)
        .transpose()?;
    if config.pitch_context {
        if let Some(reader) = text_reader.as_mut() {
            regions.extend(headers::opening_annotations(&records, &regions, reader)?);
        }
    }
    for region in &mut regions {
        if !document_header(text(region, "kind", "")) {
            if let Some(i) = nearest(&records, region) {
                region["part_id"] = records[i]["part_id"].clone();
            }
        }
        if region["source_page"].is_null() {
            if let Some(r) = records.iter().find(|r| r["page"] == region["page"]) {
                region["source_page"] = r["source_page"].clone();
            }
        }
    }
    if !config.pitch_context {
        regions.retain(|r| !["clef", "transposition", "annotation"].contains(&text(r, "kind", "")));
    }
    let mut chord_reader = config
        .chord_evidence_model
        .as_ref()
        .map(|path| chord_evidence::ChordEvidenceReader::new(path))
        .transpose()?;
    let mut clef_reader = config
        .clef_spatial_model
        .as_deref()
        .map(clef_spatial::ClefSpatialReader::new)
        .transpose()?;
    let mut predictions = recognize_regions_with_evidence(
        engine,
        &regions,
        config.image_policy.as_ref(),
        cancelled,
        chord_reader.as_mut(),
        clef_reader.as_mut(),
    )?;
    if let (Some(reader), Some(path)) = (
        text_reader.as_mut(),
        config.chord_text_roles_model.as_deref(),
    ) {
        let mut roles = crate::chord_text_roles::ChordTextRoleReader::new(path)?;
        annotations::recover_chord_names(
            engine,
            &mut predictions,
            reader,
            &mut roles,
            config.image_policy.as_ref(),
            cancelled,
        )?;
    }
    headers::recover_context(
        engine,
        &records,
        &regions,
        &mut predictions,
        config.image_policy.as_ref(),
        cancelled,
    )?;
    if config.staff_profile {
        let mut reader = None;
        for part in &parts {
            if let Some(cached) = part.get("staff_prediction").filter(|v| v.is_object()) {
                let mut prediction = cached.clone();
                prediction["part_id"] = part["id"].clone();
                predictions.push(prediction);
                continue;
            }
            let rows: Vec<Value> = records
                .iter()
                .filter(|r| r["part_id"] == part["id"])
                .cloned()
                .collect();
            engine.report(&format!(
                "正在读取乐器：{}",
                text(part, "name", "未命名声部")
            ))?;
            if reader.is_none() {
                reader = Some(staff_profile::StaffProfileReader::new(config)?);
            }
            if let Some(prediction) = reader.as_mut().unwrap().read(
                engine,
                &rows,
                config.image_policy.as_ref(),
                cancelled,
            )? {
                predictions.push(prediction);
            }
        }
    }
    annotations::prepare_regions(
        &mut records,
        &mut predictions,
        config.structure_output.as_deref(),
    )?;
    let mut values = vec![];
    let mut profiles = vec![];
    let mut contexts = vec![];
    let mut resolved = vec![];
    for (index, part) in parts.iter().enumerate() {
        let id = text(part, "id", "part-1");
        let local_predictions: Vec<Value> = predictions
            .iter()
            .filter(|p| {
                document_header(text(p, "kind", ""))
                    || text(p, "part_id", "part-1") == id
                    || (flag(p, "inside_header")
                        && matches!(text(&p["parsed"], "kind", ""), "title" | "credit"))
            })
            .cloned()
            .collect();
        let mut metadata = metadata_from_predictions(&local_predictions);
        headers::resolve_title(&mut metadata, &local_predictions, &config.fallback_title);
        let rows: Vec<Value> = records
            .iter()
            .filter(|r| text(r, "part_id", "part-1") == id)
            .cloned()
            .collect();
        if rows.is_empty() {
            continue;
        }
        let has_tab = rows.iter().any(|r| {
            matches!(
                text(r, "mode", text(layout, "mode", "notation")),
                "tab" | "both"
            )
        });
        // Metadata uses the last staff prediction. Its count, visual evidence,
        // and adapter provenance must describe the same local crop.
        let staff_prediction = local_predictions
            .iter()
            .rev()
            .find(|p| p["kind"] == "staff")
            .unwrap_or(&Value::Null);
        let label_evidence = staff_prediction.get("staff_label_evidence");
        let original_part = part.get("structure_prediction").unwrap_or(part);
        let effective_name = label_evidence
            .map(|e| {
                if e["status"] == "read" {
                    text(e, "text", "")
                } else {
                    ""
                }
            })
            .unwrap_or_else(|| text(part, "name", ""))
            .to_owned();
        if label_evidence.is_some() {
            metadata["name"] = json!(effective_name);
            metadata["staff_label_evidence"] = label_evidence.unwrap().clone();
        }
        let count_evidence = if let Some(evidence) = part
            .get("string_count_evidence")
            .filter(|_| !(index == 0 && config.tuning.is_some()))
        {
            evidence.clone()
        } else {
            staff_evidence::resolve_string_count(
                has_tab,
                &metadata["string_count"],
                &original_part["strings"],
                &staff_prediction["staff_evidence"],
                if index == 0 {
                    config.tuning.as_ref().map(Vec::len)
                } else {
                    None
                },
                flag(staff_prediction, "task_adapter_applied"),
            )
        };
        let count = count_evidence["value"].as_u64().map(|n| n as usize);
        metadata["string_count"] = json!(count);
        metadata["string_count_source"] = count_evidence["source"].clone();
        metadata["string_count_evidence"] = count_evidence.clone();
        metadata["string_count_needs_review"] = count_evidence["needs_review"].clone();
        let named: HashSet<i64> = local_predictions
            .iter()
            .filter(|p| p["kind"] == "transposition" && p["parsed"]["kind"] == "instrument")
            .filter_map(|p| program_from_visible_name(text(&p["parsed"], "text", "")))
            .collect();
        let structure_instrument = text(part, "instrument", "guitar");
        let mut identity_evidence = if let Some(evidence) = part
            .get("instrument_evidence")
            .filter(|_| !(index == 0 && config.instrument.is_some()))
        {
            evidence.clone()
        } else {
            staff_evidence::resolve_instrument_with_label(
                has_tab,
                &metadata["instrument"],
                &original_part["instrument"],
                if index == 0 {
                    config.instrument.as_deref()
                } else {
                    None
                },
                label_evidence,
                None,
            )
        };
        let named_pitched =
            !has_tab && named.len() == 1 && !(index == 0 && config.instrument.is_some());
        if named_pitched {
            identity_evidence["candidates"]["visible_instrument_annotation"] = json!("pitched");
            identity_evidence["value"] = json!("pitched");
            identity_evidence["source"] = json!("visible_instrument_annotation");
            identity_evidence["status"] = json!("resolved");
            identity_evidence["needs_review"] = json!(false);
            identity_evidence["provisional"] = json!(false);
            identity_evidence["provisional_source"] = Value::Null;
        }
        let instrument = if named_pitched {
            "pitched"
        } else {
            identity_evidence["value"].as_str().unwrap_or(
                if part["anonymous_relation_group"] == true && !has_tab {
                    "pitched"
                } else {
                    "guitar"
                },
            )
        }
        .to_owned();
        let identity_needs_review = !named_pitched && flag(&identity_evidence, "needs_review");
        metadata["instrument_evidence"] = identity_evidence.clone();
        metadata["instrument_evidence"]["names"] = json!({
            "staff_model":label_evidence.map(|e| &e["staff_prediction"]["name"]).unwrap_or(&metadata["name"]),
            "structure_model":original_part["name"], "label_text":label_evidence.map(|e| &e["text"])});
        metadata["instrument_source"] = if named_pitched {
            json!("visible_instrument_annotation")
        } else {
            identity_evidence["source"].clone()
        };
        metadata["instrument_needs_review"] = json!(identity_needs_review);
        if identity_needs_review {
            metadata["instrument_issue"] =
                json!("乐器身份候选不一致或未知，请确认乐器；各模型候选均已保留。");
        }
        metadata["instrument"] = json!(instrument);
        if label_evidence.is_none() && identity_needs_review && !part["name"].is_null() {
            metadata["name"] = part["name"].clone();
        }
        if !["guitar", "bass", "pitched", "drums"].contains(&instrument.as_str()) {
            return Err(format!("Unsupported instrument: {instrument}"));
        }
        let tuning_override = if index == 0 {
            config.tuning.clone()
        } else {
            None
        };
        if tuning_override.as_ref().is_some_and(|t| t.is_empty())
            && matches!(instrument.as_str(), "guitar" | "bass")
        {
            return Err("Tuning must contain 1..12 MIDI open pitches".into());
        }
        let named_tuning = tuning_from_name(text(&metadata, "tuning_name", ""), &instrument, count);
        let tuning = if matches!(instrument.as_str(), "pitched" | "drums") {
            vec![]
        } else {
            tuning_override
                .clone()
                .or_else(|| named_tuning.clone())
                .or_else(|| {
                    part["tuning"]
                        .as_array()
                        .and_then(|_| integers(&part["tuning"]).ok())
                })
                .map(Ok)
                .unwrap_or_else(|| standard_tuning(&instrument, count))
                .unwrap_or_default()
        };
        if tuning.len() > 12 || tuning.iter().any(|n| !(0..=127).contains(n)) {
            return Err("Tuning must contain 1..12 MIDI open pitches".into());
        }
        let unresolved_tuning =
            matches!(instrument.as_str(), "guitar" | "bass") && tuning.is_empty();
        let unreadable_printed_tuning = matches!(instrument.as_str(), "guitar" | "bass")
            && !text(&metadata, "tuning_name", "").trim().is_empty()
            && named_tuning.is_none()
            && tuning_override.is_none();
        let tuning_source = if unresolved_tuning {
            "unresolved"
        } else if tuning_override.is_some() {
            "manual"
        } else if identity_needs_review || unreadable_printed_tuning {
            "provisional"
        } else if named_tuning.is_some() {
            "printed"
        } else {
            "default"
        };
        let mut warnings: Vec<Value> = metadata["title_warning"]
            .as_str()
            .map(|s| vec![json!(s)])
            .unwrap_or_default();
        if unresolved_tuning {
            metadata["tuning_issue"] = json!(format!(
                "弦数识别为 {}，无法确定调弦。请对照谱面选择调弦，或填写各弦音高；曲名、作者和速度已保留。",
                count
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "未知".into())
            ));
        }
        if unreadable_printed_tuning {
            metadata["tuning_issue"] = json!(format!(
                "未能解析谱面的“{}”调弦标记，当前空弦音仅为暂定值，请核对各弦音高。",
                text(&metadata, "tuning_name", "")
            ));
            warnings.push(json!(format!("Could not resolve the printed {} tuning for this instrument; check open-string pitches",text(&metadata,"tuning_name",""))));
        }
        let mut candidates = if unresolved_tuning {
            vec![]
        } else {
            vec![tuning.clone()]
        };
        if identity_needs_review && tuning_override.is_none() {
            let mut instrument_tunings = vec![];
            for candidate in identity_evidence["candidates"]
                .as_object()
                .into_iter()
                .flat_map(|v| v.values())
            {
                let Some(candidate) = candidate
                    .as_str()
                    .filter(|v| ["guitar", "bass"].contains(v))
                else {
                    continue;
                };
                let alternative =
                    tuning_from_name(text(&metadata, "tuning_name", "Standard"), candidate, count)
                        .or_else(|| standard_tuning(candidate, count).ok());
                if let Some(alternative) = alternative {
                    let entry = json!({"instrument":candidate,"tuning":alternative});
                    if !instrument_tunings.contains(&entry) {
                        instrument_tunings.push(entry);
                    }
                }
            }
            metadata["instrument_tuning_candidates"] = json!(instrument_tunings);
            metadata["tuning_midi_high_to_low"] = Value::Null;
            metadata["tuning_issue"] = json!(
                "乐器身份尚未确认，当前调弦仅为既有声部的暂定值；请核对各弦音高。Standard tuning 不能单独确定乐器及调弦。"
            );
            warnings.push(json!(
                "乐器身份候选不一致，当前声部与调弦仅临时保留；请核对谱面乐器名称和各弦音高。"
            ));
        } else if unreadable_printed_tuning {
            metadata["tuning_midi_high_to_low"] = Value::Null;
        } else if !tuning.is_empty() {
            metadata["tuning_midi_high_to_low"] = json!(tuning);
        }
        metadata["tuning_needs_review"] =
            json!(unresolved_tuning || tuning_source == "provisional");
        metadata["tuning_evidence"] = json!({"value":tuning,"source":tuning_source,
            "candidates":candidates,"needs_review":metadata["tuning_needs_review"],
            "instrument_evidence":identity_evidence});
        if !has_tab
            && tuning_override.is_none()
            && count.is_none()
            && matches!(instrument.as_str(), "guitar" | "bass")
        {
            for n in tuning.len() + 1..8 {
                if let Some(t) = tuning_from_name(
                    text(&metadata, "tuning_name", "Standard"),
                    &instrument,
                    Some(n),
                ) {
                    if !candidates.contains(&t) {
                        candidates.push(t);
                    }
                }
            }
        }
        let program = if index == 0 {
            config.midi_program.map(i64::from)
        } else {
            None
        }
        .or_else(|| {
            if instrument == "pitched" && named.len() == 1 {
                named.iter().next().copied()
            } else {
                None
            }
        })
        .or_else(|| {
            (!identity_needs_review)
                .then(|| {
                    program_from_visible_name(text(&metadata, "name", effective_name.as_str()))
                })
                .flatten()
        })
        .unwrap_or_else(|| {
            if instrument == structure_instrument {
                part["program"]
                    .as_i64()
                    .unwrap_or(match instrument.as_str() {
                        "guitar" => 25,
                        "bass" => 33,
                        _ => 0,
                    })
            } else {
                match instrument.as_str() {
                    "guitar" => 25,
                    "bass" => 33,
                    _ => 0,
                }
            }
        });
        if !(0..=127).contains(&program) {
            return Err("Invalid MIDI program".into());
        }
        let capos: BTreeSet<i64> = local_predictions
            .iter()
            .filter(|p| p["parsed"]["kind"] == "capo")
            .filter_map(|p| p["parsed"]["capo"].as_i64())
            .collect();
        let capo = if index == 0 {
            config.capo.map(i64::from)
        } else {
            None
        }
        .unwrap_or(if capos.len() == 1 {
            *capos.first().unwrap()
        } else {
            0
        });
        if capos.len() > 1 {
            warnings.push(json!("Multiple printed capo positions require review"));
        }
        // A transposition suffix is meaningful only after its visible
        // instrument label has been accepted, not from an unresolved fragment.
        let pitch_name = if label_evidence.is_none_or(|e| flag(&e["identity"], "accepted")) {
            effective_name.as_str()
        } else {
            ""
        };
        let transpose = if index == 0 {
            config.transpose.map(i64::from)
        } else {
            None
        }
        .or_else(|| {
            if conventional_octave(pitch_name).is_none() {
                explicit_transposition(pitch_name)
            } else {
                None
            }
        });
        let mut local = rows.clone();
        for r in &mut local {
            r["part_name"] = json!(effective_name);
            r["instrument"] = json!(instrument);
            r["midi_program"] = json!(program);
            r["tuning"] = json!(tuning);
            r["string_count"] = json!(count);
            r["capo"] = json!(capo);
        }
        let mut local = apply_pitch_regions(
            &local,
            &local_predictions,
            &instrument,
            transpose,
            conventional_octave(pitch_name),
        )?;
        let unusual = instrument == "guitar"
            && matches!(tuning.len(), 4 | 5)
            && tuning_source == "default"
            && local.iter().all(|r| r["mode"] == "tab");
        if flag(&count_evidence, "needs_review") {
            warnings.push(json!(
                "TAB 弦数候选不一致或未知，请核对调弦；原模型与视觉模型候选均已保留。"
            ));
            for r in &mut local {
                r["pitch_needs_review"] = json!(true);
            }
        }
        if identity_needs_review {
            for r in &mut local {
                r["instrument_evidence"] = metadata["instrument_evidence"].clone();
                r["instrument_needs_review"] = json!(true);
                r["tuning_needs_review"] = metadata["tuning_needs_review"].clone();
                r["instrument_tuning_candidates"] =
                    metadata["instrument_tuning_candidates"].clone();
                r["pitch_needs_review"] = json!(true);
            }
        }
        if flag(&metadata, "tuning_needs_review") {
            for r in &mut local {
                r["tuning_needs_review"] = json!(true);
                r["pitch_needs_review"] = json!(true);
            }
        }
        if unusual {
            warnings.push(json!(
                "This guitar string count has no unique standard tuning; confirm the open strings"
            ));
            for r in &mut local {
                r["pitch_needs_review"] = json!(true);
            }
        }
        if instrument == "pitched"
            && transpose.is_none()
            && conventional_octave(effective_name.as_str()).is_none()
            && program_from_visible_name(effective_name.as_str())
                .is_some_and(|p| [56, 60, 64, 65, 66, 67, 69, 71].contains(&p))
            && !local_predictions
                .iter()
                .any(|p| p["parsed"]["kind"] == "instrument" && !p["parsed"]["semitones"].is_null())
        {
            warnings.push(json!(
                "Instrument key was not explicit; written pitches were retained for review"
            ));
            for r in &mut local {
                r["pitch_needs_review"] = json!(true);
            }
        }
        metadata["pitch_instructions"] = json!(local_predictions
            .iter()
            .filter(|p| matches!(text(p, "kind", ""), "clef" | "transposition"))
            .collect::<Vec<_>>());
        metadata["score_annotations"] = json!(local_predictions
            .iter()
            .filter(|p| p["kind"] == "annotation")
            .collect::<Vec<_>>());
        if local.iter().any(|r| flag(r, "pitch_needs_review")) {
            warnings.push(json!(
                "Some pitch instructions need review; check clef, transposition, and capo"
            ));
        }
        if !warnings.is_empty() {
            metadata["warnings"] = json!(warnings);
        }
        for r in &local {
            let mut profile = json!({"instrument":instrument,"midi_program":program,"tuning":tuning,"string_count":count,"capo":capo,"tuning_source":tuning_source,"tuning_explicit":matches!(tuning_source,"manual"|"printed"),"fingering_tunings":candidates});
            for key in [
                "measure_number",
                "part_id",
                "part_name",
                "staff_id",
                "bar_index",
                "row_index",
                "system_index",
                "instrument_evidence",
                "instrument_needs_review",
                "tuning_needs_review",
                "instrument_tuning_candidates",
            ] {
                if let Some(v) = r.get(key) {
                    profile[key] = v.clone();
                }
            }
            profiles.push(profile);
            if config.pitch_context || transpose.is_some() {
                let mut c = json!({});
                for key in [
                    "measure_number",
                    "pitch_context",
                    "clef_observation",
                    "pitch_reference",
                    "pitch_needs_review",
                ] {
                    if let Some(v) = r.get(key) {
                        c[key] = v.clone();
                    }
                }
                contexts.push(c);
            }
        }
        resolved.extend(local);
        values.push(json!({"id":id,"name":effective_name,"mode":majority_mode(&rows,text(layout,"mode","notation")),"document_metadata":metadata,"title":metadata["title"].as_str().unwrap_or(""),"artist":metadata["artist"].as_str().unwrap_or(""),"tuning_used":tuning,"tuning_candidates":candidates,"tuning_source":tuning_source,"instrument":instrument,"midi_program":program,"capo":capo,"transpose":transpose}));
    }
    let mut manifest = values.first().cloned().ok_or("No resolved parts")?;
    manifest.as_object_mut().unwrap().remove("id");
    manifest.as_object_mut().unwrap().remove("name");
    manifest.as_object_mut().unwrap().remove("mode");
    manifest["schema_version"] = json!("1.0");
    manifest["stage"] = json!("document_info");
    manifest["layout"] = json!(config.layout_path);
    manifest["parts"] = json!(values);
    manifest["measure_profiles"] = json!(profiles);
    manifest["measure_pitch_contexts"] = json!(contexts);
    resolved.sort_by_key(|r| r["measure_number"].as_i64().unwrap_or(0));
    manifest["resolved_records"] = json!(resolved);
    check(cancelled)?;
    Ok(InformationOutput {
        manifest,
        predictions,
        structure_predictions,
    })
}

#[derive(Clone, Debug)]
pub struct MeasureConfig {
    pub image_policy: Option<ImagePolicy>,
    pub max_tokens: usize,
    pub max_tokens_ceiling: usize,
    pub maximum_attempts: usize,
    pub constrained_decoding: bool,
}
impl Default for MeasureConfig {
    fn default() -> Self {
        Self {
            image_policy: None,
            max_tokens: 4096,
            max_tokens_ceiling: 6144,
            maximum_attempts: 2,
            constrained_decoding: true,
        }
    }
}
impl MeasureConfig {
    fn validate(&self) -> Result<()> {
        if self.max_tokens == 0
            || self.max_tokens_ceiling < self.max_tokens
            || self.max_tokens_ceiling > 131072
            || !(1..=10).contains(&self.maximum_attempts)
        {
            return Err("Invalid token limits or attempt count".into());
        }
        Ok(())
    }
}
pub fn prepare_records(layout: &Value, information: &Value) -> Result<Vec<Value>> {
    require_stage(layout, "layout")?;
    require_stage(information, "document_info")?;
    let mut records = information
        .get("resolved_records")
        .and_then(Value::as_array)
        .unwrap_or_else(|| layout["records"].as_array().unwrap_or(&EMPTY_VALUES))
        .clone();
    let mut numbers = HashSet::new();
    for row in &mut records {
        let number = row["measure_number"]
            .as_i64()
            .filter(|n| *n > 0)
            .ok_or("Missing positive measure number")?;
        if row["source_id"].is_null() {
            row["source_id"] = json!("");
        }
        if row["part_id"].is_null() {
            row["part_id"] = json!("part-1");
        }
        if row["staff_id"].is_null() {
            row["staff_id"] = json!("staff-1");
        }
        if row["bar_index"].is_null() {
            row["bar_index"] = json!(bar_index(row));
        }
        if !numbers.insert(number) {
            return Err("Duplicate measure number".into());
        }
        for field in ["measure_profiles", "measure_pitch_contexts"] {
            if let Some(profile) = list(information, field)
                .iter()
                .find(|p| p["measure_number"] == number)
            {
                merge(row, profile);
            }
        }
        for key in ["instrument", "midi_program", "capo", "tuning_source"] {
            if row[key].is_null() {
                row[key] = information.get(key).cloned().unwrap_or(match key {
                    "instrument" => json!("guitar"),
                    "midi_program" => json!(25),
                    "capo" => json!(0),
                    _ => json!("default"),
                });
            }
        }
        if row["mode"].is_null() {
            row["mode"] = layout["mode"].clone();
        }
        if !["tab", "notation", "both"].contains(&text(row, "mode", "")) {
            return Err(format!("Measure {number} has no notation mode"));
        }
        if row["tuning"].is_null() {
            row["tuning"] = information["tuning_used"].clone();
        }
        let tuning = integers(&row["tuning"])?;
        if tuning.is_empty() && matches!(text(row, "instrument", ""), "guitar" | "bass") {
            return Err(format!(
                "请先在“校对谱面信息”中填写“{}”的调弦，再识别小节。",
                text(row, "part_name", "当前音轨")
            ));
        }
        if tuning.len() > 12 || tuning.iter().any(|n| !(0..=127).contains(n)) {
            return Err("Invalid tuning".into());
        }
        if row["tuning_explicit"].is_null() {
            row["tuning_explicit"] =
                json!(["manual", "printed"].contains(&text(row, "tuning_source", "default")));
        }
        if list(row, "fingering_tunings").is_empty() {
            row["fingering_tunings"] = if !list(information, "tuning_candidates").is_empty() {
                information["tuning_candidates"].clone()
            } else {
                json!([tuning])
            };
        }
        if row["pitch_context"].is_object() {
            row["pitch_context"]["capo"] = row["capo"].clone();
        }
    }
    Ok(records)
}
static EMPTY_VALUES: Vec<Value> = Vec::new();
fn valid_signature(v: &Value) -> bool {
    let time = v["time"].as_str();
    (v["time"].is_null() || time.is_some_and(|s| time_signature(s).is_ok()))
        && (v["key"].is_null() || v["key"].as_i64().is_some_and(|n| (-7..=7).contains(&n)))
}
fn time_signature(time: &str) -> Result<(i64, i64)> {
    let (n, d) = time.split_once('/').ok_or("Invalid time signature")?;
    let n = n.parse::<i64>().map_err(|_| "Invalid numerator")?;
    let d = d.parse::<i64>().map_err(|_| "Invalid denominator")?;
    if !(1..=255).contains(&n) || ![1, 2, 4, 8, 16, 32, 64].contains(&d) {
        return Err("Unsupported time signature".into());
    }
    Ok((n, d))
}
pub fn read_score_states(
    reader: &mut SignatureReader,
    records: &mut [Value],
    cancelled: &dyn Fn() -> bool,
) -> Result<Value> {
    let mut predictions = Map::new();
    for rows in records.chunks(8) {
        check(cancelled)?;
        let images = rows.iter().map(record_image).collect::<Result<Vec<_>>>()?;
        let output = reader.predict(&images)?;
        if output.len() != rows.len() {
            return Err("Signature model returned wrong batch length".into());
        }
        for (row, mut value) in rows.iter().zip(output) {
            if !valid_signature(&value) {
                return Err("Invalid signature prediction".into());
            }
            value["raw"] = json!(format!(
                "S2 time={} key={}",
                text(&value, "time", "-"),
                value["key"].as_i64().map_or("-".into(), |n| n.to_string())
            ));
            value["tokens"] = json!(0);
            value["source"] = json!("visual_classifier");
            predictions.insert(row["measure_number"].to_string(), value);
        }
    }
    let predictions = Value::Object(predictions);
    for row in records {
        row["printed_state"] = predictions[row["measure_number"].to_string()].clone();
    }
    Ok(predictions)
}
const MAJOR_KEYS: [&str; 17] = [
    "FMajorFlat",
    "CMajorFlat",
    "GMajorFlat",
    "DMajorFlat",
    "AMajorFlat",
    "EMajorFlat",
    "BMajorFlat",
    "FMajor",
    "CMajor",
    "GMajor",
    "DMajor",
    "AMajor",
    "EMajor",
    "BMajor",
    "FMajorSharp",
    "CMajorSharp",
    "GMajorSharp",
];
const MINOR_KEYS: [&str; 17] = [
    "DMinorFlat",
    "AMinorFlat",
    "EMinorFlat",
    "BMinorFlat",
    "FMinor",
    "CMinor",
    "GMinor",
    "DMinor",
    "AMinor",
    "EMinor",
    "BMinor",
    "FMinorSharp",
    "CMinorSharp",
    "GMinorSharp",
    "DMinorSharp",
    "AMinorSharp",
    "EMinorSharp",
];
fn key_parts(name: &str) -> Result<(i64, bool)> {
    for (minor, names) in [(false, &MAJOR_KEYS), (true, &MINOR_KEYS)] {
        if let Some(i) = names.iter().position(|n| *n == name) {
            return Ok((i as i64 - 8, minor));
        }
    }
    Err(format!("Unknown key signature: {name}"))
}
fn key_name(fifths: i64, minor: bool) -> Result<&'static str> {
    if !(-8..=8).contains(&fifths) {
        return Err("Key accidental count outside supported range".into());
    }
    Ok(if minor {
        MINOR_KEYS[(fifths + 8) as usize]
    } else {
        MAJOR_KEYS[(fifths + 8) as usize]
    })
}
fn transpose_key(name: &str, shift: i64) -> Result<String> {
    if shift.rem_euclid(12) == 0 {
        return Ok(name.into());
    }
    let (fifths, minor) = key_parts(name)?;
    let tonic = (7 * fifths + if minor { 9 } else { 0 } + shift).rem_euclid(12);
    let chosen = (-8i64..=8)
        .filter(|f| (7 * f + if minor { 9 } else { 0 }).rem_euclid(12) == tonic)
        .min_by_key(|f| (f.abs(), (f - fifths).abs()))
        .ok_or("No transposed key spelling")?;
    Ok(key_name(chosen, minor)?.into())
}
fn ornament(position: &str) -> Option<(Option<i64>, Option<i64>)> {
    let re = Regex::new(r"^(?:f(\d+))?(?:p(\d+))?$").unwrap();
    let c = re.captures(position)?;
    Some((
        c.get(1).and_then(|m| m.as_str().parse().ok()),
        c.get(2).and_then(|m| m.as_str().parse().ok()),
    ))
}
pub fn convert_pitch_target(target: &str, context: &Value, mode: &str) -> Result<String> {
    convert_pitch_direction(target, context, mode, 1)
}
fn convert_pitch_direction(
    target: &str,
    context: &Value,
    mode: &str,
    direction: i64,
) -> Result<String> {
    let mut m = score::parse_measure_target(target)?;
    let instrument = context["instrument_transpose"].as_i64().unwrap_or(0);
    let base = instrument + context["clef_octave"].as_i64().unwrap_or(0)
        - context["capo"].as_i64().unwrap_or(0);
    if !(-108..=108).contains(&base) {
        return Err("Invalid pitch context range".into());
    }
    if let Some(key) = m["key_signature"].as_str() {
        m["key_signature"] = json!(transpose_key(key, direction * instrument)?);
    }
    for voice in m["voices"].as_array_mut().unwrap() {
        for event in voice["events"].as_array_mut().unwrap() {
            let octaves: Vec<i64> = list(event, "effects")
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|s| s.strip_prefix("ottava:"))
                .map(|s| s.parse::<i64>().map_err(|_| "Invalid ottava".to_owned()))
                .collect::<Result<_>>()?;
            if octaves.len() > 1 || octaves.iter().any(|n| ![-24, -12, 12, 24].contains(n)) {
                return Err("Invalid local octave instruction".into());
            }
            let shift = direction * (base + octaves.first().copied().unwrap_or(0));
            for note in event["notes"].as_array_mut().unwrap() {
                if list(note, "effects").iter().any(|e| e == "dead") {
                    continue;
                }
                let pitch = note["pitch"].as_i64().ok_or("Missing written pitch")? + shift;
                if !(0..=127).contains(&pitch) {
                    return Err("Transposed pitch is outside MIDI range".into());
                }
                note["pitch"] = json!(pitch);
                for effect in note["effects"].as_array_mut().unwrap() {
                    let raw = effect.as_str().unwrap();
                    let mut parts: Vec<String> = raw.split(':').map(str::to_owned).collect();
                    if parts.len() > 1 && matches!(parts[0].as_str(), "grace" | "trill") {
                        if let Some((fret, Some(p))) = ornament(&parts[1]) {
                            let p = p + shift;
                            if !(0..=127).contains(&p) {
                                return Err("Transposed ornament is outside MIDI range".into());
                            }
                            parts[1] = format!(
                                "{}p{p}",
                                fret.map(|f| format!("f{f}")).unwrap_or_default()
                            );
                        }
                    }
                    *effect = json!(parts.join(":"));
                }
            }
        }
    }
    score::format_measure_target(&m, mode, true)
}
pub fn resolve_fretted_pitches(
    target: &str,
    tuning: &[i64],
    verified: Option<&HashSet<i64>>,
) -> Result<(String, bool)> {
    let mut m = score::parse_measure_target(target)?;
    let mut changed = false;
    for voice in m["voices"].as_array_mut().unwrap() {
        for event in voice["events"].as_array_mut().unwrap() {
            for note in event["notes"].as_array_mut().unwrap() {
                let Some(s) = note["string"]
                    .as_i64()
                    .filter(|s| *s >= 1 && (*s as usize) <= tuning.len())
                else {
                    continue;
                };
                if verified.is_some_and(|v| !v.contains(&s)) {
                    continue;
                }
                if note["fret"] == "x"
                    || list(note, "effects")
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|e| ["dead", "tie"].contains(&e) || e.starts_with("harm:"))
                {
                    continue;
                }
                if let Some(f) = note["fret"].as_i64().filter(|f| (0..=36).contains(f)) {
                    let p = tuning[s as usize - 1] + f;
                    if (0..=127).contains(&p) && note["pitch"] != p {
                        note["pitch"] = json!(p);
                        changed = true;
                    }
                }
                for effect in note["effects"].as_array_mut().unwrap() {
                    let mut parts: Vec<String> = effect
                        .as_str()
                        .unwrap()
                        .split(':')
                        .map(str::to_owned)
                        .collect();
                    if parts.len() > 1 && matches!(parts[0].as_str(), "grace" | "trill") {
                        if let Some((Some(f), _)) = ornament(&parts[1]) {
                            let p = tuning[s as usize - 1] + f;
                            if (0..=36).contains(&f) && (0..=127).contains(&p) {
                                let new = format!("f{f}p{p}");
                                if parts[1] != new {
                                    parts[1] = new;
                                    *effect = json!(parts.join(":"));
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if changed {
        Ok((score::format_measure_target(&m, "both", true)?, true))
    } else {
        Ok((target.into(), false))
    }
}
pub fn rhythm_warnings(target: &str, time: &str) -> Result<Vec<String>> {
    let m = score::parse_measure_target(target)?;
    let time = m["time_signature"].as_str().unwrap_or(time);
    let (n, d) = time_signature(time)?;
    let mut warnings = vec![];
    for voice in list(&m, "voices") {
        let mut over = false;
        for e in list(voice, "events") {
            let duration = score::duration_ticks(&e["duration"])?;
            let start = e["start"].as_i64().ok_or("Invalid event start")? as i128;
            if (start * duration.denominator + duration.numerator) * i128::from(d)
                > i128::from(3840 * n) * duration.denominator
            {
                over = true;
            }
        }
        if over {
            warnings.push(format!(
                "Voice {} exceeds the {time} measure duration; check onsets and durations",
                voice["voice"].as_i64().unwrap_or(0) + 1
            ));
        }
    }
    Ok(warnings)
}
fn meter_warnings(target: &str, row: &Value) -> Result<Vec<String>> {
    let measure = score::parse_measure_target(target)?;
    let state = signature_state::trusted_state(row);
    let time = measure["time_signature"]
        .as_str()
        .or_else(|| state["time"].as_str());
    match time {
        Some(time) => rhythm_warnings(target, time),
        None => Ok(vec![]),
    }
}

/// Read one crop only. Missing musical evidence remains null until assembly.
pub fn recognize_measure<G: Generator + ?Sized>(
    engine: &mut G,
    record: &Value,
    config: &MeasureConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<Value> {
    config.validate()?;
    check(cancelled)?;
    let mut row = record.clone();
    let prompt = scorelib::observation::observation_prompt(text(&row, "mode", ""))?;
    let messages = image_messages(
        &[record_image(&row)?],
        &prompt,
        config.image_policy.as_ref(),
    )?;
    let mut budget = config.max_tokens;
    row["observation"] = Value::Null;
    row["target"] = Value::Null;
    row["written_target"] = Value::Null;
    row["ir"] = Value::Null;
    for attempt in 1..=config.maximum_attempts {
        check(cancelled)?;
        let response = engine.generate(
            messages.clone(),
            budget,
            config
                .constrained_decoding
                .then(scorelib::observation::observation_schema),
            None,
        )?;
        check(cancelled)?;
        row["raw_observation_text"] = json!(response.content);
        row["recognition_attempts"] = json!(attempt);
        row["generated_token_count"] = json!(response.completion_tokens);
        match scorelib::observation::parse_observation(&json!(response.content)) {
            Ok(observation) => {
                row["observation"] = observation;
                row["recognition_errors"] = json!([]);
                break;
            }
            Err(error) => {
                row["recognition_errors"] = json!([error]);
                budget = (budget * 2).min(config.max_tokens_ceiling);
            }
        }
    }
    row["needs_review"] = json!(true);
    Ok(row)
}

#[derive(Default)]
struct TuningEvidence {
    counts: BTreeMap<i64, BTreeMap<i64, usize>>,
    bars: HashMap<(i64, i64), HashSet<i64>>,
    frets: HashMap<(i64, i64), HashSet<i64>>,
}
impl TuningEvidence {
    fn winner(&self, string: i64) -> Option<(i64, usize, usize)> {
        let c = self.counts.get(&string)?;
        let (&pitch, &count) = c
            .iter()
            .max_by_key(|(pitch, count)| (**count, std::cmp::Reverse(**pitch)))?;
        Some((pitch, count, c.values().sum()))
    }
    fn reliable(&self, string: i64, pitch: i64, count: usize, total: usize) -> bool {
        count >= 6
            && count as f64 / total as f64 >= 0.85
            && self
                .bars
                .get(&(string, pitch))
                .is_some_and(|b| b.len() >= 3)
            && self
                .frets
                .get(&(string, pitch))
                .is_some_and(|f| f.len() >= 2)
    }
}
pub fn reconcile_score_tuning(records: &mut [Value]) -> Result<()> {
    let mut parts: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        if r["mode"] == "both" && flag(r, "visual_pitch") && r["target"].is_string() {
            parts
                .entry(text(r, "part_id", "part-1").into())
                .or_default()
                .push(i);
        }
    }
    for (part, indices) in parts {
        if records.iter().any(|row| {
            text(row, "part_id", "part-1") == part && flag(row, "instrument_needs_review")
        }) {
            // Note/fret agreement cannot silently settle a disputed instrument
            // profile or promote its provisional tuning to confirmed pitches.
            continue;
        }
        let mut e = TuningEvidence::default();
        for &i in &indices {
            let row = &records[i];
            let measure = score::parse_measure_target(text(row, "target", ""))?;
            for voice in list(&measure, "voices") {
                for event in list(voice, "events") {
                    for note in list(event, "notes") {
                        if list(note, "effects")
                            .iter()
                            .filter_map(Value::as_str)
                            .any(|s| matches!(s, "dead" | "tie") || s.starts_with("harm:"))
                        {
                            continue;
                        }
                        if let (Some(string), Some(fret), Some(pitch)) = (
                            note["string"].as_i64(),
                            note["fret"].as_i64(),
                            note["pitch"].as_i64(),
                        ) {
                            let open = pitch - fret;
                            if !(0..=127).contains(&open) {
                                continue;
                            }
                            *e.counts.entry(string).or_default().entry(open).or_default() += 1;
                            e.bars
                                .entry((string, open))
                                .or_default()
                                .insert(bar_index(row));
                            e.frets.entry((string, open)).or_default().insert(fret);
                        }
                    }
                }
            }
        }
        let mut tuning = integers(&records[indices[0]]["tuning"])?;
        if tuning.is_empty() {
            continue;
        }
        let explicit = indices
            .iter()
            .any(|&i| flag(&records[i], "tuning_explicit"));
        let unusual =
            records[indices[0]]["instrument"] == "guitar" && matches!(tuning.len(), 4 | 5);
        let mut conflicts = HashSet::new();
        let mut confirmed = HashSet::new();
        let mut altered = false;
        if unusual && !explicit {
            let reliable: BTreeMap<i64, i64> = e
                .counts
                .keys()
                .filter_map(|&s| {
                    let (p, c, t) = e.winner(s)?;
                    (s >= 1 && s as usize <= tuning.len() && e.reliable(s, p, c, t))
                        .then_some((s, p))
                })
                .collect();
            if reliable.len() >= 2 {
                let mut candidates = HashSet::new();
                for instrument in ["guitar", "bass"] {
                    if let Ok(v) = standard_tuning(instrument, Some(tuning.len())) {
                        if reliable.iter().all(|(&s, &p)| v[s as usize - 1] == p) {
                            candidates.insert(v);
                        }
                    }
                }
                if candidates.len() == 1 {
                    let inferred = candidates.into_iter().next().unwrap();
                    altered = tuning != inferred;
                    tuning = inferred;
                    confirmed.extend(reliable.keys().copied());
                }
            }
        }
        let mut shifts: HashMap<i64, Vec<i64>> = HashMap::new();
        for &s in e.counts.keys() {
            let (p, c, t) = e.winner(s).unwrap();
            if s >= 1 && s as usize <= tuning.len() && e.reliable(s, p, c, t) {
                shifts
                    .entry(p - tuning[s as usize - 1])
                    .or_default()
                    .push(s);
            }
        }
        let uniform: HashSet<i64> = shifts
            .iter()
            .filter(|(shift, strings)| **shift != 0 && strings.len() >= 3)
            .flat_map(|(_, s)| s.iter().copied())
            .collect();
        for &s in e.counts.keys() {
            let (p, c, t) = e.winner(s).unwrap();
            if s < 1 || s as usize > tuning.len() {
                conflicts.insert(s);
                continue;
            }
            let reliable = e.reliable(s, p, c, t);
            if explicit {
                confirmed.insert(s);
            } else if uniform.contains(&s) {
                conflicts.insert(s);
            } else if reliable && (p - tuning[s as usize - 1]).abs() <= 7 {
                altered |= p != tuning[s as usize - 1];
                tuning[s as usize - 1] = p;
                confirmed.insert(s);
            } else if reliable || (c >= 6 && c as f64 / (t as f64) < 0.65) {
                conflicts.insert(s);
            }
        }
        for row in records
            .iter_mut()
            .filter(|r| text(r, "part_id", "part-1") == part)
        {
            row["tuning"] = json!(tuning);
            if !explicit && altered {
                row["tuning_source"] = json!("notation_tab_consensus");
            }
            row["tuning_evidence"] = json!(e.counts);
            if row["mode"] == "both" && row["target"].is_string() && !flag(row, "manually_edited") {
                if conflicts.is_empty() && (explicit || confirmed.len() >= 2) {
                    let (target, changed) = resolve_fretted_pitches(
                        text(row, "target", ""),
                        &tuning,
                        if explicit { None } else { Some(&confirmed) },
                    )?;
                    row["target"] = json!(target);
                    if changed {
                        row["pitch_reconciled"] = json!(true);
                    }
                } else if !conflicts.is_empty() {
                    let mut conflicts: Vec<i64> = conflicts.iter().copied().collect();
                    conflicts.sort();
                    row["needs_review"] = json!(true);
                    row["tuning_needs_review"] = json!(conflicts);
                }
            }
        }
    }
    Ok(())
}
fn product_pitch_context(row: &Value) -> Value {
    let mut context = if row["pitch_context"].is_object() {
        row["pitch_context"].clone()
    } else {
        json!({})
    };
    context["capo"] = row["capo"].clone();
    if !row["score_state"]["clef_octave"].is_null() {
        context["clef_octave"] = row["score_state"]["clef_octave"].clone();
    }
    if let Some(base) = context["instrument_transpose_base"].as_i64() {
        context["instrument_transpose"] =
            json!(if !flag(&context, "instrument_transpose_explicit")
                && base != 0
                && context["clef_octave"] == base
            {
                0
            } else {
                base
            });
    }
    context
}

fn product_technique_overlay(resolved: &Value) -> Value {
    let mut anchors: BTreeMap<(i64, i64), BTreeSet<String>> = BTreeMap::new();
    for mark in list(resolved, "visual_techniques") {
        let Some(voice_id) = mark["voice"].as_i64() else {
            continue;
        };
        let Some(index) = mark["event"].as_u64() else {
            continue;
        };
        let Some(voice) = list(resolved, "voices")
            .iter()
            .find(|v| v["voice"] == voice_id)
        else {
            continue;
        };
        let Some(event) = list(voice, "events").get(index as usize) else {
            continue;
        };
        let Some(onset) = event["start"].as_i64() else {
            continue;
        };
        let Some(effect) = mark["effect"]
            .as_str()
            .filter(|e| matches!(*e, "pm" | "let"))
        else {
            continue;
        };
        if !list(event, "notes").is_empty() {
            anchors
                .entry((voice_id, onset))
                .or_default()
                .insert(effect.to_owned());
        }
    }
    if anchors.is_empty() {
        return Value::Null;
    }
    json!({"protocol":scorelib::visual_techniques::PROTOCOL,"scope":"visible_event","events":anchors.into_iter().map(|((voice,onset),effects)|
        json!({"voice":voice,"onset":onset,"effects":effects,"status":"classified"})).collect::<Vec<_>>()})
}

fn assembly_auxiliary(row: &Value) -> Result<Value> {
    let mut state = json!({});
    for field in ["time", "key"] {
        let printed = &row["printed_state"];
        let evidence = &printed["evidence"][field];
        if evidence["status"] == "observed" {
            state[field] = evidence.clone();
        } else if printed["source"] == "manual" && !printed[field].is_null() {
            state[field] = json!({"status":"observed","value":printed[field]});
        }
        if row["score_state_evidence"][field]["source"] == "manual"
            && row["score_state_evidence"][field]["status"] == "observed"
        {
            state[field] = row["score_state_evidence"][field].clone();
        }
    }
    // Only a locally observed clef is new evidence. Inherited information
    // context must not erase a change that the measure recognizer can see.
    for field in ["clef", "clef_octave"] {
        if row["clef_observation"][field]["status"] == "observed" {
            state[field] = row["clef_observation"][field].clone();
        }
    }
    for field in ["time", "key", "clef", "clef_octave"] {
        if row["manual_state"][field].is_object() {
            state[field] = row["manual_state"][field].clone();
            state[field]["source"] = json!("manual");
        }
    }
    if !list(row, "tuning").is_empty() && !tuning_needs_review(row) {
        state["tuning"] = json!({"status":"observed","value":row["tuning"]});
    }
    let mut result = json!({"state":state});
    if flag(row, "manually_edited") {
        let target = row["target"]
            .as_str()
            .ok_or("Manually edited record is missing music")?;
        let mode = text(row, "mode", "tab");
        let written = if mode == "tab" || row["instrument"] == "drums" {
            target.to_owned()
        } else {
            let context = product_pitch_context(row);
            convert_pitch_direction(target, &context, mode, -1)?
        };
        let mut measure = score::parse_measure_target(&written)?;
        if !measure["time_signature"].is_null() {
            result["state"]["time"] =
                json!({"status":"observed","value":measure["time_signature"],"source":"manual"});
        }
        if let Some(key) = measure["key_signature"].as_str() {
            result["state"]["key"] =
                json!({"status":"observed","value":key_parts(key)?.0,"source":"manual"});
        }
        for voice in measure["voices"].as_array_mut().unwrap() {
            for event in voice["events"].as_array_mut().unwrap() {
                event["duration"] = json!(score::duration_token(&event["duration"])?);
                for note in event["notes"].as_array_mut().unwrap() {
                    note["written_pitch"] = note["pitch"].clone();
                }
            }
        }
        result["resolved_voices"] = measure["voices"].clone();
        result["source"] = json!("manual");
    }
    Ok(result)
}

/// Every retry reassembles the entire score from local evidence. A changed tie
/// or scope can therefore update its neighbours without another model call.
pub fn finalize_records(records: &mut [Value], information: &mut Value) -> Result<()> {
    let previous = records.to_vec();
    let inputs: Vec<Value> = records
        .iter()
        .map(|row| {
            let mut input = row.clone();
            if !row["observation"].is_object() {
                // An unreadable crop interrupts continuity and carries no events.
                input["observation"] = json!({"state":{},"voices":[],"relations":[]});
            }
            input
        })
        .collect();
    let auxiliary: Vec<Value> = records
        .iter()
        .map(assembly_auxiliary)
        .collect::<Result<_>>()?;
    let assembled = scorelib::observation::assemble(&inputs, Some(&auxiliary))?;
    information["relations"] = assembled["relations"].clone();
    information["recognition_mechanism"] = json!("visible_observation");
    for (index, (row, assembly)) in records
        .iter_mut()
        .zip(list(&assembled, "records"))
        .enumerate()
    {
        for key in ["resolved", "unresolved", "ir"] {
            row[key] = assembly[key].clone();
        }
        row["assembly_relations"] = json!(list(&assembled, "relations")
            .iter()
            .filter(|chain| list(chain, "segments")
                .iter()
                .any(|segment| segment["record"] == index))
            .cloned()
            .collect::<Vec<_>>());
        if !flag(row, "manually_edited") {
            row["visual_techniques"] = if assembly["ir"].is_string() {
                product_technique_overlay(&assembly["resolved"])
            } else {
                Value::Null
            };
        }
        row["visual_pitch"] = json!(true);
        row["score_state_evidence"] = assembly["resolved"]["state"].clone();
        row["score_state"] = json!({});
        for field in ["time", "key", "clef", "clef_octave"] {
            row["score_state"][field] = assembly["resolved"]["state"][field]["value"].clone();
        }
        row["state_needs_review"] = json!(
            row["score_state"]["time"].is_null()
                || (row["mode"] != "tab" && row["score_state"]["key"].is_null())
        );
        row["written_target"] = assembly["ir"].clone();
        if !flag(row, "manually_edited") {
            row["target"] = Value::Null;
            if let Some(written) = assembly["ir"].as_str() {
                let mode = text(row, "mode", "tab").to_owned();
                let converted = (|| -> Result<String> {
                    let mut measure = score::parse_measure_target(written)?;
                    if mode != "tab" {
                        if let Some(fifths) = row["score_state"]["key"].as_i64() {
                            measure["key_signature"] = json!(key_name(fifths, false)?);
                            measure["print_key_signature"] =
                                json!(row["observation"]["state"]["key"]["status"] == "observed");
                        }
                    }
                    let written = score::format_measure_target(&measure, &mode, true)?;
                    row["written_target"] = json!(written);
                    if mode == "tab" || row["instrument"] == "drums" {
                        return Ok(written);
                    }
                    if row["score_state"]["clef_octave"].is_null() {
                        return Err(
                            "Clef octave is unreadable; confirm the visible clef modifier".into(),
                        );
                    }
                    let context = product_pitch_context(row);
                    convert_pitch_target(&written, &context, &mode)
                })();
                match converted {
                    Ok(target) => row["target"] = json!(target),
                    Err(error) => row["unresolved"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"reason":"pitch_conversion","detail":error})),
                }
            }
            row["pitch_reference"] = json!(if row["instrument"] == "drums" {
                "percussion_key"
            } else if matches!(text(row, "instrument", ""), "guitar" | "bass") {
                "before_capo"
            } else {
                "sounding"
            });
        }
        row["recognition_needs_review"] = json!(
            row["target"].is_null()
                || !list(row, "unresolved").is_empty()
                || !list(row, "recognition_errors").is_empty()
        );
        if !flag(row, "manually_edited") {
            row["needs_review"] = json!(
                flag(row, "recognition_needs_review")
                    || flag(row, "state_needs_review")
                    || flag(row, "alignment_needs_review")
                    || flag(row, "structure_needs_review")
                    || flag(row, "pitch_needs_review")
                    || flag(row, "instrument_needs_review")
                    || tuning_needs_review(row)
            );
        }
        row["timing_errors"] = json!([]);
    }
    reconcile_score_tuning(records)?;
    let mut annotations = vec![];
    let parts = if list(information, "parts").is_empty() {
        vec![information.clone()]
    } else {
        list(information, "parts").to_vec()
    };
    for part in parts {
        for annotation in list(&part["document_metadata"], "score_annotations") {
            let mut p = annotation.clone();
            p["part_id"] = json!(text(&part, "id", "part-1"));
            annotations.push(p);
        }
    }
    attach_chord_annotations(records, &annotations)?;
    let opening = records.first().map(bar_index);
    let tempo = information["document_metadata"]["tempo_quarter"]
        .as_i64()
        .filter(|n| (20..=400).contains(n));
    for row in records.iter_mut() {
        if !row["target"].is_string() {
            continue;
        }
        if !flag(row, "manually_edited") && Some(bar_index(row)) == opening {
            if let Some(tempo) = tempo {
                let mut m = score::parse_measure_target(text(row, "target", ""))?;
                if m["tempo_quarter"].is_null() {
                    m["tempo_quarter"] = json!(tempo);
                }
                row["target"] = json!(score::format_measure_target(
                    &m,
                    text(row, "mode", "notation"),
                    true
                )?);
            }
        }
        let target = text(row, "target", "");
        let measure = score::parse_measure_target(target)?;
        let mut warnings = scorelib::gp5_constraints::timing_errors(&measure)?;
        warnings.extend(meter_warnings(target, row)?);
        warnings.extend(chord_recognition_errors(&measure));
        row["timing_errors"] = json!(warnings);
        if !list(row, "timing_errors").is_empty() {
            row["needs_review"] = json!(true);
        }
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        groups
            .entry(text(r, "part_id", "part-1").into())
            .or_default()
            .push(i);
    }
    for (part, indices) in groups {
        let first = &records[indices[0]];
        if first["mode"] == "notation"
            && matches!(text(first, "instrument", ""), "guitar" | "bass")
            && indices.iter().all(|&i| records[i]["mode"] == "notation")
            && !indices
                .iter()
                .any(|&i| flag(&records[i], "instrument_needs_review"))
        {
            let candidates = if list(first, "fingering_tunings").is_empty() {
                vec![integers(&first["tuning"])?]
            } else {
                list(first, "fingering_tunings")
                    .iter()
                    .map(integers)
                    .collect::<Result<Vec<_>>>()?
            };
            let mut evaluation = vec![];
            for tuning in &candidates {
                let mut failures = vec![];
                for &i in &indices {
                    if !records[i]["target"].is_string() {
                        failures.push(vec![]);
                        continue;
                    }
                    failures.push(scorelib::gp5_constraints::notation_fingering_errors(
                        &score::parse_measure_target(text(&records[i], "target", ""))?,
                        tuning,
                    )?);
                }
                evaluation.push(failures);
            }
            let selected = (0..candidates.len())
                .min_by_key(|&i| (evaluation[i].iter().filter(|e| !e.is_empty()).count(), i))
                .unwrap();
            for (&i, errors) in indices.iter().zip(&evaluation[selected]) {
                records[i]["tuning"] = json!(candidates[selected]);
                records[i]["fingering_errors"] = json!(errors);
                if !errors.is_empty() && flag(&records[i], "tuning_explicit") {
                    records[i]["needs_review"] = json!(true);
                }
            }
        }
        if let Some(parts) = information["parts"].as_array_mut() {
            if let Some(p) = parts.iter_mut().find(|p| text(p, "id", "") == part) {
                p["tuning_used"] = records[indices[0]]["tuning"].clone();
                p["tuning_source"] = records[indices[0]]["tuning_source"].clone();
            }
        }
    }
    for (row, before) in records.iter_mut().zip(&previous) {
        if flag(row, "manually_edited") {
            continue;
        }
        let changed = row["target"] != before["target"]
            || row["assembly_relations"] != before["assembly_relations"]
            || row["unresolved"] != before["unresolved"];
        if !changed && flag(before, "reviewed") {
            row["reviewed"] = json!(true);
            row["needs_review"] = before["needs_review"].clone();
        } else if changed || flag(row, "needs_review") {
            row["reviewed"] = json!(false);
        }
    }
    if let Some(first) = records.first() {
        information["tuning_used"] = first["tuning"].clone();
    }
    Ok(())
}

/// Rebuild musical drafts after editing local evidence or document context.
pub fn reassemble(source: &mut Value) -> Result<()> {
    let mut records = source["records"]
        .as_array()
        .ok_or("Recognition is missing records")?
        .clone();
    finalize_records(&mut records, source)?;
    source["records"] = json!(records);
    Ok(())
}

fn unquote(value: &str) -> String {
    let mut out = vec![];
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(high), Some(low)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                let n = (high * 16 + low) as u8;
                out.push(n);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
fn chord_key(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace('♭', "b")
        .replace('♯', "#")
}

/// Preserve accepted/manual OCR records after compatible metadata edits.
/// Rebuild derived pitches from retained observations when context changes.
pub fn update_information(source: &Value, information: &Value) -> Result<Option<Value>> {
    let mut result = source.clone();
    for field in [
        "title",
        "artist",
        "instrument",
        "midi_program",
        "tuning_used",
        "capo",
        "transpose",
        "document_metadata",
        "parts",
        "measure_profiles",
        "measure_pitch_contexts",
        "tuning_source",
        "tuning_candidates",
    ] {
        if let Some(value) = information.get(field) {
            result[field] = value.clone();
        }
    }
    let records = result["records"]
        .as_array_mut()
        .ok_or("Recognition is missing records")?;
    let updated_layout = list(information, "resolved_records");
    if !updated_layout.is_empty() && updated_layout.len() != records.len() {
        return Ok(None);
    }
    for row in records.iter_mut() {
        let number = row["measure_number"]
            .as_i64()
            .ok_or("Recognition record is missing measure number")?;
        if !updated_layout.is_empty() {
            let Some(updated) = updated_layout
                .iter()
                .find(|r| r["measure_number"] == number)
            else {
                return Ok(None);
            };
            // A regrouped staff can reuse its observation; a changed crop cannot.
            if ["bbox", "mode", "source_page", "recognition_bbox"]
                .into_iter()
                .any(|field| {
                    updated
                        .get(field)
                        .is_some_and(|value| row.get(field) != Some(value))
                })
            {
                return Ok(None);
            }
            for field in [
                "source_id",
                "part_id",
                "staff_id",
                "bar_index",
                "system_index",
                "page",
            ] {
                if let Some(value) = updated.get(field) {
                    row[field] = value.clone();
                }
            }
        }
        let profile = list(information, "measure_profiles")
            .iter()
            .find(|r| r["measure_number"] == number);
        if let Some(profile) = profile {
            merge(row, profile);
        } else {
            let part = list(information, "parts")
                .iter()
                .find(|p| p["id"] == row["part_id"])
                .unwrap_or(information);
            for field in ["instrument", "midi_program", "capo", "tuning_source"] {
                if let Some(value) = part.get(field) {
                    row[field] = value.clone();
                }
            }
            if let Some(tuning) = part.get("tuning_used") {
                row["tuning"] = tuning.clone();
            }
        }
        if let Some(context) = list(information, "measure_pitch_contexts")
            .iter()
            .find(|r| r["measure_number"] == number)
        {
            merge(row, context);
        }
        if row["pitch_context"].is_object() {
            row["pitch_context"]["capo"] = row["capo"].clone();
        }
    }
    reassemble(&mut result)?;
    Ok(Some(result))
}

fn event_chord_name(event: &Value) -> Option<String> {
    list(event, "effects")
        .iter()
        .filter_map(Value::as_str)
        .find_map(|effect| effect.strip_prefix("chord:"))
        .map(unquote)
}
pub fn is_chord_symbol(name: &str) -> bool {
    use std::sync::OnceLock;
    use unicode_normalization::UnicodeNormalization;
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    static MODIFIER: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        let accidental = r"(?:##|bb|#|b|x)?";
        let root = format!(r"(?:[A-Ha-h]{accidental}|{accidental}(?:[IViv]+|[1-7]))");
        let quality = r"(?:[mM]aj|[mM]in|dim|aug|sus|dom|m|M|\+|-|°|ø|Δ|o)?";
        let extension = r"(?:2|4|5|6|7|9|11|13)?";
        let modifier = r"(?:[#b](?:5|6|7|9|11|13)|(?:[mM]aj|M)(?:7|9|11|13)|(?:sus|add|omit|no)(?:2|3|4|5|6|7|9|11|13)|alt)";
        let suffix = format!(r"{quality}{extension}(?:M|\+|-|°|ø)?(?:{modifier}){{0,4}}");
        let group = format!(r"(?:\((?:{modifier}|{extension})(?:,?(?:{modifier}|{extension})){{0,3}}\))?");
        Regex::new(&format!(r"\A{root}{suffix}{group}(?:/(?:{root}|9|11|13))?\z")).unwrap()
    });
    let modifier = MODIFIER
        .get_or_init(|| Regex::new(r"(?:[#b]|sus|add|omit|no)(?:2|3|4|5|6|7|9|11|13)").unwrap());
    let normalized = chord_key(&name.nfkc().collect::<String>())
        .replace("major", "maj")
        .replace("minor", "min");
    if ["nc", "n.c", "n.c.", "nochord"].contains(&normalized.to_lowercase().as_str()) {
        return true;
    }
    let pieces: Vec<_> = modifier.find_iter(&normalized).collect();
    let repeated = pieces
        .windows(2)
        .any(|p| p[0].end() == p[1].start() && p[0].as_str() == p[1].as_str());
    pattern.is_match(&normalized) && !repeated
}

pub fn chord_recognition_errors(measure: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    for voice in list(measure, "voices") {
        for event in list(voice, "events") {
            let Some(name) = event_chord_name(event) else {
                continue;
            };
            if !is_chord_symbol(&name) {
                errors.push(format!("Uncertain chord symbol {:?}: re-read its visible name and onset; titles, section labels, instrument names and technique instructions belong in text, not chord. Keep all musical events and do not invent a replacement chord", name.chars().take(64).collect::<String>()));
            }
        }
    }
    errors
}

#[cfg(test)]
mod observation_pipeline_tests {
    use super::*;

    fn record(bar: i64, fret: Value, relations: Value) -> Value {
        json!({"source_id":"score", "part_id":"guitar", "staff_id":"tab", "bar_index":bar,
            "measure_number":bar+1,"mode":"tab","instrument":"guitar","capo":0,
            "tuning":[64,59,55,50,45,40],"tuning_source":"manual","tuning_explicit":true,
            "observation":{"state":{"time":{"status":if bar==0 {"observed"}else{"absent"},
                "value":if bar==0 {json!("4/4")}else{Value::Null}}},
                "voices":[{"voice":0,"events":[{"start":0,"duration":"q","status":"normal",
                    "notes":[{"string":1,"fret":fret}],"effects":[]}]}],"relations":relations}})
    }
    fn information() -> Value {
        json!({"parts":[],"document_metadata":{}})
    }
    struct LocalOnly {
        output: String,
        calls: usize,
    }
    impl Generator for LocalOnly {
        fn generate(
            &mut self,
            messages: Value,
            _: usize,
            schema: Option<Value>,
            grammar: Option<&str>,
        ) -> Result<llama::Generation> {
            self.calls += 1;
            let content = messages[0]["content"].as_array().unwrap();
            assert_eq!(
                content
                    .iter()
                    .filter(|part| part["type"] == "image_url")
                    .count(),
                1
            );
            let prompt = content
                .iter()
                .find_map(|part| part["text"].as_str())
                .unwrap();
            assert!(prompt.contains("Do not infer information outside the image"));
            assert!(!prompt.contains("Effective printed state"));
            assert!(schema.is_some());
            assert!(grammar.is_none());
            Ok(llama::Generation {
                content: self.output.clone(),
                completion_tokens: 90,
            })
        }
    }
    #[test]
    fn single_crop_keeps_missing_fret_and_invalid_json_does_not_become_rest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("crop.png");
        RgbImage::from_pixel(100, 100, Rgb([255, 255, 255]))
            .save(&path)
            .unwrap();
        let mut row = record(0, Value::Null, json!([]));
        row["image"] = json!(path);
        row["previous_image"] = json!("/not/an/input.png");
        row["next_image"] = json!("/not/an/input.png");
        let mut engine = LocalOnly {
            output: row["observation"].to_string(),
            calls: 0,
        };
        let result =
            recognize_measure(&mut engine, &row, &MeasureConfig::default(), &|| false).unwrap();
        assert!(result["observation"]["voices"][0]["events"][0]["notes"][0]["fret"].is_null());
        assert!(result["target"].is_null());
        engine.output = "unreadable".into();
        let result =
            recognize_measure(&mut engine, &row, &MeasureConfig::default(), &|| false).unwrap();
        assert!(result["observation"].is_null());
        let mut rows = vec![result];
        finalize_records(&mut rows, &mut information()).unwrap();
        assert!(rows[0]["target"].is_null());
        assert_eq!(rows[0]["needs_review"], true);
    }
    #[test]
    fn reassembly_updates_tied_neighbour_from_manual_value_and_stops_at_gap() {
        let mut rows = vec![
            record(
                0,
                json!(5),
                json!([{"kind":"tie","shape":"curve","start":[0,0,0],"end":"right","voice":0}]),
            ),
            record(
                1,
                Value::Null,
                json!([{"kind":"tie","shape":"curve","start":"left","end":[0,0,0],"voice":0}]),
            ),
        ];
        rows[0]["observation"]["voices"][0]["events"][0]["duration"] = json!("w");
        finalize_records(&mut rows, &mut information()).unwrap();
        assert!(rows[1]["target"].as_str().unwrap().contains("s1f5(tie)"));
        rows[0]["target"] = json!("M | V0{@0:w:s1f9}");
        rows[0]["manually_edited"] = json!(true);
        rows[0]["needs_review"] = json!(true);
        rows[0]["visual_techniques"] =
            json!([{"voice":0,"event":0,"effect":"pm","source":"manual"}]);
        finalize_records(&mut rows, &mut information()).unwrap();
        assert!(rows[1]["target"].as_str().unwrap().contains("s1f9(tie)"));
        assert_eq!(
            rows[0]["observation"]["voices"][0]["events"][0]["notes"][0]["fret"],
            5
        );
        assert_eq!(rows[0]["needs_review"], true);
        assert_eq!(rows[0]["visual_techniques"][0]["source"], "manual");
        rows[1]["bar_index"] = json!(3);
        finalize_records(&mut rows, &mut information()).unwrap();
        assert!(rows[1]["target"].is_null());
    }
    #[test]
    fn notation_is_converted_after_assembly_and_can_be_reassembled() {
        let mut row = record(0, json!(0), json!([]));
        row["mode"] = json!("notation");
        row["capo"] = json!(2);
        row["pitch_context"] = json!({"clef":"G2","clef_octave":0,"instrument_transpose":-12,"instrument_transpose_base":-12,"instrument_transpose_explicit":false,"capo":2});
        row["observation"]["state"]["key"] = json!({"status":"observed","value":0});
        row["observation"]["state"]["clef"] = json!({"status":"observed","value":"G2"});
        row["observation"]["state"]["clef_octave"] = json!({"status":"observed","value":0});
        row["observation"]["voices"][0]["events"][0]["notes"] =
            json!([{"staff_position":0,"accidental":null}]);
        let mut source = information();
        source["records"] = json!([row]);
        reassemble(&mut source).unwrap();
        let row = &source["records"][0];
        assert!(row["written_target"].as_str().unwrap().contains("p64"));
        assert!(row["target"].as_str().unwrap().contains("p50"));
        let before = source["records"][0]["target"].clone();
        reassemble(&mut source).unwrap();
        assert_eq!(source["records"][0]["target"], before);
        // The same conventional guitar octave must not be applied twice when
        // the measure model sees a small 8 missed by information detection.
        source["records"][0]["observation"]["state"]["clef_octave"] =
            json!({"status":"observed","value":-12});
        reassemble(&mut source).unwrap();
        assert!(source["records"][0]["target"]
            .as_str()
            .unwrap()
            .contains("p50"));
        source["records"][0]["pitch_context"]["instrument_transpose_explicit"] = json!(true);
        reassemble(&mut source).unwrap();
        assert!(source["records"][0]["target"]
            .as_str()
            .unwrap()
            .contains("p38"));

        // A new printed clef survives the next crop despite stale inherited
        // information context; the next crop explicitly sees no new clef.
        let mut second = source["records"][0].clone();
        second["measure_number"] = json!(2);
        second["bar_index"] = json!(1);
        second["observation"]["state"]["clef"] = json!({"status":"observed","value":"F4"});
        let mut third = second.clone();
        third["measure_number"] = json!(3);
        third["bar_index"] = json!(2);
        third["observation"]["state"]["clef"] = json!({"status":"absent","value":null});
        source["records"]
            .as_array_mut()
            .unwrap()
            .extend([second, third]);
        reassemble(&mut source).unwrap();
        assert_eq!(
            source["records"][2]["resolved"]["state"]["clef"],
            json!({"status":"inherited","value":"F4"})
        );

        // Manual shared metadata remains authoritative over the unchanged image.
        source["records"][0]["manual_state"] =
            json!({"time":{"status":"observed","value":"3/4","source":"manual"}});
        source["records"][1]["observation"]["state"]["time"] =
            json!({"status":"absent","value":null});
        source["records"][2]["observation"]["state"]["time"] =
            json!({"status":"absent","value":null});
        reassemble(&mut source).unwrap();
        reassemble(&mut source).unwrap();
        assert_eq!(source["records"][2]["score_state"]["time"], "3/4");
        source["records"][2]["observation"]["state"]["clef_octave"] =
            json!({"status":"unknown","value":null});
        reassemble(&mut source).unwrap();
        assert!(source["records"][2]["target"].is_null());
        assert_eq!(source["records"][2]["needs_review"], true);
    }
}
