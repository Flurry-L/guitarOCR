//! Native OCR stages. This module owns model prompts and score semantics, never
//! session persistence, task state, revision checks, or the model process.
//! Inputs are explicit local crops and schema-1.0 stage values. No Python path.
use crate::auxiliary_onnx::SignatureReader;
use crate::image_boundary;
use guitarocr_engine::llama;
use image::{Rgb, RgbImage};
use regex::Regex;
use scorelib::score;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    io::Cursor,
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, String>;
pub const CANCELLED: &str = "Recognition cancelled; completed records can be retained";

/// Injection point for a native engine, also allowing model-free contract tests.
pub trait Generator {
    fn generate(
        &mut self,
        messages: Value,
        max_tokens: usize,
        schema: Option<Value>,
        grammar: Option<&str>,
    ) -> Result<llama::Generation>;
}
impl Generator for guitarocr_engine::runtime::Session<'_> {
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
    pub notation_spacing: f64,
    pub tab_spacing: f64,
    pub max_pixels: u64,
    pub max_upscale: f64,
    pub patch_multiple: u32,
    pub page_max_pixels: u64,
    pub small_crop_min_side: f64,
    pub small_crop_max_dimension: u32,
    pub small_crop_max_aspect: f64,
    pub small_crop_max_upscale: f64,
}
impl Default for ImagePolicy {
    fn default() -> Self {
        Self {
            notation_spacing: 12.,
            tab_spacing: 18.,
            max_pixels: 768 * 768,
            max_upscale: 2.5,
            patch_multiple: 28,
            page_max_pixels: 768 * 768,
            small_crop_min_side: 0.,
            small_crop_max_dimension: 512,
            small_crop_max_aspect: 2.,
            small_crop_max_upscale: 3.,
        }
    }
}
impl ImagePolicy {
    pub fn from_value(v: &Value) -> Result<Self> {
        if v["version"] != 1 {
            return Err("Unsupported score image policy version".into());
        }
        let mut p = Self::default();
        p.notation_spacing = v["notation_spacing"]
            .as_f64()
            .ok_or("Missing notation_spacing")?;
        p.tab_spacing = v["tab_spacing"].as_f64().ok_or("Missing tab_spacing")?;
        p.max_pixels = v["max_pixels"].as_u64().ok_or("Missing max_pixels")?;
        p.max_upscale = v["max_upscale"].as_f64().ok_or("Missing max_upscale")?;
        p.patch_multiple = v["patch_multiple"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("Invalid patch_multiple")?;
        p.page_max_pixels = v["page_max_pixels"].as_u64().unwrap_or(p.max_pixels);
        p.small_crop_min_side = v["small_crop_min_side"].as_f64().unwrap_or(0.);
        p.small_crop_max_dimension = v["small_crop_max_dimension"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(512);
        p.small_crop_max_aspect = v["small_crop_max_aspect"].as_f64().unwrap_or(2.);
        p.small_crop_max_upscale = v["small_crop_max_upscale"].as_f64().unwrap_or(3.);
        p.validate()?;
        Ok(p)
    }
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        if data.len() > 65536 {
            return Err("Image policy exceeds 64 KiB".into());
        }
        Self::from_value(&serde_json::from_slice(&data).map_err(|e| e.to_string())?)
    }
    fn validate(&self) -> Result<()> {
        if self.patch_multiple == 0
            || self.patch_multiple > 256
            || self.max_pixels < 112 * 112
            || self.page_max_pixels < 112 * 112
            || self.max_pixels > image_boundary::MAX_PAGE_PIXELS
            || self.page_max_pixels > image_boundary::MAX_PAGE_PIXELS
            || [
                self.notation_spacing,
                self.tab_spacing,
                self.max_upscale,
                self.small_crop_max_aspect,
                self.small_crop_max_upscale,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0. || *v > 1024.)
            || !self.small_crop_min_side.is_finite()
            || !(0.0..=4096.0).contains(&self.small_crop_min_side)
        {
            return Err("Invalid image policy bounds".into());
        }
        Ok(())
    }
}
fn gray(p: &Rgb<u8>) -> u8 {
    ((u32::from(p[0]) * 19595 + u32::from(p[1]) * 38470 + u32::from(p[2]) * 7471 + 32768) >> 16)
        as u8
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    } else {
        v[n / 2]
    }
}
/// Same long-stroke statistic and tie-breaking as research/common/score_image.py.
pub fn staff_scale(image: &RgbImage) -> Option<(f64, usize)> {
    if image.width().min(image.height()) < 24 {
        return None;
    }
    let ink: Vec<f64> = (0..image.height())
        .map(|y| {
            (0..image.width())
                .filter(|&x| gray(image.get_pixel(x, y)) < 225)
                .count() as f64
                / f64::from(image.width())
        })
        .collect();
    let threshold = 0.22f64.max(ink.iter().copied().fold(0., f64::max) * 0.5);
    let rows: Vec<usize> = ink
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v > threshold).then_some(i))
        .collect();
    if rows.len() < 5 {
        return None;
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    for row in rows {
        if groups.last().is_some_and(|g| g.last() == Some(&(row - 1))) {
            groups.last_mut().unwrap().push(row)
        } else {
            groups.push(vec![row]);
        }
    }
    let centers: Vec<f64> = groups
        .iter()
        .map(|g| {
            g.iter().map(|&i| i as f64 * ink[i]).sum::<f64>()
                / g.iter().map(|&i| ink[i]).sum::<f64>()
        })
        .collect();
    let strengths: Vec<f64> = groups
        .iter()
        .map(|g| g.iter().map(|&i| ink[i]).fold(0., f64::max))
        .collect();
    let mut best: Option<(f64, f64, usize)> = None;
    for start in 0..centers.len().saturating_sub(4) {
        for count in 5..=8 {
            if start + count > centers.len() {
                continue;
            }
            let mut gaps: Vec<f64> = centers[start..start + count]
                .windows(2)
                .map(|p| p[1] - p[0])
                .collect();
            let spacing = median(&mut gaps);
            if !(4.0..=48.0).contains(&spacing)
                || gaps
                    .iter()
                    .any(|g| (g - spacing).abs() > 1.3f64.max(spacing * 0.13))
            {
                continue;
            }
            let candidate = (
                strengths[start..start + count].iter().sum::<f64>(),
                spacing,
                count,
            );
            if best.is_none_or(|b| candidate > b) {
                best = Some(candidate);
            }
        }
    }
    best.map(|(_, spacing, lines)| (spacing, lines))
}
pub fn normalize_score_image(image: &RgbImage, policy: &ImagePolicy) -> Result<RgbImage> {
    image_boundary::check_dimensions(image.width(), image.height())?;
    policy.validate()?;
    let mut scale = staff_scale(image)
        .map(|(gap, lines)| {
            policy.max_upscale.min(
                if lines >= 6 {
                    policy.tab_spacing
                } else {
                    policy.notation_spacing
                } / gap,
            )
        })
        .unwrap_or(1.);
    let (short, long) = (
        image.width().min(image.height()),
        image.width().max(image.height()),
    );
    if policy.small_crop_min_side > 0.
        && long <= policy.small_crop_max_dimension
        && f64::from(long) / f64::from(short) <= policy.small_crop_max_aspect
    {
        scale = scale.max(
            policy
                .small_crop_max_upscale
                .min(policy.small_crop_min_side / f64::from(short)),
        );
    }
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
pub const HEADER_PROMPT:&str="Read only visible song title, artist or author credit (including composer, arranger or studio), and tuning label. Preserve the original language and spelling. Return one JSON object with title, artist, tuning_name; use null for absent fields. Tempo, playing techniques and chord names are not titles or author credits.";
pub const TEMPO_PROMPT:&str="Read the printed quarter-note tempo. Return one JSON object with tempo_quarter; use null if unreadable.";
pub const STAFF_PROMPT:&str="Read the first staff and its instrument label. Return JSON with instrument (guitar, bass, pitched, drums, or null if unknown) and string_count (number of TAB lines, or null if TAB is absent), and name (the visible instrument label, preserving abbreviations, or null if absent). Use the percussion clef for drums. Use pitched for other melodic instruments or unlabelled melodic notation; use null for unnamed TAB. Do not infer a string count from the five lines of standard notation.";
pub const CLEF_PROMPT:&str="Read the visible clef. Return JSON with clef (G2, F4, C3, C4, percussion, tab, or null) and clef_octave (0, -12, 12, -24, 24, or null if unreadable). A small 8 or 15 below the clef lowers its pitches; above raises them. Use tab with clef_octave 0 for a TAB symbol.";
pub const ANNOTATION_PROMPT:&str="Classify and read the visible score annotation. Return JSON with kind (instrument, ottava, capo, chord, chord_diagram, technique, tempo, title, credit, other, or null if unreadable), semitones (sounding minus written pitch, or null), capo (fret number, or null), and text (visible words, or null). A chord name such as Bb or F# and a chord fingering diagram are NOT transposition instructions. Let ring, P.M., vibrato and their continuation lines are techniques, NOT ottava. Use title for a song title and credit for a composer, arranger, artist or studio credit; preserve the original language. Use null pitch fields for non-pitch annotations. For ottava use 12 for 8va, -12 for 8vb, 24 for 15ma, -24 for 15mb. For instrument transposition use only an explicit instruction or an unambiguous instrument label. Do not treat a key signature or tuning as an instrument transposition. For chord_diagram also return diagram with base_fret, frets (absolute fret numbers, 0 open, \"x\" muted, LOW to HIGH string), fingers (visible numbers or null per string), barres ([absolute fret, zero-based low string index, high string index]). Preserve chord accidentals and slash bass. Read only visible dots and finger numbers; never derive a fingering from the chord name. Transcribe the evidence; do not invent missing words.";
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
        "annotation" | "transposition" => {
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
    if matches!(kind, "annotation" | "transposition") {
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
pub fn program_from_visible_name(name: &str) -> Option<i64> {
    let name = Regex::new("[a-z]+")
        .unwrap()
        .find_iter(&name.to_lowercase())
        .map(|v| v.as_str().to_owned())
        .collect::<Vec<_>>()
        .join(" ");
    for (n, p) in [
        ("pno", 0),
        ("el pno", 4),
        ("hpsi", 6),
        ("clav", 7),
        ("org", 16),
        ("acc", 21),
        ("harm", 22),
        ("chr", 52),
        ("syn chr", 54),
        ("vln", 40),
        ("vla", 41),
        ("vlc", 42),
        ("fl", 73),
        ("picc", 72),
        ("ob", 68),
        ("clar", 71),
        ("bn", 70),
        ("tpt", 56),
        ("tbn", 57),
        ("tba", 58),
        ("fhn", 60),
        ("hp", 46),
        ("sax", 65),
        ("saxophone", 65),
        ("vib", 11),
        ("rec", 74),
        ("whis", 78),
    ] {
        if name == n {
            return Some(p);
        }
    }
    for (pattern, p) in [
        (r"\bpiano\b", 0),
        (r"\bviolin\b", 40),
        (r"\bviola\b", 41),
        (r"\b(?:cello|violoncello)\b", 42),
        (r"\bcontrabass\b", 43),
        (r"\bflute\b", 73),
        (r"\boboe\b", 68),
        (r"\bclarinet\b", 71),
        (r"\bbassoon\b", 70),
        (r"\b(?:english horn|cor anglais)\b", 69),
        (r"\bsoprano sax(?:ophone)?\b", 64),
        (r"\balto sax(?:ophone)?\b", 65),
        (r"\btenor sax(?:ophone)?\b", 66),
        (r"\bbaritone sax(?:ophone)?\b", 67),
        (r"\btrumpet\b", 56),
        (r"\b(?:french horn|horn in f|f horn)\b", 60),
    ] {
        if Regex::new(pattern).unwrap().is_match(&name) {
            return Some(p);
        }
    }
    None
}
fn normalize_diagram(v: &Value) -> Value {
    let Some(frets) = v["frets"]
        .as_array()
        .filter(|f| (1..=12).contains(&f.len()))
    else {
        return Value::Null;
    };
    let base = match v.get("base_fret") {
        None => 1,
        Some(v) => match v.as_i64() {
            Some(n) => n,
            None => return Value::Null,
        },
    };
    if !(1..=36).contains(&base)
        || frets.iter().any(|f| {
            f != "x"
                && !f
                    .as_i64()
                    .is_some_and(|n| (0..=36).contains(&n) && (n == 0 || n >= base))
        })
    {
        return Value::Null;
    }
    let fingers = if v["fingers"].is_null() {
        vec![Value::Null; frets.len()]
    } else {
        match v["fingers"].as_array() {
            Some(a) => a.clone(),
            None => return Value::Null,
        }
    };
    if fingers.len() != frets.len()
        || fingers
            .iter()
            .any(|v| !v.is_null() && !v.as_i64().is_some_and(|n| (0..=4).contains(&n)))
    {
        return Value::Null;
    }
    if v.get("barres").is_some_and(|b| !b.is_array()) {
        return Value::Null;
    }
    let barres = list(v, "barres");
    for b in barres {
        let Ok(b) = integers(b) else {
            return Value::Null;
        };
        if b.len() != 3
            || b[0] < base
            || b[0] > 36
            || b[1] < 0
            || b[1] >= b[2]
            || b[2] >= frets.len() as i64
        {
            return Value::Null;
        }
    }
    json!({"base_fret":base,"frets":frets,"fingers":fingers,"barres":barres})
}
pub fn parse_info_response(raw: &str, kind: &str) -> Value {
    let Ok(v) = json_response(raw) else {
        return json!({});
    };
    match kind {
        k if header_field(k).is_some() => json!({header_field(k).unwrap(): cleaned(&v["text"])}),
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
        "annotation" | "transposition" => {
            let kind = text(&v, "kind", "");
            let words = cleaned(&v["text"]);
            let words_str = words.as_str().unwrap_or("");
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
        } else if ["header", "staff"].contains(&kind)
            || (kind == "tempo" && m["tempo_quarter"].is_null())
        {
            merge(&mut m, &p["parsed"]);
        } else if kind == "annotation"
            && p["parsed"]["kind"] == "tempo"
            && m["tempo_quarter"].is_null()
        {
            if let Some(c) = Regex::new(r"♩\s*=\s*(\d{2,3})(?:\D|$)")
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
    if let Some(t) = tuning_from_name(text(&m, "tuning_name", ""), "guitar", None) {
        m["tuning_midi_high_to_low"] = json!(t);
    }
    m
}
pub fn standard_tuning(instrument: &str, count: Option<usize>) -> Result<Vec<i64>> {
    let count = count.unwrap_or(if instrument == "bass" { 4 } else { 6 });
    let pitches=match(instrument,count){("pitched"|"drums",_)=>vec![],("guitar",4)=>vec![64,59,55,50],("guitar",5)=>vec![64,59,55,50,45],("guitar",6)=>vec![64,59,55,50,45,40],("guitar",7)=>vec![64,59,55,50,45,40,35],("guitar",8)=>vec![64,59,55,50,45,40,35,30],("bass",4)=>vec![43,38,33,28],("bass",5)=>vec![43,38,33,28,23],("bass",6)=>vec![48,43,38,33,28,23],("bass",7)=>vec![53,48,43,38,33,28,23],_=>return Err(format!("No unambiguous default tuning for {count}-string {instrument}; enter open-string pitches"))};
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
    let label = label.strip_suffix(" tuning").unwrap_or(&label);
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
    pub compact_structure: bool,
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
            compact_structure: false,
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
    let mut result = vec![];
    for region in regions {
        check(cancelled)?;
        let kind = text(region, "kind", "");
        let mut image = region_image(region)?;
        if header_field(kind).is_some() {
            // Small text strips otherwise occupy too few vision patches.
            let scale = (144. / image.height() as f64)
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
        let response = engine.generate(
            image_messages(std::slice::from_ref(&image), region_prompt(kind)?, policy)?,
            512,
            Some(information_schema(kind)?),
            None,
        )?;
        check(cancelled)?;
        let mut parsed = parse_info_response(response.content.trim(), kind);
        if matches!(kind, "annotation" | "transposition") && parsed["kind"] == "chord_diagram" {
            let raw_diagram = json_response(&response.content)
                .ok()
                .and_then(|v| v.get("diagram").cloned())
                .unwrap_or(Value::Null);
            if let Some(refined) =
                crate::pixel_refinement::refine_diagram_image(&image, &raw_diagram)?
            {
                parsed["diagram"] = normalize_diagram(&refined);
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
        p["inside_header"] = json!(regions
            .iter()
            .any(|header| { header["kind"] == "header" && region_inside(region, header) }));
        merge(
            &mut p,
            &json!({"kind":resolved,"candidate_kind":kind,"raw":response.content.trim(),"parsed":parsed,"generated_token_count":response.completion_tokens}),
        );
        result.push(p);
    }
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

/// Keep printed names legible next to a full-width first staff row.
pub fn focus_staff(image: &RgbImage) -> RgbImage {
    let (w, h) = image.dimensions();
    let ink: Vec<bool> = image.pixels().map(|p| gray(p) < 210).collect();
    let density: Vec<f64> = (0..h)
        .map(|y| (0..w).filter(|&x| ink[(y * w + x) as usize]).count() as f64 / f64::from(w))
        .collect();
    let threshold = 0.55f64.max(density.iter().copied().fold(0., f64::max) * 0.8);
    let strong: Vec<u32> = (0..h)
        .filter(|&y| density[y as usize] > threshold)
        .collect();
    let mut label = None;
    if strong.len() >= 4 {
        let columns: Vec<u32> = (0..w)
            .filter(|&x| {
                strong
                    .iter()
                    .filter(|&&y| ink[(y * w + x) as usize])
                    .count() as f64
                    / strong.len() as f64
                    > 0.75
            })
            .collect();
        let mut runs: Vec<Vec<u32>> = vec![];
        for x in columns {
            if runs
                .last()
                .is_some_and(|r| r.last().is_some_and(|&p| p + 1 == x))
            {
                runs.last_mut().unwrap().push(x);
            } else {
                runs.push(vec![x]);
            }
        }
        if let Some(line) = runs.iter().find(|r| r.len() >= 20) {
            let left = line[0].saturating_sub(2);
            let exclude: Vec<bool> = (0..left)
                .map(|x| {
                    (0..h).filter(|&y| ink[(y * w + x) as usize]).count() as f64 / f64::from(h)
                        > 0.65
                })
                .collect();
            let mut points = vec![];
            for y in 0..h {
                for x in 0..left {
                    if !exclude[x as usize] && ink[(y * w + x) as usize] {
                        points.push((x, y));
                    }
                }
            }
            if points.len() >= 15 {
                let x0 = points.iter().map(|p| p.0).min().unwrap().saturating_sub(2);
                let y0 = points.iter().map(|p| p.1).min().unwrap().saturating_sub(2);
                let x1 = (points.iter().map(|p| p.0).max().unwrap() + 3).min(w);
                let y1 = (points.iter().map(|p| p.1).max().unwrap() + 3).min(h);
                let mut crop =
                    image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
                if f64::from(crop.height()) > f64::from(crop.width()) * 1.25 && crop.width() < 45 {
                    crop = image::imageops::rotate90(&crop);
                }
                let scale = 3.0f64.min(48. / f64::from(crop.height()));
                let nw = (f64::from(crop.width()) * scale).round_ties_even().max(1.) as u32;
                let nh = (f64::from(crop.height()) * scale).round_ties_even().max(1.) as u32;
                label = Some(crate::image_transforms::resize_rgb_lanczos(&crop, nw, nh));
            }
        }
    }
    let excerpt =
        image::imageops::crop_imm(image, 0, 0, w.min(560.max(h.saturating_mul(2))), h).to_image();
    let Some(label) = label else {
        return excerpt;
    };
    let mut result = RgbImage::from_pixel(
        excerpt.width().max(label.width() + 24),
        excerpt.height() + label.height() + 24,
        Rgb([255; 3]),
    );
    image::imageops::replace(&mut result, &label, 12, 8);
    image::imageops::replace(&mut result, &excerpt, 0, i64::from(label.height() + 24));
    result
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
        focus_staff(&crop),
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
pub fn structure_image(page: &RgbImage, rows: &[Vec<Value>], compact: bool) -> Result<RgbImage> {
    let mut marked = page.clone();
    for (i, rows) in rows.iter().enumerate() {
        let mut top = f64::INFINITY;
        let mut bottom = 0.0f64;
        for row in rows {
            let b = bbox(row)?;
            top = top.min(b[1]);
            bottom = bottom.max(b[1] + b[3]);
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
    if compact {
        let left = (f64::from(marked.width()) * 0.32).round_ties_even() as u32;
        let right = (f64::from(marked.width()) * 0.08).round_ties_even() as u32;
        let mut out = RgbImage::from_pixel(left + right + 24, marked.height(), Rgb([255; 3]));
        image::imageops::replace(
            &mut out,
            &image::imageops::crop_imm(&marked, 0, 0, left, marked.height()).to_image(),
            0,
            0,
        );
        image::imageops::replace(
            &mut out,
            &image::imageops::crop_imm(&marked, marked.width() - right, 0, right, marked.height())
                .to_image(),
            i64::from(left + 24),
            0,
        );
        Ok(out)
    } else {
        Ok(marked)
    }
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
    for (page, rows) in pages {
        check(cancelled)?;
        let path = rows[0][0]["source_page"]
            .as_str()
            .ok_or("Structure requires source_page images")?;
        let original = image_boundary::open_rgb(Path::new(path))?;
        let marked = structure_image(&original, &rows, false)?;
        let systems = crate::staff_geometry::marked_systems(&marked, rows.len())?;
        let image = if config.compact_structure {
            structure_image(&original, &rows, true)?
        } else {
            marked
        };
        let modes: Vec<String> = rows.iter().map(|r| majority_mode(r, "notation")).collect();
        let mut prompt = structure_prompt(rows.len(), &modes);
        if let Some(systems) = &systems {
            prompt += &format!(
                " Continuous barlines connect the rows into systems {:?}.",
                systems
            );
        }
        let mut schema = structure_schema(rows.len());
        if let Some(systems) = &systems {
            schema["properties"]["rows"]
                .as_object_mut()
                .unwrap()
                .remove("items");
            schema["properties"]["rows"]["prefixItems"] = json!(systems.iter().map(|system| json!({"type":"array","minItems":3,"maxItems":3,"prefixItems":[{"const":system},{"type":"integer","minimum":0,"maximum":rows.len()-1},{"type":"integer","minimum":0,"maximum":15}]})).collect::<Vec<_>>());
        }
        let initial = image_messages(&[image], &prompt, config.image_policy.as_ref())?;
        let mut messages = initial.clone();
        let mut parsed = None;
        let mut raw = String::new();
        let mut attempts = vec![];
        for attempt in 1..=3 {
            check(cancelled)?;
            let response = engine.generate(messages.clone(), 2048, Some(schema.clone()), None)?;
            raw = response.content;
            match crate::score_structure::parse_structure(
                &raw,
                rows.len(),
                systems.as_deref(),
                Some(&modes),
            ) {
                Ok(value) => {
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
        let parsed = parsed.ok_or_else(|| {
            format!("Could not resolve score parts on page {page}; check staff boxes and retry")
        })?;
        predictions.push(json!({"page":page,"source_page":path,"raw":raw,"parsed":parsed,"systems_from_image":systems,"attempts":attempts}));
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
        for p in changes.get(&i).into_iter().flatten() {
            let parsed = &p["parsed"];
            if p["kind"] == "clef" {
                if parsed["clef"] == "tab" {
                    continue;
                }
                for field in ["clef", "clef_octave"] {
                    if parsed[field].is_null() {
                        pending.insert(field.into());
                    } else {
                        state[field] = parsed[field].clone();
                        pending.remove(field);
                    }
                }
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
        let mut effective = state.clone();
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
        let (p, r) = crate::score_grid::resolve_document_reconciled(
            &records,
            &mut structure_predictions,
            output,
        )?;
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
    if config.pitch_context {
        regions.extend(opening_annotations(&records, &regions)?);
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
    let mut predictions =
        recognize_regions(engine, &regions, config.image_policy.as_ref(), cancelled)?;
    if config.staff_profile {
        for part in &parts {
            let rows: Vec<Value> = records
                .iter()
                .filter(|r| r["part_id"] == part["id"])
                .cloned()
                .collect();
            if let Some((mut region, image)) = staff_image(&rows)? {
                check(cancelled)?;
                let response = engine.generate(
                    image_messages(&[image], STAFF_PROMPT, config.image_policy.as_ref())?,
                    512,
                    Some(information_schema("staff")?),
                    None,
                )?;
                region["parsed"] = parse_info_response(&response.content, "staff");
                region["raw"] = json!(response.content.trim());
                region["candidate_kind"] = json!("staff");
                predictions.push(region);
            }
        }
    }
    if config.pitch_context {
        predictions = crate::ottava_geometry::link_octave_continuations_with_images(
            &records,
            &predictions,
            &mut |prediction| {
                check(cancelled)?;
                region_image(prediction).map(Some)
            },
        )?;
    }
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
        // The native port must preserve the multi-measure line-count evidence
        // used by research/inference/information, ahead of a VLM's answer.
        let visible_count = crate::staff_classifier::part_tab_strings(&rows)?;
        let count = if has_tab {
            visible_count
                .or_else(|| metadata["string_count"].as_u64().map(|n| n as usize))
                .or_else(|| part["strings"].as_u64().map(|n| n as usize))
        } else {
            None
        };
        metadata["string_count"] = json!(count);
        metadata["string_count_source"] = json!(if visible_count.is_some() {
            "staff_lines"
        } else {
            "model"
        });
        let named: HashSet<i64> = local_predictions
            .iter()
            .filter(|p| p["kind"] == "transposition" && p["parsed"]["kind"] == "instrument")
            .filter_map(|p| program_from_visible_name(text(&p["parsed"], "text", "")))
            .collect();
        let staff_instrument = metadata["instrument"].as_str();
        let structure_instrument = text(part, "instrument", "guitar");
        let instrument = if index == 0 {
            config.instrument.as_deref()
        } else {
            None
        }
        .unwrap_or_else(|| {
            if has_tab {
                if matches!(staff_instrument, Some("guitar" | "bass")) {
                    staff_instrument.unwrap()
                } else if ["guitar", "bass"].contains(&structure_instrument) {
                    structure_instrument
                } else if matches!(metadata["string_count"].as_u64(), Some(4 | 5)) {
                    "bass"
                } else {
                    "guitar"
                }
            } else if named.len() == 1 {
                "pitched"
            } else {
                staff_instrument.unwrap_or(structure_instrument)
            }
        })
        .to_owned();
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
        let tuning_source = if unresolved_tuning {
            "unresolved"
        } else if tuning_override.is_some() {
            "manual"
        } else if named_tuning.is_some() {
            "printed"
        } else {
            "default"
        };
        let mut warnings: Vec<Value> = vec![];
        if unresolved_tuning {
            metadata["tuning_issue"] = json!(format!(
                "弦数识别为 {}，无法确定调弦。请对照谱面选择调弦，或填写各弦音高；曲名、作者和速度已保留。",
                count.map(|n| n.to_string()).unwrap_or_else(|| "未知".into())
            ));
        }
        if !text(&metadata, "tuning_name", "").is_empty()
            && named_tuning.is_none()
            && tuning_override.is_none()
        {
            warnings.push(json!(format!("Could not resolve the printed {} tuning for this instrument; check open-string pitches",text(&metadata,"tuning_name",""))));
        }
        let mut candidates = if unresolved_tuning {
            vec![]
        } else {
            vec![tuning.clone()]
        };
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
        .or_else(|| program_from_visible_name(text(&metadata, "name", text(part, "name", ""))))
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
        let transpose = if index == 0 {
            config.transpose.map(i64::from)
        } else {
            None
        }
        .or_else(|| {
            if conventional_octave(text(part, "name", "")).is_none() {
                explicit_transposition(text(part, "name", ""))
            } else {
                None
            }
        });
        let mut local = rows.clone();
        for r in &mut local {
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
            conventional_octave(text(part, "name", "")),
        )?;
        let unusual = instrument == "guitar"
            && matches!(tuning.len(), 4 | 5)
            && tuning_source == "default"
            && local.iter().all(|r| r["mode"] == "tab");
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
            && conventional_octave(text(part, "name", "")).is_none()
            && program_from_visible_name(text(part, "name", ""))
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
        values.push(json!({"id":id,"name":part.get("name").cloned().unwrap_or(json!(instrument)),"mode":majority_mode(&rows,text(layout,"mode","notation")),"document_metadata":metadata,"title":metadata["title"].as_str().filter(|s|!s.is_empty()).unwrap_or(&config.fallback_title),"artist":metadata["artist"].as_str().unwrap_or(""),"tuning_used":tuning,"tuning_candidates":candidates,"tuning_source":tuning_source,"instrument":instrument,"midi_program":program,"capo":capo,"transpose":transpose}));
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
    pub visual_pitch: bool,
    pub pitch_context: bool,
    pub written_pitch: bool,
    pub constrained_decoding: bool,
    pub retry_constraints: bool,
}
impl Default for MeasureConfig {
    fn default() -> Self {
        Self {
            image_policy: None,
            max_tokens: 2048,
            max_tokens_ceiling: 4096,
            maximum_attempts: 3,
            visual_pitch: true,
            pitch_context: true,
            written_pitch: true,
            constrained_decoding: false,
            retry_constraints: true,
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
pub struct MeasureOutput {
    pub record: Value,
    pub diagnostics: Vec<Value>,
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
    attach_neighbours(&mut records)?;
    Ok(records)
}
static EMPTY_VALUES: Vec<Value> = Vec::new();
pub fn attach_neighbours(records: &mut [Value]) -> Result<()> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        groups.entry(staff_key(r)).or_default().push(i);
    }
    for indices in groups.values_mut() {
        indices.sort_by_key(|&i| bar_index(&records[i]));
        for (i, &index) in indices.iter().enumerate() {
            for (field, n) in [
                ("previous_image", i.checked_sub(1)),
                (
                    "next_image",
                    i.checked_add(1).filter(|&n| n < indices.len()),
                ),
            ] {
                let other = n
                    .map(|n| indices[n])
                    .filter(|&n| (bar_index(&records[n]) - bar_index(&records[index])).abs() <= 1)
                    .unwrap_or(index);
                records[index][field] = records[other]["image"].clone();
                let geometry = format!("{field}_record");
                let mut neighbour = records[other].clone();
                if let Some(o) = neighbour.as_object_mut() {
                    o.remove("previous_image_record");
                    o.remove("next_image_record");
                }
                records[index][geometry] = neighbour;
            }
        }
    }
    Ok(())
}
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
    resolve_score_states(records, &predictions)?;
    Ok(predictions)
}
pub fn resolve_score_states(records: &mut [Value], predictions: &Value) -> Result<()> {
    let mut shared: BTreeMap<(String, i64), String> = BTreeMap::new();
    let lanes: HashSet<String> = records.iter().map(staff_key).collect();
    if lanes.len() > 1 && records.iter().all(|r| r["bar_index"].as_i64().is_some()) {
        let mut votes: BTreeMap<(String, i64), HashMap<String, usize>> = BTreeMap::new();
        for r in records.iter() {
            let key = (text(r, "source_id", "").to_owned(), bar_index(r));
            let counter = votes.entry(key).or_default();
            let p = &predictions[r["measure_number"].to_string()];
            if let Some(time) = p["time"].as_str() {
                *counter.entry(time.into()).or_default() += 1;
            }
        }
        let mut current: HashMap<String, Option<String>> = HashMap::new();
        for ((source, bar), counts) in votes {
            let state = current.entry(source.clone()).or_default();
            let mut counts: Vec<_> = counts.into_iter().collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1));
            if let Some(top) = counts.first() {
                *state = if counts.len() == 1 || top.1 > counts[1].1 {
                    Some(top.0.clone())
                } else {
                    None
                };
            }
            if let Some(time) = state {
                shared.insert((source, bar), time.clone());
            }
        }
    }
    let mut states: HashMap<String, (Value, HashSet<String>)> = HashMap::new();
    for row in records {
        let id = row["measure_number"].to_string();
        let p = predictions
            .get(&id)
            .ok_or_else(|| format!("Missing signature state for measure {id}"))?;
        if !valid_signature(p) {
            return Err("Invalid signature state".into());
        }
        let (state, uncertain) = states
            .entry(staff_key(row))
            .or_insert_with(|| (json!({"time":"4/4","key":0}), HashSet::new()));
        if !p["error"].is_null() {
            uncertain.extend(["time".into(), "key".into()]);
        }
        for field in ["time", "key"] {
            if !p[field].is_null() {
                state[field] = p[field].clone();
                uncertain.remove(field);
            }
        }
        if let Some(time) = shared.get(&(text(row, "source_id", "").to_owned(), bar_index(row))) {
            state["time"] = json!(time);
            uncertain.remove("time");
        }
        row["score_state"] = state.clone();
        row["state_needs_review"] = json!(!uncertain.is_empty());
        row["printed_state"] = p.clone();
    }
    Ok(())
}

fn ordered_json(value: &Value, keys: &[&str]) -> String {
    format!(
        "{{{}}}",
        keys.iter()
            .filter_map(|key| value.get(*key).map(|v| {
                let rendered = if *key == "octave_spans" {
                    format!(
                        "[{}]",
                        v.as_array()
                            .into_iter()
                            .flatten()
                            .map(|s| ordered_json(s, &["semitones", "start", "end"]))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                } else {
                    v.to_string()
                };
                format!("{}:{}", serde_json::to_string(key).unwrap(), rendered)
            }))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn prompt_pitch_context(context: &Value) -> Value {
    let mut p = json!({});
    for key in [
        "clef",
        "clef_octave",
        "instrument_transpose",
        "octave_spans",
    ] {
        if let Some(v) = context.get(key) {
            p[key] = if key == "octave_spans" {
                json!(v
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|s| json!({"semitones":s["semitones"],"start":s["start"],"end":s["end"]}))
                    .collect::<Vec<_>>())
            } else {
                v.clone()
            };
        }
    }
    p
}
pub fn state_prompt(record: &Value) -> Result<String> {
    let mode = text(record, "mode", "");
    let instrument = text(record, "instrument", "guitar");
    if !["tab", "notation", "both"].contains(&mode)
        || !["guitar", "bass", "pitched", "drums"].contains(&instrument)
        || (matches!(instrument, "pitched" | "drums") && mode != "notation")
    {
        return Err("Unsupported instrument/display-mode combination".into());
    }
    let visual = flag(record, "visual_pitch");
    let mut state = record
        .get("score_state")
        .cloned()
        .unwrap_or(json!({"time":"4/4","key":0}));
    if !list(record, "tuning").is_empty() {
        state["tuning"] = record["tuning"].clone();
    }
    if mode == "tab" {
        state["key"] = json!(0);
    }
    if visual && mode != "tab" {
        state.as_object_mut().unwrap().remove("tuning");
    }
    let fields = match mode {
        "tab" => "string, fret",
        "notation" => "written MIDI pitch",
        _ => {
            if visual {
                "string, fret, written MIDI pitch"
            } else {
                "string, fret, MIDI pitch"
            }
        }
    };
    let mut prompt=format!("Independent {instrument} {mode} measure recognition. The first image is the target measure. The second and third images show the previous and next measures for visual context only. Return exactly one M2 fragment for the FIRST image. Preserve every voice, event start, duration, {fields}, rest, tie and visible technique. Use neighbouring images to resolve cross-bar ties and voice continuity; do not transcribe them. ");
    if bar_index(record) == 0 {
        prompt += "This is the first measure of the score. ";
    }
    prompt += &format!(
        "Effective printed state: {}. ",
        ordered_json(&state, &["time", "key", "tuning"])
    );
    if instrument == "drums" {
        prompt += "Pitch fields are General MIDI drum keys. ";
    } else if mode == "notation" || (visual && mode == "both") {
        prompt +=
            "Return written pitches, before instrument transposition, clef octave and ottava. ";
        if mode == "both" {
            prompt+="Read pitches from the notation and string/fret from TAB independently. Do not assume a tuning. ";
        }
    }
    if record["pitch_context"].is_object() && mode != "tab" && instrument != "drums" {
        prompt+=&format!("Pitch context: {}. Preserve ottava:12, ottava:-12, ottava:24 or ottava:-24 on each affected event. ",ordered_json(&prompt_pitch_context(&record["pitch_context"]), &["clef", "clef_octave", "instrument_transpose", "octave_spans"]));
    }
    prompt+="Print time/key metadata only at score start or a printed change; use the effective state to read notes.";
    Ok(prompt)
}
pub fn measure_messages(record: &Value, policy: Option<&ImagePolicy>) -> Result<Value> {
    let target = record_image(record)?;
    let mut images = vec![target.clone()];
    for field in ["previous_image", "next_image"] {
        images.push(if let Some(path) = record[field].as_str() {
            image_boundary::open_rgb(Path::new(path))?
        } else if record[format!("{field}_record")].is_object() {
            record_image(&record[format!("{field}_record")])?
        } else {
            target.clone()
        });
    }
    image_messages(&images, &state_prompt(record)?, policy)
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
pub fn resolve_measure_state(target: &str, record: &Value) -> Result<(String, bool)> {
    let state = &record["score_state"];
    if !state.is_object() || flag(record, "state_needs_review") {
        return Ok((target.into(), false));
    }
    let mut measure = score::parse_measure_target(target)?;
    let mut changed = false;
    if !measure["time_signature"].is_null() && measure["time_signature"] != state["time"] {
        measure["time_signature"] = state["time"].clone();
        changed = true;
    }
    if let Some(key) = measure["key_signature"].as_str() {
        if record["mode"] != "tab" {
            let original = key_parts(key).ok();
            let minor = original.map(|(_, m)| m).unwrap_or(false);
            let fifths = original.map(|(f, _)| {
                if f < -7 {
                    f + 12
                } else if f > 7 {
                    f - 12
                } else {
                    f
                }
            });
            if fifths != state["key"].as_i64() {
                measure["key_signature"] = json!(key_name(
                    state["key"].as_i64().ok_or("Missing effective key")?,
                    minor
                )?);
                changed = true;
            }
        }
    }
    if changed {
        Ok((
            score::format_measure_target(&measure, text(record, "mode", "notation"), true)?,
            true,
        ))
    } else {
        Ok((target.into(), false))
    }
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
    let mut m = score::parse_measure_target(target)?;
    let instrument = context["instrument_transpose"].as_i64().unwrap_or(0);
    let base = instrument + context["clef_octave"].as_i64().unwrap_or(0)
        - context["capo"].as_i64().unwrap_or(0);
    if !(-108..=108).contains(&base) {
        return Err("Invalid pitch context range".into());
    }
    if let Some(key) = m["key_signature"].as_str() {
        m["key_signature"] = json!(transpose_key(key, instrument)?);
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
            let shift = base + octaves.first().copied().unwrap_or(0);
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
pub fn full_measure_rest(time: &str) -> Result<String> {
    let (n, d) = time_signature(time)?;
    let mut remaining = (3840.0 * n as f64 / d as f64).round_ties_even() as i64;
    let durations = [
        (3840, "w"),
        (2880, "h."),
        (1920, "h"),
        (1440, "q."),
        (960, "q"),
        (720, "e."),
        (480, "e"),
        (360, "s."),
        (240, "s"),
        (180, "t."),
        (120, "t"),
        (60, "f"),
        (40, "f[3:2]"),
        (30, "d128"),
        (20, "d128[3:2]"),
    ];
    let mut start = 0;
    let mut events = vec![];
    while remaining > 0 {
        let (ticks, token) = durations
            .iter()
            .find(|(t, _)| *t <= remaining && remaining - *t != 10)
            .ok_or("Rest duration cannot be represented exactly")?;
        events.push(format!("@{start}:{token}:r"));
        start += ticks;
        remaining -= ticks;
    }
    Ok(format!("M2 | V0{{{}}}", events.join(" ")))
}

fn alternatives(values: impl IntoIterator<Item = String>) -> String {
    values
        .into_iter()
        .map(|v| serde_json::to_string(&v).unwrap())
        .collect::<Vec<_>>()
        .join(" | ")
}
fn onset_rule(ticks: Option<i64>) -> String {
    let Some(ticks) = ticks else {
        return "integer".into();
    };
    let limit = (ticks - 1).max(0).to_string();
    let mut choices = vec!["\"0\"".into()];
    for size in 1..limit.len() {
        choices.push(format!(
            "[1-9]{}",
            if size > 1 {
                format!(" [0-9]{{{}}}", size - 1)
            } else {
                String::new()
            }
        ));
    }
    for (index, digit) in limit.bytes().enumerate() {
        let low = if index == 0 { 1 } else { 0 };
        let high = i32::from(digit - b'0') - 1;
        if low <= high {
            let prefix = if index > 0 {
                format!("{} ", serde_json::to_string(&limit[..index]).unwrap())
            } else {
                String::new()
            };
            let tail = limit.len() - index - 1;
            choices.push(format!(
                "{prefix}[{low}-{high}]{}",
                if tail > 0 {
                    format!(" [0-9]{{{tail}}}")
                } else {
                    String::new()
                }
            ));
        }
    }
    choices.push(serde_json::to_string(&limit).unwrap());
    choices.join(" | ")
}
pub fn measure_grammar(mode: &str, strings: usize, ticks: Option<i64>) -> Result<String> {
    let fields = match mode {
        "notation" => "pitch",
        "tab" => "position",
        "both" => "position pitch",
        _ => return Err("Unknown M2 display mode".into()),
    };
    let mut grammar = include_str!("../tests/fixtures/native_recognition/m2_template.gbnf")
        .replace("NOTE_FIELDS", fields);
    for (name, rule) in [
        ("onset", onset_rule(ticks)),
        ("voice-id", alternatives((0..16).map(|n| n.to_string()))),
        (
            "string-id",
            alternatives((1..=strings.clamp(1, 12)).map(|n| n.to_string())),
        ),
        ("fret", alternatives((0..37).map(|n| n.to_string()))),
        ("midi", alternatives((0..128).map(|n| n.to_string()))),
        ("tuplet", alternatives((1..33).map(|n| n.to_string()))),
        (
            "key-name",
            alternatives(MAJOR_KEYS.into_iter().chain(MINOR_KEYS).map(str::to_owned)),
        ),
    ] {
        grammar += &format!("{name} ::= {rule}\n");
    }
    Ok(grammar)
}
fn repair_truncated_text(target: &str) -> (String, bool) {
    let re = Regex::new(r"[<,]text:").unwrap();
    let Some(m) = re.find_iter(target).last() else {
        return (target.into(), false);
    };
    if target[m.start()..].contains('>') {
        return (target.into(), false);
    }
    let mut repaired = target[..m.start()].trim_end().to_owned();
    if target.as_bytes()[m.start()] == b',' {
        repaired.push('>');
    }
    let close = repaired
        .matches('{')
        .count()
        .saturating_sub(repaired.matches('}').count());
    repaired.push_str(&"}".repeat(close));
    (repaired, true)
}
/// Runs one independent crop, returning accepted diagnostics and a reviewable
/// rest placeholder only after hard musical validation fails all attempts.
/// Transport/runtime errors are propagated; they are never replaced by rests.
pub fn recognize_measure<G: Generator + ?Sized>(
    engine: &mut G,
    record: &Value,
    config: &MeasureConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<MeasureOutput> {
    config.validate()?;
    check(cancelled)?;
    let mut row = record.clone();
    row["visual_pitch"] = json!(config.visual_pitch);
    if !config.pitch_context && !row["pitch_context"].is_null() {
        row["pitch_needs_review"] = json!(true);
        row.as_object_mut()
            .ok_or("Invalid measure record")?
            .remove("pitch_context");
    }
    if !config.written_pitch
        && (row["mode"] == "notation" || (row["mode"] == "both" && config.visual_pitch))
    {
        return Err("Independent native prompts require a written-pitch model for notation; use matching model capabilities".into());
    }
    if !row["score_state"].is_object() {
        return Err("Read and merge printed signatures before measure inference".into());
    }
    let mode = text(&row, "mode", "").to_owned();
    let instrument = text(&row, "instrument", "guitar").to_owned();
    let tuning = integers(&row["tuning"])?;
    let time = text(&row["score_state"], "time", "4/4").to_owned();
    let (n, d) = time_signature(&time)?;
    let ticks = (!flag(&row, "state_needs_review")).then_some(3840 * n / d);
    let grammar = measure_grammar(&mode, tuning.len().max(1), ticks)?;
    let initial = measure_messages(&row, config.image_policy.as_ref())?;
    let mut messages = initial.clone();
    let mut budget = config.max_tokens;
    let mut diagnostics = vec![];
    let mut final_target = String::new();
    let mut written = String::new();
    let mut final_errors = vec![];
    let mut attempt_count = 0;
    let mut annotation_only: Option<String> = None;
    for attempt in 1..=config.maximum_attempts {
        check(cancelled)?;
        attempt_count = attempt;
        let constrained = config.constrained_decoding || (config.retry_constraints && attempt > 1);
        let response = engine.generate(
            messages.clone(),
            budget,
            None,
            constrained.then_some(grammar.as_str()),
        )?;
        check(cancelled)?;
        let raw = response.content.trim();
        let target = raw.find("M2").map(|i| &raw[i..]).unwrap_or(raw);
        let (mut model_target, truncated) = repair_truncated_text(target);
        let mut repairs = vec![];
        if truncated {
            repairs.push("drop_unterminated_text");
        }
        let mut annotation_error = None;
        if let Some(original) = annotation_only.as_deref() {
            match merge_chord_review(original, &model_target, &mode) {
                Ok(merged) => model_target = merged,
                Err(error) => {
                    model_target = original.to_owned();
                    annotation_error = Some(format!(
                        "Annotation review failed: {}",
                        error.chars().take(160).collect::<String>()
                    ));
                }
            }
            repairs.push("preserve_music_during_chord_review");
        }
        if let Ok((aligned, changed)) = resolve_measure_state(&model_target, &row) {
            model_target = aligned;
            if changed {
                repairs.push("align_printed_signatures");
            }
        }
        let mut target = model_target.clone();
        let mut conversions = vec![];
        if mode == "both" && !config.visual_pitch {
            if let Ok((derived, changed)) = resolve_fretted_pitches(&target, &tuning, None) {
                target = derived;
                if changed {
                    repairs.push("derive_pitch_from_fingering");
                }
            }
        }
        if (mode == "notation" || (mode == "both" && config.visual_pitch))
            && instrument != "drums"
            && row["pitch_context"].is_object()
        {
            match convert_pitch_target(&model_target, &row["pitch_context"], &mode) {
                Ok(converted) => target = converted,
                Err(e) => conversions.push(format!("pitch_conversion: {e}")),
            }
        }
        let (parsed, mut errors) = score::validate_measure_target(
            &target,
            &mode,
            if mode == "both" && config.visual_pitch {
                None
            } else {
                Some(&tuning)
            },
            Some(tuning.len()),
        );
        errors.extend(conversions);
        let structural = !errors.is_empty();
        if errors.is_empty() {
            let timing = scorelib::gp5_constraints::timing_errors(&target)?;
            if !timing.is_empty() {
                errors.push("Events overlap within a voice. Re-read durations and onsets; simultaneous notes belong in a chord or another voice".into());
            }
            if mode == "notation"
                && matches!(instrument.as_str(), "guitar" | "bass")
                && flag(&row, "tuning_explicit")
            {
                let options = list(&row, "fingering_tunings");
                let candidates = if options.is_empty() {
                    vec![tuning.clone()]
                } else {
                    options.iter().map(integers).collect::<Result<Vec<_>>>()?
                };
                let mut failures = vec![];
                for tuning in &candidates {
                    failures.push(scorelib::gp5_constraints::notation_fingering_errors(
                        parsed.as_ref().unwrap(),
                        tuning,
                    )?);
                }
                if failures.iter().all(|f| !f.is_empty()) {
                    errors.extend(
                        failures
                            .into_iter()
                            .min_by_key(Vec::len)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|e| format!("Check written pitches after transposition: {e}")),
                    );
                }
            }
        }
        let hit_limit = response.completion_tokens >= budget as u64;
        if truncated {
            errors.push("Text annotation was cut off. Re-read ALL musical events and chord names; keep free text brief".into());
        } else if hit_limit {
            errors.push("Generation reached the output limit; return the complete measure".into());
        }
        if errors.is_empty() {
            let chord_errors =
                chord_recognition_errors(parsed.as_ref().expect("validated measure"));
            if !chord_errors.is_empty() {
                annotation_only.get_or_insert_with(|| model_target.clone());
                errors.extend(chord_errors);
            }
        }
        if let Some(error) = annotation_error {
            errors.push(error);
        }
        let warnings = if !structural {
            rhythm_warnings(&target, &time)?
        } else {
            vec![]
        };
        diagnostics.push(json!({"measure_number":row["measure_number"],"mode":mode,"image":row["image"],"attempt":attempt,"raw":response.content,"target":target,"accepted":errors.is_empty(),"generated_token_count":response.completion_tokens,"token_budget":budget,"hit_token_limit":hit_limit,"constraint_errors":errors,"deterministic_repairs":repairs,"needs_review":!warnings.is_empty(),"fallback_reason":warnings,"constrained_decoding":constrained,"score_state":row["score_state"],"written_target":model_target,"tuning":tuning}));
        written = model_target.clone();
        final_target = target.clone();
        final_errors = errors.clone();
        if errors.is_empty() {
            break;
        }
        if attempt < config.maximum_attempts {
            messages = initial.clone();
            messages.as_array_mut().unwrap().extend([json!({"role":"assistant","content":[{"type":"text","text":model_target}]}),json!({"role":"user","content":[{"type":"text","text":format!("Correct the FIRST measure only. {}. Return one complete M2 fragment.",errors.iter().take(8).cloned().collect::<Vec<_>>().join("; ").chars().take(2048).collect::<String>())}]})]);
            if hit_limit && !truncated {
                budget = config.max_tokens_ceiling.min(budget.saturating_mul(2));
            }
            continue;
        }
        if structural {
            final_target = full_measure_rest(&time)?;
            let (_, fallback_errors) = score::validate_measure_target(
                &final_target,
                &mode,
                Some(&tuning),
                Some(tuning.len()),
            );
            if !fallback_errors.is_empty() {
                return Err(format!(
                    "Invalid failure placeholder: {}",
                    fallback_errors.join("; ")
                ));
            }
            diagnostics.push(json!({"measure_number":row["measure_number"],"mode":mode,"image":row["image"],"attempt":attempt+1,"raw":"","target":final_target,"accepted":true,"needs_review":true,"fallback_reason":errors,"deterministic_repairs":["fallback_full_measure_rest"],"token_budget":0,"generated_token_count":0}));
        } else {
            let (review_target, uncertain) = retain_uncertain_chords_as_text(&final_target, &mode)?;
            if !uncertain.is_empty() {
                final_target = review_target;
                written = retain_uncertain_chords_as_text(&written, &mode)?.0;
                repairs.push("retain_unconfirmed_chord_as_review_text");
            }
            let mut accepted = diagnostics.last().unwrap().clone();
            accepted["accepted"] = json!(true);
            accepted["needs_review"] = json!(true);
            accepted["fallback_reason"] = json!(errors);
            accepted["target"] = json!(final_target);
            accepted["written_target"] = json!(written);
            accepted["deterministic_repairs"] = json!(repairs);
            if !uncertain.is_empty() {
                accepted["uncertain_chords"] = json!(uncertain);
            }
            diagnostics.push(accepted);
        }
    }
    let warnings = rhythm_warnings(&final_target, &time)?;
    final_errors.extend(warnings);
    final_errors.sort();
    final_errors.dedup();
    row["target"] = json!(final_target);
    row["written_target"] = json!(written);
    row["recognition_attempts"] = json!(attempt_count);
    row["fallback_reason"] = json!(final_errors);
    row["needs_review"] = json!(
        !final_errors.is_empty()
            || flag(&row, "pitch_needs_review")
            || flag(&row, "state_needs_review")
    );
    row["timing_errors"] = json!(scorelib::gp5_constraints::timing_errors(text(
        &row, "target", ""
    ))?);
    if !list(&row, "timing_errors").is_empty() {
        row["needs_review"] = json!(true);
    }
    if let Some(obj) = row.as_object_mut() {
        obj.remove("previous_image_record");
        obj.remove("next_image_record");
    }
    Ok(MeasureOutput {
        record: row,
        diagnostics,
    })
}

fn note_with_pitch(note: &Value, tuning: &[i64]) -> Value {
    let mut n = note.clone();
    if n["pitch"].is_null() {
        if let (Some(s), Some(f)) = (n["string"].as_i64(), n["fret"].as_i64()) {
            if s >= 1 && s as usize <= tuning.len() {
                n["pitch"] = json!(tuning[s as usize - 1] + f);
            }
        }
    }
    n
}
pub fn resolve_ties(records: &mut [Value]) -> Result<()> {
    let mut last_by_part: HashMap<String, HashMap<i64, Vec<Value>>> = HashMap::new();
    let mut indices = HashMap::new();
    for row in records {
        let key = staff_key(row);
        let index = bar_index(row);
        if indices.get(&key).is_some_and(|last| index != last + 1) {
            last_by_part.remove(&key);
        }
        indices.insert(key.clone(), index);
        let last = last_by_part.entry(key).or_default();
        let mut m = score::parse_measure_target(text(row, "target", ""))?;
        let voices: HashSet<i64> = list(&m, "voices")
            .iter()
            .filter_map(|v| v["voice"].as_i64())
            .collect();
        last.retain(|id, _| voices.contains(id));
        let tuning = integers(&row["tuning"])?;
        let mut changed = false;
        for voice in m["voices"].as_array_mut().unwrap() {
            let id = voice["voice"].as_i64().unwrap();
            let previous = last.entry(id).or_default();
            for event in voice["events"].as_array_mut().unwrap() {
                if matches!(text(event, "status", ""), "rest" | "empty") {
                    previous.clear();
                    continue;
                }
                for note in event["notes"].as_array_mut().unwrap() {
                    if flag(row, "manually_edited")
                        || !list(note, "effects").iter().any(|e| e == "tie")
                    {
                        continue;
                    }
                    let candidates: Vec<&Value> = if !note["string"].is_null() {
                        let exact: Vec<_> = previous
                            .iter()
                            .filter(|n| n["string"] == note["string"])
                            .collect();
                        if !exact.is_empty() {
                            exact
                        } else {
                            let pitch = note_with_pitch(note, &tuning)["pitch"].clone();
                            previous
                                .iter()
                                .filter(|n| {
                                    n["string"].is_null() && !pitch.is_null() && n["pitch"] == pitch
                                })
                                .collect()
                        }
                    } else {
                        let exact: Vec<_> = previous
                            .iter()
                            .filter(|n| n["pitch"] == note["pitch"])
                            .collect();
                        if !exact.is_empty() {
                            exact
                        } else if let Some(p) = note["pitch"].as_i64() {
                            previous
                                .iter()
                                .filter(|n| n["pitch"].as_i64().is_some_and(|q| (q - p).abs() <= 2))
                                .collect()
                        } else {
                            vec![]
                        }
                    };
                    if candidates.len() == 1 {
                        for field in ["fret", "pitch"] {
                            if !note[field].is_null()
                                && !candidates[0][field].is_null()
                                && note[field] != candidates[0][field]
                            {
                                note[field] = candidates[0][field].clone();
                                changed = true;
                            }
                        }
                    }
                }
                *previous = list(event, "notes")
                    .iter()
                    .map(|n| note_with_pitch(n, &tuning))
                    .collect();
            }
        }
        if changed {
            row["target"] = json!(score::format_measure_target(
                &m,
                text(row, "mode", "notation"),
                true
            )?);
            row["boundary_resolved"] = json!(true);
        }
    }
    Ok(())
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
        if r["mode"] == "both" && flag(r, "visual_pitch") {
            parts
                .entry(text(r, "part_id", "part-1").into())
                .or_default()
                .push(i);
        }
    }
    for (part, indices) in parts {
        let mut e = TuningEvidence::default();
        for &i in &indices {
            let row = &records[i];
            if !list(row, "fallback_reason").is_empty() {
                continue;
            }
            let mut target = text(row, "target", "").to_owned();
            if let Some(written) = row["written_target"]
                .as_str()
                .filter(|_| !flag(row, "manually_edited"))
            {
                let Ok(t) = convert_pitch_target(written, &row["pitch_context"], "both") else {
                    continue;
                };
                target = t;
            }
            let measure = score::parse_measure_target(&target)?;
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
            if row["mode"] == "both" && !flag(row, "manually_edited") {
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
/// Final deterministic pass after all independent calls; per-measure checkpoints
/// should be persisted by the caller before this function is invoked.
pub fn finalize_records(records: &mut [Value], information: &mut Value) -> Result<()> {
    reconcile_score_tuning(records)?;
    resolve_ties(records)?;
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
        if Some(bar_index(row)) == opening {
            if let Some(tempo) = tempo {
                let mut m = score::parse_measure_target(text(row, "target", ""))?;
                m["tempo_quarter"] = json!(tempo);
                row["target"] = json!(score::format_measure_target(
                    &m,
                    text(row, "mode", "notation"),
                    true
                )?);
            }
        }
        row["timing_errors"] = json!(scorelib::gp5_constraints::timing_errors(text(
            row, "target", ""
        ))?);
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
    if let Some(first) = records.first() {
        information["tuning_used"] = first["tuning"].clone();
    }
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
fn diagram_effect(v: &Value) -> Result<String> {
    let v = normalize_diagram(v);
    if v.is_null() {
        return Err("Invalid chord diagram".into());
    }
    let scalar = |v: &Value| {
        v.as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| v.to_string())
    };
    let frets = list(&v, "frets")
        .iter()
        .map(scalar)
        .collect::<Vec<_>>()
        .join("/");
    let fingers = list(&v, "fingers")
        .iter()
        .map(|v| if v.is_null() { "-".into() } else { scalar(v) })
        .collect::<Vec<_>>()
        .join("/");
    let barres = list(&v, "barres")
        .iter()
        .map(|b| {
            b.as_array()
                .unwrap()
                .iter()
                .map(scalar)
                .collect::<Vec<_>>()
                .join("/")
        })
        .collect::<Vec<_>>()
        .join(";");
    Ok(format!(
        "diagram:{}:{frets}:{fingers}:{}",
        v["base_fret"],
        if barres.is_empty() { "-" } else { &barres }
    ))
}
pub fn attach_chord_annotations(records: &mut [Value], predictions: &[Value]) -> Result<()> {
    let annotations: Vec<&Value> = predictions
        .iter()
        .filter(|p| matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram"))
        .collect();
    for row in records {
        let part: Vec<&Value> = annotations
            .iter()
            .copied()
            .filter(|p| text(p, "part_id", "part-1") == text(row, "part_id", "part-1"))
            .collect();
        let b = bbox(row).unwrap_or([0.; 4]);
        let local: Vec<&Value> = part
            .iter()
            .copied()
            .filter(|p| {
                if p["page"] != row["page"] {
                    return false;
                }
                let a = bbox(p).unwrap_or([0.; 4]);
                b[0] - 8. <= a[0] + a[2] / 2.
                    && a[0] + a[2] / 2. <= b[0] + b[2] + 8.
                    && b[1] - 50.0f64.max(b[3] * 0.65) <= a[1] + a[3]
                    && a[1] + a[3] <= b[1] + b[3]
            })
            .collect();
        if !local.is_empty() {
            row["chord_annotations"] = json!(local);
        }
        let mut m = score::parse_measure_target(text(row, "target", ""))?;
        let mut changed = false;
        for voice in m["voices"].as_array_mut().unwrap() {
            for event in voice["events"].as_array_mut().unwrap() {
                let name = list(event, "effects")
                    .iter()
                    .filter_map(Value::as_str)
                    .find_map(|s| s.strip_prefix("chord:"))
                    .map(unquote);
                let Some(name) = name else {
                    continue;
                };
                if list(event, "effects")
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|s| s.starts_with("diagram:"))
                {
                    continue;
                }
                let matching = |values: &[&Value]| -> Vec<String> {
                    values
                        .iter()
                        .filter(|p| chord_key(text(&p["parsed"], "text", "")) == chord_key(&name))
                        .filter_map(|p| {
                            let d = normalize_diagram(&p["parsed"]["diagram"]);
                            if d.is_null()
                                || (!list(row, "tuning").is_empty()
                                    && list(&d, "frets").len() != list(row, "tuning").len())
                            {
                                None
                            } else {
                                diagram_effect(&d).ok()
                            }
                        })
                        .collect()
                };
                let candidates = matching(&local);
                let candidates = if candidates.is_empty() {
                    matching(&part)
                } else {
                    candidates
                };
                let diagrams: HashSet<String> = candidates.into_iter().collect();
                if diagrams.len() == 1 {
                    event["effects"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!(diagrams.into_iter().next().unwrap()));
                    changed = true;
                }
            }
        }
        if changed {
            row["target"] = json!(score::format_measure_target(
                &m,
                text(row, "mode", "notation"),
                true
            )?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::VecDeque;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/native_recognition/golden.json"
        ))
        .unwrap()
    }
    #[test]
    fn python_golden_image_policy_and_pixels() {
        let f = fixture();
        let policy = ImagePolicy::from_value(&f["policy"]).unwrap();
        for (name, bytes) in [
            (
                "notation",
                include_bytes!("../tests/fixtures/native_recognition/notation.png").as_slice(),
            ),
            (
                "tab",
                include_bytes!("../tests/fixtures/native_recognition/tab.png").as_slice(),
            ),
            (
                "tiny",
                include_bytes!("../tests/fixtures/native_recognition/tiny.png").as_slice(),
            ),
            (
                "page",
                include_bytes!("../tests/fixtures/native_recognition/page.png").as_slice(),
            ),
        ] {
            let image = image::load_from_memory(bytes).unwrap().to_rgb8();
            let output = normalize_score_image(&image, &policy).unwrap();
            assert_eq!(
                json!([output.width(), output.height()]),
                f[name]["size"],
                "{name}"
            );
            let digest = format!("{:x}", Sha256::digest(output.as_raw()));
            assert_eq!(digest, f[name]["pixels_sha256"].as_str().unwrap(), "{name}");
            let scale = staff_scale(&image);
            if f[name]["staff_scale"].is_null() {
                assert_eq!(scale, None);
            } else {
                let (gap, lines) = scale.unwrap();
                assert!((gap - f[name]["staff_scale"][0].as_f64().unwrap()).abs() < 1e-10);
                assert_eq!(lines, f[name]["staff_scale"][1].as_u64().unwrap() as usize);
            }
        }
    }
    #[test]
    fn python_golden_metadata_pitch_and_rest() {
        let f = fixture();
        for c in f["parses"].as_array().unwrap() {
            assert_eq!(
                parse_info_response(c["raw"].as_str().unwrap(), c["kind"].as_str().unwrap()),
                c["expected"]
            );
        }
        for c in f["conversions"].as_array().unwrap() {
            assert_eq!(
                convert_pitch_target(
                    c["target"].as_str().unwrap(),
                    &c["context"],
                    c["mode"].as_str().unwrap()
                )
                .unwrap(),
                c["expected"].as_str().unwrap()
            );
        }
        for c in f["rests"].as_array().unwrap() {
            assert_eq!(
                full_measure_rest(c["time"].as_str().unwrap()).unwrap(),
                c["expected"].as_str().unwrap()
            );
        }
        for k in f["keys"].as_array().unwrap() {
            assert_eq!(
                key_parts(k["name"].as_str().unwrap()).unwrap(),
                (k["fifths"].as_i64().unwrap(), k["minor"].as_bool().unwrap())
            );
        }
    }
    #[test]
    fn pitch_context_does_not_double_count_conventional_octave() {
        let rows = vec![
            json!({"measure_number":1,"page":1,"bbox":[100,100,100,40],"system_index":0,"mode":"notation"}),
        ];
        let predictions = vec![
            json!({"kind":"clef","page":1,"bbox":[80,100,20,40],"parsed":{"clef":"G2","clef_octave":-12}}),
        ];
        let rows = apply_pitch_regions(&rows, &predictions, "guitar", None, None).unwrap();
        assert_eq!(rows[0]["pitch_context"]["instrument_transpose"], 0);
        assert_eq!(rows[0]["pitch_context"]["clef_octave"], -12);
        let explicit = apply_pitch_regions(&rows, &predictions, "guitar", Some(-12), None).unwrap();
        assert_eq!(explicit[0]["pitch_context"]["instrument_transpose"], -12);
    }
    #[test]
    fn signatures_are_staff_scoped_with_shared_time() {
        let mut rows = vec![
            json!({"measure_number":1,"part_id":"a","staff_id":"s","bar_index":0}),
            json!({"measure_number":2,"part_id":"b","staff_id":"s","bar_index":0}),
            json!({"measure_number":3,"part_id":"a","staff_id":"s","bar_index":1}),
            json!({"measure_number":4,"part_id":"b","staff_id":"s","bar_index":1}),
        ];
        resolve_score_states(&mut rows,&json!({"1":{"time":"3/4","key":2},"2":{"time":null,"key":-3},"3":{"time":null,"key":null},"4":{"time":null,"key":null}})).unwrap();
        assert_eq!(rows[3]["score_state"], json!({"time":"3/4","key":-3}));
        assert_eq!(rows[2]["score_state"], json!({"time":"3/4","key":2}));
    }
    #[test]
    fn neighbours_stay_with_part_and_do_not_cross_gaps() {
        let mut r = vec![
            json!({"part_id":"a","bar_index":0,"image":"a0"}),
            json!({"part_id":"b","bar_index":0,"image":"b0"}),
            json!({"part_id":"a","bar_index":1,"image":"a1"}),
            json!({"part_id":"a","bar_index":3,"image":"a3"}),
        ];
        attach_neighbours(&mut r).unwrap();
        assert_eq!(r[0]["next_image"], "a1");
        assert_eq!(r[2]["next_image"], "a1");
        assert_eq!(r[1]["previous_image"], "b0");
    }
    struct Scripted {
        outputs: VecDeque<Result<llama::Generation>>,
        requests: Vec<(Value, usize, bool)>,
    }
    impl Generator for Scripted {
        fn generate(
            &mut self,
            m: Value,
            t: usize,
            _s: Option<Value>,
            grammar: Option<&str>,
        ) -> Result<llama::Generation> {
            self.requests.push((m, t, grammar.is_some()));
            self.outputs.pop_front().expect("unexpected inference")
        }
    }
    fn scripted(outputs: &[(&str, u64)]) -> Scripted {
        Scripted {
            outputs: outputs
                .iter()
                .map(|(s, n)| {
                    Ok(llama::Generation {
                        content: (*s).into(),
                        completion_tokens: *n,
                    })
                })
                .collect(),
            requests: vec![],
        }
    }
    fn crop_record(path: &Path) -> Value {
        RgbImage::from_pixel(80, 60, Rgb([255; 3]))
            .save(path)
            .unwrap();
        json!({"measure_number":1,"mode":"notation","instrument":"pitched","tuning":[],"image":path,"score_state":{"time":"4/4","key":0},"pitch_context":{"instrument_transpose":-2,"clef_octave":0,"capo":0},"tuning_source":"default"})
    }
    #[test]
    fn real_measure_orchestration_retries_and_preserves_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let row = crop_record(&tmp.path().join("crop.png"));
        let mut engine = scripted(&[
            ("M2 | V0{@0:q:p999}", 2048),
            ("M2 time=3/4 key=GMajor | V0{@0:q:p62}", 24),
        ]);
        let out =
            recognize_measure(&mut engine, &row, &MeasureConfig::default(), &|| false).unwrap();
        assert_eq!(
            out.record["target"],
            "M2 time=4/4 key=BMajorFlat | V0{@0:q:p60}"
        );
        assert_eq!(out.record["recognition_attempts"], 2);
        assert_eq!(
            engine.requests[0].0[0]["content"].as_array().unwrap().len(),
            4
        );
        assert_eq!(engine.requests[1].1, 4096);
        assert!(engine.requests[1].2);
        assert!(!flag(&out.record, "needs_review"));
        assert!(out.diagnostics.last().unwrap()["accepted"]
            .as_bool()
            .unwrap());
    }
    #[test]
    fn invalid_output_is_reviewed_rest_but_transport_errors_propagate() {
        let tmp = tempfile::tempdir().unwrap();
        let row = crop_record(&tmp.path().join("crop.png"));
        let config = MeasureConfig {
            maximum_attempts: 1,
            ..MeasureConfig::default()
        };
        let mut engine = scripted(&[("garbage", 12)]);
        let out = recognize_measure(&mut engine, &row, &config, &|| false).unwrap();
        assert_eq!(out.record["target"], "M2 | V0{@0:w:r}");
        assert!(flag(&out.record, "needs_review"));
        let mut engine = Scripted {
            outputs: VecDeque::from([Err("native engine stopped".into())]),
            requests: vec![],
        };
        assert_eq!(
            recognize_measure(&mut engine, &row, &config, &|| false)
                .err()
                .unwrap(),
            "native engine stopped"
        );
        let mut engine = scripted(&[]);
        assert_eq!(
            recognize_measure(&mut engine, &row, &config, &|| true)
                .err()
                .unwrap(),
            CANCELLED
        );
        assert!(engine.requests.is_empty());
    }
    #[test]
    fn incorrect_string_count_does_not_discard_header_information() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("staff.png");
        for lines in [6, 2] {
            let mut image = RgbImage::from_pixel(500, 200, Rgb([255; 3]));
            for line in 0..lines {
                for x in 20..480 {
                    image.put_pixel(x, 100 + line * 12, Rgb([20; 3]));
                }
            }
            image.save(&path).unwrap();
            let layout = json!({"schema_version":"1.0","stage":"layout","mode":"tab","records":[
                {"measure_number":1,"page":1,"bbox":[20,80,460,120],"image":path,"source_page":path,"mode":"tab"}
            ],"regions":[
                {"kind":"header","page":1,"bbox":[0,0,500,80],"image":path},
                {"kind":"annotation","page":1,"bbox":[300,20,150,30],"image":path},
                {"kind":"tempo","page":1,"bbox":[20,60,60,20],"image":path}
            ]});
            let mut engine = scripted(&[
                (r#"{"title":"示例曲","artist":null,"tuning_name":null}"#, 20),
                (
                    r#"{"kind":"credit","text":"示例工作室","semitones":null,"capo":null}"#,
                    20,
                ),
                (r#"{"tempo_quarter":96}"#, 8),
                (
                    r#"{"instrument":"guitar","string_count":2,"name":null}"#,
                    20,
                ),
            ]);
            let config = InformationConfig {
                score_structure: false,
                pitch_context: true,
                ..Default::default()
            };
            let out = recognize_information(&mut engine, &layout, &config, &|| false).unwrap();
            assert_eq!(out.manifest["title"], "示例曲");
            assert_eq!(out.manifest["artist"], "示例工作室");
            assert_eq!(out.manifest["document_metadata"]["tempo_quarter"], 96);
            if lines == 6 {
                assert_eq!(out.manifest["tuning_used"], json!([64, 59, 55, 50, 45, 40]));
                assert_eq!(
                    out.manifest["document_metadata"]["string_count_source"],
                    "staff_lines"
                );
                assert!(prepare_records(&layout, &out.manifest).is_ok());
            } else {
                assert_eq!(out.manifest["tuning_source"], "unresolved");
                assert_eq!(out.manifest["tuning_used"], json!([]));
                assert!(out.manifest["document_metadata"]["tuning_issue"].is_string());
                assert!(prepare_records(&layout, &out.manifest)
                    .unwrap_err()
                    .contains("校对谱面信息"));
            }
        }
    }
    #[test]
    fn metadata_generates_real_requests_and_schema_one_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("crop.png");
        let mut row = crop_record(&path);
        row["page"] = json!(1);
        row["bbox"] = json!([0, 0, 80, 60]);
        row["system_index"] = json!(0);
        row["part_id"] = json!("part-1");
        let layout = json!({"schema_version":"1.0","stage":"layout","info_source":"image","mode":"notation","records":[row],"regions":[{"kind":"header","image":path,"page":1,"bbox":[0,0,80,60]}],"parts":[{"id":"part-1","name":"Piano","instrument":"pitched","program":0,"strings":null}]});
        let mut engine = scripted(&[(
            r#"{"title":"Example","artist":"Artist","tuning_name":null}"#,
            15,
        )]);
        let config = InformationConfig {
            score_structure: false,
            staff_profile: false,
            layout_path: tmp.path().join("layout.json"),
            fallback_title: "fallback".into(),
            ..InformationConfig::default()
        };
        let out = recognize_information(&mut engine, &layout, &config, &|| false).unwrap();
        assert_eq!(out.manifest["schema_version"], "1.0");
        assert_eq!(out.manifest["title"], "Example");
        assert_eq!(out.manifest["parts"][0]["instrument"], "pitched");
        assert_eq!(out.manifest["measure_profiles"][0]["measure_number"], 1);
        assert_eq!(prepare_records(&layout, &out.manifest).unwrap().len(), 1);
    }
    #[test]
    fn finalization_keeps_parts_ties_and_tempo() {
        let mut rows = vec![
            json!({"measure_number":1,"part_id":"a","bar_index":0,"mode":"notation","instrument":"pitched","tuning":[],"target":"M2 | V0{@0:q:p61}","tuning_source":"default"}),
            json!({"measure_number":2,"part_id":"a","bar_index":1,"mode":"notation","instrument":"pitched","tuning":[],"target":"M2 | V0{@0:q:p60(tie)}","tuning_source":"default"}),
            json!({"measure_number":3,"part_id":"b","bar_index":0,"mode":"notation","instrument":"pitched","tuning":[],"target":"M2 | V0{@0:q:p70}","tuning_source":"default"}),
        ];
        let mut info = json!({"document_metadata":{"tempo_quarter":120},"parts":[]});
        finalize_records(&mut rows, &mut info).unwrap();
        assert!(rows[0]["target"].as_str().unwrap().contains("tempo=120"));
        assert!(rows[2]["target"].as_str().unwrap().contains("tempo=120"));
        assert!(rows[1]["target"].as_str().unwrap().contains("p61(tie)"));
        assert!(flag(&rows[1], "boundary_resolved"));
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    #[test]
    fn prompts_match_training_contract() {
        let f: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/native_recognition/golden.json"
        ))
        .unwrap();
        for case in f["prompts"].as_array().unwrap() {
            assert_eq!(
                state_prompt(&case["row"]).unwrap(),
                case["expected"].as_str().unwrap()
            );
        }
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(unquote("%♭"), "%♭");
    }
    #[test]
    fn printed_diagrams_enrich_only_matching_timed_chords() {
        let mut r = vec![
            json!({"measure_number":1,"part_id":"p","page":1,"bbox":[20,50,300,90],"mode":"tab","tuning":[64,59,55,50,45,40],"target":"M2 | V0{@0:q:s1f0<chord:C>}"}),
        ];
        let diagram = json!({"base_fret":1,"frets":["x",3,2,0,1,0],"fingers":[null,3,2,null,1,null],"barres":[]});
        let p = vec![
            json!({"part_id":"p","page":1,"bbox":[25,5,40,30],"parsed":{"kind":"chord_diagram","text":"C","diagram":diagram}}),
        ];
        attach_chord_annotations(&mut r, &p).unwrap();
        assert!(r[0]["target"]
            .as_str()
            .unwrap()
            .contains("diagram:1:x/3/2/0/1/0:-/3/2/-/1/-:-"));
        assert!(r[0]["chord_annotations"].is_array());
    }
    #[test]
    fn repeated_visual_evidence_recovers_one_detuned_string() {
        let mut rows = vec![];
        for n in 0..3 {
            rows.push(json!({"measure_number":n+1,"part_id":"p","bar_index":n,"mode":"both","instrument":"guitar","visual_pitch":true,"tuning":[64,59,55,50,45,40],"tuning_source":"default","target":"M2 | V0{@0:q:s6f0p38 @960:q:s6f2p40}","fallback_reason":[]}));
        }
        reconcile_score_tuning(&mut rows).unwrap();
        assert_eq!(rows[0]["tuning"][5], 38);
        assert_eq!(rows[2]["tuning_source"], "notation_tab_consensus");
        assert!(!flag(&rows[0], "needs_review"));
    }
    #[test]
    fn structure_vlm_request_reaches_native_geometry_mapping() {
        struct Model {
            calls: usize,
        }
        impl Generator for Model {
            fn generate(
                &mut self,
                messages: Value,
                _tokens: usize,
                schema: Option<Value>,
                _: Option<&str>,
            ) -> Result<llama::Generation> {
                self.calls += 1;
                assert!(messages[0]["content"][0]["image_url"]["url"]
                    .as_str()
                    .unwrap()
                    .starts_with("data:image/png;base64,"));
                assert!(schema.is_some());
                Ok(llama::Generation{content:r#"{"parts":[{"name":"Piano","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0]]}"#.into(),completion_tokens:40})
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("page.png");
        RgbImage::from_pixel(500, 300, Rgb([255; 3]))
            .save(&path)
            .unwrap();
        let layout = json!({"schema_version":"1.0","stage":"layout","mode":"notation","info_source":"image","records":[{"measure_number":1,"page":1,"source_page":path,"image":path,"mode":"notation","bbox":[20,100,400,70],"system_index":0,"system_measure_index":0}],"regions":[]});
        let mut engine = Model { calls: 0 };
        let config = InformationConfig {
            staff_profile: false,
            layout_path: tmp.path().join("layout.json"),
            structure_output: Some(tmp.path().join("info")),
            ..InformationConfig::default()
        };
        let out = recognize_information(&mut engine, &layout, &config, &|| false).unwrap();
        assert_eq!(engine.calls, 1);
        assert_eq!(out.manifest["parts"][0]["instrument"], "pitched");
        assert_eq!(out.manifest["resolved_records"][0]["part_id"], "part-1");
        assert_eq!(out.structure_predictions.len(), 1);
    }
}

/// Recover opening text above the measure, with a detected clef as an anchor.
/// High notes and slurs can extend above the clef but remain inside the measure.
pub fn opening_annotations(records: &[Value], regions: &[Value]) -> Result<Vec<Value>> {
    let mut first = BTreeMap::new();
    for row in records {
        first
            .entry(text(row, "part_id", "part-1").to_owned())
            .or_insert(row);
    }
    let mut result = vec![];
    for (part, row) in first {
        let Some(path) = row["source_page"].as_str() else {
            continue;
        };
        let b = bbox(row)?;
        let clefs: Vec<[f64; 4]> = regions
            .iter()
            .filter(|r| r["kind"] == "clef" && r["page"] == row["page"])
            .filter_map(|r| bbox(r).ok())
            .filter(|c| {
                b[0] - 20. <= c[0]
                    && c[0] <= b[0] + 120.0f64.min(b[2] / 3.)
                    && b[1] <= c[1]
                    && c[1] < b[1] + b[3]
            })
            .collect();
        if clefs.is_empty() {
            continue;
        }
        let bottom = (clefs
            .iter()
            .map(|c| c[1])
            .fold(f64::INFINITY, f64::min)
            - 4.)
            .min(b[1])
            .trunc();
        let mut top = (b[1].trunc() - 80.).max(0.);
        for prev in records.iter().filter(|r| r["page"] == row["page"]) {
            if let Ok(p) = bbox(prev) {
                if p[1] + p[3] < b[1] {
                    top = top.max((p[1] + p[3]).trunc() + 8.);
                }
            }
        }
        if bottom - top < 8. {
            continue;
        }
        let image = image_boundary::open_rgb(Path::new(path))?;
        let left = b[0].trunc().max(0.);
        let right = (b[0] + b[2].max(f64::from(image.width()) * 0.4))
            .trunc()
            .min(f64::from(image.width()));
        if right <= left {
            continue;
        }
        let crop = direct_crop(&image, [left, top, right - left, bottom - top])?;
        let (w, h) = crop.dimensions();
        let ink: Vec<bool> = crop.pixels().map(|p| gray(p) < 200).collect();
        let rows: Vec<u32> = (0..h)
            .filter(|&y| (0..w).filter(|&x| ink[(y * w + x) as usize]).count() >= 3)
            .collect();
        let mut bands: Vec<Vec<u32>> = vec![];
        for y in rows {
            if bands.last().is_some_and(|r| y - r.last().unwrap() <= 3) {
                bands.last_mut().unwrap().push(y);
            } else {
                bands.push(vec![y]);
            }
        }
        for band in bands {
            let bh = band.last().unwrap() - band[0] + 1;
            if band.len() < 6 || bh > 65 {
                continue;
            }
            let columns: Vec<u32> = (0..w)
                .filter(|&x| (band[0]..=band[band.len() - 1]).any(|y| ink[(y * w + x) as usize]))
                .collect();
            let gap = 24.0f64.max(f64::from(bh) * 1.5);
            let mut words: Vec<Vec<u32>> = vec![];
            for x in columns {
                if words
                    .last()
                    .is_some_and(|r| f64::from(x - r.last().unwrap()) <= gap)
                {
                    words.last_mut().unwrap().push(x);
                } else {
                    words.push(vec![x]);
                }
            }
            for word in words {
                let bw = word.last().unwrap() - word[0] + 1;
                if word.len() < 3 || f64::from(bw) < f64::from(bh) * 1.5 {
                    continue;
                }
                let pixels = (band[0]..=band[band.len() - 1])
                    .map(|y| word.iter().filter(|&&x| ink[(y * w + x) as usize]).count())
                    .sum::<usize>();
                if pixels < 20 {
                    continue;
                }
                let a = [
                    left + f64::from(word[0]),
                    top + f64::from(band[0]),
                    f64::from(bw),
                    f64::from(bh),
                ];
                let covered = regions
                    .iter()
                    .filter(|r| {
                        r["page"] == row["page"]
                            && matches!(
                                text(r, "kind", ""),
                                "annotation"
                                    | "transposition"
                                    | "tempo"
                                    | "title"
                                    | "subtitle"
                                    | "credit"
                                    | "tuning"
                                    | "header_text"
                            )
                    })
                    .filter_map(|r| bbox(r).ok())
                    .any(|r| {
                        ((a[0] + a[2]).min(r[0] + r[2]) - a[0].max(r[0])).max(0.)
                            * ((a[1] + a[3]).min(r[1] + r[3]) - a[1].max(r[1])).max(0.)
                            / (a[2] * a[3])
                            > 0.7
                    });
                if !covered {
                    let x0 = (a[0] - 6.).max(0.);
                    let y0 = (a[1] - 6.).max(0.);
                    let x1 = (a[0] + a[2] + 6.).min(f64::from(image.width()));
                    let y1 = (a[1] + a[3] + 6.).min(f64::from(image.height()));
                    result.push(json!({"kind":"annotation","page":row["page"],"bbox":a,"crop_bbox":[x0,y0,x1-x0,y1-y0],"part_id":part,"source_page":path,"source":"opening_text_line"}));
                }
            }
        }
    }
    Ok(result)
}

/// Preserve accepted/manual OCR records after compatible metadata edits.
/// Returning None means pitches must be recognized again; no source is mutated.
pub fn update_information(source: &Value, information: &Value) -> Result<Option<Value>> {
    fn part_contexts(value: &Value) -> Result<BTreeMap<String, Value>> {
        let mut result = BTreeMap::new();
        for part in list(value, "parts") {
            let id = part["id"].as_str().ok_or("Part is missing its id")?;
            if result.insert(id.to_owned(),json!({"instrument":part["instrument"],"tuning_used":part["tuning_used"],"transpose":part["transpose"]})).is_some(){return Err("Duplicate part id".into());}
        }
        Ok(result)
    }
    fn pitch_contexts(value: &Value) -> Result<BTreeMap<i64, Value>> {
        let mut result = BTreeMap::new();
        for row in list(value, "measure_pitch_contexts") {
            let number = row["measure_number"]
                .as_i64()
                .ok_or("Pitch context is missing its measure number")?;
            if result.insert(number, row.clone()).is_some() {
                return Err("Duplicate pitch context measure".into());
            }
        }
        Ok(result)
    }
    if part_contexts(source)? != part_contexts(information)?
        || source["tuning_used"] != information["tuning_used"]
        || text(source, "instrument", "guitar") != text(information, "instrument", "guitar")
        || source["transpose"] != information["transpose"]
        || pitch_contexts(source)? != pitch_contexts(information)?
    {
        return Ok(None);
    }
    let mut result = source.clone();
    let previous_tempo = source["document_metadata"]["tempo_quarter"].clone();
    let global_capo_changed = list(information, "measure_profiles").is_empty()
        && source.get("capo").unwrap_or(&Value::from(0))
            != information.get("capo").unwrap_or(&Value::from(0))
        && matches!(text(source, "instrument", "guitar"), "guitar" | "bass");
    let fallback_mode = text(source, "mode", "notation").to_owned();
    if global_capo_changed {
        for row in result["records"]
            .as_array_mut()
            .ok_or("Recognition is missing records")?
        {
            if text(row, "mode", &fallback_mode) != "tab" {
                row["pitch_needs_review"] = json!(true);
                row["needs_review"] = json!(true);
            }
        }
    }
    for field in [
        "title",
        "artist",
        "tuning_used",
        "capo",
        "document_metadata",
    ] {
        result[field] = information
            .get(field)
            .cloned()
            .ok_or_else(|| format!("Document information is missing {field}"))?;
    }
    result["instrument"] = json!(text(information, "instrument", "guitar"));
    result["midi_program"] = information
        .get("midi_program")
        .cloned()
        .unwrap_or(json!(25));
    result["parts"] = json!(list(information, "parts"));
    let profiles: BTreeMap<i64, &Value> = list(information, "measure_profiles")
        .iter()
        .map(|p| {
            p["measure_number"]
                .as_i64()
                .map(|n| (n, p))
                .ok_or("Profile is missing measure number".to_owned())
        })
        .collect::<Result<_>>()?;
    let records = result["records"]
        .as_array_mut()
        .ok_or("Recognition is missing records")?;
    for row in records.iter_mut() {
        let Some(number) = row["measure_number"].as_i64() else {
            return Err("Recognition record is missing measure number".into());
        };
        let Some(profile) = profiles.get(&number) else {
            continue;
        };
        if row["capo"].as_i64().unwrap_or(0) != profile["capo"].as_i64().unwrap_or(0)
            && row["mode"] != "tab"
        {
            row["pitch_needs_review"] = json!(true);
            row["needs_review"] = json!(true);
        }
        for key in [
            "part_name",
            "instrument",
            "midi_program",
            "tuning",
            "capo",
            "tuning_explicit",
            "fingering_tunings",
        ] {
            if let Some(v) = profile.get(key) {
                row[key] = v.clone();
            }
        }
        if row["pitch_context"].is_object() {
            row["pitch_context"]["capo"] = profile["capo"].clone();
        }
    }
    let tempo = information["document_metadata"]["tempo_quarter"]
        .as_i64()
        .filter(|n| *n != 0);
    if let Some(tempo) =
        tempo.filter(|_| information["document_metadata"]["tempo_quarter"] != previous_tempo)
    {
        if let Some(first) = records.first() {
            let opening = first["bar_index"].as_i64().unwrap_or(0);
            for (index, row) in records.iter_mut().enumerate() {
                if row["bar_index"].as_i64().unwrap_or(index as i64) != opening {
                    continue;
                }
                let mut measure = score::parse_measure_target(text(row, "target", ""))?;
                measure["tempo_quarter"] = json!(tempo);
                row["target"] = json!(score::format_measure_target(
                    &measure,
                    text(row, "mode", &fallback_mode),
                    true
                )?);
            }
        }
    }
    Ok(Some(result))
}

#[cfg(test)]
mod metadata_update_tests {
    use super::*;
    fn values() -> (Value, Value) {
        let source = json!({"mode":"notation","instrument":"guitar","midi_program":25,"transpose":null,"tuning_used":[64,59,55,50,45,40],"capo":0,"title":"Old","artist":"Artist","document_metadata":{"tempo_quarter":100},"parts":[{"id":"a","instrument":"guitar","tuning_used":[64,59,55,50,45,40],"transpose":null},{"id":"b","instrument":"guitar","tuning_used":[64,59,55,50,45,40],"transpose":null}],"measure_pitch_contexts":[],"records":[{"measure_number":1,"part_id":"a","bar_index":0,"mode":"notation","target":"M2 tempo=100 | V0{@0:q:p60}","manually_edited":true,"reviewed":true},{"measure_number":2,"part_id":"a","bar_index":1,"mode":"notation","target":"M2 | V0{@0:q:p61}"},{"measure_number":3,"part_id":"b","bar_index":0,"mode":"notation","target":"M2 tempo=100 | V0{@0:q:p62}"}]});
        let mut info = source.clone();
        info.as_object_mut().unwrap().remove("records");
        (source, info)
    }
    #[test]
    fn compatible_title_and_tempo_keep_edits_and_update_every_opening_part() {
        let (source, mut info) = values();
        info["title"] = json!("New");
        info["document_metadata"]["tempo_quarter"] = json!(132);
        let updated = update_information(&source, &info).unwrap().unwrap();
        assert_eq!(updated["title"], "New");
        assert!(flag(&updated["records"][0], "manually_edited"));
        assert!(flag(&updated["records"][0], "reviewed"));
        assert!(updated["records"][0]["target"]
            .as_str()
            .unwrap()
            .contains("tempo=132"));
        assert!(updated["records"][2]["target"]
            .as_str()
            .unwrap()
            .contains("tempo=132"));
        assert_eq!(
            updated["records"][1]["target"],
            source["records"][1]["target"]
        );
        assert_eq!(source["title"], "Old");
    }
    #[test]
    fn incompatible_pitch_identity_invalidates_without_mutation() {
        let (source, mut info) = values();
        info["transpose"] = json!(-2);
        assert!(update_information(&source, &info).unwrap().is_none());
        let (source, mut info) = values();
        info["parts"][1]["tuning_used"] = json!([43, 38, 33, 28]);
        assert!(update_information(&source, &info).unwrap().is_none());
    }
    #[test]
    fn capo_review_is_scoped_and_profile_capo_updates_context() {
        let (mut source, mut info) = values();
        source["records"][1]["mode"] = json!("tab");
        info["capo"] = json!(2);
        let updated = update_information(&source, &info).unwrap().unwrap();
        assert!(flag(&updated["records"][0], "pitch_needs_review"));
        assert!(!flag(&updated["records"][1], "pitch_needs_review"));
        source["records"][0]["pitch_context"] = json!({"capo":0});
        info["measure_profiles"] = json!([{"measure_number":1,"capo":3,"part_name":"Guitar II"}]);
        let updated = update_information(&source, &info).unwrap().unwrap();
        assert_eq!(updated["records"][0]["pitch_context"]["capo"], 3);
        assert_eq!(updated["records"][0]["part_name"], "Guitar II");
        assert!(!flag(&updated["records"][2], "pitch_needs_review"));
    }
}

#[cfg(test)]
mod opening_tests {
    use super::*;
    #[test]
    fn opening_text_recovery_requires_a_clef_and_avoids_duplicate_boxes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("page.png");
        let mut image = RgbImage::from_pixel(500, 300, Rgb([255; 3]));
        for y in 50..60 {
            for x in 60..150 {
                if x % 10 < 6 {
                    image.put_pixel(x, y, Rgb([0; 3]));
                }
            }
        }
        image.save(&path).unwrap();
        let records =
            vec![json!({"part_id":"part-1","page":1,"bbox":[50,120,400,80],"source_page":path})];
        assert!(opening_annotations(&records, &[]).unwrap().is_empty());
        let mut regions = vec![json!({"kind":"clef","page":1,"bbox":[50,135,20,40]})];
        let recovered = opening_annotations(&records, &regions).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0]["bbox"], json!([60., 50., 86., 10.]));
        assert_eq!(region_image(&recovered[0]).unwrap().dimensions(), (98, 22));
        regions.push(json!({"kind":"annotation","page":1,"bbox":[60,50,86,10]}));
        assert!(opening_annotations(&records, &regions).unwrap().is_empty());
    }
    #[test]
    fn invalid_diagram_fields_are_not_silently_filled() {
        for bad in [
            json!({"frets":[0,0,0],"base_fret":false}),
            json!({"frets":[0,0,0],"fingers":3}),
            json!({"frets":[0,0,0],"barres":null}),
        ] {
            assert!(normalize_diagram(&bad).is_null());
        }
    }
}

fn annotation_effect(effect: &str) -> bool {
    ["chord:", "diagram:", "text:"]
        .iter()
        .any(|prefix| effect.starts_with(prefix))
}
fn event_chord_name(event: &Value) -> Option<String> {
    list(event, "effects")
        .iter()
        .filter_map(Value::as_str)
        .find_map(|effect| effect.strip_prefix("chord:"))
        .map(unquote)
}
fn quote_annotation(value: &str) -> String {
    let mut result = String::new();
    for byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric() || b"-_.~".contains(byte) {
            result.push(*byte as char);
        } else {
            result.push_str(&format!("%{byte:02X}"));
        }
    }
    result
}
/// OCR confidence checks only. Custom/manual chord names remain legal score IR.
pub fn chord_recognition_errors(measure: &Value) -> Vec<String> {
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
    let mut errors = Vec::new();
    for voice in list(measure, "voices") {
        for event in list(voice, "events") {
            let Some(name) = event_chord_name(event) else {
                continue;
            };
            let normalized = chord_key(&name.nfkc().collect::<String>())
                .replace("major", "maj")
                .replace("minor", "min");
            if ["nc", "n.c", "n.c.", "nochord"].contains(&normalized.to_lowercase().as_str()) {
                continue;
            }
            let pieces: Vec<_> = modifier.find_iter(&normalized).collect();
            let repeated = pieces
                .windows(2)
                .any(|p| p[0].end() == p[1].start() && p[0].as_str() == p[1].as_str());
            if !pattern.is_match(&normalized) || repeated {
                errors.push(format!("Uncertain chord symbol {:?}: re-read its visible name and onset; titles, section labels, instrument names and technique instructions belong in text, not chord. Keep all musical events and do not invent a replacement chord", name.chars().take(64).collect::<String>()));
            }
        }
    }
    errors
}

/// Splice only annotation suffixes into validated original event tokens. In
/// particular, never route retained music through the M2 formatter here.
/// A repeated '^' payload is expanded from the original lexical payload only
/// when otherwise an edited previous annotation would accidentally propagate.
fn patch_event_annotations(
    original: &str,
    changes: &BTreeMap<(i64, i64), Vec<String>>,
    section: Option<&str>,
) -> Result<String> {
    let measure = score::parse_measure_target(original)?;
    let mut originals: BTreeMap<(i64, i64), &Value> = BTreeMap::new();
    for voice in list(&measure, "voices") {
        for event in list(voice, "events") {
            let key = (
                voice["voice"].as_i64().ok_or("Missing voice id")?,
                event["start"].as_i64().ok_or("Missing event onset")?,
            );
            if originals.insert(key, event).is_some() {
                return Err("Ambiguous original voice/onset during chord review".into());
            }
        }
    }
    let delimiter = original.find('|').ok_or("Missing M2 voice separator")?;
    let whitespace = Regex::new(r"\S+").unwrap();
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let body = &original[delimiter + 1..];
    let mut offset = delimiter + 1;
    for part in body.split("||") {
        let open = part.find('{').ok_or("Missing voice opening brace")?;
        let close = part.rfind('}').ok_or("Missing voice closing brace")?;
        let voice = part[..open]
            .trim()
            .strip_prefix('V')
            .ok_or("Invalid voice prefix")?
            .parse::<i64>()
            .map_err(|_| "Invalid voice id")?;
        let mut previous_original = String::new();
        let mut previous_output = String::new();
        for token in whitespace.find_iter(&part[open + 1..close]) {
            let raw = token.as_str();
            let first = raw.find(':').ok_or("Missing event onset separator")?;
            let onset = raw[..first]
                .strip_prefix('@')
                .ok_or("Invalid event prefix")?
                .parse::<i64>()
                .map_err(|_| "Invalid event onset")?;
            let mut bracket = 0i32;
            let payload_offset = raw[first + 1..]
                .char_indices()
                .find_map(|(i, c)| {
                    match c {
                        '[' => bracket += 1,
                        ']' => bracket -= 1,
                        ':' if bracket == 0 => return Some(first + 1 + i + 1),
                        _ => {}
                    }
                    None
                })
                .ok_or("Missing payload separator")?;
            let payload = &raw[payload_offset..];
            let effective = if payload == "^" {
                if previous_original.is_empty() {
                    return Err("Repeated payload has no predecessor".into());
                }
                previous_original.clone()
            } else {
                payload.to_owned()
            };
            let original_event = originals
                .get(&(voice, onset))
                .ok_or("Original event identity was not retained")?;
            let current_effects: Vec<String> = list(original_event, "effects")
                .iter()
                .map(|v| {
                    v.as_str()
                        .ok_or("Invalid original effect")
                        .map(str::to_owned)
                })
                .collect::<std::result::Result<_, _>>()?;
            let desired_effects = changes.get(&(voice, onset)).unwrap_or(&current_effects);
            let desired = if desired_effects == &current_effects {
                effective.clone()
            } else {
                let notes = if effective.ends_with('>') {
                    effective
                        .rsplit_once('<')
                        .ok_or("Invalid original annotation suffix")?
                        .0
                } else {
                    &effective
                };
                format!(
                    "{notes}{}",
                    if desired_effects.is_empty() {
                        String::new()
                    } else {
                        format!("<{}>", desired_effects.join(","))
                    }
                )
            };
            let replacement = if payload == "^" && desired == previous_output {
                "^".to_owned()
            } else {
                desired.clone()
            };
            if replacement != payload {
                let start = offset + open + 1 + token.start() + payload_offset;
                edits.push((start, start + payload.len(), replacement));
            }
            previous_original = effective;
            previous_output = desired;
        }
        offset += part.len() + 2;
    }
    if let Some(section) = section.filter(|s| !s.is_empty()) {
        if measure["section"].as_str() != Some(section) {
            let value = format!("section={}", quote_annotation(section));
            if let Some(token) = whitespace
                .find_iter(&original[..delimiter])
                .find(|t| t.as_str().starts_with("section="))
            {
                edits.push((token.start(), token.end(), value));
            } else {
                let end = original[..delimiter].trim_end().len();
                edits.push((end, end, format!(" {value}")));
            }
        }
    }
    edits.sort_by_key(|e| e.0);
    let mut result = original.to_owned();
    for (start, end, replacement) in edits.into_iter().rev() {
        result.replace_range(start..end, &replacement);
    }
    // Reject a malformed annotation without falling back to regenerating music.
    score::parse_measure_target(&result)?;
    Ok(result)
}

/// Apply only chord/diagram/text and section changes from the reread. Every
/// musical event, note, onset, duration, velocity and technique comes from the
/// original, including musical fields in candidate rows that happen to match.
pub fn merge_chord_review(original: &str, reread: &str, mode: &str) -> Result<String> {
    let measure = score::parse_measure_target(original)?;
    let candidate = score::parse_measure_target(reread)?;
    let mut positions = BTreeMap::new();
    for voice in list(&measure, "voices") {
        for event in list(voice, "events") {
            let key = (
                voice["voice"].as_i64().ok_or("Missing voice id")?,
                event["start"].as_i64().ok_or("Missing event onset")?,
            );
            if positions.insert(key, event).is_some() {
                return Err("Ambiguous original event onset".into());
            }
        }
    }
    let mut changes = BTreeMap::new();
    let mut seen = HashSet::new();
    for voice in list(&candidate, "voices") {
        for event in list(voice, "events") {
            let at = (
                voice["voice"]
                    .as_i64()
                    .ok_or("Missing candidate voice id")?,
                event["start"]
                    .as_i64()
                    .ok_or("Missing candidate event onset")?,
            );
            if !seen.insert(at) {
                return Err(
                    "The reread repeats a voice/onset; annotation assignment is ambiguous".into(),
                );
            }
            if event_chord_name(event).is_some_and(|s| !s.is_empty())
                && !positions.contains_key(&at)
            {
                return Err(
                    "The reread chord onset does not match an existing musical event".into(),
                );
            }
            let Some(original_event) = positions.get(&at) else {
                continue;
            };
            let mut effects: Vec<String> = list(original_event, "effects")
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| !annotation_effect(s))
                .map(str::to_owned)
                .collect();
            let previous_annotations: Vec<&str> = list(original_event, "effects")
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| annotation_effect(s))
                .collect();
            let next_annotations: Vec<&str> = list(event, "effects")
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| annotation_effect(s))
                .collect();
            if previous_annotations == next_annotations {
                continue;
            }
            effects.extend(next_annotations.into_iter().map(str::to_owned));
            changes.insert(at, effects);
        }
    }
    let result = patch_event_annotations(original, &changes, candidate["section"].as_str())?;
    let (_, before_errors) = score::validate_measure_target(original, mode, None, Some(12));
    let (_, after_errors) = score::validate_measure_target(&result, mode, None, Some(12));
    let added: Vec<_> = after_errors
        .into_iter()
        .filter(|error| !before_errors.contains(error))
        .collect();
    if !added.is_empty() {
        return Err(format!("Invalid reread annotation: {}", added.join("; ")));
    }
    Ok(result)
}

/// After visual retries, retain uncertain chord text as review evidence without
/// allowing the uncertainty to erase or regenerate an otherwise valid measure.
pub fn retain_uncertain_chords_as_text(target: &str, _mode: &str) -> Result<(String, Vec<Value>)> {
    let measure = score::parse_measure_target(target)?;
    let mut changes = BTreeMap::new();
    let mut uncertain = vec![];
    for voice in list(&measure, "voices") {
        for event in list(voice, "events") {
            if chord_recognition_errors(&json!({"voices":[{"events":[event]}]})).is_empty() {
                continue;
            }
            let name = event_chord_name(event).ok_or("Uncertain chord has no name")?;
            let voice = voice["voice"].as_i64().ok_or("Missing voice id")?;
            let start = event["start"].as_i64().ok_or("Missing event onset")?;
            uncertain.push(json!({"voice":voice,"start":start,"text":name}));
            let mut captions: Vec<String> = list(event, "effects")
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|s| s.strip_prefix("text:"))
                .map(unquote)
                .collect();
            captions.push(name);
            let mut seen = HashSet::new();
            captions.retain(|s| seen.insert(s.clone()));
            let mut effects: Vec<String> = list(event, "effects")
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.starts_with("chord:") && !s.starts_with("text:"))
                .map(str::to_owned)
                .collect();
            effects.push(format!("text:{}", quote_annotation(&captions.join("; "))));
            changes.insert((voice, start), effects);
        }
    }
    if uncertain.is_empty() {
        Ok((target.to_owned(), uncertain))
    } else {
        Ok((patch_event_annotations(target, &changes, None)?, uncertain))
    }
}

#[cfg(test)]
mod chord_review_tests {
    use super::*;
    use std::collections::VecDeque;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/native_recognition/chord_review.json"
        ))
        .unwrap()
    }
    fn annotation_order(mut m: Value) -> Value {
        for voice in m["voices"].as_array_mut().unwrap() {
            for event in voice["events"].as_array_mut().unwrap() {
                let effects = event["effects"].as_array_mut().unwrap();
                effects.sort_by_key(|effect| annotation_effect(effect.as_str().unwrap()));
            }
        }
        m
    }
    fn music_only(target: &str) -> Value {
        let mut m = score::parse_measure_target(target).unwrap();
        m.as_object_mut().unwrap().remove("section");
        for voice in m["voices"].as_array_mut().unwrap() {
            for event in voice["events"].as_array_mut().unwrap() {
                event["effects"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|e| !annotation_effect(e.as_str().unwrap()));
            }
        }
        m
    }
    #[test]
    fn python_oracle_chord_diffs_preserve_every_musical_field() {
        let f = fixture();
        for c in list(&f, "merges") {
            let original = text(c, "original", "");
            let merged =
                merge_chord_review(original, text(c, "candidate", ""), text(c, "mode", ""))
                    .unwrap();
            assert_eq!(
                annotation_order(score::parse_measure_target(&merged).unwrap()),
                annotation_order(c["oracle_parsed"].clone()),
                "{}",
                c["name"]
            );
            assert_eq!(music_only(original), music_only(&merged));
            assert!(merged.contains(
                "@0000:q:s1f0p64(vel:87,grace:f2p66:32:hammer:0:0,trill:f3p67:16,hammer)"
            ));
            assert!(merged.contains("@1920:h:s2f1p60(vib)"));
            if c["name"] == "unchanged_annotations_leave_exact_tokens" {
                assert_eq!(merged, original);
            }
            if c["name"] == "changed_prior_annotation_does_not_leak" {
                assert!(!merged.contains("@960:q:^"));
                assert!(merged.contains("@960:q:s1f0p64(vel:87,grace:f2p66:32:hammer:0:0,trill:f3p67:16,hammer)<chord:Allegro,text:old,dyn:90>"));
            }
        }
    }
    #[test]
    fn python_oracle_uncertain_chords_and_unicode_classification() {
        let f = fixture();
        for case in list(&f, "errors") {
            assert_eq!(
                chord_recognition_errors(&case["measure"]).len(),
                case["error_count"].as_u64().unwrap() as usize,
                "{}",
                case["name"]
            );
        }
        let c = &f["retained"];
        let (retained, uncertain) =
            retain_uncertain_chords_as_text(text(c, "original", ""), "both").unwrap();
        assert_eq!(
            score::parse_measure_target(&retained).unwrap(),
            c["oracle_parsed"]
        );
        assert_eq!(uncertain, c["uncertain"].as_array().unwrap().clone());
        assert_eq!(music_only(text(c, "original", "")), music_only(&retained));
        assert!(retained
            .contains("@0000:q:s1f0p64(vel:87,grace:f2p66:32:hammer:0:0,trill:f3p67:16,hammer)"));
    }
    #[test]
    fn misplaced_or_invalid_annotation_never_replaces_music() {
        let f = fixture();
        let original = text(&f["merges"][0], "original", "");
        assert!(merge_chord_review(original, "M2 | V0{@777:q:s1f0p64<chord:C>}", "both").is_err());
        assert!(merge_chord_review(
            original,
            "M2 | V0{@0:q:s1f0p64<chord:C,diagram:bad>}",
            "both"
        )
        .is_err());
        assert!(merge_chord_review(
            original,
            "M2 | V0{@0:q:s1f0p64<chord:C> @0:q:s1f0p64<chord:D>}",
            "both"
        )
        .is_err());
    }
    struct Mock {
        values: VecDeque<String>,
        calls: usize,
    }
    impl Generator for Mock {
        fn generate(
            &mut self,
            _: Value,
            _: usize,
            _: Option<Value>,
            _: Option<&str>,
        ) -> Result<llama::Generation> {
            self.calls += 1;
            Ok(llama::Generation {
                content: self.values.pop_front().unwrap(),
                completion_tokens: 80,
            })
        }
    }
    fn row(path: &Path) -> Value {
        RgbImage::from_pixel(70, 40, Rgb([255; 3]))
            .save(path)
            .unwrap();
        json!({"measure_number":1,"mode":"both","instrument":"guitar","tuning":[64,59,55,50,45,40],"image":path,"score_state":{"time":"4/4","key":0},"tuning_source":"default"})
    }
    #[test]
    fn mocked_chord_retry_changes_annotations_only() {
        let f = fixture();
        let tmp = tempfile::tempdir().unwrap();
        let row = row(&tmp.path().join("crop.png"));
        let original = text(&f["merges"][0], "original", "");
        let candidate = text(&f["merges"][0], "candidate", "");
        let mut engine = Mock {
            values: VecDeque::from([original.to_owned(), candidate.to_owned()]),
            calls: 0,
        };
        let out =
            recognize_measure(&mut engine, &row, &MeasureConfig::default(), &|| false).unwrap();
        assert_eq!(engine.calls, 2);
        assert_eq!(
            music_only(text(&out.record, "target", "")),
            music_only(original)
        );
        assert_eq!(out.record["written_target"], out.record["target"]);
        assert!(!flag(&out.record, "needs_review"));
        assert!(
            list(out.diagnostics.last().unwrap(), "deterministic_repairs")
                .iter()
                .any(|v| v == "preserve_music_during_chord_review")
        );
    }
    #[test]
    fn exhausted_bad_chord_retry_keeps_music_as_review_text() {
        let f = fixture();
        let tmp = tempfile::tempdir().unwrap();
        let row = row(&tmp.path().join("crop.png"));
        let original = text(&f["merges"][0], "original", "");
        let mut engine = Mock {
            values: VecDeque::from([
                original.to_owned(),
                "M2 | V0{@777:q:s1f0p64<chord:C>}".into(),
            ]),
            calls: 0,
        };
        let config = MeasureConfig {
            maximum_attempts: 2,
            ..MeasureConfig::default()
        };
        let out = recognize_measure(&mut engine, &row, &config, &|| false).unwrap();
        assert_eq!(
            music_only(text(&out.record, "target", "")),
            music_only(original)
        );
        assert!(flag(&out.record, "needs_review"));
        assert!(text(&out.record, "target", "").contains("text:old%3B%20Allegro"));
        assert!(!text(&out.record, "target", "").contains("@777"));
        assert!(
            list(out.diagnostics.last().unwrap(), "deterministic_repairs")
                .iter()
                .any(|v| v == "retain_unconfirmed_chord_as_review_text")
        );
        assert!(!out
            .diagnostics
            .iter()
            .any(|v| list(v, "deterministic_repairs")
                .iter()
                .any(|s| s == "fallback_full_measure_rest")));
    }
}

#[cfg(test)]
mod geometry_wiring_tests {
    use super::*;
    use std::collections::VecDeque;
    struct Mock(VecDeque<String>);
    impl Generator for Mock {
        fn generate(
            &mut self,
            _: Value,
            _: usize,
            _: Option<Value>,
            _: Option<&str>,
        ) -> Result<llama::Generation> {
            Ok(llama::Generation {
                content: self.0.pop_front().expect("Unexpected metadata inference"),
                completion_tokens: 50,
            })
        }
    }
    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pixel-geometry")
    }
    #[test]
    fn recognized_diagrams_use_pixels_and_keep_model_on_ambiguous_geometry() {
        let cases: Value =
            serde_json::from_str(include_str!("../tests/fixtures/pixel-geometry/chords.json"))
                .unwrap();
        for case in [&cases[0], &cases[9]] {
            let raw=json!({"kind":"chord_diagram","semitones":null,"capo":null,"text":"C","diagram":case["diagram"]}).to_string();
            let mut model = Mock(VecDeque::from([raw.clone()]));
            let regions = vec![
                json!({"kind":"annotation","image":root().join(text(case,"image","")),"page":1,"bbox":[0,0,100,100]}),
            ];
            let results = recognize_regions(&mut model, &regions, None, &|| false).unwrap();
            let expected = if case["expected"].is_null() {
                normalize_diagram(&case["diagram"])
            } else {
                normalize_diagram(&case["expected"])
            };
            assert_eq!(
                results[0]["parsed"]["diagram"], expected,
                "{}",
                case["name"]
            );
            assert_eq!(results[0]["raw"], raw);
        }
    }
    #[test]
    fn information_links_real_dashes_before_resolving_pitch_contexts() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/pixel-geometry/octaves.json"
        ))
        .unwrap();
        let case = &cases[0];
        let mut rows = list(case, "records").to_vec();
        for (i, row) in rows.iter_mut().enumerate() {
            row["measure_number"] = json!(i + 1);
        }
        let mut regions = list(case, "predictions").to_vec();
        let mut responses = VecDeque::new();
        for region in &mut regions {
            responses.push_back(region["parsed"].to_string());
            region.as_object_mut().unwrap().remove("parsed");
            region["image"] = json!(root().join(text(region, "image", "")));
        }
        let layout = json!({"schema_version":"1.0","stage":"layout","mode":"notation","info_source":"image","records":rows,"regions":regions,"parts":[{"id":"part-1","name":"Piano","instrument":"pitched","program":0,"strings":null}]});
        let config = InformationConfig {
            score_structure: false,
            staff_profile: false,
            layout_path: root().join("fixture-layout.json"),
            fallback_title: "Fixture".into(),
            ..InformationConfig::default()
        };
        let mut model = Mock(responses);
        let output = recognize_information(&mut model, &layout, &config, &|| false).unwrap();
        assert!(model.0.is_empty());
        assert_eq!(output.predictions[2]["parsed"]["semitones"], 12);
        assert_eq!(output.predictions[2]["model_parsed"]["kind"], "technique");
        assert_eq!(output.predictions[2]["octave_continuation"]["page"], 1);
        let second = &output.manifest["measure_pitch_contexts"][1]["pitch_context"];
        assert_eq!(second["octave_spans"][0]["semitones"], 12);
        assert_eq!(second["octave_spans"][0]["start"], 0.0);
        assert_eq!(second["octave_spans"][0]["end"], 1.0);
    }
}
