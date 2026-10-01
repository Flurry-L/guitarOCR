//! Literal raster/geometry port of shared/octave_lines.py. A continuation is
//! never inferred from a musical name: an explicit octave mark, visible bare
//! dashes, staff identity, and the original margin/gap rules must all agree.
use crate::image_boundary::{open_rgb, round_even, Result};
use image::RgbImage;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
#[derive(Debug, Clone, PartialEq)]
pub struct DashedLine {
    pub y: f64,
    pub left: usize,
    pub right: usize,
    pub hook: bool,
    pub bare: bool,
}
fn groups(values: impl IntoIterator<Item = usize>) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![];
    for v in values {
        if out
            .last()
            .and_then(|g| g.last())
            .map(|p| v > *p + 1)
            .unwrap_or(true)
        {
            out.push(vec![v]);
        } else {
            out.last_mut().unwrap().push(v);
        }
    }
    out
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}
pub fn dashed_line_pixels(image: &RgbImage) -> Option<DashedLine> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w < 60 || w < h * 2 {
        return None;
    }
    let ink: Vec<bool> = image
        .pixels()
        .map(|p| {
            ((19595u32 * p[0] as u32 + 38470u32 * p[1] as u32 + 7471u32 * p[2] as u32 + 32768)
                >> 16)
                < 180
        })
        .collect();
    let strengths: Vec<usize> = (0..h)
        .map(|y| ink[y * w..(y + 1) * w].iter().filter(|&&b| b).count())
        .collect();
    let max = *strengths.iter().max()?;
    if (max as f64) < w as f64 * 0.2 {
        return None;
    }
    let peak = strengths.iter().position(|&v| v == max)?;
    let bands = groups((0..h).filter(|&y| strengths[y] as f64 >= max as f64 * 0.65));
    let band = bands.iter().find(|b| b.contains(&peak))?;
    let center = band
        .iter()
        .map(|&y| y as f64 * strengths[y] as f64)
        .sum::<f64>()
        / band.iter().map(|&y| strengths[y]).sum::<usize>() as f64;
    let radius = 2.max(band.len());
    let upper = round_even(center - radius as f64).max(0.) as usize;
    let lower = (round_even(center + radius as f64) + 1.).min(h as f64) as usize;
    let strokes = groups((0..w).filter(|&x| (upper..lower).any(|y| ink[y * w + x])));
    let thin: Vec<_> = strokes
        .iter()
        .filter(|g| g.len() >= 1 && g.len() as f64 <= 18f64.max(w as f64 * 0.045))
        .collect();
    if thin.len() < 4 || (thin.iter().map(|g| g.len()).sum::<usize>() as f64) < w as f64 * 0.2 {
        return None;
    }
    let left = strokes.first()?[0];
    let right = *strokes.last()?.last()?;
    if ((right - left) as f64) < w as f64 * 0.6 {
        return None;
    }
    let terminal_width = 4.max(round_even(w as f64 * 0.015) as usize);
    let terminal_left = right.saturating_sub(terminal_width);
    let hook = (terminal_left..=right)
        .map(|x| (0..h).filter(|&y| ink[y * w + x]).count())
        .max()
        .unwrap_or(0)
        >= 5.max(band.len() * 2 + 1);
    let total = ink.iter().filter(|&&p| p).count();
    let outside = (0..h)
        .filter(|&y| y < upper || y >= lower)
        .map(|y| {
            (0..w)
                .filter(|&x| (x < terminal_left || x > right) && ink[y * w + x])
                .count()
        })
        .sum::<usize>();
    Some(DashedLine {
        y: center,
        left,
        right,
        hook,
        bare: outside as f64 <= 4f64.max(total as f64 * 0.06),
    })
}
pub fn dashed_line(path: Option<&Path>) -> Result<Option<DashedLine>> {
    let Some(path) = path.filter(|p| p.is_file()) else {
        return Ok(None);
    };
    Ok(dashed_line_pixels(&open_rgb(path)?))
}
fn bbox(value: &Value) -> Result<[f64; 4]> {
    let a = value["bbox"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("Octave geometry needs a four-number bbox")?;
    let mut out = [0.; 4];
    for (i, v) in a.iter().enumerate() {
        out[i] = v
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or("Octave bbox must be finite")?;
    }
    if out[2] <= 0. || out[3] <= 0. {
        return Err("Octave bbox must have positive dimensions".into());
    }
    Ok(out)
}
fn string(value: Option<&Value>, default: &str) -> String {
    match value {
        None => default.into(),
        Some(Value::String(v)) => v.clone(),
        Some(Value::Null) => "None".into(),
        Some(Value::Bool(v)) => if *v { "True" } else { "False" }.into(),
        Some(v) => v.to_string(),
    }
}
type Staff = (String, String);
type Row = (u64, u64, Staff);
fn staff(value: &Value) -> Staff {
    (
        string(value.get("part_id"), "part-1"),
        string(value.get("staff_id"), "staff-1"),
    )
}
fn row_key(value: &Value) -> Result<Row> {
    Ok((
        value["page"]
            .as_u64()
            .ok_or("Octave record needs a page number")?,
        value["system_index"]
            .as_u64()
            .ok_or("Octave record needs a system index")?,
        staff(value),
    ))
}
#[derive(Clone)]
struct Segment {
    index: usize,
    explicit: bool,
    line: Option<DashedLine>,
    left: f64,
    right: f64,
    y: f64,
    height: f64,
    above: bool,
}
/// Return copied predictions. Only linked bare segments become transposition
/// predictions; original model_parsed and the immediate reference page/bbox
/// remain attached. TAB-only records are excluded exactly as in Python.
pub fn link_octave_continuations(records: &[Value], predictions: &[Value]) -> Result<Vec<Value>> {
    link_octave_continuations_with_images(records, predictions, &mut |prediction| {
        let Some(path) = prediction["image"]
            .as_str()
            .map(Path::new)
            .filter(|p| p.is_file())
        else {
            return Ok(None);
        };
        open_rgb(path).map(Some)
    })
}

/// Same geometry with an image resolver for native in-memory/source-page crops.
/// The caller must return precisely the original padded annotation crop.
pub fn link_octave_continuations_with_images(
    records: &[Value],
    predictions: &[Value],
    resolve_image: &mut dyn FnMut(&Value) -> Result<Option<RgbImage>>,
) -> Result<Vec<Value>> {
    let mut result = predictions.to_vec();
    let records: Vec<_> = records.iter().filter(|r| r["mode"] != "tab").collect();
    let mut rows: BTreeMap<Row, Vec<[f64; 4]>> = BTreeMap::new();
    for r in &records {
        rows.entry(row_key(r)?).or_default().push(bbox(r)?);
    }
    let mut segments: BTreeMap<Row, Vec<Segment>> = BTreeMap::new();
    for (index, p) in result.iter().enumerate() {
        if !matches!(p["kind"].as_str(), Some("annotation" | "transposition"))
            || p["bbox"].as_array().map(|b| b.is_empty()).unwrap_or(true)
        {
            continue;
        }
        let explicit = p["parsed"]["kind"] == "ottava"
            && matches!(p["parsed"]["semitones"].as_i64(), Some(-24 | -12 | 12 | 24));
        let line = resolve_image(p)?.as_ref().and_then(dashed_line_pixels);
        let region = bbox(p)?;
        let candidates: Vec<_> = records.iter().filter(|r| r["page"] == p["page"]).collect();
        if candidates.is_empty() {
            continue;
        }
        let distance = |r: &Value| -> Result<(f64, f64, f64)> {
            let b = bbox(r)?;
            let center = region[1] + region[3] / 2.;
            Ok((
                (b[1] - center).max(center - b[1] - b[3]).max(0.),
                (center - b[1] - b[3] / 2.).abs(),
                (b[0] - region[0]).abs(),
            ))
        };
        let mut anchor = *candidates[0];
        let mut best = distance(anchor)?;
        for &&r in candidates.iter().skip(1) {
            let d = distance(r)?;
            if d.0
                .total_cmp(&best.0)
                .then(d.1.total_cmp(&best.1))
                .then(d.2.total_cmp(&best.2))
                .is_lt()
            {
                anchor = r;
                best = d;
            }
        }
        segments.entry(row_key(anchor)?).or_default().push(Segment {
            index,
            explicit,
            line,
            left: region[0],
            right: region[0] + region[2],
            y: region[1] + region[3] / 2.,
            height: region[3],
            above: false,
        });
    }
    let mut previous: BTreeMap<Staff, Segment> = BTreeMap::new();
    for (key, measures) in rows {
        let left = measures.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
        let right = measures
            .iter()
            .map(|b| b[0] + b[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let center = median(measures.iter().map(|b| b[1] + b[3] / 2.).collect());
        let gap = 32f64.max(median(measures.iter().map(|b| b[2]).collect()) * 0.08);
        let mut carried = previous.remove(&key.2);
        let mut active: Option<Segment> = None;
        let mut row_segments = segments.remove(&key).unwrap_or_default();
        row_segments.sort_by(|a, b| a.left.total_cmp(&b.left));
        for segment in row_segments {
            if segment.explicit {
                let upward = result[segment.index]["parsed"]["semitones"]
                    .as_i64()
                    .unwrap()
                    > 0;
                active = if segment.line.is_some() && (!upward || segment.y < center) {
                    Some(segment)
                } else {
                    None
                };
                carried = None;
                continue;
            }
            if segment.line.as_ref().map(|l| !l.bare).unwrap_or(true) {
                if active
                    .as_ref()
                    .map(|a| (segment.y - a.y).abs() <= 8f64.max(segment.height * 0.55))
                    .unwrap_or(false)
                {
                    active = None;
                }
                if segment.left <= left + gap {
                    carried = None;
                }
                continue;
            }
            let mut predecessor = active.clone();
            let mut connected = predecessor
                .as_ref()
                .map(|p| {
                    !p.line.as_ref().unwrap().hook
                        && (-8.0..=gap.max(segment.height * 2.)).contains(&(segment.left - p.right))
                        && (segment.y - p.y).abs() <= 8f64.max(segment.height * 0.55)
                })
                .unwrap_or(false);
            if predecessor.is_none() {
                if let Some(c) = carried.as_ref() {
                    predecessor = Some(c.clone());
                    connected = segment.left <= left + gap && (segment.y < center) == c.above;
                }
            }
            carried = None;
            if !connected {
                active = None;
                continue;
            }
            let reference = predecessor.unwrap().index;
            let semitones = result[reference]["parsed"]["semitones"].clone();
            let continuation =
                json!({"page":result[reference]["page"],"bbox":result[reference]["bbox"]});
            let prediction = &mut result[segment.index];
            prediction["model_parsed"] = prediction.get("parsed").cloned().unwrap_or(json!({}));
            prediction["kind"] = json!("transposition");
            prediction["parsed"] =
                json!({"kind":"ottava","semitones":semitones,"capo":null,"text":null});
            prediction["octave_continuation"] = continuation;
            active = Some(segment);
        }
        if let Some(mut a) = active {
            if !a.line.as_ref().unwrap().hook && a.right >= right - gap {
                a.above = a.y < center;
                previous.insert(key.2, a);
            }
        }
    }
    Ok(result)
}
