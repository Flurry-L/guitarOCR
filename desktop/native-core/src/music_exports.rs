//! Native score projections follow the existing file-codec contracts.
//! MusicXML retains legacy 960 divisions and integer tick projection; IR stays exact.
//! GP5.10 output uses a native binary encoder; unsupported techniques fail closed.
use crate::score::duration_ticks;
mod cp936_compat;
mod diagram;
mod fingering;
mod gp5;
mod gp5_duration;
mod gp5_techniques;
mod navigation;
mod readback;
mod xml_harmony;
use serde_json::{json, Value};
use std::{collections::HashMap, path::Path};
type R<T> = Result<T, String>;
fn arr<'a>(v: &'a Value, k: &str) -> R<&'a Vec<Value>> {
    v[k].as_array().ok_or_else(|| format!("Missing array: {k}"))
}
fn num(v: &Value, k: &str, d: i64) -> i64 {
    v[k].as_i64().unwrap_or(d)
}
fn txt<'a>(v: &'a Value, k: &str, d: &'a str) -> &'a str {
    v[k].as_str().unwrap_or(d)
}
fn esc(s: &str) -> R<String> {
    if s.chars().any(|c| {
        !(c == '\t' || c == '\n' || c == '\r' || c >= ' ' && c != '\u{fffe}' && c != '\u{ffff}')
    }) {
        return Err("XML text contains forbidden control characters".into());
    }
    Ok(s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;"))
}
fn tag(k: &str, v: impl ToString) -> R<String> {
    Ok(format!("<{k}>{}</{k}>", esc(&v.to_string())?))
}
fn effects(v: &Value) -> Vec<&str> {
    v["effects"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}
fn pitch(n: &Value, p: &Value) -> R<i64> {
    let base = if let Some(x) = n["pitch"].as_i64() {
        x
    } else {
        let s = num(n, "string", 0);
        let t = arr(p, "tuning")?;
        if s < 1 || s as usize > t.len() {
            return Err(format!("Missing tuning for string {s}"));
        }
        t[s as usize - 1].as_i64().ok_or("Invalid tuning")? + num(n, "fret", 0)
    };
    let x = base
        + if ["guitar", "bass"].contains(&txt(p, "instrument", "guitar")) {
            num(p, "capo", 0)
        } else {
            0
        };
    if !(0..=127).contains(&x) {
        Err(format!("Pitch outside MIDI range: {x}"))
    } else {
        Ok(x)
    }
}
fn named_pitch(p: i64, flat: bool) -> R<String> {
    let sharp = [
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    ];
    let flats = [
        ("C", 0),
        ("D", -1),
        ("D", 0),
        ("E", -1),
        ("E", 0),
        ("F", 0),
        ("G", -1),
        ("G", 0),
        ("A", -1),
        ("A", 0),
        ("B", -1),
        ("B", 0),
    ];
    let (s, a) = if flat { flats } else { sharp }[p.rem_euclid(12) as usize];
    Ok(format!(
        "<pitch>{}{}{}</pitch>",
        tag("step", s)?,
        if a != 0 {
            tag("alter", a)?
        } else {
            String::new()
        },
        tag("octave", p.div_euclid(12) - 1)?
    ))
}
fn key(s: &str) -> R<(i64, &str)> {
    if let Some((f, mode)) = s.split_once(':') {
        let f = f.parse::<i64>().map_err(|_| "Invalid key fifths")?;
        if (-8..=8).contains(&f) && ["major", "minor"].contains(&mode) {
            return Ok((f, mode));
        }
    }
    let normalized = s
        .replace("MajorFlat", "bMajor")
        .replace("MajorSharp", "#Major")
        .replace("MinorFlat", "bMinor")
        .replace("MinorSharp", "#Minor");
    if normalized == "FbMajor" {
        return Ok((-8, "major"));
    }
    if normalized == "G#Major" {
        return Ok((8, "major"));
    }
    if normalized == "DbMinor" {
        return Ok((-8, "minor"));
    }
    if normalized == "E#Minor" {
        return Ok((8, "minor"));
    }
    let major = [
        "CbMajor", "GbMajor", "DbMajor", "AbMajor", "EbMajor", "BbMajor", "FMajor", "CMajor",
        "GMajor", "DMajor", "AMajor", "EMajor", "BMajor", "F#Major", "C#Major",
    ];
    let minor = [
        "AbMinor", "EbMinor", "BbMinor", "FMinor", "CMinor", "GMinor", "DMinor", "AMinor",
        "EMinor", "BMinor", "F#Minor", "C#Minor", "G#Minor", "D#Minor", "A#Minor",
    ];
    for (names, mode) in [(&major, "major"), (&minor, "minor")] {
        if let Some(i) = names.iter().position(|x| *x == normalized) {
            return Ok((i as i64 - 7, mode));
        }
    }
    Err(format!("Unsupported key signature {s}"))
}
fn percent_decode(s: &str) -> R<String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let h = std::str::from_utf8(&b[i + 1..i + 3]).map_err(|_| "Invalid text escape")?;
            out.push(u8::from_str_radix(h, 16).map_err(|_| "Invalid text escape")?);
            i += 3
        } else {
            out.push(b[i]);
            i += 1
        }
    }
    String::from_utf8(out).map_err(|_| "Invalid UTF-8 text effect".into())
}
/// Convert recognition manifest or accept a canonical score document unchanged.
/// Reject inconsistent part metadata instead of silently picking the first row.
pub fn score_document(input: &Value) -> R<Value> {
    let score = if input["parts"].as_array().is_some() && input["timeline"].as_array().is_some() {
        input.clone()
    } else {
        crate::score::score_document(input)?
    };
    validate_projection(&score)?;
    Ok(score)
}
fn validate_projection(score: &Value) -> R<()> {
    let timeline = arr(score, "timeline")?;
    if timeline.len() > 100_000 {
        return Err("Export exceeds 100000-measure resource limit".into());
    }
    for (i, t) in timeline.iter().enumerate() {
        if num(t, "index", i as i64) != i as i64 {
            return Err("Score timeline must be contiguous and ordered".into());
        }
    }
    for p in arr(score, "parts")? {
        if !(0..=24).contains(&num(p, "capo", 0)) || !(0..=127).contains(&num(p, "midi_program", 0))
        {
            return Err("Invalid capo or MIDI program".into());
        }
        if arr(p, "tuning")?
            .iter()
            .any(|n| n.as_i64().is_none_or(|v| !(0..=127).contains(&v)))
        {
            return Err("Invalid MIDI tuning pitch".into());
        }
        for staff in arr(p, "staves")? {
            let mut seen = std::collections::HashSet::new();
            for m in arr(staff, "measures")? {
                let index = num(m, "index", -1);
                if index < 0 || index as usize >= timeline.len() || !seen.insert(index) {
                    return Err("Duplicate/out-of-range score measure; refusing note loss".into());
                }
                if !(-127..=127).contains(&num(&m["pitch_context"], "instrument_transpose", 0)) {
                    return Err("Invalid instrument transposition".into());
                }
                let octave = num(&m["pitch_context"], "clef_octave", 0);
                if octave % 12 != 0 || !(-24..=24).contains(&octave) {
                    return Err("Invalid clef octave".into());
                }
                let mut voices = std::collections::HashSet::new();
                for v in arr(m, "voices")? {
                    let voice = num(v, "voice", -1);
                    if !(0..=15).contains(&voice) || !voices.insert(voice) {
                        return Err("Duplicate/out-of-range voice; refusing note loss".into());
                    }
                    for e in arr(v, "events")? {
                        if num(e, "start", 0) < 0 {
                            return Err("Negative event start".into());
                        }
                        if !e["effects"].is_null()
                            && e["effects"]
                                .as_array()
                                .is_none_or(|es| es.iter().any(|ef| !ef.is_string()))
                        {
                            return Err("Malformed event effects".into());
                        }
                        for n in arr(e, "notes")? {
                            if n["pitch"].as_i64().is_some_and(|p| !(0..=127).contains(&p)) {
                                return Err("Invalid note pitch".into());
                            }
                            if n["fret"].as_i64().is_some_and(|p| !(0..=127).contains(&p)) {
                                return Err("Invalid fret".into());
                            }
                            if !n["effects"].is_null()
                                && n["effects"]
                                    .as_array()
                                    .is_none_or(|es| es.iter().any(|ef| !ef.is_string()))
                            {
                                return Err("Malformed note effects".into());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// MusicXML 4.0 projection preserving the legacy 960-division/int-tick contract.
pub fn score_musicxml(input: &Value) -> R<Vec<u8>> {
    let score = score_document(input)?;
    let parts = arr(&score, "parts")?;
    if parts.is_empty() {
        return Err("At least one score part is required".into());
    }
    let timeline = arr(&score, "timeline")?;
    let scale = 1i128;
    let mut out=format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<score-partwise version=\"4.0\"><work>{}</work><identification><creator type=\"composer\">{}</creator></identification><part-list>",tag("work-title",txt(&score,"title",""))?,esc(txt(&score,"artist",""))?);
    let mut melodic_channel = 0;
    for (pi, p) in parts.iter().enumerate() {
        if txt(p, "instrument", "") == "drums" {
            let mut pitches = std::collections::BTreeSet::new();
            for staff in arr(p, "staves")? {
                for m in arr(staff, "measures")? {
                    for v in arr(m, "voices")? {
                        for e in arr(v, "events")? {
                            for n in arr(e, "notes")? {
                                pitches.insert(pitch(n, p)?);
                            }
                        }
                    }
                }
            }
            out += &format!(
                "<score-part id=\"P{}\">{}",
                pi + 1,
                format!(
                    "{}{}",
                    tag("part-name", txt(p, "name", "Drums"))?,
                    tag(
                        "part-abbreviation",
                        txt(p, "name", "Drums").chars().take(12).collect::<String>()
                    )?
                )
            );
            for pitch in &pitches {
                out += &format!(
                    "<score-instrument id=\"P{}-D{pitch}\">{}</score-instrument>",
                    pi + 1,
                    tag("instrument-name", format!("Drum {pitch}"))?
                );
            }
            for pitch in &pitches {
                out += &format!("<midi-instrument id=\"P{}-D{pitch}\"><midi-channel>10</midi-channel>{}</midi-instrument>",pi+1,tag("midi-unpitched",pitch+1)?);
            }
            out += "</score-part>";
            continue;
        }
        let channel = if melodic_channel < 15 {
            melodic_channel + 1
        } else {
            1
        };
        melodic_channel += 1;
        let channel = if channel >= 10 { channel + 1 } else { channel };
        out+=&format!("<score-part id=\"P{}\">{}<score-instrument id=\"P{}-I1\">{}</score-instrument><midi-instrument id=\"P{}-I1\">{}{}</midi-instrument></score-part>",pi+1,format!("{}{}",tag("part-name",txt(p,"name","Instrument"))?,tag("part-abbreviation",txt(p,"name","Instrument").chars().take(12).collect::<String>())?),pi+1,tag("instrument-name",txt(p,"name","Instrument"))?,pi+1,tag("midi-channel",channel)?,tag("midi-program",num(p,"midi_program",0)+1)?)
    }
    out += "</part-list>";
    for (pi, p) in parts.iter().enumerate() {
        let staves = arr(p, "staves")?;
        out += &format!("<part id=\"P{}\">", pi + 1);
        let mut contexts = vec![json!({}); staves.len()];
        let mut keys = vec!["CMajor".to_owned(); staves.len()];
        // Precompute tie starts by actual sounding pitch, retaining voice/staff boundaries.
        let mut tie_starts = std::collections::HashSet::new();
        let mut previous: HashMap<(usize, i64, i64), (i64, usize, usize)> = HashMap::new();
        for (si, s) in staves.iter().enumerate() {
            let mut ms = arr(s, "measures")?.iter().collect::<Vec<_>>();
            ms.sort_by_key(|m| num(m, "index", 0));
            for m in ms {
                for v in arr(m, "voices")? {
                    let vi = num(v, "voice", 0);
                    for (ei, e) in arr(v, "events")?.iter().enumerate() {
                        let mut current = vec![];
                        for (ni, n) in arr(e, "notes")?.iter().enumerate() {
                            let pp = pitch(n, p)?;
                            if effects(n).contains(&"tie") {
                                if let Some((mi, pe, pn)) = previous.get(&(si, vi, pp)) {
                                    tie_starts.insert((si, *mi, vi, *pe, *pn));
                                }
                            }
                            current.push((pp, (num(m, "index", 0), ei, ni)));
                        }
                        previous.retain(|(ss, vv, _), _| *ss != si || *vv != vi);
                        for (pp, at) in current {
                            previous.insert((si, vi, pp), at);
                        }
                    }
                }
            }
        }
        for (bi, t) in timeline.iter().enumerate() {
            let old_contexts = contexts.clone();
            let old_keys = keys.clone();
            let time = txt(t, "time_signature", "4/4");
            let (a, b) = time.split_once('/').ok_or("Invalid time signature")?;
            let a = a.parse::<i128>().map_err(|_| "Invalid meter numerator")?;
            let b = b.parse::<i128>().map_err(|_| "Invalid meter denominator")?;
            if a < 1 || b < 1 {
                return Err("Meter cannot be represented exactly".into());
            }
            let length = a * 3840 * scale / b;
            out += &format!(
                "<measure number=\"{}\" id=\"p{}m{}\">",
                bi + 1,
                pi + 1,
                bi + 1
            );
            let attributes_start = out.len();
            out += "<attributes>";
            if bi == 0 {
                out += &tag("divisions", 960)?;
            }
            let rows = staves
                .iter()
                .map(|s| {
                    s["measures"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|m| num(m, "index", -1) == bi as i64)
                })
                .collect::<Vec<_>>();
            for (si, row) in rows.iter().enumerate() {
                if let Some(m) = row {
                    if !m["pitch_context"].is_null() {
                        contexts[si] = m["pitch_context"].clone()
                    }
                    if let Some(k) = m["key_signature"].as_str() {
                        keys[si] = k.to_owned();
                    }
                }
                let (f, mode) =
                    written_key(&keys[si], num(&contexts[si], "instrument_transpose", 0))?;
                if bi == 0 || old_keys[si] != keys[si] {
                    out += &format!(
                        "<key number=\"{}\">{}{}</key>",
                        si + 1,
                        tag("fifths", f)?,
                        tag("mode", mode)?
                    );
                }
            }
            if bi == 0 || txt(&timeline[bi - 1], "time_signature", "4/4") != time {
                out += &format!("<time>{}{}</time>", tag("beats", a)?, tag("beat-type", b)?);
            }
            if bi == 0 && staves.len() > 1 {
                out += &tag("staves", staves.len())?;
            }
            for (si, c) in contexts.iter().enumerate() {
                if bi > 0
                    && c["clef"] == old_contexts[si]["clef"]
                    && c["clef_octave"] == old_contexts[si]["clef_octave"]
                {
                    continue;
                }
                let name = txt(
                    c,
                    "clef",
                    if txt(p, "instrument", "") == "drums" {
                        "percussion"
                    } else if txt(p, "instrument", "") == "bass" {
                        "F4"
                    } else {
                        "G2"
                    },
                );
                let (sign, line) = match name {
                    "G2" => ("G", Some(2)),
                    "F4" => ("F", Some(4)),
                    "C3" => ("C", Some(3)),
                    "C4" => ("C", Some(4)),
                    "TAB" => ("TAB", None),
                    "percussion" => ("percussion", None),
                    _ => return Err(format!("Unsupported clef {name}")),
                };
                out += &format!(
                    "<clef number=\"{}\">{}{}{}</clef>",
                    si + 1,
                    tag("sign", sign)?,
                    line.map(|x| tag("line", x))
                        .transpose()?
                        .unwrap_or_default(),
                    if num(c, "clef_octave", 0) != 0 {
                        tag("clef-octave-change", num(c, "clef_octave", 0) / 12)?
                    } else {
                        String::new()
                    }
                );
            }
            for (si, c) in contexts.iter().enumerate() {
                let shift = num(c, "instrument_transpose", 0);
                if bi == 0 && shift == 0
                    || bi > 0 && shift == num(&old_contexts[si], "instrument_transpose", 0)
                {
                    continue;
                }
                let octave = shift / 12;
                let chromatic = shift - octave * 12;
                let diatonic = [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6]
                    [chromatic.unsigned_abs() as usize]
                    * chromatic.signum();
                out += &format!(
                    "<transpose number=\"{}\">{}{}{}</transpose>",
                    si + 1,
                    tag("diatonic", diatonic)?,
                    tag("chromatic", chromatic)?,
                    if octave != 0 {
                        tag("octave-change", octave)?
                    } else {
                        String::new()
                    }
                );
            }
            if out.ends_with("<attributes>") {
                out.truncate(attributes_start);
            } else {
                out += "</attributes>";
            }
            if let Some(tempo) = t["tempo_quarter"].as_i64() {
                out+=&format!("<direction placement=\"above\"><direction-type><metronome><beat-unit>quarter</beat-unit>{}</metronome></direction-type><sound tempo=\"{}\"/></direction>",tag("per-minute",tempo)?,tempo)
            }
            let bars = rows
                .iter()
                .filter_map(|r| *r)
                .flat_map(|r| r["bars"].as_array().into_iter().flatten())
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            if bars.contains(&"repeat_open") {
                out+="<barline location=\"left\"><bar-style>heavy-light</bar-style><repeat direction=\"forward\"/></barline>";
            }
            let mut first = true;
            for (si, row) in rows.iter().enumerate() {
                let empty = json!({"voices":[{"voice":0,"events":[]}]});
                let row = row.unwrap_or(&empty);
                if let Some(section) = row["section"].as_str() {
                    out += &format!(
                        "<direction placement=\"above\"><direction-type>{}</direction-type></direction>",
                        tag("rehearsal", section)?
                    )
                }
                for v in arr(row, "voices")? {
                    if !first {
                        out += &format!("<backup>{}</backup>", tag("duration", length)?)
                    }
                    first = false;
                    let mut cursor = 0i128;
                    let vi = num(v, "voice", 0);
                    let vn = si as i64 * 16 + vi + 1;
                    let events = arr(v, "events")?;
                    let mut current_ottava = 0;
                    if events.is_empty() {
                        out += &format!(
                            "<note><rest measure=\"yes\"/>{}{}{}</note>",
                            tag("duration", length)?,
                            tag("voice", vn)?,
                            tag("staff", si + 1)?
                        );
                        cursor = length;
                    }
                    for (ei, e) in events.iter().enumerate() {
                        let start = num(e, "start", (cursor / scale) as i64) as i128 * scale;
                        if start != cursor {
                            let name = if start > cursor { "forward" } else { "backup" };
                            out += &format!(
                                "<{name}>{}</{name}>",
                                tag("duration", (start - cursor).abs())?
                            )
                        }
                        let d = duration_ticks(&e["duration"])?;
                        let ticks = d.numerator * scale / d.denominator;
                        let ottava = effects(e)
                            .iter()
                            .filter_map(|s| s.strip_prefix("ottava:"))
                            .map(|s| s.parse::<i64>().map_err(|_| "Invalid octave shift"))
                            .collect::<Result<Vec<_>, _>>()?
                            .iter()
                            .sum::<i64>();
                        if ![0, 12, -12, 24, -24].contains(&ottava) {
                            return Err("Unsupported octave shift".into());
                        }
                        if ottava != current_ottava {
                            if current_ottava != 0 {
                                out += &octave_direction(current_ottava, vn, si + 1, true);
                            }
                            if ottava != 0 {
                                out += &octave_direction(ottava, vn, si + 1, false);
                            }
                            current_ottava = ottava;
                        }
                        out += &xml_harmony::render(e, si + 1)?;
                        for ef in effects(e) {
                            if let Some(text) = ef.strip_prefix("text:") {
                                out+=&format!("<direction placement=\"above\"><direction-type>{}</direction-type>{}</direction>",tag("words",percent_decode(text)?)?,tag("staff",si+1)?);
                            }
                        }
                        let notes = arr(e, "notes")?;
                        let null = Value::Null;
                        let items = if notes.is_empty() {
                            vec![&null]
                        } else {
                            notes.iter().collect()
                        };
                        for (ni, n) in items.iter().enumerate() {
                            out += &format!(
                                "<note id=\"n{}_{}_{}_{}_{}_{}\">",
                                pi + 1,
                                si + 1,
                                bi + 1,
                                vn,
                                ei,
                                ni
                            );
                            if ni > 0 {
                                out += "<chord/>"
                            }
                            let tied = !n.is_null() && effects(n).contains(&"tie");
                            let starts = tie_starts.contains(&(si, bi as i64, vi, ei, ni));
                            if n.is_null() {
                                out += "<rest/>"
                            } else {
                                if txt(p, "instrument", "") == "drums" {
                                    let pp = pitch(n, p)?;
                                    let (step, octave) = match pp {
                                        35 | 36 | 41 | 43 => ("F", 4),
                                        37 | 38 | 40 | 47 => ("C", 5),
                                        45 => ("A", 4),
                                        48 => ("D", 5),
                                        50 => ("E", 5),
                                        42 | 44 | 46 => ("G", 5),
                                        49 => ("A", 5),
                                        51 | 53 => ("F", 5),
                                        _ => ("B", 4),
                                    };
                                    out += &format!(
                                        "<unpitched>{}{}</unpitched>",
                                        tag("display-step", step)?,
                                        tag("display-octave", octave)?
                                    );
                                } else {
                                    out += &named_pitch(
                                        pitch(n, p)?
                                            - num(&contexts[si], "instrument_transpose", 0),
                                        written_key(
                                            &keys[si],
                                            num(&contexts[si], "instrument_transpose", 0),
                                        )?
                                        .0 < 0,
                                    )?;
                                }
                            }
                            out += &tag("duration", ticks)?;
                            if starts {
                                out += "<tie type=\"start\"/>";
                            }
                            if tied {
                                out += "<tie type=\"stop\"/>";
                            }
                            if !n.is_null() && txt(p, "instrument", "") == "drums" {
                                out +=
                                    &format!("<instrument id=\"P{}-D{}\"/>", pi + 1, pitch(n, p)?);
                            }
                            out += &tag("voice", vn)?;
                            let typ = match num(&e["duration"], "value", 0) {
                                1 => "whole",
                                2 => "half",
                                4 => "quarter",
                                8 => "eighth",
                                16 => "16th",
                                32 => "32nd",
                                64 => "64th",
                                128 => "128th",
                                256 => "256th",
                                512 => "512th",
                                1024 => "1024th",
                                _ => return Err("MusicXML duration type unsupported".into()),
                            };
                            out += &tag("type", typ)?;
                            for _ in 0..if e["duration"]["double_dotted"] == true {
                                2
                            } else if e["duration"]["dotted"] == true {
                                1
                            } else {
                                0
                            } {
                                out += "<dot/>"
                            }
                            let enters = num(&e["duration"], "tuplet_enters", 1);
                            if enters != 1 {
                                out += &format!(
                                    "<time-modification>{}{}</time-modification>",
                                    tag("actual-notes", enters)?,
                                    tag("normal-notes", num(&e["duration"], "tuplet_times", 1))?
                                )
                            }
                            out += &tag("staff", si + 1)?;
                            if ni == 0 && !n.is_null() && num(&e["duration"], "value", 0) >= 8 {
                                let value = num(&e["duration"], "value", 0);
                                for level in 1..value.ilog2() - 1 {
                                    let joins = |other: Option<&Value>| {
                                        other.is_some_and(|other| {
                                            other["notes"]
                                                .as_array()
                                                .is_some_and(|ns| !ns.is_empty())
                                                && num(other, "start", 0) / 960
                                                    == start as i64 / 960
                                                && num(&other["duration"], "value", 0)
                                                    >= 1i64 << (level + 2)
                                        })
                                    };
                                    let left = joins(ei.checked_sub(1).and_then(|j| events.get(j)));
                                    let right = joins(events.get(ei + 1));
                                    if left || right {
                                        out += &format!(
                                            "<beam number=\"{level}\">{}</beam>",
                                            if left && right {
                                                "continue"
                                            } else if left {
                                                "end"
                                            } else {
                                                "begin"
                                            }
                                        );
                                    }
                                }
                            }
                            if !n.is_null() {
                                out += "<notations>";
                                if ni == 0 && enters != 1 {
                                    let group = events
                                        .iter()
                                        .enumerate()
                                        .filter(|(_, other)| {
                                            num(other, "start", 0) / 960 == start as i64 / 960
                                                && num(&other["duration"], "tuplet_enters", 1)
                                                    == enters
                                        })
                                        .map(|(j, _)| j)
                                        .collect::<Vec<_>>();
                                    if group.first() == Some(&ei) {
                                        out +=
                                            "<tuplet type=\"start\" number=\"1\" bracket=\"yes\"/>";
                                    }
                                    if group.last() == Some(&ei) {
                                        out += "<tuplet type=\"stop\" number=\"1\"/>";
                                    }
                                }
                                if tied {
                                    out += &format!("<tied type=\"stop\" number=\"{}\"/>", vi + 1)
                                }
                                if starts {
                                    out += &format!("<tied type=\"start\" number=\"{}\"/>", vi + 1)
                                }
                                if let Some(s) = n["string"].as_i64() {
                                    out += &format!(
                                        "<technical>{}{}</technical>",
                                        tag("string", s)?,
                                        n["fret"]
                                            .as_i64()
                                            .map(|f| tag("fret", f))
                                            .transpose()?
                                            .unwrap_or_default()
                                    )
                                }
                                for ef in effects(n) {
                                    match ef {
                                        "tie" => {}
                                        "stacc" | "accent" | "heavy" => {
                                            out += &format!(
                                                "<articulations><{}/></articulations>",
                                                match ef {
                                                    "stacc" => "staccato",
                                                    "heavy" => "strong-accent",
                                                    _ => "accent",
                                                }
                                            )
                                        }
                                        _ if ef.starts_with("trill") => {
                                            out += "<ornaments><trill-mark/></ornaments>";
                                        }
                                        _ => {
                                            out += &format!(
                                            "<other-notation type=\"single\">{}</other-notation>",
                                            esc(ef)?
                                        )
                                        }
                                    }
                                }
                                out += "</notations>"
                            }
                            out += "</note>";
                        }
                        cursor = start + ticks;
                    }
                    if current_ottava != 0 {
                        out += &octave_direction(current_ottava, vn, si + 1, true);
                    }
                    if cursor < length {
                        out += &format!("<forward>{}</forward>", tag("duration", length - cursor)?);
                    }
                }
            }
            if bars.contains(&"repeat_close") || bars.contains(&"double") {
                out += &format!(
                    "<barline location=\"right\"><bar-style>{}</bar-style>",
                    if bars.contains(&"repeat_close") {
                        "light-heavy"
                    } else {
                        "light-light"
                    }
                );
                if bars.contains(&"repeat_close") {
                    let repeat = rows
                        .iter()
                        .filter_map(|r| r.and_then(|m| m["repeat_count"].as_i64()))
                        .next()
                        .unwrap_or(2);
                    out += &format!("<repeat direction=\"backward\" times=\"{repeat}\"/>")
                }
                out += "</barline>"
            }
            out += "</measure>";
        }
        out += "</part>";
    }
    out += "</score-partwise>\n";
    Ok(out.into_bytes())
}
pub fn write_musicxml(input: &Value, output: &Path) -> R<Value> {
    let bytes = score_musicxml(input)?;
    atomic_write(output, &bytes)?;
    Ok(json!({"format":"musicxml","path":output,"encoding":"UTF-8","replacements":[]}))
}
pub fn write_gp5(input: &Value, output: &Path) -> R<Value> {
    gp5::write(input, output)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> R<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create export directory: {e}"))?;
    let mut id = [0u8; 16];
    getrandom::fill(&mut id).map_err(|e| e.to_string())?;
    let name = id.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let temporary = parent.join(format!(".guitarocr-export-{name}.tmp"));
    let result = (|| -> R<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, path).map_err(|e| format!("Cannot replace export: {e}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn octave_direction(semitones: i64, voice: i64, staff: usize, stop: bool) -> String {
    format!("<direction placement=\"{}\"><direction-type><octave-shift type=\"{}\" size=\"{}\" number=\"{}\"/></direction-type><voice>{voice}</voice><staff>{staff}</staff></direction>",if semitones>0{"above"}else{"below"},if stop{"stop"}else if semitones>0{"down"}else{"up"},if semitones.abs()==12{8}else{15},(voice-1)%16+1)
}

fn written_key(name: &str, shift: i64) -> R<(i64, &str)> {
    let (f, mode) = key(name)?;
    if shift % 12 == 0 {
        return Ok((
            if f < -7 {
                f + 12
            } else if f > 7 {
                f - 12
            } else {
                f
            },
            mode,
        ));
    }
    let tonic = (7 * f + if mode == "minor" { 9 } else { 0 } - shift).rem_euclid(12);
    let candidate = (-7i64..=7)
        .filter(|k| (7 * k + if mode == "minor" { 9 } else { 0 }).rem_euclid(12) == tonic)
        .min_by_key(|k| (k.abs(), (k - f).abs()))
        .ok_or("Cannot spell transposed key")?;
    Ok((candidate, mode))
}
