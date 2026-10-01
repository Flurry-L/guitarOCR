//! Independent GP5.10 format encoder based on published wire-format fields.
//! Format reference: https://pyguitarpro.readthedocs.io/en/stable/pyguitarpro/format.html
//! No PyGuitarPro code or interpreter is linked or executed by this module.
//! Unsupported effects fail closed before output; development tests use PyGuitarPro
//! solely as an independent reader oracle.
use super::*;
use std::collections::{BTreeSet, HashSet};
struct Writer {
    bytes: Vec<u8>,
    replacements: Vec<Value>,
    velocity_quantization: Vec<Value>,
}
impl Writer {
    fn new() -> Self {
        Self {
            bytes: vec![],
            replacements: vec![],
            velocity_quantization: vec![],
        }
    }
    fn u(&mut self, x: u8) {
        self.bytes.push(x)
    }
    fn i(&mut self, x: i32) {
        self.bytes.extend(x.to_le_bytes())
    }
    fn z(&mut self, n: usize) {
        self.bytes.resize(self.bytes.len() + n, 0)
    }
    fn text(&mut self, s: &str, field: &str, max: usize) -> R<Vec<u8>> {
        let mut b = vec![];
        let mut written = String::new();
        for c in s.chars() {
            let mut buf = [0; 4];
            let cs = c.encode_utf8(&mut buf);
            let (encoded, _, bad) = encoding_rs::GBK.encode(cs);
            let bad = bad || super::cp936_compat::incompatible(c);
            let enc = if bad {
                vec![b'?']
            } else {
                encoded.into_owned()
            };
            b.extend(enc);
            written.push(if bad { '?' } else { c });
        }
        if b.len() > max {
            b.truncate(max);
            let (decoded, _, bad) = encoding_rs::GBK.decode(&b);
            if bad {
                return Err(format!(
                    "Legacy GP5 text limit splits a CP936 character at {field}"
                ));
            }
            written = decoded.into_owned();
        }
        if written != s {
            self.replacements
                .push(json!({"field":field,"original":s,"written":written}));
        }
        Ok(b)
    }
    fn string(&mut self, s: &str, field: &str) -> R<()> {
        let b = self.text(s, field, 255)?;
        self.i(b.len() as i32 + 1);
        self.u(b.len() as u8);
        self.bytes.extend(b);
        Ok(())
    }
    fn fixed(&mut self, s: &str, field: &str, n: usize) -> R<()> {
        let b = self.text(s, field, n)?;
        self.u(b.len() as u8);
        self.bytes.extend(&b);
        self.z(n - b.len());
        Ok(())
    }
}
#[derive(Clone)]
struct Track {
    part: Value,
    staff: Value,
    voices: Vec<i64>,
    strings: Vec<usize>,
    channel: i32,
    mode: String,
    name: String,
}
fn meter(t: &Value) -> R<(i32, i32)> {
    let (a, b) = txt(t, "time_signature", "4/4")
        .split_once('/')
        .ok_or("Invalid meter")?;
    let a = a.parse::<i32>().map_err(|_| "Invalid meter")?;
    let b = b.parse::<i32>().map_err(|_| "Invalid meter")?;
    if !(1..=127).contains(&a) || ![1, 2, 4, 8, 16, 32, 64].contains(&b) {
        return Err("GP5 meter cannot be represented".into());
    }
    Ok((a, b))
}
fn rests(ticks: i64) -> R<Vec<Value>> {
    super::gp5_duration::rest_events(ticks)
}

fn projected_duration(e: &Value) -> R<crate::score::Rational> {
    let mut duration = e["duration"].clone();
    duration["double_dotted"] = json!(false);
    duration_ticks(&duration)
}
fn duration(e: &Value) -> R<(u8, i8, Option<i32>, i64)> {
    let d = &e["duration"];
    let value = num(d, "value", 0);
    if ![1, 2, 4, 8, 16, 32, 64, 128].contains(&value) {
        return Err("GP5 duration outside whole..128th".into());
    }
    let enters = num(d, "tuplet_enters", 1);
    let times = num(d, "tuplet_times", 1);
    let pair = match enters {
        1 => 1,
        3 => 2,
        5 | 6 | 7 => 4,
        9 | 10 | 11 | 12 | 13 => 8,
        _ => return Err("Unsupported GP5 tuplet".into()),
    };
    if times != pair {
        return Err("GP5 tuplet ratio not representable".into());
    }
    let ticks = projected_duration(e)?;
    Ok((
        if d["dotted"] == true { 1 } else { 0 } | if enters != 1 { 32 } else { 0 },
        value.ilog2() as i8 - 2,
        if enters == 1 {
            None
        } else {
            Some(enters as i32)
        },
        i64::try_from(ticks.numerator / ticks.denominator).map_err(|_| "Duration overflow")?,
    ))
}
fn beat(
    w: &mut Writer,
    e: &Value,
    t: &Track,
    prev: &mut HashMap<usize, i64>,
    detached: &mut Vec<Value>,
    loc: &Value,
    velocity: &mut i64,
) -> R<()> {
    let (mut flags, d, tuplet, _) = duration(e)?;
    let mut text = None;
    let mut chord = None;
    let mut diagram = None;
    let mut octave = 0u16;
    let mut tempo = None;
    for ef in effects(e) {
        if let Some(v) = ef.strip_prefix("diagram:") {
            if diagram.is_some() {
                return Err("Multiple chord diagrams at one onset".into());
            }
            diagram = Some(super::diagram::parse(v)?);
        } else if let Some(v) = ef.strip_prefix("chord:") {
            if chord.is_some() || diagram.is_some() {
                return Err("Multiple GP5 chord labels at one onset".into());
            }
            chord = Some(percent_decode(v)?.replace('♭', "b").replace('♯', "#"));
        } else if let Some(v) = ef.strip_prefix("tempo:") {
            let bpm = v.parse::<i32>().map_err(|_| "Invalid tempo")?;
            if !(1..=1000).contains(&bpm) {
                return Err("Tempo outside 1..1000".into());
            }
            tempo = Some(bpm);
        } else if let Some(v) = ef.strip_prefix("dyn:") {
            *velocity = v.parse().map_err(|_| "Invalid dynamic")?;
        } else if let Some(v) = ef.strip_prefix("text:") {
            if chord.is_some() || diagram.is_some() {
                flags |= 2;
            }
            if text.is_some() {
                return Err("Multiple text effects on one GP5 beat unsupported".into());
            }
            text = Some(percent_decode(v)?);
        } else if let Some(v) = ef.strip_prefix("ottava:") {
            octave = match v {
                "12" => 0x10,
                "-12" => 0x20,
                "24" => 0x40,
                "-24" => 0x100,
                _ => return Err("Unsupported ottava".into()),
            };
        } else if [
            "fade",
            "rasg",
            "pick_up",
            "pick_down",
            "stroke_up",
            "stroke_down",
        ]
        .contains(&ef)
            || ef.starts_with("slap:")
        {
            // Encoded by the beat-technique payload below.
        } else {
            return Err(format!(
                "GP5 event effect {ef:?} not yet implemented at {loc}"
            ));
        }
    }
    let notes = arr(e, "notes")?;
    let status = match txt(e, "status", "normal") {
        "empty" => 0,
        "rest" => 2,
        _ => 1,
    };
    let rest = status != 1;
    if rest {
        flags |= 64
    }
    if chord.is_some() || diagram.is_some() {
        flags |= 2;
    }
    if text.is_some() {
        flags |= 4
    }
    if tempo.is_some() {
        flags |= 16
    }
    let beat_payload = super::gp5_techniques::beat(e)?;
    if !beat_payload.is_empty() {
        flags |= 8;
    }
    w.u(flags);
    if rest {
        w.u(status)
    }
    w.u(d as u8);
    if let Some(n) = tuplet {
        w.i(n)
    }
    if diagram.is_none() && chord.is_some() {
        diagram = Some(super::diagram::Diagram {
            base: 1,
            frets: vec![-1; t.strings.len()],
            fingers: vec![-2; t.strings.len()],
            barres: vec![],
        });
    }
    if let Some(diagram) = diagram {
        let strings = if ["pitched", "drums"].contains(&txt(&t.part, "instrument", "")) {
            (0..t.strings.len()).collect::<Vec<_>>()
        } else {
            t.strings.clone()
        };
        let diagram = diagram.project(&strings);
        if diagram.barres.len() > 5 {
            return Err("GP5 can store at most five barres per chord diagram".into());
        }
        w.u(1);
        w.z(7);
        w.i(0);
        w.i(0);
        w.u(0);
        w.fixed(chord.as_deref().unwrap_or(""), &format!("{loc} chord"), 22)?;
        w.z(3);
        w.i(diagram.base as i32);
        for i in 0..7 {
            w.i(diagram.frets.iter().rev().nth(i).copied().unwrap_or(-1) as i32);
        }
        w.u(diagram.barres.len() as u8);
        for field in 0..3 {
            for i in 0..5 {
                w.u(diagram
                    .barres
                    .get(i)
                    .map(|b| match field {
                        0 => b[0],
                        1 => diagram.frets.len() as i64 - b[2],
                        _ => diagram.frets.len() as i64 - b[1],
                    })
                    .unwrap_or(0) as u8);
            }
        }
        w.z(8);
        for i in 0..7 {
            w.u(diagram.fingers.iter().rev().nth(i).copied().unwrap_or(-2) as u8);
        }
        w.u(1);
    }
    if let Some(s) = text {
        w.string(&s, &format!("{loc} text"))?;
    }
    w.bytes.extend(beat_payload);
    if let Some(bpm) = tempo {
        w.u(255);
        for _ in 0..4 {
            w.i(-1)
        }
        for _ in 0..6 {
            w.u(255)
        }
        w.string("", "tempo name")?;
        w.i(bpm);
        w.u(0);
        w.u(0);
        w.u(0);
        w.u(255);
        w.string("", "tempo effect")?;
        w.string("", "tempo category")?;
    }
    let mut positions = vec![];
    let mut used = HashSet::new();
    for n in notes {
        let source = num(n, "string", 0);
        if source < 1 {
            return Err("GP5 pitch-only fingering is not yet implemented; score retained".into());
        }
        let local = t
            .strings
            .iter()
            .position(|s| *s == source as usize - 1)
            .ok_or("Internal GP5 string partition mismatch")?
            + 1;
        if !used.insert(local) {
            return Err(format!(
                "GP5 simultaneous notes share string {source} at {loc}"
            ));
        }
        let fret = num(n, "fret", 0);
        let is_dead = n["fret"] == "x" || effects(n).contains(&"dead");
        if !(0..=if txt(&t.part, "instrument", "") == "drums" {
            99
        } else {
            30
        })
            .contains(&fret)
        {
            return Err("GP5 melodic fret outside native importer range 0..30".into());
        }
        if let Some(p) = n["pitch"].as_i64() {
            let tuning = t.part["tuning"][source as usize - 1]
                .as_i64()
                .ok_or("Invalid tuning")?;
            if !is_dead && p != tuning + fret {
                return Err("GP5 pitch and string/fret disagree".into());
            }
        }
        positions.push((local, fret, is_dead, n));
    }
    positions.sort_by_key(|n| n.0);
    w.u(positions.iter().fold(0u8, |f, n| f | (1 << (7 - n.0))));
    for (s, f, dead, n) in positions {
        let effects = effects(n);
        let source = t.strings[s - 1];
        let open = t.part["tuning"][source].as_i64().ok_or("Invalid tuning")?;
        let payload =
            super::gp5_techniques::note(n, open, f, txt(&t.part, "instrument", "") == "drums")?;
        let nf = 0x30 | payload.note_flags;
        let swap =
            if t.mode != "tab" && (n["swap_accidentals"] == true || effects.contains(&"accswap")) {
                2
            } else {
                0
            };
        let mut typ = if dead {
            3
        } else if effects.contains(&"tie") {
            2
        } else {
            1
        };
        if typ == 2 && prev.get(&s) != Some(&f) {
            typ = 1;
            let mut report = loc.clone();
            report["string"] = json!(s);
            detached.push(report);
        }
        w.u(nf);
        w.u(typ);
        let vel = effects
            .iter()
            .find_map(|e| e.strip_prefix("vel:"))
            .map(|s| s.parse::<i64>())
            .transpose()
            .map_err(|_| "Invalid note velocity")?
            .unwrap_or_else(|| {
                let value = num(n, "velocity", 95);
                if value == 95 {
                    *velocity
                } else {
                    value
                }
            });
        if !(0..=127).contains(&vel) {
            return Err("GP5 velocity outside 0..127".into());
        }
        let packed = (vel + 1) / 16;
        let written = packed * 16 - 1;
        if written != vel {
            w.velocity_quantization
                .push(json!({"location":loc,"string":s,"original":vel,"written":written}));
        }
        w.u(packed as u8);
        w.u(if typ == 2 { 0 } else { f as u8 });
        w.u(swap);
        w.bytes.extend(payload.bytes);
        if !dead {
            prev.insert(s, f);
        } else {
            prev.remove(&s);
        }
    }
    w.bytes.extend(octave.to_le_bytes());
    Ok(())
}
pub(super) fn encode(input: &Value) -> R<(Vec<u8>, Value)> {
    let mut score = score_document(input)?;
    super::fingering::assign(&mut score)?;
    let timeline = arr(&score, "timeline")?;
    if timeline.is_empty() {
        return Err("At least one measure required".into());
    }
    let mut tracks = vec![];
    let mut channels: HashMap<String, (i32, i64)> = HashMap::new();
    for p in arr(&score, "parts")? {
        if !["guitar", "bass", "pitched", "drums"].contains(&txt(p, "instrument", "")) {
            return Err("Unknown GP5 instrument".into());
        }
        let tuning = arr(p, "tuning")?;
        if tuning.is_empty()
            || tuning
                .iter()
                .any(|v| v.as_i64().is_none_or(|n| !(0..=127).contains(&n)))
        {
            return Err("Invalid GP5 tuning".into());
        }
        let program = num(p, "midi_program", 25);
        let channel = if txt(p, "instrument", "") == "drums" {
            9
        } else {
            let id = txt(p, "id", "").to_owned();
            if !channels.contains_key(&id) {
                let used = channels.values().map(|(ch, _)| *ch).collect::<HashSet<_>>();
                let available = (0..16)
                    .find(|ch| *ch != 9 && !used.contains(ch))
                    .or_else(|| {
                        channels
                            .values()
                            .find(|(_, pr)| *pr == program)
                            .map(|(ch, _)| *ch)
                    })
                    .ok_or("GP5 MIDI channels cannot preserve every instrument")?;
                channels.insert(id.clone(), (available, program));
            }
            channels[&id].0
        };
        for s in arr(p, "staves")? {
            let modes = arr(s, "measures")?
                .iter()
                .map(|m| txt(m, "mode", "both"))
                .collect::<HashSet<_>>();
            let mode = if modes.len() > 1 {
                "notation"
            } else {
                modes.iter().next().copied().unwrap_or("both")
            }
            .to_owned();
            let voices = arr(s, "measures")?
                .iter()
                .flat_map(|m| m["voices"].as_array().into_iter().flatten())
                .map(|v| num(v, "voice", 0))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            for vs in voices.chunks(2) {
                for start in (0..tuning.len()).step_by(7) {
                    let instrument = txt(p, "instrument", "guitar");
                    let default_name = match instrument {
                        "guitar" => "Guitar",
                        "bass" => "Bass",
                        "drums" => "Drums",
                        _ => match program {
                            0 => "Piano",
                            56 => "Trumpet",
                            60 => "Horn",
                            64 => "Soprano Saxophone",
                            65 => "Alto Saxophone",
                            66 => "Tenor Saxophone",
                            67 => "Baritone Saxophone",
                            69 => "English Horn",
                            _ => "Melodic instrument",
                        },
                    };
                    let mut name = if txt(p, "name", instrument) == instrument {
                        default_name
                    } else {
                        txt(p, "name", default_name)
                    }
                    .to_owned();
                    if arr(p, "staves")?.len() > 1 {
                        name.push(' ');
                        name += match txt(s, "id", "") {
                            "staff-1" => "R.H.",
                            "staff-2" => "L.H.",
                            other => other,
                        };
                    }
                    if voices.len() > 2 {
                        name += &format!(
                            " V{}",
                            vs.iter()
                                .map(|v| (v + 1).to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                    }
                    if tuning.len() > 7 {
                        name += &format!(
                            " strings {}",
                            (start..(start + 7).min(tuning.len()))
                                .map(|s| (s + 1).to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                    }
                    tracks.push(Track {
                        part: p.clone(),
                        staff: s.clone(),
                        voices: vs.to_vec(),
                        strings: (start..(start + 7).min(tuning.len())).collect(),
                        channel,
                        mode: mode.clone(),
                        name,
                    });
                }
            }
        }
    }
    if tracks.is_empty() {
        return Err("No GP5 voices".into());
    }
    let mut w = Writer::new();
    w.fixed("FICHIER GUITAR PRO v5.10", "version", 30)?;
    for field in [
        "title",
        "subtitle",
        "artist",
        "album",
        "words",
        "music",
        "copyright",
        "tab",
        "instructions",
    ] {
        w.string(txt(&score, field, ""), field)?;
    }
    w.i(0);
    w.i(0);
    for _ in 0..5 {
        w.i(1);
        w.i(0)
    }
    w.i(100);
    w.i(0);
    w.z(11);
    for v in [210, 297, 10, 10, 15, 10, 100] {
        w.i(v)
    }
    w.u(255);
    w.u(1);
    for text in [
        "%title%",
        "%subtitle%",
        "%artist%",
        "%album%",
        "Words by %words%",
        "Music by %music%",
        "Words & Music by %WORDSMUSIC%",
        "Copyright %copyright%",
        "All Rights Reserved - International Copyright Secured",
        "Page %N%/%P%",
    ] {
        w.string(text, "page setup")?;
    }
    w.string("", "tempo name")?;
    let tempo = arr(&tracks[0].staff, "measures")?
        .iter()
        .find_map(|m| m["tempo_quarter"].as_i64())
        .unwrap_or(120);
    w.i(tempo as i32);
    w.u(0);
    let first_key = if tracks[0].mode == "tab" {
        0
    } else {
        key(arr(&tracks[0].staff, "measures")?
            .iter()
            .find(|m| num(m, "index", -1) == 0)
            .map(|m| txt(m, "key_signature", "CMajor"))
            .unwrap_or("CMajor"))?
        .0
    };
    w.u(first_key as u8);
    w.i(0);
    for ch in 0..64 {
        let program = tracks
            .iter()
            .find(|t| t.channel == ch)
            .map(|t| num(&t.part, "midi_program", 24))
            .unwrap_or(0);
        w.i(if ch % 16 == 9 { -1 } else { program as i32 });
        w.bytes.extend([13, 8, 0, 0, 0, 0, 0, 0]);
    }
    let mut navigation_positions = [-1i16; 19];
    for track in tracks.iter().take(1) {
        for row in arr(&track.staff, "measures")? {
            for sign in super::navigation::signs(&[Some(row)])? {
                let i = super::navigation::SIGNS
                    .iter()
                    .position(|s| *s == sign)
                    .unwrap();
                let position = i16::try_from(num(row, "index", 0) + 1)
                    .map_err(|_| "Navigation location exceeds GP5 signed-short range")?;
                if navigation_positions[i] != -1 && navigation_positions[i] != position {
                    return Err(format!("GP5 permits only one destination for {sign}"));
                }
                navigation_positions[i] = position;
            }
        }
    }
    for position in navigation_positions {
        w.bytes.extend(position.to_le_bytes());
    }
    w.i(0);
    w.i(timeline.len() as i32);
    w.i(tracks.len() as i32);
    let mut last_key = "CMajor".to_owned();
    for (bi, t) in timeline.iter().enumerate() {
        if bi > 0 {
            w.u(0)
        }
        let (a, b) = meter(t)?;
        let rows = tracks
            .iter()
            .take(1)
            .filter_map(|tr| {
                tr.staff["measures"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|m| num(m, "index", -1) == bi as i64)
            })
            .collect::<Vec<_>>();
        let ks = rows
            .iter()
            .filter_map(|r| r["key_signature"].as_str())
            .collect::<HashSet<_>>();
        if ks.len() > 1 {
            return Err("GP5 tracks disagree on key signature".into());
        }
        if tracks[0].mode != "tab" {
            if let Some(k) = ks.iter().next() {
                last_key = (*k).to_owned()
            }
        }
        let (f, mode) = key(&last_key)?;
        let bars = rows
            .iter()
            .flat_map(|r| r["bars"].as_array().into_iter().flatten())
            .filter_map(Value::as_str)
            .collect::<HashSet<_>>();
        let section = rows.iter().find_map(|r| r["section"].as_str());
        // Legacy GP5 uses the already-sounding note values; written-pitch
        // context is preserved by score JSON/MusicXML, not applied again here.
        let alternative = rows
            .iter()
            .map(|r| num(r, "alternate_endings", 0))
            .max()
            .unwrap_or(0);
        if !(0..=255).contains(&alternative) {
            return Err("GP5 alternate ending mask outside byte range".into());
        }
        let flags = 3
            | 64
            | if bars.contains("repeat_open") { 4 } else { 0 }
            | if bars.contains("repeat_close") { 8 } else { 0 }
            | if alternative > 0 { 16 } else { 0 }
            | if section.is_some() { 32 } else { 0 }
            | if bars.contains("double") { 128 } else { 0 };
        w.u(flags);
        w.u(a as u8);
        w.u(b as u8);
        if flags & 8 != 0 {
            let repeat = rows
                .iter()
                .map(|r| num(r, "repeat_count", 2))
                .max()
                .unwrap_or(2);
            if !(1..=127).contains(&repeat) {
                return Err("GP5 repeat count outside byte range".into());
            }
            w.u(repeat as u8)
        }
        if let Some(section) = section {
            w.string(section, &format!("measure {} marker", bi + 1))?;
            w.bytes.extend([255, 0, 0, 0])
        }
        w.u(f as u8);
        w.u(if mode == "minor" { 1 } else { 0 });
        if alternative > 0 {
            w.u(alternative as u8)
        }
        w.bytes.extend([2, 2, 2, 2]);
        if alternative == 0 {
            w.u(0)
        }
        w.u(super::navigation::feel(
            &rows.iter().map(|r| Some(*r)).collect::<Vec<_>>(),
        )?);
    }
    for (ti, t) in tracks.iter().enumerate() {
        if ti == 0 {
            w.u(0)
        }
        w.u(if txt(&t.part, "instrument", "") == "drums" {
            9
        } else {
            8
        });
        w.fixed(&t.name, &format!("track {} name", ti + 1), 40)?;
        w.i(t.strings.len() as i32);
        for i in 0..7 {
            w.i(t
                .strings
                .get(i)
                .map(|s| t.part["tuning"][*s].as_i64().unwrap() as i32)
                .unwrap_or(0));
        }
        w.i(1);
        w.i(t.channel + 1);
        w.i(t.channel + 1);
        w.i(24);
        w.i(num(&t.part, "capo", 0) as i32);
        w.bytes.extend([255, 0, 0, 0]);
        let mode = t.mode.as_str();
        let display = match mode {
            "tab" => 0xc9u16,
            "notation" => 0xcau16,
            "both" => 0xcbu16,
            _ => return Err("Invalid GP5 display mode".into()),
        };
        w.bytes.extend(display.to_le_bytes());
        w.z(3);
        let clef = if t.part["tuning"][*t.strings.last().unwrap()]
            .as_i64()
            .unwrap()
            < 35
        {
            12
        } else {
            0
        };
        w.i(clef);
        w.i(clef);
        w.i(100);
        w.z(12);
        w.i(-1);
        w.i(-1);
        w.i(-1);
        w.i(-1);
        w.z(4);
        w.string("", "effect")?;
        w.string("", "effect category")?;
    }
    w.u(0);
    let mut prev = vec![vec![HashMap::new(); 2]; tracks.len()];
    let mut velocities = vec![[95; 2]; tracks.len()];
    let mut detached = vec![];
    let mut duration_projections = vec![];

    let mut expected = vec![];
    for (bi, timing) in timeline.iter().enumerate() {
        let (a, b) = meter(timing)?;
        let length = a as i64 * 3840 / b as i64;
        for (ti, t) in tracks.iter().enumerate() {
            let row = t.staff["measures"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| num(m, "index", -1) == bi as i64);
            for vi in 0..2 {
                let source = t.voices.get(vi).and_then(|id| {
                    row.and_then(|r| {
                        r["voices"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|v| num(v, "voice", -1) == *id)
                    })
                });
                let mut beats = vec![];
                let mut cursor = 0;
                if let Some(v) = source {
                    let source_events = arr(v, "events")?;
                    for (source_event_index, e) in source_events.iter().enumerate() {
                        let start = num(e, "start", cursor);
                        if start < cursor {
                            return Err(format!(
                                "GP5 overlapping event in measure {} voice {vi}",
                                bi + 1
                            ));
                        }
                        beats.extend(rests(start - cursor)?);
                        let mut e = e.clone();
                        let mut kept = vec![];
                        for n in arr(&e, "notes")? {
                            let s = num(n, "string", 0);
                            if s < 1 {
                                return Err(
                                    "GP5 pitch-only fingering is not yet implemented".into()
                                );
                            }
                            if s as usize > arr(&t.part, "tuning")?.len() {
                                return Err("GP5 string has no tuning".into());
                            }
                            if t.strings.contains(&(s as usize - 1)) {
                                kept.push(n.clone())
                            }
                        }
                        e["notes"] = json!(kept);
                        // String partitions turn filtered-out notes into rests, as the
                        // legacy whole-score projector does; do not rewrite true empty beats.
                        if t.part["tuning"].as_array().unwrap().len() > 7
                            && arr(&e, "notes")?.is_empty()
                        {
                            e["status"] = json!("rest");
                        }
                        let exact = duration_ticks(&e["duration"])?;
                        let projected_exact = projected_duration(&e)?;
                        if projected_exact.denominator != 1
                            && source_event_index + 1 < source_events.len()
                        {
                            return Err("GP5 cannot preserve the following integer onset after a fractional tuplet, matching the legacy writer".into());
                        }
                        let projected = duration(&e)?.3;
                        cursor = start + projected;
                        if exact.numerator * projected_exact.denominator
                            != projected_exact.numerator * exact.denominator
                        {
                            duration_projections.push(json!({"track":ti+1,"measure":bi+1,"voice":vi+1,"source_start":start,"source_duration":e["duration"],"source_ticks":{"numerator":exact.numerator.to_string(),"denominator":exact.denominator.to_string()},"written_ticks":projected,"reason":"Legacy GP5 duration projection retains only one dot flag; score IR remains unchanged"}));
                        }
                        beats.push(e);
                    }
                }
                // A missing whole voice-group gets V0's full-bar placeholder; an
                // absent V1 or a present under/overfull voice is not padded/trimmed.
                let group_present = row.is_some_and(|r| {
                    r["voices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|v| t.voices.contains(&num(v, "voice", -1)))
                });
                if vi == 0 && !group_present {
                    beats = rests(length)?;
                }
                w.i(beats.len() as i32);
                let mut expected_cursor = 0;
                for (ei, e) in beats.iter().enumerate() {
                    let before = prev[ti][vi].clone();
                    let ns = arr(e, "notes")?;
                    if !ns.is_empty()
                        && e["start"]
                            .as_i64()
                            .is_some_and(|start| start != expected_cursor)
                    {
                        return Err(format!("GP5 readback would change note onset in measure {} voice {} (legacy empty-beat timing)",bi+1,vi+1));
                    }
                    let mut expected_notes = vec![];
                    for n in ns {
                        let source = num(n, "string", 0) as usize - 1;
                        let local = t
                            .strings
                            .iter()
                            .position(|s| *s == source)
                            .ok_or("Partition mismatch")?
                            + 1;
                        let f = num(n, "fret", 0);
                        let dead = n["fret"] == "x" || effects(n).contains(&"dead");
                        let typ = if dead {
                            3
                        } else if effects(n).contains(&"tie") && before.get(&local) == Some(&f) {
                            2
                        } else {
                            1
                        };
                        expected_notes.push(json!([local, f, typ]));
                    }
                    expected_notes.sort_by_key(|n| n[0].as_i64());
                    let dt = duration(e)?.3;
                    let exact_duration = projected_duration(e)?;
                    expected.push(json!({"measure":bi,"track":ti,"voice":vi,"start":expected_cursor,"duration":dt,"duration_fraction":[exact_duration.numerator.to_string(),exact_duration.denominator.to_string()],"status":match txt(e,"status","normal"){"empty"=>0,"rest"=>2,_=>1},"notes":expected_notes}));
                    if txt(e, "status", "normal") != "empty" {
                        expected_cursor += dt;
                    }
                    beat(
                        &mut w,
                        e,
                        t,
                        &mut prev[ti][vi],
                        &mut detached,
                        &json!({"track":ti+1,"measure":bi+1,"voice":vi+1,"beat":ei+1}),
                        &mut velocities[ti][vi],
                    )?;
                }
            }
            w.u(0);
        }
    }
    super::readback::verify(&w.bytes, &json!(expected))?;
    let report = json!({"duration_projections":duration_projections,"velocity_quantization":w.velocity_quantization,"readback_verified":true,"encoding":"cp936","replacements":w.replacements,"detached_ties":detached,"tracks":tracks.len(),"native":true,"readback_scope":["notes","ties","durations","voice_timing"],"mapping":tracks.iter().enumerate().map(|(i,t)|json!({"track":i+1,"part_id":t.part["id"],"staff_id":t.staff["id"],"voices":t.voices,"strings":t.strings.iter().map(|s|s+1).collect::<Vec<_>>(),"tuning":t.strings.iter().map(|s|t.part["tuning"][*s].clone()).collect::<Vec<_>>()})).collect::<Vec<_>>()});
    Ok((w.bytes, report))
}
pub(super) fn write(input: &Value, output: &Path) -> R<Value> {
    let (bytes, report) = encode(input)?;
    let report_path = output.with_file_name(format!(
        "{}.encoding.json",
        output
            .file_name()
            .ok_or("Output filename missing")?
            .to_string_lossy()
    ));
    // Encoding and readback have already succeeded before any destination changes.
    // Write the report first, so an encoding-report failure never publishes a GP5.
    atomic_write(
        &report_path,
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    atomic_write(output, &bytes)?;
    Ok(json!({"format":"gp5","path":output,"report_path":report_path,"report":report}))
}
