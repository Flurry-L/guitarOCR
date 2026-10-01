//! GP5 note and beat technique payloads. The public M2 effect strings remain
//! authoritative; payload construction rejects unknown values rather than guessing.
use super::*;
#[derive(Default)]
pub(super) struct NotePayload {
    pub note_flags: u8,
    pub bytes: Vec<u8>,
}
fn int(s: &str) -> R<i64> {
    s.parse()
        .map_err(|_| format!("Invalid technique integer {s:?}"))
}
fn byte(n: i64) -> R<u8> {
    u8::try_from(n).map_err(|_| "Technique byte out of range".into())
}
fn i32_bytes(out: &mut Vec<u8>, n: i64) -> R<()> {
    out.extend(
        i32::try_from(n)
            .map_err(|_| "Technique integer overflow")?
            .to_le_bytes(),
    );
    Ok(())
}
pub(super) fn ornament_fret(token: &str, open: i64, fret: i64, max: i64) -> R<i64> {
    if token == "x" {
        return Ok(fret.max(0));
    }
    let (f, p) = if let Some(rest) = token.strip_prefix('f') {
        if let Some((f, p)) = rest.split_once('p') {
            (Some(int(f)?), Some(int(p)?))
        } else {
            (Some(int(rest)?), None)
        }
    } else if let Some(p) = token.strip_prefix('p') {
        (None, Some(int(p)?))
    } else {
        return Err("Invalid ornament position".into());
    };
    if let (Some(f), Some(p)) = (f, p) {
        if open + f != p {
            return Err("Ornament pitch/fret mismatch on assigned string".into());
        }
    }
    let f = match f {
        Some(f) => f,
        None => p.ok_or("Missing ornament position")? - open,
    };
    if !(0..=max).contains(&f) {
        return Err("Ornament cannot fit assigned string".into());
    }
    Ok(f)
}
pub(super) fn bend(effect: &str) -> R<Vec<u8>> {
    let fields = effect.split(':').collect::<Vec<_>>();
    let name = fields.get(1).copied().unwrap_or("bend");
    let kind = match name.to_ascii_lowercase().as_str() {
        "none" => 0,
        "bend" => 1,
        "bendrelease" => 2,
        "bendreleasebend" => 3,
        "prebend" => 4,
        "prebendrelease" => 5,
        "dip" => 6,
        "dive" => 7,
        "releaseup" => 8,
        "inverteddip" => 9,
        "return_" | "return" => 10,
        "releasedown" => 11,
        _ => return Err(format!("Unknown bend type {name}")),
    };
    let value = fields.get(2).map(|s| int(s)).transpose()?.unwrap_or(100);
    let pv = ((value as f64 / 25.).round() as i64).max(0);
    let points = match kind {
        2 => vec![(0, 0), (30, pv), (60, 0)],
        3 => vec![(0, 0), (20, pv), (40, 0), (60, pv)],
        4 => vec![(0, pv), (60, pv)],
        5 => vec![(0, pv), (30, pv), (60, 0)],
        _ => vec![(0, 0), (30, pv), (60, pv)],
    };
    let mut out = vec![kind];
    i32_bytes(&mut out, value)?;
    i32_bytes(&mut out, points.len() as i64)?;
    for (x, y) in points {
        i32_bytes(&mut out, x)?;
        i32_bytes(&mut out, y * 25)?;
        out.push(0)
    }
    Ok(out)
}
pub(super) fn note(note: &Value, open: i64, fret: i64, percussion: bool) -> R<NotePayload> {
    let max = if percussion { 99 } else { 30 };
    let mut plan = NotePayload::default();
    let (mut f1, mut f2) = (0u8, 0u8);
    let (mut bend_bytes, mut grace, mut trem, mut harmonic, mut trill) =
        (None, None, None, None, None);
    let mut slides = 0u8;
    for ef in effects(note) {
        let fs = ef.split(':').collect::<Vec<_>>();
        match ef {
            "tie" | "dead" | "tap" | "accswap" => {}
            "ghost" => plan.note_flags |= 4,
            "heavy" => plan.note_flags |= 2,
            "accent" => plan.note_flags |= 64,
            "hammer" => f1 |= 2,
            "let" | "letRing" => f1 |= 8,
            "stacc" => f2 |= 1,
            "pm" | "palmMute" => f2 |= 2,
            "vib" | "vibrato" => f2 |= 64,
            "sl" => slides |= 2,
            "ss" => slides |= 1,
            "sib" => slides |= 16,
            "sia" => slides |= 32,
            "sod" => slides |= 4,
            "sou" => slides |= 8,
            _ if ef.starts_with("vel:") => {}
            _ if ef.starts_with("slide:") => {
                slides |= match fs[1] {
                    "shiftSlideTo" => 1,
                    "legatoSlideTo" => 2,
                    "outDownwards" => 4,
                    "outUpwards" => 8,
                    "intoFromBelow" => 16,
                    "intoFromAbove" => 32,
                    _ => return Err(format!("Unknown slide effect {ef}")),
                }
            }
            _ if ef.starts_with("bend:") => {
                bend_bytes = Some(bend(ef)?);
                f1 |= 1;
            }
            "grace" => {
                grace = Some(vec![byte(fret.max(0))?, 6, 0, 2, 0]);
                f1 |= 16;
            }
            "trill" => {
                if fret + 1 > max {
                    return Err("Trill auxiliary pitch outside assigned string".into());
                }
                trill = Some(vec![byte(fret + 1)?, 1]);
                f2 |= 32;
            }
            "trem" => {
                trem = Some(2);
                f2 |= 4;
            }
            _ if ef.starts_with("trem:") => {
                trem = Some(match int(fs[1])? {
                    8 => 1,
                    16 => 2,
                    32 => 3,
                    _ => return Err("Unsupported GP5 tremolo duration".into()),
                });
                f2 |= 4;
            }
            _ if ef.starts_with("grace:") => {
                let f = ornament_fret(fs.get(1).ok_or("Missing grace position")?, open, fret, max)?;
                let mut fs = fs.clone();
                if fs.len() == 5 {
                    fs.insert(2, "32");
                }
                if fs.len() != 6 {
                    return Err("Malformed grace effect".into());
                }
                let duration = int(fs[2])?;
                if ![8, 16, 32, 64, 128].contains(&duration) {
                    return Err("Unsupported grace duration".into());
                }
                let transition = match fs[3] {
                    "none" => 0,
                    "slide" => 1,
                    "bend" => 2,
                    "hammer" => 3,
                    _ => return Err("Unknown grace transition".into()),
                };
                grace = Some(vec![
                    byte(f)?,
                    6,
                    transition,
                    (7 - duration.ilog2()) as u8,
                    if fs[5] == "1" { 1 } else { 0 } | if fs[4] == "1" { 2 } else { 0 },
                ]);
                f1 |= 16;
            }
            _ if ef.starts_with("trill:") => {
                let f = ornament_fret(fs.get(1).ok_or("Missing trill position")?, open, fret, max)?;
                let duration = if fs.len() == 3 { int(fs[2])? } else { 32 };
                let period = match duration {
                    16 => 1,
                    32 => 2,
                    64 => 3,
                    _ => return Err("Unsupported GP5 trill duration".into()),
                };
                trill = Some(vec![byte(f)?, period]);
                f2 |= 32;
            }
            _ if ef.starts_with("harm:") => {
                let payload = match fs.get(1).copied().unwrap_or("") {
                    "natural" => vec![1],
                    "pinch" => vec![4],
                    "semi" => vec![5],
                    "tapped" => vec![
                        3,
                        byte(if fs.len() >= 3 {
                            int(fs[2])?
                        } else {
                            (fret + 12).clamp(0, 255)
                        })?,
                    ],
                    "artificial" => {
                        if fs.len() >= 5 {
                            let just = int(fs[2])?.rem_euclid(12);
                            let acc = int(fs[3])?;
                            let octave = int(fs[4])?;
                            if !(-2..=2).contains(&acc) || !(0..=4).contains(&octave) {
                                return Err("Invalid artificial harmonic pitch/octave".into());
                            }
                            vec![2, just as u8, acc as u8, octave as u8]
                        } else {
                            let pitch = (open + fret).rem_euclid(12);
                            let acc = if [1, 3, 6, 8, 10].contains(&pitch) {
                                1
                            } else {
                                0
                            };
                            vec![2, (pitch - acc) as u8, acc as u8, 1]
                        }
                    }
                    _ => return Err("Unknown GP5 harmonic".into()),
                };
                harmonic = Some(payload);
                f2 |= 16;
            }
            _ => return Err(format!("Unknown GP5 note technique {ef:?}")),
        }
    }
    if slides != 0 {
        f2 |= 8
    }
    if f1 != 0 || f2 != 0 {
        plan.note_flags |= 8;
        plan.bytes.extend([f1, f2]);
        if let Some(b) = bend_bytes {
            plan.bytes.extend(b)
        }
        if let Some(g) = grace {
            plan.bytes.extend(g)
        }
        if let Some(t) = trem {
            plan.bytes.push(t)
        }
        if slides != 0 {
            plan.bytes.push(slides)
        }
        if let Some(h) = harmonic {
            plan.bytes.extend(h)
        }
        if let Some(t) = trill {
            plan.bytes.extend(t)
        }
    }
    Ok(plan)
}
pub(super) fn beat(event: &Value) -> R<Vec<u8>> {
    let (mut f1, mut f2) = (0u8, 0u8);
    let (mut slap, mut stroke, mut pick) = (0u8, 0u8, 0u8);
    for ef in effects(event) {
        match ef {
            "fade" => f1 |= 16,
            "rasg" => f2 |= 1,
            "pick_up" => {
                f2 |= 2;
                pick = 1
            }
            "pick_down" => {
                f2 |= 2;
                pick = 2
            }
            "stroke_up" => {
                f1 |= 64;
                stroke = 1
            }
            "stroke_down" => {
                f1 |= 64;
                stroke = 2
            }
            _ if ef.starts_with("slap:") => {
                slap = match ef.split_once(':').unwrap().1 {
                    "none" => 0,
                    "tapping" => 1,
                    "slapping" => 2,
                    "popping" => 3,
                    _ => return Err("Unknown slap technique".into()),
                };
                if slap != 0 {
                    f1 |= 32
                }
            }
            _ => {}
        }
    }
    if arr(event, "notes")?
        .iter()
        .any(|n| effects(n).contains(&"tap"))
    {
        slap = 1;
        f1 |= 32
    }
    if f1 == 0 && f2 == 0 {
        return Ok(vec![]);
    }
    let mut out = vec![f1, f2];
    if f1 & 32 != 0 {
        out.push(slap)
    }
    if f1 & 64 != 0 {
        out.extend(if stroke == 1 { [5, 0] } else { [0, 5] })
    }
    if f2 & 2 != 0 {
        out.push(pick)
    }
    Ok(out)
}
