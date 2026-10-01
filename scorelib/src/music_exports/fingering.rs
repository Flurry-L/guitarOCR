//! Interval-constrained native fingering. Tie chains own one string until their
//! last continuation; the solver never steals that string for intervening notes.
use super::*;
#[derive(Clone)]
struct Chain {
    start: usize,
    end: usize,
    notes: Vec<(usize, usize, usize)>,
    candidates: Vec<usize>,
}
fn solve(
    order: &[usize],
    _at: usize,
    chains: &[Chain],
    assigned: &mut [Option<usize>],
    budget: &mut usize,
) -> bool {
    if order.is_empty() {
        return true;
    }
    let mut at = 0;
    let mut choices = vec![0; order.len()];
    loop {
        let ci = order[at];
        let c = &chains[ci];
        let mut found = false;
        while choices[at] < c.candidates.len() {
            let s = c.candidates[choices[at]];
            choices[at] += 1;
            let mut conflict = false;
            for (j, other) in chains.iter().enumerate() {
                if *budget == 0 {
                    return false;
                }
                *budget -= 1;
                if assigned[j] == Some(s) && c.start <= other.end && other.start <= c.end {
                    conflict = true;
                    break;
                }
            }
            if !conflict {
                assigned[ci] = Some(s);
                found = true;
                break;
            }
        }
        if found {
            at += 1;
            if at == order.len() {
                return true;
            }
            choices[at] = 0;
        } else {
            assigned[ci] = None;
            choices[at] = 0;
            if at == 0 {
                return false;
            }
            at -= 1;
            assigned[order[at]] = None;
        }
    }
}

pub(super) fn assign(score: &mut Value) -> R<()> {
    for p in score["parts"].as_array_mut().ok_or("Missing score parts")? {
        let instrument = txt(p, "instrument", "guitar").to_owned();
        let virtual_slots = instrument == "pitched" || instrument == "drums";
        if virtual_slots {
            let mut pitches = std::collections::BTreeSet::new();
            for s in arr(p, "staves")? {
                for m in arr(s, "measures")? {
                    for v in arr(m, "voices")? {
                        for e in arr(v, "events")? {
                            for n in arr(e, "notes")? {
                                let pitch = n["pitch"]
                                    .as_i64()
                                    .ok_or("Pitched/percussion note missing pitch")?;
                                if !(0..=127).contains(&pitch) {
                                    return Err("Pitch outside MIDI range".into());
                                }
                                pitches.insert(pitch);
                                for ef in effects(n) {
                                    if ef == "trill" {
                                        pitches.insert(pitch + 1);
                                    } else if ef.starts_with("grace:") || ef.starts_with("trill:") {
                                        if let Some(pos) = ef.split(':').nth(1) {
                                            if let Some((_, p)) = pos.split_once('p') {
                                                pitches.insert(
                                                    p.parse::<i64>()
                                                        .map_err(|_| "Invalid ornament pitch")?,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Stable 31-semitone bands avoid GP's silent >30 melodic-fret clamping.
            // Seven slots per band preserve all simultaneous notes; bands become tracks.
            let mut bases = vec![];
            if instrument == "drums" {
                bases = vec![0; 7]
            } else if let Some(legacy) = legacy_storage_bases(p)? {
                bases = legacy;
            } else {
                let mut covered = -1;
                for pitch in pitches {
                    if pitch > covered {
                        bases.extend([pitch; 7]);
                        covered = pitch + 30;
                    }
                }
                if bases.is_empty() {
                    bases = vec![60; 7]
                }
            }
            p["tuning"] = json!(bases);
            p["capo"] = json!(0);
        }
        let tuning = arr(p, "tuning")?
            .iter()
            .map(|v| v.as_i64().ok_or_else(|| "Invalid tuning".to_owned()))
            .collect::<R<Vec<_>>>()?;
        for staff in p["staves"].as_array_mut().ok_or("Missing staves")? {
            let ms = staff["measures"].as_array_mut().ok_or("Missing measures")?;
            ms.sort_by_key(|m| num(m, "index", 0));
            let voices = ms
                .iter()
                .flat_map(|m| m["voices"].as_array().into_iter().flatten())
                .map(|v| num(v, "voice", 0))
                .collect::<std::collections::BTreeSet<_>>();
            for voice in voices {
                let mut chains: Vec<Chain> = vec![];
                let mut previous: HashMap<i64, Vec<usize>> = HashMap::new();
                let mut event_index = 0;
                for (mi, m) in ms.iter().enumerate() {
                    let Some(v) = m["voices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|v| num(v, "voice", 0) == voice)
                    else {
                        continue;
                    };
                    for (ei, e) in arr(v, "events")?.iter().enumerate() {
                        let mut tied_used = std::collections::HashSet::new();
                        let mut updated: HashMap<i64, Vec<usize>> = HashMap::new();
                        for (ni, n) in arr(e, "notes")?.iter().enumerate() {
                            let s = n["string"].as_i64();
                            let f = num(n, "fret", 0);
                            if !virtual_slots {
                                if let (Some(source), Some(explicit), Some(fret)) =
                                    (s, n["pitch"].as_i64(), n["fret"].as_i64())
                                {
                                    if source < 1
                                        || tuning.get(source as usize - 1).map(|t| t + fret)
                                            != Some(explicit)
                                    {
                                        return Err("Explicit pitch conflicts with string/fret; refusing fingering rewrite".into());
                                    }
                                }
                            }
                            let pitch = n["pitch"]
                                .as_i64()
                                .or_else(|| {
                                    s.and_then(|s| {
                                        if s > 0 {
                                            tuning.get(s as usize - 1).map(|p| p + f)
                                        } else {
                                            None
                                        }
                                    })
                                })
                                .ok_or("Cannot resolve note pitch for fingering")?;
                            let mut candidates = if let Some(s) = s.filter(|_| !virtual_slots) {
                                if s < 1 || s as usize > tuning.len() {
                                    return Err("String outside tuning".into());
                                }
                                vec![s as usize - 1]
                            } else {
                                tuning
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, base)| {
                                        if instrument == "drums" {
                                            (0..=99).contains(&(pitch - **base))
                                        } else {
                                            (0..=30).contains(&(pitch - **base))
                                        }
                                    })
                                    .map(|(i, _)| i)
                                    .collect::<Vec<_>>()
                            };
                            candidates.retain(|slot| {
                                let open = tuning[*slot];
                                let f = pitch - open;
                                let max = if instrument == "drums" { 99 } else { 30 };
                                effects(n).iter().all(|ef| {
                                    if *ef == "trill" {
                                        f + 1 <= max
                                    } else if ef.starts_with("grace:") || ef.starts_with("trill:") {
                                        super::gp5_techniques::ornament_fret(
                                            ef.split(':').nth(1).unwrap_or(""),
                                            open,
                                            f,
                                            max,
                                        )
                                        .is_ok()
                                    } else {
                                        true
                                    }
                                })
                            });
                            if candidates.is_empty() {
                                return Err(format!(
                                    "No playable native GP5 fingering for pitch {pitch}"
                                ));
                            }
                            let tie = effects(n).contains(&"tie");
                            let origin = if tie {
                                previous
                                    .get(&pitch)
                                    .and_then(|cs| cs.iter().find(|c| !tied_used.contains(*c)))
                                    .copied()
                            } else {
                                None
                            };
                            let ci = if let Some(ci) = origin {
                                tied_used.insert(ci);
                                chains[ci].end = event_index;
                                chains[ci].notes.push((mi, ei, ni));
                                chains[ci].candidates.retain(|s| candidates.contains(s));
                                if chains[ci].candidates.is_empty() {
                                    return Err("Tie chain requires incompatible strings; refusing pitch substitution".into());
                                }
                                ci
                            } else {
                                chains.push(Chain {
                                    start: event_index,
                                    end: event_index,
                                    notes: vec![(mi, ei, ni)],
                                    candidates,
                                });
                                chains.len() - 1
                            };
                            updated.entry(pitch).or_default().push(ci);
                        }
                        for (pitch, cs) in updated {
                            previous.insert(pitch, cs);
                        }
                        event_index += 1;
                    }
                }
                if chains.len() > 20000 {
                    return Err("Native GP5 fingering exceeds 20000-chain resource limit".into());
                }
                let mut order = (0..chains.len()).collect::<Vec<_>>();
                order.sort_by_key(|i| {
                    (
                        chains[*i].candidates.len(),
                        std::cmp::Reverse(chains[*i].end - chains[*i].start),
                        chains[*i].start,
                    )
                });
                let mut assigned = vec![None; chains.len()];
                let mut budget = 20_000_000;
                if !solve(&order, 0, &chains, &mut assigned, &mut budget) {
                    return Err(if budget==0{"Native GP5 fingering search limit reached; no approximate assignment written"}else{"No tie-safe GP5 string assignment exists for this tuning"}.into());
                }
                for (ci, c) in chains.iter().enumerate() {
                    let string = assigned[ci].unwrap();
                    for &(mi, ei, ni) in &c.notes {
                        let v = ms[mi]["voices"]
                            .as_array_mut()
                            .unwrap()
                            .iter_mut()
                            .find(|v| num(v, "voice", 0) == voice)
                            .unwrap();
                        let n = &mut v["events"][ei]["notes"][ni];
                        if let Some(pitch) = n["pitch"].as_i64() {
                            n["fret"] = json!(pitch - tuning[string]);
                        }
                        n["string"] = json!(string + 1);
                    }
                }
            }
        }
    }
    Ok(())
}
/// The same seven-slot coordinate search used by the existing pitched exporter.
/// Chord multiplicity contributes to cost, so this does not unnecessarily split
/// a wide piano staff merely because its overall range exceeds thirty semitones.
fn legacy_storage_bases(part: &Value) -> R<Option<Vec<i64>>> {
    let mut chords = vec![];
    let mut all = vec![];
    for staff in arr(part, "staves")? {
        for m in arr(staff, "measures")? {
            for v in arr(m, "voices")? {
                for e in arr(v, "events")? {
                    let mut chord = vec![];
                    for n in arr(e, "notes")? {
                        let pitch = n["pitch"].as_i64().ok_or("Missing pitched note")?;
                        let mut values = vec![pitch];
                        for ef in effects(n) {
                            if ef == "trill" {
                                values.push(pitch + 1);
                            } else if ef.starts_with("grace:") || ef.starts_with("trill:") {
                                if let Some((_, p)) =
                                    ef.split(':').nth(1).unwrap_or("").split_once('p')
                                {
                                    values.push(p.parse().map_err(|_| "Invalid ornament pitch")?);
                                }
                            }
                        }
                        let lo = *values.iter().min().unwrap();
                        let hi = *values.iter().max().unwrap();
                        chord.push((lo, hi));
                        all.extend([lo, hi]);
                    }
                    chord.sort();
                    chords.push(chord);
                }
            }
        }
    }
    let low = all.iter().min().copied().unwrap_or(60);
    let high = all.iter().max().copied().unwrap_or(60);
    if high - low <= 30 {
        return Ok(Some(vec![low; 7]));
    }
    let candidates = all
        .iter()
        .flat_map(|p| [low.max(p - 0), low.max(p - 15), low.max(p - 30)])
        .collect::<std::collections::BTreeSet<_>>();
    let cost = |bases: &[i64]| -> usize {
        chords
            .iter()
            .map(|chord| {
                let mut available = bases.to_vec();
                available.sort();
                let mut missing = 0;
                for &(lo, hi) in chord {
                    if let Some(i) = available.iter().position(|b| *b <= lo && hi <= *b + 30) {
                        available.remove(i);
                    } else {
                        missing += 1
                    }
                }
                missing
            })
            .sum()
    };
    let mut bases = (0..7)
        .map(|i| low.max(high - 30 - 12 * i))
        .collect::<Vec<_>>();
    let mut current = cost(&bases);
    for _ in 0..14 {
        if current == 0 {
            return Ok(Some(bases));
        }
        let mut best = current;
        let mut replacement = None;
        for slot in 0..7 {
            for &base in &candidates {
                let mut proposal = bases.clone();
                proposal[slot] = base;
                let c = cost(&proposal);
                if c < best {
                    best = c;
                    replacement = Some(proposal);
                }
            }
        }
        if let Some(next) = replacement {
            bases = next;
            current = best;
        } else {
            break;
        }
    }
    if current == 0 {
        Ok(Some(bases))
    } else {
        Ok(None)
    }
}
