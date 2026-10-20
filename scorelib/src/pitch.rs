//! Written/sounding pitch conversion and key spelling shared by data generation and export.
use crate::score;
use serde_json::{json, Value};
type Result<T> = std::result::Result<T, String>;
fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value[key].as_array().map(Vec::as_slice).unwrap_or(&[])
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
pub fn key_parts(name: &str) -> Result<(i64, bool)> {
    if let Some((fifths, mode)) = name.split_once(':') {
        let fifths = fifths.parse::<i64>().map_err(|_| "Invalid key fifths")?;
        if (-8..=8).contains(&fifths) && ["major", "minor"].contains(&mode) {
            return Ok((fifths, mode == "minor"));
        }
    }
    let identity = |value: &str| {
        value
            .replace("Major", "")
            .replace("Minor", "")
            .replace("Sharp", "#")
            .replace("Flat", "b")
            .replace('♯', "#")
            .replace('♭', "b")
    };
    let names = if name.contains("Minor") {
        &MINOR_KEYS
    } else {
        &MAJOR_KEYS
    };
    let wanted = identity(name);
    names
        .iter()
        .position(|candidate| identity(candidate) == wanted)
        .map(|index| (index as i64 - 8, name.contains("Minor")))
        .ok_or_else(|| format!("Unknown key signature: {name}"))
}
pub fn key_name(fifths: i64, minor: bool) -> Result<&'static str> {
    if !(-8..=8).contains(&fifths) {
        return Err("Key accidental count outside supported range".into());
    }
    Ok(if minor {
        MINOR_KEYS[(fifths + 8) as usize]
    } else {
        MAJOR_KEYS[(fifths + 8) as usize]
    })
}
pub fn transpose_key(name: &str, shift: i64) -> Result<String> {
    if shift.rem_euclid(12) == 0 {
        return Ok(name.into());
    }
    let (fifths, minor) = key_parts(name)?;
    Ok(key_name(transpose_fifths(fifths, shift), minor)?.into())
}

/// Spell a transposed key using at most seven accidentals.
pub fn transpose_fifths(fifths: i64, shift: i64) -> i64 {
    if shift.rem_euclid(12) == 0 {
        return if fifths < -7 {
            fifths + 12
        } else if fifths > 7 {
            fifths - 12
        } else {
            fifths
        };
    }
    let tonic = (7 * fifths + shift.rem_euclid(12)).rem_euclid(12);
    (-7i64..=7)
        .filter(|f| (7 * f).rem_euclid(12) == tonic)
        .min_by_key(|f| (f.abs(), (f - fifths).abs()))
        .unwrap()
}
pub fn convert_pitch(
    target: &str,
    context: &Value,
    mode: &str,
    to_written: bool,
) -> Result<String> {
    let mut m = score::parse_measure_target(target)?;
    convert_measure_pitch(&mut m, context, if to_written { -1 } else { 1 })?;
    score::format_measure_target(&m, mode, true)
}

pub fn convert_measure_pitch(m: &mut Value, context: &Value, direction: i64) -> Result<()> {
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
                // A partial export can retain a visible TAB position without
                // a notation pitch. Its string/fret already identifies sound.
                if let Some(pitch) = note["pitch"].as_i64() {
                    let pitch = pitch + shift;
                    if !(0..=127).contains(&pitch) {
                        return Err("Transposed pitch is outside MIDI range".into());
                    }
                    note["pitch"] = json!(pitch);
                } else if !(note["string"].is_i64()
                    && (note["fret"].is_i64() || note["fret"] == "x"))
                {
                    return Err("Missing written pitch".into());
                }
                for effect in note["effects"].as_array_mut().unwrap() {
                    let raw = effect.as_str().unwrap();
                    let mut parts: Vec<String> = raw.split(':').map(str::to_owned).collect();
                    if parts.len() > 1 && matches!(parts[0].as_str(), "grace" | "trill") {
                        if let (fret, Some(p)) = score::ornament_position(&parts[1])? {
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
    Ok(())
}
