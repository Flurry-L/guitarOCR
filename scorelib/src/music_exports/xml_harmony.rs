use super::*;
use std::collections::BTreeMap;
pub(super) fn render(event: &Value, staff: usize) -> R<String> {
    let names = effects(event)
        .into_iter()
        .filter_map(|s| s.strip_prefix("chord:"))
        .collect::<Vec<_>>();
    let diagrams = effects(event)
        .into_iter()
        .filter_map(|s| s.strip_prefix("diagram:"))
        .collect::<Vec<_>>();
    if names.len() > 1 || diagrams.len() > 1 {
        return Err("Multiple chord names/diagrams at one onset are not supported".into());
    }
    if names.is_empty() && diagrams.is_empty() {
        return Ok(String::new());
    }
    let name = percent_decode(names.first().copied().unwrap_or(""))?;
    let normalized = name
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace('♭', "b")
        .replace('♯', "#");
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^([A-Ga-g])((?:##|bb|#|b|x)?)(.*?)(?:/([A-Ga-g])((?:##|bb|#|b|x)?))?$")
            .unwrap()
    });
    let alter = |s: &str| match s {
        "#" => 1,
        "b" => -1,
        "##" | "x" => 2,
        "bb" => -2,
        _ => 0,
    };
    let mut out = "<harmony>".to_owned();
    let mut bass = None;
    let suffix;
    if let Some(c) = RE.captures(&normalized) {
        out += &format!(
            "<root>{}{}</root>",
            tag("root-step", c[1].to_uppercase())?,
            if c[2].is_empty() {
                String::new()
            } else {
                tag("root-alter", alter(&c[2]))?
            }
        );
        suffix = c[3].to_owned();
        if let Some(b) = c.get(4) {
            bass = Some((
                b.as_str().to_uppercase(),
                alter(c.get(5).map(|s| s.as_str()).unwrap_or("")),
            ))
        }
    } else {
        out += &tag("function", &name)?;
        suffix = String::new();
    }
    let kind = if !RE.is_match(&normalized) {
        "other"
    } else {
        match suffix.as_str() {
            "" => "major",
            "m" | "min" => "minor",
            "7" => "dominant",
            "maj7" | "M7" => "major-seventh",
            "m7" => "minor-seventh",
            "dim" => "diminished",
            "dim7" => "diminished-seventh",
            "aug" | "+" => "augmented",
            "sus4" => "suspended-fourth",
            "sus2" => "suspended-second",
            "6" => "major-sixth",
            "m6" => "minor-sixth",
            "9" => "dominant-ninth",
            "maj9" => "major-ninth",
            "m9" => "minor-ninth",
            "11" => "dominant-11th",
            "13" => "dominant-13th",
            "m7b5" => "half-diminished",
            "5" => "power",
            _ => "other",
        }
    };
    out += &format!("<kind text=\"{}\">{kind}</kind>", esc(&suffix)?);
    if let Some((step, a)) = bass {
        out += &format!(
            "<bass>{}{}</bass>",
            tag("bass-step", step)?,
            if a != 0 {
                tag("bass-alter", a)?
            } else {
                String::new()
            }
        );
    }
    if let Some(d) = diagrams.first() {
        let diagram = crate::chords::parse_diagram(d)?;
        let crate::chords::Diagram {
            base,
            frets,
            fingers,
            barres,
        } = diagram;
        let mut positions = BTreeMap::new();
        for (i, f) in frets.iter().enumerate() {
            if *f >= 0 {
                positions.insert((i, *f), (fingers[i] >= 0).then_some(fingers[i]));
            }
        }
        for b in &barres {
            positions.entry((b[1] as usize, b[0])).or_insert(None);
            positions.entry((b[2] as usize, b[0])).or_insert(None);
        }
        out += &format!(
            "<frame>{}{}{}",
            tag("frame-strings", frets.len())?,
            tag(
                "frame-frets",
                4.max(frets.iter().max().copied().unwrap_or(0) - base + 1)
            )?,
            tag("first-fret", base)?
        );
        for ((i, f), finger) in positions {
            out += &format!(
                "<frame-note>{}{}",
                tag("string", frets.len() - i)?,
                tag("fret", f)?
            );
            if let Some(finger) = finger {
                out += &tag("fingering", finger)?
            }
            for b in &barres {
                if b[0] == f && (b[1] as usize == i || b[2] as usize == i) {
                    out += &format!(
                        "<barre type=\"{}\"/>",
                        if b[1] as usize == i { "start" } else { "stop" }
                    );
                }
            }
            out += "</frame-note>";
        }
        out += "</frame>";
    }
    out += &tag("staff", staff)?;
    out += "</harmony>";
    Ok(out)
}
