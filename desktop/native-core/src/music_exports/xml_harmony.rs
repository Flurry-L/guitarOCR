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
        let fields = d.split(':').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err("Invalid chord diagram".into());
        }
        let parse = |s: &str| {
            s.parse::<i64>()
                .map_err(|_| "Invalid chord diagram integer".to_owned())
        };
        let base = parse(fields[0])?;
        let frets = fields[1]
            .split('/')
            .map(|s| {
                if s == "x" {
                    Ok(None)
                } else {
                    parse(s).map(Some)
                }
            })
            .collect::<R<Vec<_>>>()?;
        let fingers = fields[2]
            .split('/')
            .map(|s| {
                if s == "-" {
                    Ok(None)
                } else {
                    parse(s).map(Some)
                }
            })
            .collect::<R<Vec<_>>>()?;
        if !(1..=12).contains(&frets.len())
            || frets.len() != fingers.len()
            || !(1..=36).contains(&base)
            || frets
                .iter()
                .flatten()
                .any(|f| !(0..=36).contains(f) || *f > 0 && *f < base)
            || fingers.iter().flatten().any(|f| !(0..=4).contains(f))
        {
            return Err("Invalid chord diagram range".into());
        }
        let barres = if fields[3] == "-" {
            vec![]
        } else {
            fields[3]
                .split(';')
                .map(|s| s.split('/').map(parse).collect::<R<Vec<_>>>())
                .collect::<R<Vec<_>>>()?
        };
        let mut positions = BTreeMap::new();
        for (i, f) in frets.iter().enumerate() {
            if let Some(f) = f {
                positions.insert((i, *f), fingers[i]);
            }
        }
        for b in &barres {
            if b.len() != 3
                || b[0] < base
                || b[0] > 36
                || b[1] < 0
                || b[1] >= b[2]
                || b[2] >= frets.len() as i64
            {
                return Err("Invalid chord barre".into());
            }
            positions.entry((b[1] as usize, b[0])).or_insert(None);
            positions.entry((b[2] as usize, b[0])).or_insert(None);
        }
        out += &format!(
            "<frame>{}{}{}</frame-placeholder>",
            tag("frame-strings", frets.len())?,
            tag(
                "frame-frets",
                4.max(frets.iter().flatten().max().copied().unwrap_or(0) - base + 1)
            )?,
            tag("first-fret", base)?
        );
        out = out.replace("</frame-placeholder>", "");
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
