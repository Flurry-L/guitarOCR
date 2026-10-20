//! Printed chord diagrams and explicit review decisions shared by recognition and export.
use regex::Regex;
use serde_json::{json, Value};

pub fn decision(annotation: &Value) -> Option<&str> {
    let review = &annotation["chord_review"];
    (review["source"] == "manual")
        .then(|| review["status"].as_str())
        .flatten()
}

pub fn pending(annotation: &Value) -> bool {
    if matches!(decision(annotation), Some("confirmed" | "rejected")) {
        return false;
    }
    let parsed = &annotation["parsed"];
    let evidence = &annotation["chord_evidence"];
    let has_candidate = [
        &parsed["diagram"],
        &evidence["diagram"],
        &evidence["proposed_diagram"],
        &evidence["secondary_diagram"],
        &evidence["original_diagram"],
        &evidence["rejected_diagram_candidate"],
    ]
    .iter()
    .any(|value| value.is_object());
    let is_chord = matches!(parsed["kind"].as_str(), Some("chord" | "chord_diagram"));
    let unresolved = evidence["review_required"] == true
        || evidence["name_needs_review"] == true
        || evidence["accepted"]["geometry"] == false
        || evidence["accepted"]["full"] == false
        // A fully accepted dedicated reading resolves the older text-model
        // geometry; its differences remain audit evidence, not a pending choice.
        || (evidence["accepted"]["full"] != true
            && evidence["conflict_fields"]
                .as_array()
                .is_some_and(|fields| !fields.is_empty()))
        || (has_candidate && parsed["diagram"].is_null());
    unresolved && (has_candidate || (is_chord && evidence["name_needs_review"] == true))
}

type R<T> = Result<T, String>;

pub fn normalize_diagram(value: &Value) -> Value {
    Diagram::from_annotation(value)
        .map(|diagram| diagram.value())
        .unwrap_or(Value::Null)
}

pub fn diagram_effect(value: &Value) -> R<String> {
    Ok(Diagram::from_annotation(value)?.effect())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Diagram {
    pub base: i64,
    pub frets: Vec<i64>,
    pub fingers: Vec<i64>,
    pub barres: Vec<[i64; 3]>,
}
pub(crate) fn parse_diagram(text: &str) -> R<Diagram> {
    let fs = text.split(':').collect::<Vec<_>>();
    if fs.len() != 4 {
        return Err("Invalid chord diagram fields".into());
    }
    let integer = |s: &str| {
        s.parse::<i64>()
            .ok()
            .filter(|n| *n >= 0)
            .ok_or_else(|| "Invalid diagram integer".to_owned())
    };
    let base = integer(fs[0])?;
    let frets = fs[1]
        .split('/')
        .map(|s| if s == "x" { Ok(-1) } else { integer(s) })
        .collect::<R<Vec<_>>>()?;
    let fingers = fs[2]
        .split('/')
        .map(|s| if s == "-" { Ok(-2) } else { integer(s) })
        .collect::<R<Vec<_>>>()?;
    let mut barres = vec![];
    if fs[3] != "-" {
        for b in fs[3].split(';') {
            let b = b.split('/').map(integer).collect::<R<Vec<_>>>()?;
            barres.push(b.try_into().map_err(|_| "Invalid chord barre")?);
        }
    }
    Diagram {
        base,
        frets,
        fingers,
        barres,
    }
    .validate()
}
impl Diagram {
    pub fn from_annotation(value: &Value) -> R<Self> {
        let number = |v: &Value| {
            v.as_i64()
                .filter(|n| *n >= 0)
                .ok_or_else(|| "Invalid diagram integer".to_owned())
        };
        let frets = value["frets"]
            .as_array()
            .ok_or("Missing array: frets")?
            .iter()
            .map(|v| if v == "x" { Ok(-1) } else { number(v) })
            .collect::<R<Vec<_>>>()?;
        let fingers = match value.get("fingers") {
            None | Some(Value::Null) => vec![-2; frets.len()],
            Some(v) => v
                .as_array()
                .ok_or("Chord fingers must be an array")?
                .iter()
                .map(|v| if v.is_null() { Ok(-2) } else { number(v) })
                .collect::<R<Vec<_>>>()?,
        };
        let barres = match value.get("barres") {
            None => vec![],
            Some(v) => v
                .as_array()
                .ok_or("Chord barres must be an array")?
                .iter()
                .map(|b| {
                    b.as_array()
                        .ok_or("Chord barre must be an array")?
                        .iter()
                        .map(number)
                        .collect::<R<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| "Invalid chord barre".to_owned())
                })
                .collect::<R<Vec<_>>>()?,
        };
        Self {
            base: value.get("base_fret").map(number).transpose()?.unwrap_or(1),
            frets,
            fingers,
            barres,
        }
        .validate()
    }

    fn validate(self) -> R<Self> {
        if !(1..=12).contains(&self.frets.len())
            || self.frets.len() != self.fingers.len()
            || !(1..=36).contains(&self.base)
            || self
                .frets
                .iter()
                .any(|n| !(-1..=36).contains(n) || *n > 0 && *n < self.base)
            || self
                .fingers
                .iter()
                .any(|n| *n != -2 && !(0..=4).contains(n))
        {
            return Err("Invalid chord diagram range".into());
        }
        if self.barres.iter().any(|b| {
            b[0] < self.base
                || b[0] > 36
                || b[1] < 0
                || b[1] >= b[2]
                || b[2] >= self.frets.len() as i64
        }) {
            return Err("Invalid chord barre".into());
        }
        Ok(self)
    }

    pub fn value(&self) -> Value {
        json!({
            "base_fret": self.base,
            "frets": self.frets.iter().map(|n| if *n < 0 { json!("x") } else { json!(n) }).collect::<Vec<_>>(),
            "fingers": self.fingers.iter().map(|n| if *n < 0 { Value::Null } else { json!(n) }).collect::<Vec<_>>(),
            "barres": self.barres,
        })
    }

    pub fn effect(&self) -> String {
        format!(
            "diagram:{}:{}:{}:{}",
            self.base,
            self.frets
                .iter()
                .map(|n| if *n < 0 {
                    "x".to_owned()
                } else {
                    n.to_string()
                })
                .collect::<Vec<_>>()
                .join("/"),
            self.fingers
                .iter()
                .map(|n| if *n < 0 {
                    "-".to_owned()
                } else {
                    n.to_string()
                })
                .collect::<Vec<_>>()
                .join("/"),
            if self.barres.is_empty() {
                "-".to_owned()
            } else {
                self.barres
                    .iter()
                    .map(|b| b.iter().map(i64::to_string).collect::<Vec<_>>().join("/"))
                    .collect::<Vec<_>>()
                    .join(";")
            }
        )
    }

    pub fn project(&self, high_to_low: &[usize]) -> Self {
        let indexes = high_to_low
            .iter()
            .rev()
            .map(|i| self.frets.len() as i64 - 1 - *i as i64)
            .collect::<Vec<_>>();
        let at = |source: &[i64], i: i64, missing: i64| {
            if i >= 0 {
                source.get(i as usize).copied().unwrap_or(missing)
            } else {
                missing
            }
        };
        let mut barres = vec![];
        for b in &self.barres {
            let covered = indexes
                .iter()
                .enumerate()
                .filter(|(_, i)| b[1] <= **i && **i <= b[2])
                .map(|(j, _)| j as i64)
                .collect::<Vec<_>>();
            if covered.len() > 1 {
                barres.push([b[0], covered[0], *covered.last().unwrap()]);
            }
        }
        Self {
            base: self.base,
            frets: indexes.iter().map(|i| at(&self.frets, *i, -1)).collect(),
            fingers: indexes.iter().map(|i| at(&self.fingers, *i, -2)).collect(),
            barres,
        }
    }
}

pub fn chord_key(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace('♭', "b")
        .replace('♯', "#")
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
