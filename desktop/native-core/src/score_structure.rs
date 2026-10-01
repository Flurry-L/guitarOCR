//! Pure score-system repair and cross-page part identity matching.
//!
//! Mirrors layout/structure.py. Geometry supplies optional system indices;
//! visible names and row modes constrain repairs. This module never requests
//! model output, reads images, writes crops, or guesses missing instruments.
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;
use unicode_casefold::UnicodeCaseFold;

pub type Result<T> = std::result::Result<T, String>;
pub type StaffRows = Vec<Vec<Value>>;
pub type PageTemplate = Vec<(usize, usize, String)>;

#[derive(Clone, Debug)]
pub struct StructurePage {
    pub page: u64,
    pub structure: Value,
    pub rows: StaffRows,
}

#[derive(Clone, Debug)]
pub struct ResolvedSystem {
    pub page: u64,
    pub system: usize,
    /// (left-to-right boxes, global part index, zero-based staff index)
    pub members: Vec<(Vec<Value>, usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct ResolvedPages {
    pub parts: Vec<Value>,
    pub systems: Vec<ResolvedSystem>,
}

fn parts(value: &Value) -> Result<&[Value]> {
    value["parts"]
        .as_array()
        .filter(|p| !p.is_empty())
        .map(Vec::as_slice)
        .ok_or_else(|| "Structure needs parts and staff rows".into())
}
fn rows(value: &Value) -> Result<&[Value]> {
    value["rows"]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "Structure needs parts and staff rows".into())
}
fn triples(value: &Value) -> Option<Vec<[i64; 3]>> {
    value["rows"]
        .as_array()?
        .iter()
        .map(|row| {
            let row = row.as_array().filter(|r| r.len() == 3)?;
            Some([row[0].as_i64()?, row[1].as_i64()?, row[2].as_i64()?])
        })
        .collect()
}
fn require_triples(value: &Value) -> Result<Vec<[i64; 3]>> {
    triples(value).ok_or_else(|| "Structure indices must be integer triples".into())
}
fn group_indices(indices: impl IntoIterator<Item = i64>) -> Vec<(i64, Vec<usize>)> {
    let mut groups: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut slots = HashMap::new();
    for (index, system) in indices.into_iter().enumerate() {
        let slot = *slots.entry(system).or_insert_with(|| {
            groups.push((system, Vec::new()));
            groups.len() - 1
        });
        groups[slot].1.push(index);
    }
    groups
}
fn normalized_name(name: &str) -> String {
    name.case_fold()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Identity(String, String, Option<i64>, i64);
fn identity(part: &Value) -> Identity {
    Identity(
        normalized_name(part["name"].as_str().unwrap_or("")),
        part["instrument"].as_str().unwrap_or("").into(),
        part["strings"].as_i64(),
        part["program"].as_i64().unwrap_or(-1),
    )
}
fn grand_name(part: &Value) -> bool {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(r"(?i)\b(?:piano|keyboard|organ|harp)\b").unwrap())
        .is_match(part["name"].as_str().unwrap_or(""))
}
fn grand_program(part: &Value) -> bool {
    part["instrument"] == "pitched"
        && part["program"]
            .as_i64()
            .is_some_and(|p| (0..8).contains(&p) || (16..21).contains(&p) || p == 46)
}
fn fretted(part: &Value) -> bool {
    part["instrument"] == "guitar" || part["instrument"] == "bass"
}
fn complementary(modes: &[String], indices: &[usize]) -> bool {
    indices.len() == 2
        && ((modes[indices[0]] == "notation" && modes[indices[1]] == "tab")
            || (modes[indices[0]] == "tab" && modes[indices[1]] == "notation"))
}
fn repeated_rows(groups: &[(i64, Vec<usize>)], template: &[(usize, usize)]) -> Value {
    json!(groups
        .iter()
        .flat_map(|(system, _)| template.iter().map(move |&(p, s)| json!([system, p, s])))
        .collect::<Vec<_>>())
}

/// Printed instrument names take precedence over a conflicting model program.
pub fn program_from_visible_name(name: &str) -> Option<i64> {
    let lowered = name.to_lowercase();
    let tokens: Vec<&str> = lowered
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let text = tokens.join(" ");
    let abbreviations = [
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
    ];
    if let Some((_, program)) = abbreviations.iter().find(|(abbr, _)| *abbr == text) {
        return Some(*program);
    }
    let padded = format!(" {text} ");
    for (names, program) in [
        (&["piano"][..], 0),
        (&["violin"][..], 40),
        (&["viola"][..], 41),
        (&["cello", "violoncello"][..], 42),
        (&["contrabass"][..], 43),
        (&["flute"][..], 73),
        (&["oboe"][..], 68),
        (&["clarinet"][..], 71),
        (&["bassoon"][..], 70),
        (&["english horn", "cor anglais"][..], 69),
        (&["soprano sax", "soprano saxophone"][..], 64),
        (&["alto sax", "alto saxophone"][..], 65),
        (&["tenor sax", "tenor saxophone"][..], 66),
        (&["baritone sax", "baritone saxophone"][..], 67),
        (&["trumpet"][..], 56),
        (&["french horn", "horn in f", "f horn"][..], 60),
    ] {
        if names
            .iter()
            .any(|name| padded.contains(&format!(" {name} ")))
        {
            return Some(program);
        }
    }
    None
}

/// Group page/row identities, ordering each row's boxes from left to right.
pub fn staff_rows(records: &[Value]) -> Result<StaffRows> {
    let mut grouped: BTreeMap<(u64, u64), Vec<(f64, Value)>> = BTreeMap::new();
    for record in records {
        let page = record["page"]
            .as_u64()
            .ok_or("Record needs a nonnegative page index")?;
        let row = record
            .get("row_index")
            .unwrap_or(&record["system_index"])
            .as_u64()
            .ok_or("Record needs row_index/system_index")?;
        let x = record["bbox"][0]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("Record needs finite bbox x")?;
        grouped
            .entry((page, row))
            .or_default()
            .push((x, record.clone()));
    }
    Ok(grouped
        .into_values()
        .map(|mut row| {
            row.sort_by(|a, b| a.0.total_cmp(&b.0));
            row.into_iter().map(|(_, v)| v).collect()
        })
        .collect())
}

/// Returns at most two candidates: two is enough to establish ambiguity.
pub fn compatible_templates(parts: &[Value], modes: &[String]) -> Vec<Vec<(usize, usize)>> {
    fn visit(
        parts: &[Value],
        modes: &[String],
        part: usize,
        position: usize,
        template: &mut Vec<(usize, usize)>,
        candidates: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if candidates.len() > 1 {
            return;
        }
        if part == parts.len() {
            if position == modes.len() {
                candidates.push(template.clone());
            }
            return;
        }
        let patterns: Vec<(&[&str], &[usize])> = if fretted(&parts[part]) {
            vec![
                (&["both"], &[0]),
                (&["tab"], &[0]),
                (&["notation"], &[0]),
                (&["notation", "tab"], &[0, 0]),
                (&["tab", "notation"], &[0, 0]),
            ]
        } else if parts[part]["instrument"] == "pitched"
            && (grand_name(&parts[part]) || grand_program(&parts[part]))
        {
            vec![(&["notation"], &[0]), (&["notation", "notation"], &[0, 1])]
        } else {
            vec![(&["notation"], &[0])]
        };
        for (pattern, staves) in patterns {
            if position + pattern.len() <= modes.len()
                && modes[position..position + pattern.len()]
                    .iter()
                    .map(String::as_str)
                    .eq(pattern.iter().copied())
            {
                let length = template.len();
                template.extend(staves.iter().map(|&staff| (part, staff)));
                visit(
                    parts,
                    modes,
                    part + 1,
                    position + pattern.len(),
                    template,
                    candidates,
                );
                template.truncate(length);
            }
        }
    }
    let mut candidates = Vec::new();
    visit(parts, modes, 0, 0, &mut Vec::new(), &mut candidates);
    candidates
}

/// Repair only layouts supported by a repeated geometric system width.
pub fn constrain_rows(
    value: &mut Value,
    systems: &[usize],
    modes: Option<&[String]>,
) -> Result<()> {
    if modes.is_some_and(|m| m.len() != systems.len()) {
        return Err("Row mode count does not match systems".into());
    }
    let groups = group_indices(systems.iter().map(|&s| s as i64));
    let Some((_, first)) = groups.first() else {
        return Ok(());
    };
    let width = first.len();
    if groups.iter().any(|(_, g)| g.len() != width) {
        return Ok(());
    }
    let mut profiles = parts(value)?.to_vec();
    let ids: Vec<_> = profiles.iter().map(identity).collect();
    if profiles.len() > width {
        let Some(period) =
            (1..=width).find(|&n| ids.iter().enumerate().all(|(i, id)| *id == ids[i % n]))
        else {
            return Ok(());
        };
        profiles.truncate(period);
    }
    let mut template: Vec<(usize, usize)> = triples(value)
        .filter(|r| !r.is_empty())
        .map(|r| {
            r.iter()
                .filter(|row| row[0] == r[0][0])
                .map(|row| (row[1] as usize, row[2] as usize))
                .collect()
        })
        .unwrap_or_default();
    let all_parts: HashSet<_> = (0..profiles.len()).collect();
    let mut grand: HashSet<usize> = profiles
        .iter()
        .enumerate()
        .filter_map(|(i, p)| grand_name(p).then_some(i))
        .collect();
    let mut valid = template.len() == width
        && template.iter().map(|r| r.0).collect::<HashSet<_>>() == all_parts
        && template.iter().all(|r| r.1 < 16);
    if valid {
        let mut occurrences: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, &slot) in template.iter().enumerate() {
            occurrences.entry(slot).or_default().push(first[i]);
        }
        valid = occurrences
            .values()
            .all(|indices| indices.len() == 1 || modes.is_some_and(|m| complementary(m, indices)));
    }
    if !valid {
        if let Some(modes) = modes {
            let patterns: HashSet<Vec<String>> = groups
                .iter()
                .map(|(_, indices)| indices.iter().map(|&i| modes[i].clone()).collect())
                .collect();
            if patterns.len() == 1 {
                let candidates = compatible_templates(&profiles, patterns.iter().next().unwrap());
                if candidates.len() == 1 {
                    value["parts"] = json!(profiles);
                    value["rows"] = repeated_rows(&groups, &candidates[0]);
                    return Ok(());
                }
            }
        }
    }
    if valid {
        value["parts"] = json!(profiles);
        value["rows"] = repeated_rows(&groups, &template);
        return Ok(());
    }
    if profiles.len() + grand.len() != width {
        grand.extend(
            profiles
                .iter()
                .enumerate()
                .filter_map(|(i, p)| grand_program(p).then_some(i)),
        );
    }
    let grand_template: Vec<_> = (0..profiles.len())
        .flat_map(|i| (0..if grand.contains(&i) { 2 } else { 1 }).map(move |staff| (i, staff)))
        .collect();
    if !grand.is_empty() && grand_template.len() == width {
        template = grand_template.clone();
    } else if template.len() != width
        || template.iter().collect::<HashSet<_>>().len() != width
        || template.iter().map(|r| r.0).collect::<HashSet<_>>() != all_parts
        || template.iter().any(|r| r.1 >= 16)
    {
        template = (0..profiles.len()).map(|p| (p, 0)).collect();
        if template.len() != width {
            template = grand_template;
        }
    }
    if template.len() == width {
        value["parts"] = json!(profiles);
        value["rows"] = repeated_rows(&groups, &template);
    }
    Ok(())
}

/// Preserve simultaneous guitars/basses whose modes cannot form a paired row.
pub fn separate_fretted_lanes(value: &mut Value, modes: Option<&[String]>) -> Result<()> {
    let Some(modes) = modes else {
        return Ok(());
    };
    let Some(rows) = triples(value) else {
        return Ok(());
    };
    let profiles = parts(value)?.to_vec();
    if rows.len() != modes.len()
        || rows
            .iter()
            .any(|r| r[1] < 0 || r[1] as usize >= profiles.len())
    {
        return Ok(());
    }
    let groups = group_indices(rows.iter().map(|r| r[0]));
    let templates: HashSet<Vec<_>> = groups
        .iter()
        .map(|(_, indices)| {
            indices
                .iter()
                .map(|&i| (rows[i][1], rows[i][2], modes[i].clone()))
                .collect()
        })
        .collect();
    if templates.len() != 1 {
        return Ok(());
    }
    let first = &groups[0].1;
    let mut slots: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for &i in first {
        slots.entry((rows[i][1], rows[i][2])).or_default().push(i);
    }
    let duplicated: HashSet<_> = slots
        .iter()
        .filter_map(|(&slot, indices)| {
            (indices.len() > 1
                && fretted(&profiles[slot.0 as usize])
                && !complementary(modes, indices))
            .then_some(slot)
        })
        .collect();
    if duplicated.is_empty() {
        return Ok(());
    }
    if duplicated.iter().any(|slot| {
        slots[slot].iter().any(|&i| modes[i] == "notation")
            && slots[slot].iter().any(|&i| modes[i] == "tab")
    }) {
        return Ok(());
    }
    let mut new_profiles = Vec::new();
    let mut mapping = HashMap::new();
    let mut occurrences = HashMap::new();
    let mut template = Vec::new();
    for &i in first {
        let part = rows[i][1] as usize;
        let staff = rows[i][2] as usize;
        let slot = (rows[i][1], rows[i][2]);
        let occurrence = occurrences.entry(slot).or_insert(0usize);
        let lane = if duplicated.contains(&slot) {
            *occurrence
        } else {
            0
        };
        *occurrence += 1;
        let mapped = *mapping.entry((part, lane)).or_insert_with(|| {
            let mut profile = profiles[part].clone();
            if lane > 0 {
                profile["name"] = json!(format!(
                    "{} ({})",
                    profile["name"].as_str().unwrap_or(""),
                    lane + 1
                ));
            }
            new_profiles.push(profile);
            new_profiles.len() - 1
        });
        template.push((mapped, staff));
    }
    value["parts"] = json!(new_profiles);
    value["rows"] = repeated_rows(&groups, &template);
    value["separated_fretted_lanes"] = json!(true);
    Ok(())
}

fn json_object(raw: &str) -> Result<Value> {
    let start = raw
        .find('{')
        .ok_or("Structure response has no JSON object")?;
    let end = raw
        .rfind('}')
        .filter(|&i| i >= start)
        .ok_or("Structure response has no complete JSON object")?;
    serde_json::from_str(&raw[start..=end]).map_err(|e| format!("Invalid structure JSON: {e}"))
}
fn integer_like(value: &Value) -> Option<i64> {
    if let Some(v) = value.as_i64() {
        Some(v)
    } else if let Some(v) = value.as_str() {
        v.trim().parse().ok()
    } else if let Some(v) = value.as_bool() {
        Some(i64::from(v))
    } else {
        value
            .as_f64()
            .filter(|v| v.is_finite() && *v >= i64::MIN as f64 && *v < i64::MAX as f64)
            .map(|v| v.trunc() as i64)
    }
}
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Number(_) => value.as_f64() != Some(0.),
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
    }
}

/// Parse, normalize and repair one page while preserving every detected row.
pub fn parse_structure(
    raw: &str,
    count: usize,
    systems: Option<&[usize]>,
    modes: Option<&[String]>,
) -> Result<Value> {
    if modes.is_some_and(|m| m.len() != count) {
        return Err("Row mode count does not cover every detected row".into());
    }
    let mut value = json_object(raw)?;
    parts(&value)?;
    rows(&value)?;
    for part in value["parts"].as_array_mut().unwrap() {
        if !part.is_object() {
            return Err("Invalid part profile".into());
        }
        let instrument = part["instrument"]
            .as_str()
            .filter(|s| ["guitar", "bass", "pitched", "drums"].contains(s))
            .ok_or("Invalid part instrument")?
            .to_owned();
        if !part["strings"].is_null() {
            let strings = integer_like(&part["strings"])
                .filter(|v| (1..=12).contains(v))
                .ok_or("Invalid string count")?;
            part["strings"] = json!(strings);
        } else {
            part["strings"] = Value::Null;
        }
        let name = if truthy(&part["name"]) {
            part["name"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| part["name"].to_string())
        } else {
            instrument.clone()
        };
        let name: String = name.chars().take(160).collect();
        part["name"] = json!(name);
        let mut program = integer_like(&part["program"])
            .filter(|v| (0..=127).contains(v))
            .ok_or("Invalid MIDI program")?;
        if let Some(visible) = program_from_visible_name(&name) {
            program = visible;
        } else if instrument == "guitar" && !(24..=31).contains(&program) {
            program = 25;
        } else if instrument == "bass" && !(32..=39).contains(&program) {
            program = 33;
        }
        part["program"] = json!(program);
    }
    if let Some(systems) = systems.filter(|s| s.len() == count) {
        constrain_rows(&mut value, systems, modes)?;
    } else if let (Some(modes), Some(assignments)) =
        (modes, triples(&value).filter(|r| r.len() == count))
    {
        let groups = group_indices(assignments.iter().map(|r| r[0]));
        let mut seen: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        let mut invalid = false;
        for (index, &row) in assignments.iter().enumerate() {
            seen.entry(row).or_default().push(index);
            invalid |= row[1] < 0 || row[1] as usize >= parts(&value)?.len();
        }
        invalid |= seen
            .values()
            .any(|indices| indices.len() > 1 && !complementary(modes, indices));
        let patterns: HashSet<Vec<String>> = groups
            .iter()
            .map(|(_, g)| g.iter().map(|&i| modes[i].clone()).collect())
            .collect();
        if invalid && patterns.len() == 1 {
            let candidates = compatible_templates(parts(&value)?, patterns.iter().next().unwrap());
            if candidates.len() == 1 {
                value["rows"] = repeated_rows(&groups, &candidates[0]);
            }
        }
    }
    separate_fretted_lanes(&mut value, modes)?;
    if value["separated_fretted_lanes"] == true {
        let inferred: Vec<usize>;
        let actual = if let Some(s) = systems {
            s
        } else {
            inferred = require_triples(&value)?
                .iter()
                .map(|r| r[0] as usize)
                .collect();
            &inferred
        };
        constrain_rows(&mut value, actual, modes)?;
    }
    let assignments = require_triples(&value)?;
    if assignments.len() != count {
        return Err("Structure does not cover every detected staff row".into());
    }
    if let Some(systems) = systems.filter(|s| s.len() == count) {
        let mut order = HashMap::new();
        let mut actual = Vec::new();
        for row in &assignments {
            let next = order.len();
            actual.push(*order.entry(row[0]).or_insert(next));
        }
        if actual != systems {
            return Err(format!("Visible barlines group rows into systems {systems:?}; each connected system needs all its parts and staves"));
        }
    }
    let mut previous = -1;
    let mut seen: HashMap<[i64; 3], Option<usize>> = HashMap::new();
    for (index, &row) in assignments.iter().enumerate() {
        let [system, part, staff] = row;
        if part < 0 || part as usize >= parts(&value)?.len() {
            return Err(format!("R{} uses nonexistent part {part}; separate notation and TAB rows of one instrument share the same part and staff",index+1));
        }
        if system < 0 || system < previous || !(0..16).contains(&staff) {
            return Err(format!(
                "R{} has invalid system/staff indices; systems must stay in page order",
                index + 1
            ));
        }
        if let Some(old) = seen.get(&row) {
            if !modes
                .zip(*old)
                .is_some_and(|(m, i)| complementary(m, &[index, i]))
            {
                return Err(format!("R{} repeats system {system}, part {part}, staff {staff}; only complementary notation/TAB rows may share this triple",index+1));
            }
            seen.insert(row, None);
        } else {
            seen.insert(row, Some(index));
        }
        previous = system;
    }
    if let Some(modes) = modes {
        for (part, profile) in value["parts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            let visible: Vec<_> = assignments
                .iter()
                .zip(modes)
                .filter_map(|(row, mode)| (row[1] as usize == part).then_some(mode))
                .collect();
            if !visible.is_empty() && visible.iter().all(|m| m.as_str() == "notation") {
                profile["strings"] = Value::Null;
            }
        }
    }
    Ok(value)
}

fn row_mode(row: &[Value]) -> Result<String> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for box_ in row {
        let mode = box_["mode"]
            .as_str()
            .ok_or("Staff record needs a display mode")?;
        if let Some((_, count)) = counts.iter_mut().find(|(v, _)| v == mode) {
            *count += 1;
        } else {
            counts.push((mode.into(), 1));
        }
    }
    let maximum = counts
        .iter()
        .map(|(_, n)| *n)
        .max()
        .ok_or("Staff row is empty")?;
    Ok(counts.into_iter().find(|(_, n)| *n == maximum).unwrap().0)
}

/// One repeated system signature, plus local part indices in first-use order.
pub fn page_template(
    structure: &Value,
    rows: &[Vec<Value>],
) -> Result<(Option<PageTemplate>, Vec<usize>)> {
    let assignments = require_triples(structure)?;
    if assignments.len() != rows.len() {
        return Err("Page structure does not cover every staff row".into());
    }
    let mut order = Vec::new();
    let mut systems: Vec<(i64, PageTemplate)> = Vec::new();
    for (assignment, boxes) in assignments.iter().zip(rows) {
        let [system, part, staff] = *assignment;
        if system < 0
            || part < 0
            || part as usize >= parts(structure)?.len()
            || !(0..16).contains(&staff)
        {
            return Err("Invalid page structure indices".into());
        }
        let part = part as usize;
        let slot = if let Some(index) = order.iter().position(|&p| p == part) {
            index
        } else {
            order.push(part);
            order.len() - 1
        };
        let position = if let Some(index) = systems.iter().position(|(s, _)| *s == system) {
            index
        } else {
            systems.push((system, Vec::new()));
            systems.len() - 1
        };
        systems[position]
            .1
            .push((slot, staff as usize, row_mode(boxes)?));
    }
    let template = systems
        .first()
        .filter(|(_, first)| systems.iter().all(|(_, s)| s == first))
        .map(|(_, s)| s.clone());
    Ok((template, order))
}

/// Correct isolated profile errors only with a strict majority of >=3 pages.
pub fn reconcile_page_profiles(pages: &mut [StructurePage]) -> Result<()> {
    let mut templates: HashMap<PageTemplate, Vec<(usize, Vec<usize>)>> = HashMap::new();
    for (index, page) in pages.iter().enumerate() {
        let (template, order) = page_template(&page.structure, &page.rows)?;
        if let Some(template) = template {
            templates.entry(template).or_default().push((index, order));
        }
    }
    for group in templates.into_values() {
        if group.len() < 3 {
            continue;
        }
        for slot in 0..group[0].1.len() {
            let candidates: Vec<Value> = group
                .iter()
                .map(|(index, ids)| pages[*index].structure["parts"][ids[slot]].clone())
                .collect();
            let keys: Vec<_> = candidates.iter().map(identity).collect();
            let mut counts: HashMap<&Identity, usize> = HashMap::new();
            for key in &keys {
                *counts.entry(key).or_default() += 1;
            }
            let Some((winner, count)) = counts.into_iter().max_by_key(|(_, count)| *count) else {
                continue;
            };
            if count * 2 <= group.len() {
                continue;
            }
            let profile = &candidates[keys.iter().position(|key| key == winner).unwrap()];
            for ((index, ids), key) in group.iter().zip(&keys) {
                if key != winner {
                    let value = &mut pages[*index].structure;
                    if value.get("model_parts").is_none() {
                        value["model_parts"] = value["parts"].clone();
                    }
                    value["parts"][ids[slot]] = profile.clone();
                }
            }
        }
    }
    Ok(())
}

/// Recover an incomplete page only from one unambiguous complete-page template.
pub fn complete_from_reference(
    raw: &str,
    rows: &[Vec<Value>],
    systems: Option<&[usize]>,
    references: &[StructurePage],
) -> Result<Option<Value>> {
    let Some(systems) = systems else {
        return Ok(None);
    };
    if systems.len() != rows.len() {
        return Err("Reference geometry does not cover all rows".into());
    }
    let modes: Vec<_> = rows.iter().map(|r| row_mode(r)).collect::<Result<_>>()?;
    let mut invalid_rows = false;
    let incomplete = match parse_structure(raw, rows.len(), None, Some(&modes)) {
        Ok(value) => value,
        Err(_) => {
            let Ok(value) = json_object(raw) else {
                return Ok(None);
            };
            let (Ok(profiles), Ok(rows)) = (parts(&value), self::rows(&value)) else {
                return Ok(None);
            };
            let mut used: Vec<usize> = rows
                .iter()
                .filter_map(|row| {
                    row.as_array()
                        .filter(|r| r.len() == 3)
                        .and_then(|r| r[1].as_u64())
                        .and_then(|i| usize::try_from(i).ok())
                        .filter(|&i| i < profiles.len())
                })
                .collect();
            used.sort_unstable();
            used.dedup();
            if used.is_empty() {
                return Ok(None);
            }
            let extracted = json!({"parts":used.iter().map(|&i|profiles[i].clone()).collect::<Vec<_>>(),"rows":(0..used.len()).map(|i|json!([i,i,0])).collect::<Vec<_>>()});
            let Ok(value) = parse_structure(&extracted.to_string(), used.len(), None, None) else {
                return Ok(None);
            };
            invalid_rows = true;
            value
        }
    };
    let groups = group_indices(systems.iter().map(|&s| s as i64));
    let patterns: HashSet<Vec<_>> = groups
        .iter()
        .map(|(_, indices)| indices.iter().map(|&i| modes[i].clone()).collect())
        .collect();
    if patterns.len() != 1 {
        return Ok(None);
    }
    let mode = patterns.iter().next().unwrap();
    let observed: HashSet<_> = parts(&incomplete)?.iter().map(identity).collect();
    let mut candidates: HashMap<(Vec<Identity>, PageTemplate), Value> = HashMap::new();
    for reference in references {
        let (Some(template), order) = page_template(&reference.structure, &reference.rows)? else {
            continue;
        };
        if template.iter().map(|r| &r.2).ne(mode.iter()) {
            continue;
        }
        let profiles: Vec<Value> = order
            .iter()
            .map(|&i| reference.structure["parts"][i].clone())
            .collect();
        let ids: Vec<_> = profiles.iter().map(identity).collect();
        if (profiles.len() <= parts(&incomplete)?.len() && !invalid_rows)
            || !observed.is_subset(&ids.iter().cloned().collect())
        {
            continue;
        }
        let assignment: Vec<_> = template.iter().map(|&(p, s, _)| (p, s)).collect();
        let candidate = json!({"parts":profiles,"rows":repeated_rows(&groups,&assignment),"reference_page":reference.page});
        candidates.insert((ids, template), candidate);
    }
    Ok(if candidates.len() == 1 {
        candidates.into_values().next()
    } else {
        None
    })
}

/// Reconcile pages and assign stable global parts, leaving geometry to score_grid.
/// Pages are sorted numerically. Unused model profiles never become output parts.
pub fn resolve_pages(pages: &mut [StructurePage]) -> Result<ResolvedPages> {
    pages.sort_by_key(|p| p.page);
    if pages.windows(2).any(|p| p[0].page == p[1].page) {
        return Err("Duplicate structure page".into());
    }
    reconcile_page_profiles(pages)?;
    let mut output = ResolvedPages {
        parts: Vec::new(),
        systems: Vec::new(),
    };
    let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
    let mut identities: HashMap<PageTemplate, Vec<usize>> = HashMap::new();
    for page in pages {
        let (template, order) = page_template(&page.structure, &page.rows)?;
        let assignments = require_triples(&page.structure)?;
        let mut mapping = HashMap::new();
        let mut used = HashSet::new();
        for (local, part) in parts(&page.structure)?.iter().enumerate() {
            if !assignments.iter().any(|row| row[1] as usize == local) {
                continue;
            }
            let name = normalized_name(part["name"].as_str().unwrap_or(""));
            let mut matched = template
                .as_ref()
                .and_then(|t| identities.get(t))
                .and_then(|ids| {
                    order
                        .iter()
                        .position(|&i| i == local)
                        .and_then(|i| ids.get(i))
                })
                .copied();
            if matched.is_none() {
                matched = by_name
                    .get(&name)
                    .and_then(|ids| ids.iter().find(|i| !used.contains(*i)))
                    .copied();
            }
            if matched.is_none() && local < output.parts.len() && !used.contains(&local) {
                let old = &output.parts[local];
                if old["instrument"] == part["instrument"]
                    && old["strings"] == part["strings"]
                    && old["program"] == part["program"]
                {
                    matched = Some(local);
                }
            }
            let matched = matched.unwrap_or_else(|| {
                let index = output.parts.len();
                let mut profile = part.clone();
                profile["id"] = json!(format!("part-{}", index + 1));
                output.parts.push(profile);
                index
            });
            let names = by_name.entry(name).or_default();
            if !names.contains(&matched) {
                names.push(matched);
            }
            used.insert(matched);
            mapping.insert(local, matched);
        }
        if let Some(template) = template {
            identities.insert(template, order.iter().map(|i| mapping[i]).collect());
        }
        let mut systems: BTreeMap<usize, Vec<(Vec<Value>, usize, usize)>> = BTreeMap::new();
        for (boxes, row) in page.rows.iter().zip(&assignments) {
            let part = *mapping
                .get(&(row[1] as usize))
                .ok_or("Unmapped structure part")?;
            systems.entry(row[0] as usize).or_default().push((
                boxes.clone(),
                part,
                row[2] as usize,
            ));
        }
        output
            .systems
            .extend(systems.into_iter().map(|(system, members)| ResolvedSystem {
                page: page.page,
                system,
                members,
            }));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Generated once from layout.structure.parse_structure; runtime is Rust-only.
    const PARSE_GOLDENS: &str = r#"[{"name":"paired","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]}","count":4,"systems":[0,0,1,1],"modes":["notation","tab","notation","tab"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25}],"rows":[[0,0,0],[0,0,0],[1,0,0],[1,0,0]]}},{"name":"grand_repair","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}, {\"name\": \"Piano\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 57}], \"rows\": [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]]}","count":6,"systems":[0,0,0,1,1,1],"modes":["both","notation","notation","both","notation","notation"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Piano","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0],[0,1,0],[0,1,1],[1,0,0],[1,1,0],[1,1,1]]}},{"name":"separate_guitars","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]}","count":4,"systems":null,"modes":["both","both","both","both"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Guitar (2)","instrument":"guitar","strings":6,"program":25}],"rows":[[0,0,0],[0,1,0],[1,0,0],[1,1,0]],"separated_fretted_lanes":true}},{"name":"separate_then_grand","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}, {\"name\": \"Piano\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]]}","count":4,"systems":[0,0,0,0],"modes":["both","both","notation","notation"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Guitar (2)","instrument":"guitar","strings":6,"program":25},{"name":"Piano","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0],[0,1,0],[0,2,0],[0,2,1]],"separated_fretted_lanes":true}},{"name":"periodic_profiles","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}, {\"name\": \"Piano\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}, {\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}, {\"name\": \"Piano\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 2, 0], [1, 3, 0], [1, 3, 1]]}","count":6,"systems":[0,0,0,1,1,1],"modes":["both","notation","notation","both","notation","notation"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Piano","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0],[0,1,0],[0,1,1],[1,0,0],[1,1,0],[1,1,1]]}},{"name":"named_guitars","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}, {\"name\": \"Guitar II\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 1, 0]]}","count":2,"systems":[0,0],"modes":["both","both"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Guitar II","instrument":"guitar","strings":6,"program":25}],"rows":[[0,0,0],[0,1,0]]}},{"name":"notation_strings","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}], \"rows\": [[0, 0, 0]]}","count":1,"systems":null,"modes":["notation"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":null,"program":25}],"rows":[[0,0,0]]}},{"name":"visible_programs","raw":"{\"parts\": [{\"name\": \"Tenor Saxophone\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 5}, {\"name\": \"Flute\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 1, 0]]}","count":2,"systems":null,"modes":["notation","notation"],"expected":{"parts":[{"name":"Tenor Saxophone","instrument":"pitched","strings":null,"program":66},{"name":"Flute","instrument":"pitched","strings":null,"program":73}],"rows":[[0,0,0],[0,1,0]]}},{"name":"repair_part_indices","raw":"{\"parts\": [{\"name\": \"Guitar\", \"instrument\": \"guitar\", \"strings\": 6, \"program\": 0}, {\"name\": \"Piano\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 21, 0], [0, 21, 0]]}","count":2,"systems":null,"modes":["both","notation"],"expected":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Piano","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0],[0,1,0]]}},{"name":"repair_grand_unknown_geometry","raw":"{\"parts\": [{\"name\": \"Piano\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 0, 0]]}","count":2,"systems":null,"modes":["notation","notation"],"expected":{"parts":[{"name":"Piano","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0],[0,0,1]]}},{"name":"grand_program_fallback","raw":"{\"parts\": [{\"name\": \"Pian0\", \"instrument\": \"pitched\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 0, 0], [0, 0, 0]]}","count":2,"systems":[0,0],"modes":["notation","notation"],"expected":{"parts":[{"name":"Pian0","instrument":"pitched","strings":null,"program":0}],"rows":[[0,0,0],[0,0,1]]}},{"name":"percussion","raw":"{\"parts\": [{\"name\": \"Drums\", \"instrument\": \"drums\", \"strings\": null, \"program\": 0}], \"rows\": [[0, 0, 0], [1, 0, 0]]}","count":2,"systems":null,"modes":["notation","notation"],"expected":{"parts":[{"name":"Drums","instrument":"drums","strings":null,"program":0}],"rows":[[0,0,0],[1,0,0]]}}]"#;

    fn profile(name: &str, instrument: &str, strings: Value, program: i64) -> Value {
        json!({"name":name,"instrument":instrument,"strings":strings,"program":program})
    }
    fn page(page: u64, profiles: Vec<Value>, slots: &[(usize, usize, &str)]) -> StructurePage {
        StructurePage{page,structure:json!({"parts":profiles,"rows":slots.iter().map(|&(p,s,_)|json!([0,p,s])).collect::<Vec<_>>()}),
            rows:slots.iter().enumerate().map(|(index,(_,_,mode))|vec![json!({"page":page,"system_index":index,"bbox":[0,index*50,100,40],"mode":mode})]).collect()}
    }

    #[test]
    fn structure_parse_matches_twelve_python_reference_cases() {
        let goldens: Value = serde_json::from_str(PARSE_GOLDENS).unwrap();
        for fixture in goldens.as_array().unwrap() {
            let systems: Option<Vec<usize>> = fixture["systems"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect());
            let modes: Vec<String> = fixture["modes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().into())
                .collect();
            let actual = parse_structure(
                fixture["raw"].as_str().unwrap(),
                fixture["count"].as_u64().unwrap() as usize,
                systems.as_deref(),
                Some(&modes),
            )
            .unwrap();
            assert_eq!(actual, fixture["expected"], "{}", fixture["name"]);
        }
    }

    #[test]
    fn invalid_structure_fails_without_silent_row_loss() {
        let profiles = vec![profile("Guitar", "guitar", json!(6), 25)];
        for rows in [
            json!([[0, 0, true]]),
            json!([[0, 0, 16]]),
            json!([[1, 0, 0], [0, 0, 0]]),
            json!([[0, 9, 0]]),
        ] {
            let raw = json!({"parts":profiles,"rows":rows});
            assert!(parse_structure(
                &raw.to_string(),
                raw["rows"].as_array().unwrap().len(),
                None,
                None
            )
            .is_err());
        }
        let raw = json!({"parts":profiles,"rows":[[0,0,0],[0,0,0],[0,0,0]]});
        assert!(parse_structure(
            &raw.to_string(),
            3,
            None,
            Some(&["notation".into(), "tab".into(), "notation".into()])
        )
        .is_err());
        assert!(parse_structure("{}", 0, None, None).is_err());
        assert!(parse_structure("not json", 1, None, None).is_err());
        assert!(parse_structure(
            &json!({"parts":profiles,"rows":[[0,0,0]]}).to_string(),
            2,
            None,
            None
        )
        .is_err());
    }

    #[test]
    fn grouping_orders_pages_rows_and_boxes_without_losing_fields() {
        let records = vec![
            json!({"page":2,"row_index":1,"bbox":[8,0,3,3],"id":"c"}),
            json!({"page":1,"system_index":3,"bbox":[9,0,3,3],"id":"b"}),
            json!({"page":1,"system_index":3,"bbox":[1,0,3,3],"id":"a"}),
        ];
        let grouped = staff_rows(&records).unwrap();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0][0]["id"], "a");
        assert_eq!(grouped[0][1]["id"], "b");
        assert_eq!(grouped[1][0]["id"], "c");
    }

    #[test]
    fn profile_majority_excludes_condensed_systems_and_preserves_model_evidence() {
        let guitar = profile("Guitar", "guitar", json!(6), 25);
        let piano = profile("Piano", "pitched", Value::Null, 0);
        let wrong = profile("Violin", "pitched", Value::Null, 40);
        let slots = [(0, 0, "both"), (1, 0, "notation"), (1, 1, "notation")];
        let mut pages = vec![
            page(1, vec![guitar.clone(), piano.clone()], &slots),
            page(2, vec![guitar.clone(), wrong.clone()], &slots),
            page(3, vec![guitar.clone(), piano.clone()], &slots),
            page(4, vec![wrong.clone()], &[(0, 0, "notation")]),
        ];
        reconcile_page_profiles(&mut pages).unwrap();
        assert_eq!(pages[1].structure["parts"][1], piano);
        assert_eq!(pages[1].structure["model_parts"][1], wrong);
        assert_eq!(pages[3].structure["parts"][0], wrong);
        assert!(pages[0].structure.get("model_parts").is_none());
        let mut two = vec![
            pages[0].clone(),
            page(2, vec![guitar, wrong.clone()], &slots),
        ];
        reconcile_page_profiles(&mut two).unwrap();
        assert_eq!(two[1].structure["parts"][1], wrong);
    }

    #[test]
    fn part_identity_survives_page_turns_names_and_condensed_pages() {
        let guitar = profile("Guitar", "guitar", json!(6), 25);
        let second = profile("Guitar II", "guitar", json!(6), 25);
        let mut pages = vec![
            page(3, vec![second.clone()], &[(0, 0, "both")]),
            page(
                1,
                vec![guitar.clone(), second],
                &[(0, 0, "both"), (1, 0, "both")],
            ),
            page(
                2,
                vec![guitar.clone(), guitar],
                &[(0, 0, "both"), (1, 0, "both")],
            ),
        ];
        let result = resolve_pages(&mut pages).unwrap();
        assert_eq!(result.parts.len(), 2);
        assert_eq!(result.parts[1]["name"], "Guitar II");
        assert_eq!(
            result.systems.iter().map(|s| s.page).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            result.systems[1]
                .members
                .iter()
                .map(|m| m.1)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(result.systems[2].members[0].1, 1);
        assert_eq!(result.parts[0]["id"], "part-1");
        assert_eq!(result.parts[1]["id"], "part-2");
    }

    #[test]
    fn missing_parts_recover_only_from_one_compatible_reference() {
        let guitar = profile("Guitar", "guitar", json!(6), 25);
        let piano = profile("Piano", "pitched", Value::Null, 0);
        let slots = [(0, 0, "both"), (1, 0, "notation"), (1, 1, "notation")];
        let reference = page(1, vec![guitar.clone(), piano], &slots);
        let raw = json!({"parts":[guitar],"rows":[[0,0,0],[0,99,0],[0,99,0]]}).to_string();
        let result = complete_from_reference(
            &raw,
            &reference.rows,
            Some(&[0, 0, 0]),
            &[reference.clone()],
        )
        .unwrap()
        .unwrap();
        assert_eq!(result["rows"], reference.structure["rows"]);
        assert_eq!(result["reference_page"], 1);
        let mut conflict = reference.clone();
        conflict.page = 2;
        conflict.structure["parts"][1]["name"] = json!("Organ");
        conflict.structure["parts"][1]["program"] = json!(16);
        assert!(complete_from_reference(
            &raw,
            &reference.rows,
            Some(&[0, 0, 0]),
            &[reference.clone(), conflict]
        )
        .unwrap()
        .is_none());
        assert!(
            complete_from_reference(&raw, &reference.rows, None, &[reference.clone()])
                .unwrap()
                .is_none()
        );
    }
}
