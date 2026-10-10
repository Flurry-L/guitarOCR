//! Validate learned staff groups and match observed part identities across pages.
//! Invalid row assignments require model evidence or manual correction.
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use unicode_casefold::UnicodeCaseFold;

mod anonymous_relations;
mod part_identity;
pub use anonymous_relations::resolve_conflicting_groups as anonymous_relation_groups;
pub(crate) use anonymous_relations::{identity_batches, resolve_identity_batch};
pub(crate) use part_identity::compatible_profile as compatible_part_profiles;

pub type Result<T> = std::result::Result<T, String>;
pub type StaffRows = Vec<Vec<Value>>;

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
fn normalized_name(name: &str) -> String {
    name.case_fold()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
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
fn complementary(modes: &[String], indices: &[usize]) -> bool {
    indices.len() == 2
        && ((modes[indices[0]] == "notation" && modes[indices[1]] == "tab")
            || (modes[indices[0]] == "tab" && modes[indices[1]] == "notation"))
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
/// Check semantic references without guessing a repeating page template.
fn validate_row_topology(
    value: &Value,
    count: usize,
    modes: Option<&[String]>,
) -> Result<Vec<[i64; 3]>> {
    if modes.is_some_and(|m| m.len() != count) {
        return Err("Row mode count does not cover every detected row".into());
    }
    let assignments = require_triples(value)?;
    if assignments.len() != count {
        return Err("Structure does not cover every detected staff row".into());
    }
    let mut previous = -1;
    let mut seen: HashMap<[i64; 3], Option<usize>> = HashMap::new();
    for (index, &row) in assignments.iter().enumerate() {
        let [system, part, staff] = row;
        if part < 0 || part as usize >= parts(value)?.len() {
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
    Ok(assignments)
}

/// Normalize visible profiles and validate the supplied staff assignments.
pub fn parse_structure(raw: &str, count: usize, modes: Option<&[String]>) -> Result<Value> {
    parse_structure_impl(raw, count, modes, None)
}

fn record_structure_change(
    audit: &mut Option<&mut Vec<Value>>,
    reason: &str,
    before: &Value,
    after: &Value,
) {
    if before["parts"] != after["parts"] || before["rows"] != after["rows"] {
        if let Some(changes) = audit.as_deref_mut() {
            changes.push(json!({"reason": reason,
                "before": {"parts": before["parts"], "rows": before["rows"]},
                "after": {"parts": after["parts"], "rows": after["rows"]}}));
        }
    }
}

fn parse_structure_impl(
    raw: &str,
    count: usize,
    modes: Option<&[String]>,
    mut audit: Option<&mut Vec<Value>>,
) -> Result<Value> {
    if modes.is_some_and(|m| m.len() != count) {
        return Err("Row mode count does not cover every detected row".into());
    }
    let mut value = json_object(raw)?;
    let before = value.clone();
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
            let strings = part["strings"]
                .as_i64()
                .filter(|v| (1..=12).contains(v))
                .ok_or("Invalid string count")?;
            part["strings"] = json!(strings);
        } else {
            part["strings"] = Value::Null;
        }
        let name = if let Some(name) = part["name"].as_str().filter(|s| !s.trim().is_empty()) {
            name.to_owned()
        } else {
            part["name_source"] = json!("missing");
            instrument.clone()
        };
        let name: String = name.chars().take(160).collect();
        part["name"] = json!(name);
        let mut program = part["program"]
            .as_i64()
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
    record_structure_change(
        &mut audit,
        "visible_instrument_profile_normalization",
        &before,
        &value,
    );
    let assignments = validate_row_topology(&value, count, modes)?;
    let before = value.clone();
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
    record_structure_change(&mut audit, "notation_only_string_count", &before, &value);
    Ok(value)
}

/// Compare a visual relation proposal without using it to repair model rows.
pub fn parse_structure_with_audit_proposal(
    raw: &str,
    count: usize,
    systems: Option<&[usize]>,
    modes: Option<&[String]>,
) -> Result<(Value, Value)> {
    let mut changes = vec![];
    let parsed = parse_structure_impl(raw, count, modes, Some(&mut changes))?;
    audit_structure_result(raw, count, systems, modes, parsed, changes)
}

/// Preserve semantic identities and compare independent system/part relation candidates.
pub fn parse_structure_with_relation_heads(
    raw: &str,
    count: usize,
    systems: Option<&[usize]>,
    probabilities: &[f32],
    part_probabilities: Option<&[f32]>,
    modes: Option<&[String]>,
    review_confidence: f32,
    part_review_confidence: f32,
) -> Result<(Value, Value)> {
    if modes.is_some_and(|m| m.len() != count) {
        return Err("Row mode count does not cover every detected row".into());
    }
    if !review_confidence.is_finite() || !(0.5..=1.0).contains(&review_confidence) {
        return Err("Relation review confidence must be in [0.5, 1]".into());
    }
    if !part_review_confidence.is_finite() || !(0.5..=1.0).contains(&part_review_confidence) {
        return Err("Part relation review confidence must be in [0.5, 1]".into());
    }
    // A changed learned partition cannot inherit semantic identities by their
    // position in an incompatible language-model partition. Keep the observed
    // groups anonymous until independent visual identity evidence is available.
    if let Some(result) = anonymous_relation_groups(
        raw,
        count,
        systems,
        probabilities,
        part_probabilities,
        modes,
        review_confidence,
        part_review_confidence,
    )? {
        return Ok(result);
    }
    let model = parse_structure(raw, count, modes);
    let coherent = count > 0
        && systems.is_some_and(|s| s.len() == count && s[0] == 0)
        && probabilities.len() == count.saturating_sub(1)
        && probabilities
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && systems
            .unwrap()
            .windows(2)
            .zip(probabilities)
            .all(|(s, p)| s[0].checked_add(usize::from(*p < 0.5)) == Some(s[1]));
    let reliable = coherent
        && probabilities
            .iter()
            .all(|p| p.max(1. - p) >= review_confidence);
    // With a part head, a new system partition also invalidates the old
    // positional semantic binding. The trusted two-head case returned anonymous
    // groups above; an uncertain part head cannot authorize template repair.
    let unresolved_system_identity = part_probabilities.is_some()
        && systems.filter(|s| s.len() == count).is_some_and(|systems| {
            json_object(raw)
                .ok()
                .and_then(|source| triples(&source))
                .filter(|rows| rows.len() == count)
                .is_some_and(|rows| {
                    rows.windows(2)
                        .zip(systems.windows(2))
                        .any(|(a, b)| (a[0][0] == a[1][0]) != (b[0] == b[1]))
                })
        });
    let mut changes = vec![];
    let mut relation_value = None;
    let relation = systems
        .filter(|s| s.len() == count)
        .map(|systems| -> Result<Value> {
            let mut candidate = json_object(raw)?;
            let rows = candidate["rows"]
                .as_array_mut()
                .ok_or("Structure needs staff rows")?;
            if rows.len() != count {
                return Err("Structure does not cover every detected staff row".into());
            }
            for (row, system) in rows.iter_mut().zip(systems) {
                let triple = row
                    .as_array_mut()
                    .filter(|r| r.len() == 3)
                    .ok_or("Invalid structure row")?;
                triple[0] = json!(system);
            }
            relation_value = Some(candidate.clone());
            parse_structure_impl(&candidate.to_string(), count, modes, Some(&mut changes))
        });
    let part_relation = part_probabilities.map(|probabilities| -> Result<(Value, bool, bool)> {
        let candidate = relation_value
            .as_ref()
            .filter(|_| reliable)
            .ok_or("Part relations need reliable system assignments")?;
        part_relation_candidate(
            candidate,
            count,
            probabilities,
            modes,
            part_review_confidence,
        )
    });
    let use_part = !unresolved_system_identity
        && part_relation
            .as_ref()
            .is_some_and(|r| r.as_ref().is_ok_and(|(_, confident, _)| *confident));
    let use_relation =
        !unresolved_system_identity && reliable && relation.as_ref().is_some_and(Result::is_ok);
    let selected = if use_part {
        part_relation.as_ref().unwrap().as_ref().unwrap().0.clone()
    } else if use_relation {
        relation.as_ref().unwrap().as_ref().unwrap().clone()
    } else {
        model.as_ref().map_err(Clone::clone)?.clone()
    };
    let (parsed, mut audit) = audit_structure_result(
        raw,
        count,
        systems,
        modes,
        selected,
        if use_relation { changes } else { vec![] },
    )?;
    audit["model_candidate"] = model.as_ref().ok().cloned().unwrap_or(Value::Null);
    audit["model_candidate_error"] = model
        .as_ref()
        .err()
        .map(|e| json!(e))
        .unwrap_or(Value::Null);
    audit["relation_candidate"] = relation
        .as_ref()
        .and_then(|r| r.as_ref().ok())
        .cloned()
        .unwrap_or(Value::Null);
    audit["relation_candidate_error"] = relation
        .as_ref()
        .and_then(|r| r.as_ref().err())
        .map(|e| json!(e))
        .unwrap_or(Value::Null);
    audit["relation_probabilities"] = json!(probabilities);
    audit["same_part_probabilities"] = json!(part_probabilities);
    audit["part_relation_candidate"] = part_relation
        .as_ref()
        .and_then(|r| r.as_ref().ok())
        .map(|(v, _, _)| v.clone())
        .unwrap_or(Value::Null);
    audit["part_relation_candidate_error"] = part_relation
        .as_ref()
        .and_then(|r| r.as_ref().err())
        .map(|e| json!(e))
        .unwrap_or(Value::Null);
    audit["system_indices_constrained"] = json!(false);
    audit["selected_system_source"] = json!(if use_relation || use_part {
        "visual_relation_network"
    } else {
        "structure_model"
    });
    audit["selected_part_source"] = json!(if use_part {
        "visual_relation_network"
    } else {
        "structure_model"
    });
    let invalid_original = audit["raw_rows_valid_without_geometry"] != true;
    let reasons = audit["review_reasons"].as_array_mut().unwrap();
    if systems.is_some() && !reliable {
        reasons.push(json!("uncertain_row_relation"));
    }
    if relation.as_ref().is_some_and(Result::is_err) {
        reasons.push(json!("invalid_relation_candidate"));
    }
    if systems.is_some() && !coherent {
        reasons.push(json!("invalid_relation_prediction"));
    }
    if invalid_original {
        reasons.push(json!("invalid_model_row_assignment"));
    }
    if unresolved_system_identity {
        reasons.push(json!("unresolved_part_identity_after_system_conflict"));
    }
    if let Some(result) = &part_relation {
        match result {
            Ok((_, confident, disagreement)) => {
                if !confident {
                    reasons.push(json!("uncertain_part_relation"));
                }
                if *disagreement {
                    reasons.push(json!("part_evidence_disagreement"));
                }
            }
            Err(_) => reasons.push(json!("invalid_part_relation_candidate")),
        }
    }
    if !reasons.is_empty() {
        audit["needs_review"] = json!(true);
    }
    Ok((parsed, audit))
}

fn part_relation_candidate(
    source: &Value,
    count: usize,
    probabilities: &[f32],
    modes: Option<&[String]>,
    threshold: f32,
) -> Result<(Value, bool, bool)> {
    if count == 0
        || probabilities.len() != count - 1
        || probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("Invalid same-part probabilities".into());
    }
    let original = require_triples(source)?;
    if original.len() != count {
        return Err("Part relation row count mismatch".into());
    }
    let confident = original
        .windows(2)
        .zip(probabilities)
        .all(|(r, p)| r[0][0] != r[1][0] || p.max(1. - p) >= threshold);
    let disagreement = original
        .windows(2)
        .zip(probabilities)
        .any(|(r, p)| r[0][0] == r[1][0] && ((r[0][1] == r[1][1]) != (*p >= 0.5)));
    if disagreement && confident {
        return Err(
            "Conflicting part relations require anonymous groups or new visual identity evidence"
                .into(),
        );
    }
    Ok((
        parse_structure(&source.to_string(), count, modes)?,
        confident,
        disagreement,
    ))
}

fn audit_structure_result(
    raw: &str,
    count: usize,
    systems: Option<&[usize]>,
    modes: Option<&[String]>,
    parsed: Value,
    changes: Vec<Value>,
) -> Result<(Value, Value)> {
    let source = json_object(raw)?;
    let valid_original =
        parse_structure(raw, count, modes).is_ok_and(|value| value["rows"] == source["rows"]);
    let mut review_rows = std::collections::BTreeSet::new();
    if valid_original && source["rows"] != parsed["rows"] {
        if let (Some(original), Some(resolved)) = (triples(&source), triples(&parsed)) {
            if original.len() == resolved.len() {
                for (i, first) in original.iter().enumerate() {
                    let profile = &source["parts"][first[1] as usize];
                    let grand = profile["instrument"] == "pitched"
                        && (grand_name(profile) || grand_program(profile));
                    if !grand {
                        continue;
                    }
                    for j in i + 1..original.len() {
                        let second = original[j];
                        if first[..2] == second[..2]
                            && matches!((first[2], second[2]), (0, 1) | (1, 0))
                            && resolved[i][0] != resolved[j][0]
                        {
                            review_rows.extend([i + 1, j + 1]);
                        }
                    }
                }
            }
        }
    }
    let mut disagreement = false;
    if valid_original {
        if let (Some(systems), Some(original)) =
            (systems.filter(|s| s.len() == count), triples(&source))
        {
            let mut order = HashMap::new();
            for (index, row) in original.iter().enumerate() {
                let next = order.len();
                let actual = *order.entry(row[0]).or_insert(next);
                if actual != systems[index] {
                    disagreement = true;
                    review_rows.insert(index + 1);
                }
            }
        }
    }
    let needs_review = !review_rows.is_empty();
    let audit = json!({"raw_rows":source["rows"],"parsed_rows":parsed["rows"],
        "rows_changed":source["rows"] != parsed["rows"],
        "raw_rows_valid_without_geometry":valid_original,"systems_from_image":systems,
        "changes":changes,"needs_review":needs_review,"review_rows":review_rows,
        "review_reasons":if disagreement {vec!["system_evidence_disagreement"]}
            else if needs_review {vec!["piano_grand_staff_split_by_geometry"]} else {vec![]}});
    Ok((parsed, audit))
}

/// Assign stable global parts across pages, leaving geometry to score_grid.
/// Pages are sorted numerically. Unused model profiles never become output parts.
pub fn resolve_pages(pages: &mut [StructurePage]) -> Result<ResolvedPages> {
    pages.sort_by_key(|p| p.page);
    if pages.windows(2).any(|p| p[0].page == p[1].page) {
        return Err("Duplicate structure page".into());
    }
    // Page-local semantic profiles remain evidence. A positional majority
    // cannot replace a valid name/instrument when the arrangement changes.
    let mut output = ResolvedPages {
        parts: Vec::new(),
        systems: Vec::new(),
    };
    for page in pages {
        let assignments = require_triples(&page.structure)?;
        let batches = identity_batches(
            &assignments,
            page.structure["anonymous_relation_groups"] == true,
        )?;
        let profiles = page.structure["parts"]
            .as_array_mut()
            .ok_or("Missing structure parts")?;
        let mut mapping = HashMap::new();
        let mut identity_evidence = HashMap::new();
        for (system, rows) in batches {
            let (resolved, evidence) =
                resolve_identity_batch(profiles, &rows, page.page, system, &mut output.parts)?;
            mapping.extend(resolved);
            identity_evidence.extend(evidence);
        }
        let mut evidence: Vec<_> = identity_evidence.values().cloned().collect();
        evidence.sort_by_key(|e| e["local_part"].as_u64());
        page.structure["part_identity"] = json!(evidence);
        for (boxes, assignment) in page.rows.iter_mut().zip(&assignments) {
            let evidence = &identity_evidence[&(assignment[1] as usize)];
            for row in boxes {
                row["part_identity"] = evidence.clone();
                part_identity::attach_page_staff_evidence(
                    row,
                    &page.structure["parts"][assignment[1] as usize],
                );
            }
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
