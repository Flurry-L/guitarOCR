//! Validate learned staff groups and match observed part identities across pages.
//! Invalid row assignments require model evidence or manual correction.
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use unicode_casefold::UnicodeCaseFold;

mod anonymous_relations;
mod part_identity;
pub use anonymous_relations::relation_groups;
pub(crate) use anonymous_relations::{identity_batches, resolve_identity_batch};
pub(crate) use part_identity::compatible_profile as compatible_part_profiles;
pub(crate) use part_identity::name_key as part_name_key;

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
fn complementary(modes: &[String], indices: &[usize]) -> bool {
    indices.len() == 2
        && ((modes[indices[0]] == "notation" && modes[indices[1]] == "tab")
            || (modes[indices[0]] == "tab" && modes[indices[1]] == "notation"))
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
        let batches = identity_batches(&assignments, page.structure["system_local_parts"] == true)?;
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
