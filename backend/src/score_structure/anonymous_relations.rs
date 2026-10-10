//! Build visible staff groups; semantic identity is read separately.
use super::{complementary, part_identity, validate_row_topology, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

pub(crate) fn identity_batches(
    assignments: &[[i64; 3]],
    anonymous: bool,
) -> Result<Vec<(Option<i64>, Vec<[i64; 3]>)>> {
    if !anonymous {
        return Ok(vec![(None, assignments.to_vec())]);
    }
    let mut systems = BTreeMap::<i64, Vec<[i64; 3]>>::new();
    let mut part_system = HashMap::new();
    for &row in assignments {
        if row.iter().any(|&value| value < 0)
            || part_system
                .insert(row[1], row[0])
                .is_some_and(|old| old != row[0])
        {
            return Err("Anonymous parts must belong to one nonnegative system".into());
        }
        systems.entry(row[0]).or_default().push(row);
    }
    Ok(systems
        .into_iter()
        .map(|(system, rows)| (Some(system), rows))
        .collect())
}

/// Shared by recognition's continuation preview and the actual score grid.
pub(crate) fn resolve_identity_batch(
    profiles: &mut [Value],
    assignments: &[[i64; 3]],
    page: u64,
    system: Option<i64>,
    parts: &mut Vec<Value>,
) -> Result<(HashMap<usize, usize>, HashMap<usize, Value>)> {
    let (mapping, mut evidence) = part_identity::resolve(profiles, assignments, page, parts)?;
    if let Some(system) = system {
        for (&local, item) in &mut evidence {
            item["system"] = json!(system);
            let profile = &mut profiles[local];
            let confirmed = item["review_reason"].is_null()
                && profile["instrument_evidence"]["status"] == "resolved";
            profile["identity_status"] = json!(if confirmed { "resolved" } else { "unknown" });
            profile["needs_review"] = json!(!confirmed);
        }
    }
    Ok((mapping, evidence))
}

/// Adjacent-row classifiers own grouping; low confidence is review evidence,
/// never a request for a language model to invent a replacement topology.
pub fn relation_groups(
    systems: &[usize],
    system_probabilities: &[f32],
    part_probabilities: &[f32],
    modes: &[String],
    review_confidence: f32,
    part_review_confidence: f32,
) -> Result<(Value, Value)> {
    let count = modes.len();
    let valid = |p: &[f32]| {
        p.len() == count.saturating_sub(1)
            && p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    };
    if count == 0
        || systems.len() != count
        || systems[0] != 0
        || !valid(system_probabilities)
        || !valid(part_probabilities)
        || !systems
            .windows(2)
            .zip(system_probabilities)
            .all(|(s, p)| s[0].checked_add(usize::from(*p < 0.5)) == Some(s[1]))
        || [review_confidence, part_review_confidence]
            .iter()
            .any(|v| !v.is_finite() || !(0.5..=1.0).contains(v))
    {
        return Err("Invalid staff relation prediction".into());
    }
    let mut review_rows = std::collections::BTreeSet::new();
    for (i, pair) in systems.windows(2).enumerate() {
        let s = system_probabilities[i];
        let p = part_probabilities[i];
        if s.max(1. - s) < review_confidence
            || (pair[0] == pair[1] && p.max(1. - p) < part_review_confidence)
        {
            review_rows.extend([i + 1, i + 2]);
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    for index in 0..count {
        if index == 0 || systems[index] != systems[index - 1] || part_probabilities[index - 1] < 0.5
        {
            groups.push(vec![]);
        }
        groups.last_mut().unwrap().push(index);
    }
    let mut profiles = vec![];
    let mut assignments = vec![[0usize; 3]; count];
    for (part, indices) in groups.iter().enumerate() {
        let paired = complementary(modes, indices);
        for (staff, &index) in indices.iter().enumerate() {
            assignments[index] = [systems[index], part, if paired { 0 } else { staff }];
        }
        profiles.push(
            json!({"name":"", "name_source":"missing", "instrument":null,
            "program":null,"strings":null,"identity_status":"unknown",
            "instrument_evidence":{"value":null,"status":"unknown"},
            "string_count_evidence":{"value":null,"status":"unknown"}}),
        );
    }
    let needs_review = !review_rows.is_empty();
    let parsed = json!({"parts":profiles,"rows":assignments,"system_local_parts":true});
    validate_row_topology(&parsed, count, Some(modes))?;
    let audit = json!({"parsed_rows":assignments,"systems_from_image":systems,
        "needs_review":needs_review,"review_rows":review_rows,
        "review_reasons":if needs_review {vec!["uncertain_row_relation"]} else {vec![]},
        "relation_probabilities":system_probabilities,"same_part_probabilities":part_probabilities,
        "selected_system_source":"visual_relation_network", "selected_part_source":"visual_relation_network"});
    Ok((parsed, audit))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_preserve_simultaneous_staves_and_uncertain_boundaries() {
        let modes = vec!["notation".to_owned(); 4];
        let (piano, audit) = relation_groups(
            &[0, 0, 1, 1],
            &[0.999, 0.001, 0.999],
            &[0.999, 0.001, 0.999],
            &modes,
            0.95,
            0.95,
        )
        .unwrap();
        assert_eq!(
            piano["rows"],
            json!([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
        );
        assert_eq!(audit["needs_review"], false);
        let (ensemble, audit) = relation_groups(
            &[0, 0, 1, 1],
            &[0.999, 0.001, 0.999],
            &[0.067, 0.001, 0.001],
            &modes,
            0.95,
            0.95,
        )
        .unwrap();
        assert_eq!(
            ensemble["rows"],
            json!([[0, 0, 0], [0, 1, 0], [1, 2, 0], [1, 3, 0]])
        );
        assert_eq!(audit["review_rows"], json!([1, 2]));
        let (dual, _) = relation_groups(
            &[0, 0],
            &[0.999],
            &[0.999],
            &["notation".into(), "tab".into()],
            0.95,
            0.95,
        )
        .unwrap();
        assert_eq!(dual["rows"], json!([[0, 0, 0], [0, 0, 0]]));
    }
}
