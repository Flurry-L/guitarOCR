//! Keep confident visual groups without borrowing incompatible semantic labels.
//!
//! These groups describe observed row membership, not instrument identity or
//! correspondence across systems. This path never invokes template repair.
use super::{complementary, json_object, part_identity, triples, validate_row_topology, Result};
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

/// Return an anonymous structure only when both relation heads are trustworthy
/// and their adjacent-row partition conflicts with the original model rows.
/// Uncertain/malformed proposals remain for the caller's existing review path.
pub fn resolve_conflicting_groups(
    raw: &str,
    count: usize,
    systems: Option<&[usize]>,
    system_probabilities: &[f32],
    part_probabilities: Option<&[f32]>,
    modes: Option<&[String]>,
    review_confidence: f32,
    part_review_confidence: f32,
) -> Result<Option<(Value, Value)>> {
    if modes.is_some_and(|m| m.len() != count) {
        return Err("Row mode count does not cover every detected row".into());
    }
    for (threshold, name) in [
        (review_confidence, "Relation"),
        (part_review_confidence, "Part relation"),
    ] {
        if !threshold.is_finite() || !(0.5..=1.0).contains(&threshold) {
            return Err(format!("{name} review confidence must be in [0.5, 1]"));
        }
    }
    let (Some(systems), Some(part_probabilities)) = (systems, part_probabilities) else {
        return Ok(None);
    };
    let valid_probabilities = |p: &[f32]| {
        p.len() == count.saturating_sub(1)
            && p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    };
    let coherent = count > 0
        && systems.len() == count
        && systems[0] == 0
        && valid_probabilities(system_probabilities)
        && valid_probabilities(part_probabilities)
        && systems
            .windows(2)
            .zip(system_probabilities)
            .all(|(s, p)| s[0].checked_add(usize::from(*p < 0.5)) == Some(s[1]));
    if !coherent
        || !system_probabilities
            .iter()
            .all(|p| p.max(1. - p) >= review_confidence)
        || !systems
            .windows(2)
            .zip(part_probabilities)
            .all(|(s, p)| s[0] != s[1] || p.max(1. - p) >= part_review_confidence)
    {
        return Ok(None);
    }
    let source = json_object(raw)?;
    let Some(original) = triples(&source)
        .filter(|rows| rows.len() == count && rows.iter().flatten().all(|&value| value >= 0))
    else {
        return Ok(None);
    };
    let system_conflict = original
        .windows(2)
        .zip(systems.windows(2))
        .any(|(raw, learned)| (raw[0][0] == raw[1][0]) != (learned[0] == learned[1]));
    let part_conflict = original
        .windows(2)
        .zip(systems.windows(2))
        .zip(part_probabilities)
        .any(|((raw, system), p)| {
            system[0] == system[1]
                && ((raw[0][0] == raw[1][0] && raw[0][1] == raw[1][1]) != (*p >= 0.5))
        });
    if !system_conflict && !part_conflict {
        return Ok(None);
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
    let mut group_audit = vec![];
    for (part, indices) in groups.iter().enumerate() {
        let system = systems[indices[0]];
        let paired = modes.is_some_and(|m| complementary(m, indices));
        // Only the unambiguous two-row notation/TAB representation shares a
        // staff. All other staff slots follow observed row order, never a name.
        for (staff, &index) in indices.iter().enumerate() {
            assignments[index] = [system, part, if paired { 0 } else { staff }];
        }
        profiles.push(json!({
            "name":"", "name_source":"missing", "instrument":null,
            "program":null, "strings":null, "identity_status":"unknown",
            "anonymous_relation_group":true, "needs_review":true,
            "instrument_evidence":{"value":null,"status":"unknown",
                "source":"anonymous_relation_group","needs_review":true,"candidates":{}},
            "string_count_evidence":{"value":null,"status":"unknown",
                "source":"anonymous_relation_group","needs_review":true,"candidates":{}}
        }));
        group_audit.push(json!({"system":system,"part":part,
            "row_indices":indices,"paired_notation_tab":paired,
            "identity_status":"unknown"}));
    }
    let parsed = json!({"parts":profiles,"rows":assignments,
        "anonymous_relation_groups":true,"needs_review":true});
    if validate_row_topology(&parsed, count, modes).is_err() {
        return Err("Anonymous relation groups exceed supported staff topology".into());
    }
    let mut reasons = vec![];
    if system_conflict {
        reasons.push("system_evidence_disagreement");
    }
    if part_conflict {
        reasons.push("part_evidence_disagreement");
    }
    reasons.push("anonymous_part_identity");
    let audit = json!({
        "raw_rows":source["rows"],"parsed_rows":parsed["rows"],
        "rows_changed":source["rows"] != parsed["rows"],
        "raw_rows_valid_without_geometry":validate_row_topology(&source,count,modes).is_ok(),
        "systems_from_image":systems,"changes":[],"needs_review":true,
        "review_rows":(1..=count).collect::<Vec<_>>(),"review_reasons":reasons,
        "model_candidate":source,"model_candidate_error":null,
        "relation_candidate":parsed,"relation_candidate_error":null,
        "part_relation_candidate":parsed,"part_relation_candidate_error":null,
        "relation_probabilities":system_probabilities,
        "same_part_probabilities":part_probabilities,
        "system_indices_constrained":false,
        "selected_system_source":"visual_relation_network",
        "selected_part_source":"anonymous_visual_relation_groups",
        "semantic_binding":"unknown","anonymous_groups":group_audit,
        "semantic_profiles_reused":false,"template_repair_applied":false
    });
    Ok(Some((parsed, audit)))
}
