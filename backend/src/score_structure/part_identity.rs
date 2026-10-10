//! Cross-page identity requires a unique compatible printed name.
use super::{normalized_name, Result};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

fn name_key(part: &Value) -> String {
    if part["name_source"] == "missing" {
        String::new()
    } else {
        normalized_name(part["name"].as_str().unwrap_or(""))
    }
}

pub(crate) fn compatible_profile(previous: &Value, current: &Value) -> bool {
    let identities = [
        &previous["instrument_evidence"],
        &current["instrument_evidence"],
    ];
    if !identities
        .iter()
        .all(|e| e.as_object().is_some_and(|value| !value.is_empty()))
    {
        return ["instrument", "strings", "program"]
            .iter()
            .all(|key| previous[*key] == current[*key]);
    }
    if identities.iter().all(|e| {
        e["status"] == "resolved"
            && matches!(
                e["source"].as_str(),
                Some("label_text" | "confirmed_continuation")
            )
    }) && identities[0]["value"] != identities[1]["value"]
    {
        return false;
    }
    let counts = [
        &previous["string_count_evidence"],
        &current["string_count_evidence"],
    ];
    !(counts
        .iter()
        .all(|e| e.is_object() && e["needs_review"] != true && !e["value"].is_null())
        && counts[0]["value"] != counts[1]["value"])
}

pub(crate) fn resolve(
    profiles: &[Value],
    assignments: &[[i64; 3]],
    page: u64,
    parts: &mut Vec<Value>,
) -> Result<(HashMap<usize, usize>, HashMap<usize, Value>)> {
    let active: BTreeSet<usize> = assignments.iter().map(|row| row[1] as usize).collect();
    let mut counts = HashMap::<String, usize>::new();
    let mut continuation_counts = HashMap::<String, usize>::new();
    for &local in &active {
        let profile = profiles.get(local).ok_or("Unknown local part")?;
        *counts.entry(name_key(profile)).or_default() += 1;
        let continuation = &profile["continuation_evidence"];
        if continuation["source"] == "part_continuation_network"
            && continuation["status"] == "matched"
            && profile["staff_label_evidence"]["status"] == "absent"
        {
            *continuation_counts
                .entry(
                    continuation["matched_part_id"]
                        .as_str()
                        .unwrap_or("")
                        .to_owned(),
                )
                .or_default() += 1;
        }
    }
    let previous = parts.len();
    let mut mapping = HashMap::new();
    let mut evidence = HashMap::new();
    for local in active {
        let profile = &profiles[local];
        let name = name_key(profile);
        let continuation = &profile["continuation_evidence"];
        let confirmed_continuation = continuation["source"] == "part_continuation_network"
            && continuation["status"] == "matched"
            && profile["staff_label_evidence"]["status"] == "absent";
        let candidates: Vec<usize> = (0..previous)
            .filter(|&i| {
                if confirmed_continuation {
                    parts[i]["id"] == continuation["matched_part_id"]
                } else {
                    !name.is_empty() && name_key(&parts[i]) == name
                }
            })
            .collect();
        let reason = if confirmed_continuation && candidates.len() != 1 {
            Some("invalid_part_continuation")
        } else if confirmed_continuation
            && continuation_counts[continuation["matched_part_id"].as_str().unwrap_or("")] > 1
        {
            Some("competing_part_continuations")
        } else if name.is_empty() && !confirmed_continuation {
            Some("missing_part_name")
        } else if !confirmed_continuation && (counts[&name] > 1 || candidates.len() > 1) {
            Some("ambiguous_part_name")
        } else if candidates
            .first()
            .is_some_and(|&i| !compatible_profile(&parts[i], profile))
        {
            Some("conflicting_part_profile")
        } else {
            None
        };
        let matched = (candidates.len() == 1 && reason.is_none()).then(|| candidates[0]);
        let status = if matched.is_some() {
            "matched"
        } else if reason.is_some() {
            "unresolved"
        } else {
            "new"
        };
        let index = matched.unwrap_or_else(|| {
            let index = parts.len();
            let mut part = profile.clone();
            part["id"] = json!(format!("part-{}", index + 1));
            parts.push(part);
            index
        });
        mapping.insert(local, index);
        let mut identity = json!({
            "status":status,"source":if confirmed_continuation {"part_continuation_network"} else {"printed_name"},"page":page,"local_part":local,
            "name":profile["name"],"part_id":parts[index]["id"],
            "candidates":candidates.iter().map(|&i|parts[i]["id"].clone()).collect::<Vec<_>>(),
            "review_reason":reason,
        });
        if continuation
            .as_object()
            .is_some_and(|value| !value.is_empty())
        {
            identity["continuation_evidence"] = continuation.clone();
        }
        evidence.insert(local, identity);
    }
    Ok((mapping, evidence))
}

pub(super) fn attach_page_staff_evidence(record: &mut Value, profile: &Value) {
    if !["instrument_evidence", "string_count_evidence"]
        .iter()
        .any(|key| profile[*key].as_object().is_some_and(|e| !e.is_empty()))
    {
        return;
    }
    let mut evidence = json!({});
    for key in [
        "name",
        "name_source",
        "instrument_evidence",
        "string_count_evidence",
    ] {
        if let Some(value) = profile.get(key) {
            evidence[key] = value.clone();
        }
    }
    record["page_staff_evidence"] = evidence;
    if ["instrument_evidence", "string_count_evidence"]
        .iter()
        .any(|key| profile[*key]["needs_review"] == true)
    {
        record["pitch_needs_review"] = json!(true);
        record["needs_review"] = json!(true);
    }
}
