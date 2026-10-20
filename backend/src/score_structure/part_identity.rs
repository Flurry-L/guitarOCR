//! Match parts using printed identity or learned continuation evidence.
use super::{normalized_name, Result};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

pub(crate) fn name_key(part: &Value) -> String {
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
            && matches!(
                profile["staff_label_evidence"]["status"].as_str(),
                Some("absent" | "unreadable")
            )
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
            && matches!(
                profile["staff_label_evidence"]["status"].as_str(),
                Some("absent" | "unreadable")
            );
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
            // The first staff groups establish parts even without printed names.
            // A later anonymous group needs evidence that it continues a part.
            (previous > 0).then_some("missing_part_name")
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
        // A failure to match an existing part is not evidence of a new part.
        // The first group establishes the score's parts; later groups need a
        // distinct printed identity or a resolved continuation.
        if previous > 0 && matched.is_none() && reason.is_some() {
            return Err(format!(
                "第 {page} 页的谱行尚不能确定音轨归属（{}），请检查谱行区域后重新识别谱面信息。",
                reason.unwrap()
            ));
        }
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
            "status":status,"source":if confirmed_continuation {"part_continuation_network"} else if previous == 0 && name.is_empty() {"initial_staff_group"} else {"printed_name"},"page":page,"local_part":local,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_unnamed_staff_groups_establish_parts_without_review() {
        let mut profiles: Vec<Value> = ["pitched", "drums"]
            .into_iter()
            .map(|instrument| json!({"name":"", "name_source":"missing",
                "instrument":instrument, "instrument_evidence":{"value":instrument,"status":"resolved"}}))
            .collect();
        let mut parts = vec![];
        let (mapping, evidence) = super::super::resolve_identity_batch(
            &mut profiles,
            &[[0, 0, 0], [0, 1, 0]],
            1,
            Some(0),
            &mut parts,
        )
        .unwrap();
        assert_eq!(parts.len(), 2);
        for local in 0..2 {
            assert_eq!(mapping[&local], local);
            assert_eq!(evidence[&local]["status"], "new");
            assert_eq!(evidence[&local]["source"], "initial_staff_group");
            assert!(evidence[&local]["review_reason"].is_null());
            assert_eq!(profiles[local]["identity_status"], "resolved");
            assert_eq!(profiles[local]["needs_review"], false);
        }
    }

    #[test]
    fn uncertain_continuation_does_not_create_a_track() {
        let mut parts = vec![
            json!({"id":"part-1", "name":"Guitar", "instrument":"guitar",
            "strings":6, "program":25}),
        ];
        let mut profile = json!({"name":"", "name_source":"missing", "instrument":"guitar",
            "strings":6, "program":25, "staff_label_evidence":{"status":"absent"},
            "continuation_evidence":{"source":"part_continuation_network", "status":"unresolved"}});
        assert!(resolve(&[profile.clone()], &[[0, 0, 0]], 2, &mut parts).is_err());
        assert_eq!(parts.len(), 1);
        profile["continuation_evidence"]["status"] = json!("matched");
        profile["continuation_evidence"]["matched_part_id"] = json!("part-1");
        let (mapping, _) = resolve(&[profile.clone()], &[[0, 0, 0]], 2, &mut parts).unwrap();
        assert_eq!(mapping[&0], 0);
        assert_eq!(parts.len(), 1);
        // A false label proposal with no readable name is not contrary identity evidence.
        profile["staff_label_evidence"]["status"] = json!("unreadable");
        let (mapping, _) = resolve(&[profile.clone()], &[[0, 0, 0]], 2, &mut parts).unwrap();
        assert_eq!(mapping[&0], 0);
        let identity = crate::recognition::staff_evidence::resolve_instrument_with_label(
            false,
            &json!("pitched"),
            None,
            Some(&profile["staff_label_evidence"]),
            Some(&json!({"value":"guitar","source":"label_text","status":"resolved"})),
        );
        assert_eq!(identity["value"], "guitar");
        assert_eq!(identity["source"], "confirmed_continuation");
        profile["staff_label_evidence"]["status"] = json!("read");
        assert!(resolve(&[profile], &[[0, 0, 0]], 2, &mut parts).is_err());
        assert_eq!(parts.len(), 1);
        let (mapping, _) = resolve(
            &[json!({"name":"Bass", "instrument":"bass", "strings":4, "program":33})],
            &[[0, 0, 0]],
            3,
            &mut parts,
        )
        .unwrap();
        assert_eq!(mapping[&0], 1);
    }
}
