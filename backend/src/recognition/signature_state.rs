//! Trusted signature access and explicit edits; observation assembly propagates state.
use super::*;

fn trusted(status: &Value) -> bool {
    matches!(status.as_str(), Some("observed" | "inherited"))
}

pub fn trusted_state(row: &Value) -> Value {
    let state = &row["score_state"];
    let Some(evidence) = row["score_state_evidence"].as_object() else {
        return json!({});
    };
    let mut result = json!({});
    for field in ["time", "key"] {
        if evidence.get(field).is_some_and(|v| trusted(&v["status"])) && !state[field].is_null() {
            result[field] = state[field].clone();
        }
    }
    result
}

/// Record only signatures explicitly supplied by an edit.
pub fn observe_manual_metadata(row: &mut Value, target: &str) -> Result<()> {
    let measure = score::parse_measure_target(target)?;
    let mut evidence = json!({});
    let mut state = if row["score_state"].is_object() {
        row["score_state"].clone()
    } else {
        json!({})
    };
    for field in ["time", "key"] {
        evidence[field] = if row["score_state_evidence"][field].is_object() {
            row["score_state_evidence"][field].clone()
        } else {
            json!({"status":"unknown","value":null,"source":"missing_evidence"})
        };
    }
    let mut explicit = vec![("time", measure["time_signature"].clone())];
    if row["mode"] != "tab" {
        if let Some(key) = measure["key_signature"].as_str() {
            let (fifths, _) = key_parts(key)?;
            explicit.push((
                "key",
                json!(if fifths < -7 {
                    fifths + 12
                } else if fifths > 7 {
                    fifths - 12
                } else {
                    fifths
                }),
            ));
        }
    }
    for (field, value) in explicit {
        if !value.is_null() {
            state[field] = value.clone();
            evidence[field] = json!({"status":"observed","value":value,"source":"manual","measure_number":row["measure_number"]});
            if !row["manual_state"].is_object() {
                row["manual_state"] = json!({});
            }
            row["manual_state"][field] = evidence[field].clone();
        }
    }
    row["state_needs_review"] = json!(
        !trusted(&evidence["time"]["status"])
            || (row["mode"] != "tab" && !trusted(&evidence["key"]["status"]))
    );
    row["score_state"] = state;
    row["score_state_evidence"] = evidence;
    Ok(())
}
