//! Evidence-aware signature propagation; display defaults are not facts.
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

fn observation(prediction: &Value, field: &str) -> Value {
    if prediction["evidence"][field].is_object() {
        return prediction["evidence"][field].clone();
    }
    let source = text(prediction, "source", "missing_evidence");
    let value = &prediction[field];
    if matches!(source, "manual" | "visual_ocr") && prediction["error"].is_null() {
        json!({"status":if value.is_null(){"absent"}else{"observed"},"value":value,"source":source})
    } else {
        json!({"status":"unknown","value":null,"candidate":value,"source":source})
    }
}

/// Accept edited signatures without treating omitted compatibility values as facts.
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
        }
    }
    row["state_needs_review"] = json!(
        !trusted(&evidence["time"]["status"])
            || (row["mode"] != "tab" && !trusted(&evidence["key"]["status"]))
    );
    row["score_state"] = state;
    row["score_state_evidence"] = evidence;
    row.as_object_mut()
        .ok_or("Invalid measure")?
        .remove("_signature_evidence_before_ocr");
    Ok(())
}

/// Context usable for retries; raw score_state defaults have no authority.
pub fn accepted_context(row: &Value) -> Result<Value> {
    let mut context = row.clone();
    if flag(row, "manually_edited") {
        if let Some(target) = row["target"].as_str() {
            observe_manual_metadata(&mut context, target)?;
        }
    }
    let mut evidence = json!({});
    for field in ["time", "key"] {
        evidence[field] = if context["score_state_evidence"][field].is_object() {
            context["score_state_evidence"][field].clone()
        } else {
            json!({"status":"unknown","value":null,"source":"missing_evidence"})
        };
        if evidence[field]["status"] == "unknown"
            && context["ocr_signature_observations"][field]["status"] == "absent"
        {
            evidence[field] = context["ocr_signature_observations"][field].clone();
        }
    }
    Ok(evidence)
}

/// Merge accepted facts without letting old journals erase verified observations.
pub fn seed_accepted_context(prediction: &Value, row: &Value) -> Result<Value> {
    let mut result = prediction.clone();
    let mut evidence = accepted_context(row)?;
    for field in ["time", "key"] {
        let current = observation(prediction, field);
        if evidence[field]["status"] == "inherited" {
            evidence[field]["status"] = json!("observed");
        } else if evidence[field]["status"] == "unknown"
            || (evidence[field]["status"] == "absent"
                && matches!(current["status"].as_str(), Some("observed" | "conflict")))
        {
            evidence[field] = current;
        }
    }
    result["evidence"] = evidence;
    Ok(result)
}

pub fn resolve(records: &mut [Value], predictions: &Value) -> Result<()> {
    let mut observations: HashMap<(String, i64), Vec<Value>> = HashMap::new();
    for row in records.iter() {
        let prediction = predictions
            .get(row["measure_number"].to_string())
            .ok_or("Missing signature prediction")?;
        let time = if flag(row, "manually_edited") {
            accepted_context(row)?["time"].clone()
        } else {
            observation(prediction, "time")
        };
        if (time["status"] == "observed" && !time["value"].is_null())
            || time["status"] == "conflict"
        {
            observations
                .entry((text(row, "source_id", "").into(), bar_index(row)))
                .or_default()
                .push(time);
        }
    }
    let mut states: HashMap<String, Value> = HashMap::new();
    for row in records {
        if flag(row, "manually_edited") {
            states.insert(staff_key(row), accepted_context(row)?);
            continue;
        }
        let prediction = &predictions[row["measure_number"].to_string()];
        let previous = states
            .get(&staff_key(row))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let mut current = json!({});
        for field in ["time", "key"] {
            let value = observation(prediction, field);
            current[field] = if value["status"] == "observed" && !value["value"].is_null() {
                let mut observed = value;
                observed["measure_number"] = row["measure_number"].clone();
                observed
            } else if value["status"] == "absent" && trusted(&previous[field]["status"]) {
                let mut inherited = previous[field].clone();
                inherited["status"] = json!("inherited");
                inherited["observation"] = value;
                inherited
            } else {
                let mut unknown = value;
                unknown["status"] = json!(if unknown["status"] == "conflict" {
                    "conflict"
                } else {
                    "unknown"
                });
                unknown["value"] = Value::Null;
                if previous.get(field).is_some() {
                    unknown["previous_candidate"] = previous[field]["value"].clone();
                }
                unknown
            };
        }
        if let Some(shared) = observations.get(&(text(row, "source_id", "").into(), bar_index(row)))
        {
            let alternatives: std::collections::BTreeSet<&str> = shared
                .iter()
                .flat_map(|value| {
                    value["value"]
                        .as_str()
                        .into_iter()
                        .chain(list(value, "candidates").iter().filter_map(Value::as_str))
                })
                .collect();
            if alternatives.len() > 1 || shared.iter().any(|value| value["status"] == "conflict") {
                if current["time"]["status"] == "conflict" {
                    current["time"]["candidates"] = json!(alternatives);
                } else {
                    current["time"] = json!({"status":"conflict","value":null,"candidates":alternatives,"source":"simultaneous_staves"});
                }
            } else if !alternatives.is_empty() {
                current["time"] = shared[0].clone();
                current["time"]["status"] = json!("observed");
                current["time"]["source"] = json!("shared_printed_signature");
                current["time"]["measure_number"] = row["measure_number"].clone();
            }
        }
        let mut compatible = json!({});
        for (field, default) in [("time", json!("4/4")), ("key", json!(0))] {
            compatible[field] = if !current[field]["value"].is_null() {
                current[field]["value"].clone()
            } else if !row["score_state"][field].is_null() {
                row["score_state"][field].clone()
            } else if !previous[field]["value"].is_null() {
                previous[field]["value"].clone()
            } else {
                default
            };
        }
        let needs_review = !trusted(&current["time"]["status"])
            || (row["mode"] != "tab" && !trusted(&current["key"]["status"]));
        states.insert(staff_key(row), current.clone());
        row["score_state"] = compatible;
        row["score_state_evidence"] = current;
        row["state_needs_review"] = json!(needs_review);
        row["printed_state"] = prediction.clone();
    }
    Ok(())
}

/// Fuse the main visual OCR's explicit metadata without changing its text.
pub fn observe_ocr_metadata(row: &mut Value, target: &str) -> Result<()> {
    if flag(row, "manually_edited") {
        return Ok(());
    }
    let measure = score::parse_measure_target(target)?;
    let mut candidates = vec![("time", measure["time_signature"].clone())];
    if row["mode"] != "tab" {
        candidates.push(("key", Value::Null));
        if let Some(key) = measure["key_signature"].as_str() {
            let (fifths, _) = key_parts(key)?;
            let fifths = if fifths < -7 {
                fifths + 12
            } else if fifths > 7 {
                fifths - 12
            } else {
                fifths
            };
            candidates.pop();
            candidates.push(("key", json!(fifths)));
        }
    }
    if !row["_signature_evidence_before_ocr"].is_object() {
        row["_signature_evidence_before_ocr"] = if row["score_state_evidence"].is_object() {
            row["score_state_evidence"].clone()
        } else {
            json!({})
        };
    }
    let mut evidence = row["_signature_evidence_before_ocr"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let mut ocr_observations = json!({});
    let mut state = if row["score_state"].is_object() {
        row["score_state"].clone()
    } else {
        json!({"time":"4/4","key":0})
    };
    for (field, value) in candidates {
        ocr_observations[field] = json!({"status":if value.is_null(){"absent"}else{"observed"},"value":value,"source":"visual_measure_ocr"});
        if value.is_null() {
            continue;
        }
        let previous = evidence.get(field).cloned().unwrap_or_else(|| json!({}));
        let next = if previous["status"] == "conflict" {
            let mut next = previous;
            let mut candidates = list(&next, "candidates").to_vec();
            if !candidates.contains(&value) {
                candidates.push(value.clone());
            }
            next["candidates"] = json!(candidates);
            next["ocr_candidate"] = value;
            next
        } else if trusted(&previous["status"]) && previous["value"] != value {
            json!({"status":"conflict","value":null,"candidates":[previous["value"],value],
                "source":"signature_and_measure_ocr","signature_evidence":previous,"ocr_candidate":value})
        } else {
            state[field] = value.clone();
            json!({"status":"observed","value":value,"source":"visual_measure_ocr",
                "measure_number":row["measure_number"],"signature_evidence":previous})
        };
        evidence.insert(field.into(), next);
    }
    let evidence = Value::Object(evidence);
    row["state_needs_review"] = json!(
        !trusted(&evidence["time"]["status"])
            || (row["mode"] != "tab" && !trusted(&evidence["key"]["status"]))
    );
    row["score_state"] = state;
    row["ocr_signature_observations"] = ocr_observations;
    row["score_state_evidence"] = evidence;
    Ok(())
}

pub fn finalize_ocr_states(records: &mut [Value]) -> Result<()> {
    let mut predictions = json!({});
    for row in records.iter() {
        let mut prediction = if row["printed_state"].is_object() {
            row["printed_state"].clone()
        } else {
            json!({})
        };
        let mut evidence = json!({});
        for field in ["time", "key"] {
            evidence[field] = observation(&prediction, field);
            let ocr_observation = &row["ocr_signature_observations"][field];
            if evidence[field]["status"] == "unknown" && ocr_observation["status"] == "absent" {
                evidence[field] = ocr_observation.clone();
            }
            let value = &row["score_state_evidence"][field];
            if (matches!(
                value["source"].as_str(),
                Some("visual_measure_ocr" | "manual")
            ) && value["status"] == "observed")
                || value["status"] == "conflict"
            {
                evidence[field] = value.clone();
            }
        }
        prediction["evidence"] = evidence;
        predictions[row["measure_number"].to_string()] = prediction;
    }
    resolve(records, &predictions)?;
    for row in records {
        row.as_object_mut()
            .ok_or("Invalid measure")?
            .remove("_signature_evidence_before_ocr");
    }
    Ok(())
}
