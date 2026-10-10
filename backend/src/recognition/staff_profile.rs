//! Read one complete staff profile and retain its independent visual evidence.
use super::*;

pub(super) struct StaffProfileReader {
    count: Option<staff_evidence::StaffReader>,
    label: Option<staff_label::StaffLabelReader>,
    identity: Option<instrument_text::InstrumentTextReader>,
}

impl StaffProfileReader {
    pub(super) fn new(config: &InformationConfig) -> Result<Self> {
        if config.staff_label_model.is_some() != config.instrument_text_model.is_some() {
            return Err(
                "staff_label_model and instrument_text_model must be declared together".into(),
            );
        }
        Ok(Self {
            count: config
                .staff_evidence_model
                .as_deref()
                .map(staff_evidence::StaffReader::new)
                .transpose()?,
            label: config
                .staff_label_model
                .as_deref()
                .map(staff_label::StaffLabelReader::new)
                .transpose()?,
            identity: config
                .instrument_text_model
                .as_deref()
                .map(instrument_text::InstrumentTextReader::new)
                .transpose()?,
        })
    }

    pub(super) fn read<G: Generator + ?Sized>(
        &mut self,
        engine: &mut G,
        rows: &[Value],
        policy: Option<&ImagePolicy>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Value>> {
        let Some((mut region, image)) = staff_image(rows)? else {
            return Ok(None);
        };
        check(cancelled)?;
        let proposed_label = self
            .label
            .as_mut()
            .map(|reader| reader.predict(&image))
            .transpose()?;
        let mut staff_images = vec![image.clone()];
        if let Some(evidence) = proposed_label.as_ref() {
            if let Some(crop) = staff_label::label_crop(&image, evidence)? {
                staff_images.push(crop);
            }
        }
        let response = engine.generate_task(
            OcrTask::Staff,
            image_messages(&staff_images, STAFF_PROMPT, policy)?,
            512,
            Some(information_schema("staff")?),
        )?;
        if let Some(reader) = self.count.as_mut() {
            region["staff_evidence"] = reader.predict(&image)?;
        }
        region["parsed"] = parse_info_response(&response.content, "staff");
        if let Some(mut evidence) = proposed_label {
            evidence["text"] = Value::Null;
            evidence["raw"] = Value::Null;
            evidence["status"] = json!("absent");
            evidence["text_task_adapter_applied"] =
                json!(engine.task_adapter_applied(OcrTask::Staff));
            evidence["text_source"] = json!("complete_staff_with_label_proposal");
            if staff_images.len() > 1 {
                evidence["text"] = cleaned(&region["parsed"]["name"]);
                evidence["raw"] = json!(response.content);
                evidence["status"] = json!(if evidence["text"].is_null() {
                    "unreadable"
                } else {
                    "read"
                });
            }
            evidence["staff_prediction"] = region["parsed"].clone();
            region["parsed"]["name"] = evidence["text"].clone();
            if let Some(reader) = self.identity.as_mut() {
                evidence["identity"] = reader.predict(evidence["text"].as_str())?;
                region["parsed"]["instrument"] = evidence["identity"]["value"].clone();
            }
            region["staff_label_evidence"] = evidence;
        }
        region["task_adapter_applied"] = json!(engine.task_adapter_applied(OcrTask::Staff));
        region["raw"] = json!(response.content.trim());
        region["candidate_kind"] = json!("staff");
        Ok(Some(region))
    }
}

/// Local evidence must be resolved before the profile can name a global part.
pub(super) fn apply(profile: &mut Value, prediction: &Value, has_tab: bool) {
    let original = profile.clone();
    let parsed = &prediction["parsed"];
    let label = prediction.get("staff_label_evidence");
    let count = staff_evidence::resolve_string_count(
        has_tab,
        &parsed["string_count"],
        &original["strings"],
        &prediction["staff_evidence"],
        None,
        flag(prediction, "task_adapter_applied"),
    );
    let identity = staff_evidence::resolve_instrument_with_label(
        has_tab,
        &parsed["instrument"],
        &original["instrument"],
        None,
        label,
        None,
    );
    profile["structure_prediction"] = original;
    profile["staff_prediction"] = prediction.clone();
    profile["instrument_evidence"] = identity.clone();
    profile["string_count_evidence"] = count.clone();
    profile["strings"] = count["value"].clone();
    if let Some(label) = label {
        profile["staff_label_evidence"] = label.clone();
        profile["name"] = if label["status"] == "read" {
            label["text"].clone()
        } else {
            json!("")
        };
        profile["name_source"] = json!(if text(profile, "name", "").is_empty() {
            "missing"
        } else {
            "label_text"
        });
    }
    if let Some(instrument) = identity["value"].as_str() {
        profile["instrument"] = json!(instrument);
        let program = program_from_visible_name(text(profile, "name", ""));
        profile["program"] = json!(program.unwrap_or(match instrument {
            "guitar" => 25,
            "bass" => 33,
            _ => 0,
        }));
        profile["program_source"] = json!(if program.is_some() {
            "printed_name"
        } else {
            "category_default"
        });
    } else {
        profile["program_source"] = json!("provisional");
    }
}
