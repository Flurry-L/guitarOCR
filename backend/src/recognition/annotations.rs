//! Printed chord and technique annotations: text evidence, crop context and musical ownership.
use super::*;
use crate::text_regions::TextRegionReader;

/// Verify diagram names using contextual text roles while preserving their grids.
pub(super) fn recover_chord_names<G: Generator + ?Sized>(
    engine: &mut G,
    predictions: &mut [Value],
    reader: &mut TextRegionReader,
    roles: &mut crate::chord_text_roles::ChordTextRoleReader,
    policy: Option<&ImagePolicy>,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    for prediction in predictions {
        if prediction["parsed"]["kind"] != "chord_diagram" {
            continue;
        }
        check(cancelled)?;
        let image = region_image(prediction)?;
        let mut attempts = vec![];
        let mut candidates = std::collections::BTreeSet::new();
        let proposals = reader.predict(&image)?;
        let role_predictions = roles.predict(&image, &proposals)?;
        for (mut proposal, role) in proposals.into_iter().zip(role_predictions) {
            proposal["role_evidence"] = role.clone();
            if role["is_name"] != true {
                attempts.push(proposal);
                continue;
            }
            check(cancelled)?;
            let b = bbox(&proposal)?;
            let bounds = [
                b[0].floor(),
                b[1].floor(),
                (b[0] + b[2]).ceil() - b[0].floor(),
                (b[1] + b[3]).ceil() - b[1].floor(),
            ];
            let crop = direct_crop(&image, bounds)?;
            let response = engine.generate(
                image_messages(&[crop], DIAGRAM_NAME_PROMPT, policy)?,
                64,
                Some(information_schema("header_text")?),
                None,
            )?;
            let value =
                parse_info_response(&response.content, "header_text")["header_notes"].clone();
            if let Some(name) = value.as_str().filter(|s| is_chord_symbol(s)) {
                candidates.insert(name.to_owned());
            }
            let mut attempt = proposal;
            merge(
                &mut attempt,
                &json!({"raw":response.content,"text":value,
                "generated_token_count":response.completion_tokens}),
            );
            attempts.push(attempt);
        }
        record_name_evidence(prediction, attempts, candidates);
    }
    Ok(())
}

fn record_name_evidence(
    prediction: &mut Value,
    attempts: Vec<Value>,
    candidates: std::collections::BTreeSet<String>,
) {
    prediction["name_evidence"] = json!({"source":"text_region_network_and_ocr",
            "original":prediction["parsed"]["text"], "attempts":attempts,"candidates":candidates,
            "status":if candidates.len()==1 {"unique"} else {"unresolved"}});
    if !prediction["chord_evidence"].is_object() {
        prediction["chord_evidence"] = json!({});
    }
    prediction["chord_evidence"]["name_needs_review"] = json!(candidates.len() != 1);
    if candidates.len() == 1 {
        prediction["parsed"]["text"] = json!(candidates.first().unwrap());
    } else {
        prediction["chord_evidence"]["review_required"] = json!(true);
    }
}

/// Geometry selects context only; musical ownership requires an existing
/// recognized event or an explicit manual association.
pub(super) fn prepare_regions(
    records: &mut [Value],
    predictions: &mut [Value],
    output: Option<&Path>,
) -> Result<()> {
    for p in predictions
        .iter_mut()
        .filter(|p| matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram"))
    {
        if p["scope"] == "measure" && p["scope_source"] == "manual" {
            continue;
        }
        let a = bbox(p)?;
        let candidates = records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                if r["page"] != p["page"]
                    || (!p["part_id"].is_null() && p["part_id"] != r["part_id"])
                {
                    return None;
                }
                let b = bbox(r).ok()?;
                Some((
                    i,
                    (b[1] - a[1] - a[3]).max(a[1] - b[1] - b[3]).max(0.),
                    (b[0] - a[0] - a[2]).max(a[0] - b[0] - b[2]).max(0.),
                ))
            })
            .collect::<Vec<_>>();
        let nearest = candidates
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)));
        p["scope"] = json!("library");
        p["scope_source"] = json!("unresolved");
        p["context_measures"] = json!(candidates
            .iter()
            .filter(|candidate| nearest
                .is_some_and(|best| candidate.1 == best.1 && candidate.2 == best.2))
            .map(|candidate| records[candidate.0]["measure_number"].clone())
            .collect::<Vec<_>>());
        p.as_object_mut().unwrap().remove("measure_number");
    }
    attach_technique_regions(records, predictions)?;
    let mut page_cache = None;
    for r in records {
        let annotations: Vec<Value> = predictions
            .iter()
            .filter(|p| {
                p["scope"] == "measure"
                    && p["scope_source"] == "manual"
                    && p["measure_number"] == r["measure_number"]
                    && matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram")
            })
            .cloned()
            .collect();
        let context: Vec<Value> = predictions
            .iter()
            .filter(|p| list(p, "context_measures").contains(&r["measure_number"]))
            .cloned()
            .collect();
        r["chord_annotations"] = json!(annotations);
        r["chord_context_candidates"] = json!(context);
        let technique_annotations = list(r, "technique_annotations").to_vec();
        if annotations.is_empty() && context.is_empty() && technique_annotations.is_empty() {
            continue;
        }
        let b = bbox(r)?;
        let mut bounds = [b[0], b[1], b[0] + b[2], b[1] + b[3]];
        // Complete text already intersecting the crop, while standalone grids
        // retain their separate images and cannot shrink the first bar's music.
        let local = context
            .iter()
            .filter(|p| p["parsed"]["kind"] == "chord" && p["chord_evidence"]["kind"] != "diagram")
            .filter(|p| {
                bbox(p).is_ok_and(|a| {
                    (b[0] + b[2]).min(a[0] + a[2]) > b[0].max(a[0])
                        && (b[1] + b[3]).min(a[1] + a[3]) > b[1].max(a[1])
                })
            });
        for p in annotations.iter().chain(local) {
            let a = bbox(p)?;
            bounds[0] = bounds[0].min(a[0]);
            bounds[1] = bounds[1].min(a[1]);
            bounds[2] = bounds[2].max(a[0] + a[2]);
            bounds[3] = bounds[3].max(a[1] + a[3]);
        }
        // Include the printed technique above/below the staff, but do not
        // widen into neighbouring measures along a long continuation line.
        for p in &technique_annotations {
            let a = bbox(p)?;
            bounds[1] = bounds[1].min(a[1]);
            bounds[3] = bounds[3].max(a[1] + a[3]);
        }
        let Some(output) = output else {
            continue;
        };
        if bounds == [b[0], b[1], b[0] + b[2], b[1] + b[3]] {
            continue;
        }
        let source = PathBuf::from(text(r, "source_page", ""));
        if page_cache.as_ref().is_none_or(|(path, _)| path != &source) {
            page_cache = Some((source.clone(), image_boundary::open_rgb(&source)?));
        }
        let crop = [
            bounds[0],
            bounds[1],
            bounds[2] - bounds[0],
            bounds[3] - bounds[1],
        ];
        std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
        let path = output.join(format!("measure-{}-annotations.png", r["measure_number"]));
        image_boundary::crop_measure(&page_cache.as_ref().unwrap().1, crop)?
            .save(&path)
            .map_err(|e| e.to_string())?;
        r["image"] = json!(path);
        r["content_bbox"] = json!(crop);
    }
    Ok(())
}

pub fn attach_chord_annotations(records: &mut [Value], predictions: &[Value]) -> Result<()> {
    let annotations: Vec<&Value> = predictions
        .iter()
        .filter(|p| scorelib::chords::decision(p) != Some("rejected"))
        .filter(|p| matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram"))
        .collect();
    for row in records {
        let part: Vec<&Value> = annotations
            .iter()
            .copied()
            .filter(|p| text(p, "part_id", "part-1") == text(row, "part_id", "part-1"))
            .collect();
        // Legacy automatic scope=measure records were proximity guesses.
        // Only explicit manual scope may choose a local diagram over a library.
        let manual = |p: &&Value| p["scope"] == "measure" && p["scope_source"] == "manual";
        let local: Vec<&Value> = part
            .iter()
            .copied()
            .filter(|p| manual(p) && p["measure_number"] == row["measure_number"])
            .collect();
        row["chord_annotations"] = json!(local);
        let mut choices = Vec::new();
        let mut m = score::parse_measure_target(text(row, "target", ""))?;
        let mut changed = false;
        for voice in m["voices"].as_array_mut().unwrap() {
            let voice_number = voice["voice"].clone();
            for event in voice["events"].as_array_mut().unwrap() {
                let name = list(event, "effects")
                    .iter()
                    .filter_map(Value::as_str)
                    .find_map(|s| s.strip_prefix("chord:"))
                    .map(unquote);
                let Some(name) = name else {
                    continue;
                };
                let has_diagram = list(event, "effects")
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|s| s.starts_with("diagram:"));
                let matching = |values: &[&Value]| -> (HashSet<String>, bool) {
                    let matched = values
                        .iter()
                        .filter(|p| chord_key(text(&p["parsed"], "text", "")) == chord_key(&name))
                        .collect::<Vec<_>>();
                    let pending = matched.iter().any(|p| scorelib::chords::pending(p));
                    let diagrams = matched
                        .into_iter()
                        .flat_map(|p| {
                            let evidence = &p["chord_evidence"];
                            let pending = scorelib::chords::pending(p);
                            let mut shapes = vec![&p["parsed"]["diagram"]];
                            if pending {
                                for key in [
                                    "proposed_diagram",
                                    "diagram",
                                    "secondary_diagram",
                                    "original_diagram",
                                    "rejected_diagram_candidate",
                                ] {
                                    shapes.push(&evidence[key]);
                                }
                            }
                            shapes
                                .into_iter()
                                .filter_map(|shape| {
                                    let d = normalize_diagram(shape);
                                    if d.is_null()
                                        || (!list(row, "tuning").is_empty()
                                            && list(&d, "frets").len() != list(row, "tuning").len())
                                    {
                                        None
                                    } else {
                                        diagram_effect(&d).ok()
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    (diagrams, pending)
                };
                let candidates = matching(&local);
                let (diagrams, pending) = if candidates.0.is_empty() && !candidates.1 {
                    matching(
                        &part
                            .iter()
                            .copied()
                            .filter(|p| !manual(p))
                            .collect::<Vec<_>>(),
                    )
                } else {
                    candidates
                };
                if has_diagram && (row["manually_edited"] == true || !pending) {
                    continue;
                }
                if diagrams.len() > 1 || pending {
                    let mut candidates = diagrams.iter().cloned().collect::<Vec<_>>();
                    candidates.sort();
                    choices.push(json!({"voice":voice_number,"start":event["start"],"name":name,"diagrams":candidates,
                        "source":if pending {"chord_evidence"} else {"library"},
                        "reason":if pending {"图示识别尚有未决或冲突的指法，请对照原谱选择候选并确认小节。"} else {"同名和弦存在多种指法，已完整保留在指法库，请按原谱选择当前拍的按法。"}}));
                    if pending {
                        row["needs_review"] = json!(true);
                    }
                }
                if diagrams.len() == 1 && !pending {
                    event["effects"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!(diagrams.into_iter().next().unwrap()));
                    changed = true;
                }
            }
        }
        row["chord_diagram_choices"] = json!(choices);
        if changed {
            row["target"] = json!(score::format_measure_target(
                &m,
                text(row, "mode", "notation"),
                true
            )?);
        }
    }
    Ok(())
}

fn attach_technique_regions(records: &mut [Value], predictions: &[Value]) -> Result<()> {
    for r in records.iter_mut() {
        r["technique_annotations"] = json!([]);
    }
    for p in predictions
        .iter()
        .filter(|p| p["parsed"]["kind"] == "technique")
    {
        let a = bbox(p)?;
        let owner = records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                if r["page"] != p["page"]
                    || (!p["part_id"].is_null() && p["part_id"] != r["part_id"])
                {
                    return None;
                }
                let b = bbox(r).ok()?;
                if (a[0] + a[2]).min(b[0] + b[2]) <= a[0].max(b[0]) {
                    return None;
                }
                let gap = (b[1] - a[1] - a[3]).max(a[1] - b[1] - b[3]).max(0.);
                if gap > 48f64.max(b[3] * 0.2) {
                    return None;
                }
                Some((i, gap, (a[1] + a[3] / 2. - b[1] - b[3] / 2.).abs()))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
            .map(|x| x.0);
        let Some(owner) = owner else {
            continue;
        };
        let staff = records[owner].clone();
        let b = bbox(&staff)?;
        for r in records.iter_mut() {
            let c = bbox(r)?;
            if r["page"] != staff["page"]
                || r["part_id"] != staff["part_id"]
                || r["staff_id"] != staff["staff_id"]
                || (c[1] + c[3] / 2. - b[1] - b[3] / 2.).abs() > 0.35 * b[3].min(c[3])
                || (a[0] + a[2]).min(c[0] + c[2]) <= a[0].max(c[0])
            {
                continue;
            }
            let mut evidence = p.clone();
            // The text may overhang into the next bar. Only the bar containing
            // the label's start can request a missing-mark review.
            evidence["label_starts_here"] = json!(a[0] >= c[0] - 8. && a[0] < c[0] + c[2] - 8.);
            r["technique_annotations"]
                .as_array_mut()
                .unwrap()
                .push(evidence);
        }
    }
    Ok(())
}
