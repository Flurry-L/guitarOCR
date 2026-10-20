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
            )?;
            let value = information::parse_response(&response.content, "header_text")
                ["header_notes"]
                .clone();
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
    }
}

/// Geometry selects context only; musical ownership requires an existing
/// recognized event or an explicit manual association.
fn visible_context(record: &Value, annotation: &Value) -> bool {
    if record["page"] != annotation["page"]
        || (!annotation["part_id"].is_null() && record["part_id"] != annotation["part_id"])
    {
        return false;
    }
    let Ok(a) = bbox(annotation) else {
        return false;
    };
    let Ok(b) = bbox(record) else {
        return false;
    };
    // Include annotations already entering the crop's existing 1.5 mm / 4 mm
    // margins. This selects visible context, never a beat or a duration.
    let px = 1.5 * f64::from(image_boundary::MODEL_RENDER_DPI) / 25.4;
    let py = 4. * f64::from(image_boundary::MODEL_RENDER_DPI) / 25.4;
    (b[0] + b[2] + px).min(a[0] + a[2]) > (b[0] - px).max(a[0])
        && (b[1] + b[3] + py).min(a[1] + a[3]) > (b[1] - py).max(a[1])
}

fn staff_system(record: &Value) -> Option<(i64, String, String, i64)> {
    Some((
        record["page"].as_i64()?,
        record["part_id"].as_str()?.to_owned(),
        record["staff_id"].as_str()?.to_owned(),
        record["system_index"]
            .as_i64()
            .or(record["row_index"].as_i64())?,
    ))
}

pub(super) fn prepare_regions(
    records: &mut [Value],
    predictions: &mut [Value],
    output: Option<&Path>,
) -> Result<()> {
    for p in predictions
        .iter_mut()
        .filter(|p| matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram"))
    {
        let a = bbox(p)?;
        // A margin may touch the next staff's diagrams. Establish the staff
        // from unpadded boxes and the recognized score structure first. If
        // those boxes disagree or supply no owner, retain the original crop
        // and the separate annotation instead of expanding into another row.
        let groups: Option<HashSet<_>> = records
            .iter()
            .filter(|r| {
                visible_context(r, p)
                    && bbox(r).is_ok_and(|b| {
                        (b[0] + b[2]).min(a[0] + a[2]) > b[0].max(a[0])
                            && (b[1] + b[3]).min(a[1] + a[3]) > b[1].max(a[1])
                    })
            })
            .map(staff_system)
            .collect();
        let owner = groups
            .filter(|groups| groups.len() == 1)
            .and_then(|groups| groups.into_iter().next());
        if !(p["scope"] == "measure" && p["scope_source"] == "manual") {
            p["scope"] = json!("library");
            p["scope_source"] = json!("unresolved");
            p.as_object_mut().unwrap().remove("measure_number");
        }
        p["context_measures"] = json!(records
            .iter()
            .filter(|record| owner.is_some()
                && staff_system(record) == owner
                && visible_context(record, p))
            .filter(|record| {
                let Ok(b) = bbox(record) else { return false };
                let px = 1.5 * f64::from(image_boundary::MODEL_RENDER_DPI) / 25.4;
                let py = 4. * f64::from(image_boundary::MODEL_RENDER_DPI) / 25.4;
                let top = b[1].min(a[1]) - py;
                let bottom = (b[1] + b[3]).max(a[1] + a[3]) + py;
                records.iter().all(|other| {
                    if other["page"] != record["page"] || staff_system(other) == owner {
                        return true;
                    }
                    let Some([staff_top, staff_bottom]) =
                        crate::layout_postprocess::staff_band(other)
                    else {
                        return true;
                    };
                    let Ok(other_box) = bbox(other) else {
                        return true;
                    };
                    if (b[0] + b[2] + px).min(other_box[0] + other_box[2])
                        <= (b[0] - px).max(other_box[0])
                    {
                        return true;
                    }
                    // Completing a diagram must not introduce another staff's
                    // actual music into a crop that did not contain it before.
                    bottom.min(staff_bottom) <= top.max(staff_top)
                        || (b[1] + b[3] + py).min(staff_bottom) > (b[1] - py).max(staff_top)
                })
            })
            .map(|record| record["measure_number"].clone())
            .collect::<Vec<_>>());
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
                    && text(p, "part_id", "part-1") == text(r, "part_id", "part-1")
                    && matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram")
            })
            .cloned()
            .collect();
        let context: Vec<Value> = predictions
            .iter()
            .filter(|p| {
                list(p, "context_measures").contains(&r["measure_number"]) && visible_context(r, p)
            })
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
        // Keep the music and the full height of names/grids already entering
        // this measure. Preserve its left/right boundaries: expanding them
        // would pull in neighbouring notes. The separate annotation crop
        // retains a grid's complete geometry when it crosses a bar boundary.
        for p in &context {
            let a = bbox(p)?;
            bounds[1] = bounds[1].min(a[1]);
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

fn fuse_chord_name(row: &mut Value, predictions: &[Value]) -> Result<()> {
    if flag(row, "manually_edited") {
        return Ok(());
    }
    let measure = match row["target"].as_str() {
        Some(target) => score::parse_measure_target(target)?,
        None => row["resolved"].clone(),
    };
    let mut invalid = Vec::new();
    let mut used = HashSet::new();
    for voice in list(&measure, "voices") {
        for event in list(voice, "events") {
            for effect in list(event, "effects").iter().filter_map(Value::as_str) {
                let Some(name) = effect.strip_prefix("chord:").map(unquote) else {
                    continue;
                };
                if is_chord_symbol(&name) {
                    used.insert(chord_key(&name));
                } else {
                    invalid.push((
                        voice["voice"].clone(),
                        event["start"].clone(),
                        effect.to_owned(),
                        name,
                    ));
                }
            }
        }
    }
    let [(voice, start, original_effect, original_name)] = invalid.as_slice() else {
        return Ok(());
    };
    // A damaged chord spelling has a note-name prefix. A title or an IR
    // technique such as "hammer" is a different classification error.
    let key = chord_key(original_name);
    let mut letters = key.chars();
    let root = letters.next();
    let suffix = letters.as_str();
    let chord_prefix = matches!(root, Some('A'..='H'))
        && (suffix.starts_with(['#', 'b', '/', '(', '+', '-', '°', 'ø', 'Δ'])
            || suffix.starts_with(|c: char| c.is_ascii_digit())
            || ["maj", "min", "dim", "aug", "sus", "add", "omit", "no", "m"]
                .iter()
                .any(|prefix| suffix.starts_with(prefix)));
    if !chord_prefix || original_name.chars().count() > 32 || !start.is_i64() {
        return Ok(());
    }
    let mut candidates = BTreeMap::new();
    for p in predictions {
        let name = text(&p["parsed"], "text", "");
        let name_key = chord_key(name);
        if p["page"] == row["page"]
            && text(p, "part_id", "part-1") == text(row, "part_id", "part-1")
            && list(p, "context_measures") == std::slice::from_ref(&row["measure_number"])
            && visible_context(row, p)
            && matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram")
            && scorelib::chords::decision(p) != Some("rejected")
            && p["name_evidence"]["status"] == "unique"
            && p["chord_evidence"]["name_needs_review"] != true
            && is_chord_symbol(name)
            && !used.contains(&name_key)
        {
            candidates.entry(name_key).or_insert(p);
        }
    }
    if candidates.len() != 1 {
        return Ok(());
    }
    let source = *candidates.values().next().unwrap();
    let name = text(&source["parsed"], "text", "");
    let replacement = format!(
        "chord:{}",
        percent_encoding::utf8_percent_encode(name, percent_encoding::NON_ALPHANUMERIC)
    );
    let replace = |measure: &mut Value| {
        for v in measure
            .get_mut("voices")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            if v["voice"] != *voice {
                continue;
            }
            for event in v["events"].as_array_mut().into_iter().flatten() {
                if event["start"] != *start {
                    continue;
                }
                for effect in event["effects"].as_array_mut().into_iter().flatten() {
                    if effect.as_str() == Some(original_effect.as_str()) {
                        *effect = json!(replacement);
                    }
                }
            }
        }
    };
    let mode = text(row, "mode", "notation").to_owned();
    if row["target"].is_string() {
        for field in ["target", "written_target", "ir"] {
            if let Some(target) = row[field].as_str() {
                let mut measure = score::parse_measure_target(target)?;
                replace(&mut measure);
                row[field] = json!(score::format_measure_target(&measure, &mode, true)?);
            }
        }
    }
    replace(&mut row["resolved"]);
    row["chord_name_corrections"] = json!([{"voice":voice,"start":start,
        "from":original_name,"to":name,"source":"local_name_reader",
        "page":source["page"],"bbox":source["bbox"]}]);
    Ok(())
}

pub fn attach_chord_annotations(records: &mut [Value], predictions: &[Value]) -> Result<()> {
    for row in records.iter_mut() {
        fuse_chord_name(row, predictions)?;
    }
    // Read beat evidence before updating any record. A diagram can enter two
    // crop margins while only one measure actually prints its chord event.
    let mut chord_owners = Vec::new();
    for row in records.iter() {
        let Some(target) = row["target"].as_str() else {
            continue;
        };
        let measure = score::parse_measure_target(target)?;
        let names = list(&measure, "voices")
            .iter()
            .flat_map(|voice| list(voice, "events"))
            .filter_map(event_chord_name)
            .map(|name| chord_key(&name))
            .collect::<HashSet<_>>();
        chord_owners.push((
            json!({"part_id":row["part_id"],"page":row["page"],
                "measure_number":row["measure_number"],"bbox":row["bbox"]}),
            names,
        ));
    }
    let annotations: Vec<&Value> = predictions
        .iter()
        .filter(|p| scorelib::chords::decision(p) != Some("rejected"))
        .filter(|p| matches!(text(&p["parsed"], "kind", ""), "chord" | "chord_diagram"))
        .collect();
    for row in records {
        if flag(row, "manually_edited") {
            continue;
        }
        let part: Vec<&Value> = annotations
            .iter()
            .copied()
            .filter(|p| text(p, "part_id", "part-1") == text(row, "part_id", "part-1"))
            .collect();
        // Geometry limits the candidate set. Only the measure reader's
        // visible chord event establishes musical timing.
        let manual = |p: &&Value| p["scope"] == "measure" && p["scope_source"] == "manual";
        let local: Vec<&Value> = part
            .iter()
            .copied()
            .filter(|p| manual(p) && p["measure_number"] == row["measure_number"])
            .collect();
        row["chord_annotations"] = json!(local);
        let visible: Vec<&Value> = part
            .iter()
            .copied()
            .filter(|p| !manual(p))
            .filter(|p| {
                list(p, "context_measures").contains(&row["measure_number"])
                    && visible_context(row, p)
            })
            .collect();
        let mut choices = Vec::new();
        let Some(target) = row["target"].as_str() else {
            row["chord_diagram_choices"] = json!([]);
            continue;
        };
        let mut m = score::parse_measure_target(target)?;
        let mut changed = false;
        for voice in m["voices"].as_array_mut().unwrap() {
            let voice_number = voice["voice"].clone();
            for event in voice["events"].as_array_mut().unwrap() {
                // Diagrams are supplied by the dedicated reader. Recompute
                // automatic attachments so an old guess cannot survive a
                // changed or ambiguous local candidate set.
                if let Some(effects) = event["effects"].as_array_mut() {
                    let before = effects.len();
                    effects.retain(|effect| {
                        !effect.as_str().is_some_and(|s| s.starts_with("diagram:"))
                    });
                    changed |= before != effects.len();
                }
                let name = list(event, "effects")
                    .iter()
                    .filter_map(Value::as_str)
                    .find_map(|s| s.strip_prefix("chord:"))
                    .map(unquote);
                let Some(name) = name else {
                    continue;
                };
                let same_name =
                    |p: &&Value| chord_key(text(&p["parsed"], "text", "")) == chord_key(&name);
                let mut matched = local.iter().copied().filter(same_name).collect::<Vec<_>>();
                if matched.is_empty() {
                    matched = visible.iter().copied().filter(same_name).collect();
                }
                let pending = matched.iter().any(|p| scorelib::chords::pending(p));
                let ambiguous = matched.iter().any(|p| {
                    !manual(p)
                        && p["parsed"]["diagram"].is_object()
                        && chord_owners
                            .iter()
                            .filter(|(owner, names)| {
                                text(owner, "part_id", "part-1") == text(row, "part_id", "part-1")
                                    && list(p, "context_measures")
                                        .contains(&owner["measure_number"])
                                    && visible_context(owner, p)
                                    && names.contains(&chord_key(&name))
                            })
                            .count()
                            != 1
                });
                let diagrams: HashSet<String> = matched
                    .into_iter()
                    .flat_map(|p| {
                        let evidence = &p["chord_evidence"];
                        let mut shapes = vec![&p["parsed"]["diagram"]];
                        if scorelib::chords::pending(p) {
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
                if diagrams.len() > 1 || pending || ambiguous {
                    let mut candidates = diagrams.iter().cloned().collect::<Vec<_>>();
                    candidates.sort();
                    choices.push(json!({"voice":voice_number,"start":event["start"],"name":name,"diagrams":candidates,
                        "source":if pending {"chord_evidence"} else {"library"},
                        "reason":if pending {"图示识别尚有未决或冲突的指法，请对照原谱选择候选。"} else if ambiguous {"指法图同时进入多个小节，尚不能确定归属，请按原谱选择当前拍的按法。"} else {"当前小节的同名和弦存在多种指法，请按原谱选择当前拍的按法。"}}));
                    row["needs_review"] = json!(true);
                }
                if diagrams.len() == 1 && !pending && !ambiguous {
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
                // Chord diagrams can make the content box much taller than
                // its musical staff. Locate technique marks against measured
                // staff lines whenever that geometry is available.
                let [top, bottom] =
                    crate::layout_postprocess::staff_band(r).unwrap_or([b[1], b[1] + b[3]]);
                let gap = (top - a[1] - a[3]).max(a[1] - bottom).max(0.);
                if gap > 48f64.max((bottom - top) * 0.2) {
                    return None;
                }
                Some((i, gap, (a[1] + a[3] / 2. - (top + bottom) / 2.).abs()))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
            .map(|x| x.0);
        let Some(owner) = owner else {
            continue;
        };
        let system = staff_system(&records[owner]);
        for (index, r) in records.iter_mut().enumerate() {
            let c = bbox(r)?;
            if !system.as_ref().map_or(index == owner, |system| {
                staff_system(r).as_ref() == Some(system)
            }) || (a[0] + a[2]).min(c[0] + c[2]) <= a[0].max(c[0])
            {
                continue;
            }
            r["technique_annotations"]
                .as_array_mut()
                .unwrap()
                .push(p.clone());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_local_name_corrects_only_the_existing_chord_slot() {
        let mut row = json!({"measure_number":1,"page":1,"part_id":"guitar","mode":"tab",
            "bbox":[0,0,300,200],"target":"M | V0{@0:q:s1f5<chord:E> @960:q:s1f7<chord:F%23E>}",
            "raw_observation_text":"original model answer",
            "observation":{"voices":[{"voice":0,"events":[
                {"start":0,"notes":[{"string":1,"fret":5}],"effects":["chord:E"]},
                {"start":960,"notes":[{"string":1,"fret":7}],"effects":["chord:F%23E"]}]}]}});
        row["written_target"] = row["target"].clone();
        row["resolved"] = row["observation"].clone();
        let original = row.clone();
        let candidate = json!({"page":1,"part_id":"guitar","bbox":[160,10,40,60],
            "context_measures":[1],"parsed":{"kind":"chord_diagram","text":"F♯/E"},
            "name_evidence":{"status":"unique","candidates":["F♯/E"]},
            "chord_evidence":{"name_needs_review":false},"image":"name.png"});
        fuse_chord_name(&mut row, std::slice::from_ref(&candidate)).unwrap();
        assert_eq!(row["observation"], original["observation"]);
        assert_eq!(
            row["raw_observation_text"],
            original["raw_observation_text"]
        );
        assert_eq!(row["target"], row["written_target"]);
        let measure = score::parse_measure_target(row["target"].as_str().unwrap()).unwrap();
        let event = &measure["voices"][0]["events"][1];
        assert_eq!(event["start"], 960);
        assert_eq!(event["notes"][0]["fret"], 7);
        assert_eq!(event_chord_name(event).as_deref(), Some("F♯/E"));
        assert_eq!(row["chord_name_corrections"][0]["from"], "F#E");

        let mut partial = original.clone();
        partial["target"] = Value::Null;
        let written = partial["written_target"].clone();
        fuse_chord_name(&mut partial, std::slice::from_ref(&candidate)).unwrap();
        assert!(partial["target"].is_null());
        assert_eq!(partial["written_target"], written);
        assert_eq!(partial["observation"], original["observation"]);
        assert_eq!(
            event_chord_name(&partial["resolved"]["voices"][0]["events"][1]).as_deref(),
            Some("F♯/E")
        );

        let mut crossing = candidate.clone();
        crossing["context_measures"] = json!([1, 2]);
        let mut unchanged = original.clone();
        fuse_chord_name(&mut unchanged, &[crossing]).unwrap();
        assert_eq!(unchanged, original);
        let mut other = candidate.clone();
        other["parsed"]["text"] = json!("G");
        let mut unchanged = original.clone();
        fuse_chord_name(&mut unchanged, &[candidate.clone(), other]).unwrap();
        assert_eq!(unchanged, original);
        for name in ["hammer", "Chorus"] {
            let mut unchanged = original.clone();
            unchanged["target"] = json!(format!("M | V0{{@0:q:s1f5<chord:{name}>}}"));
            let before = unchanged.clone();
            fuse_chord_name(&mut unchanged, std::slice::from_ref(&candidate)).unwrap();
            assert_eq!(unchanged, before);
        }
    }

    #[test]
    fn chord_context_preserves_music_and_attaches_only_local_evidence() {
        let output = tempfile::tempdir().unwrap();
        let page_path = output.path().join("page.png");
        let page = RgbImage::from_fn(600, 700, |x, y| {
            Rgb([(x % 251) as u8, (y % 251) as u8, ((x + y) % 251) as u8])
        });
        page.save(&page_path).unwrap();
        let measure = |number, b: [i32; 4], name: &str| {
            json!({"measure_number":number,"part_id":"guitar","staff_id":"tab",
                "page":1,"system_index":b[1],"source_page":page_path,"bbox":b,"mode":"tab",
                "tuning":[64,59,55,50,45,40],
                "target":format!("M | V0{{@0:q:s1f0{}}}",
                    if name.is_empty() {String::new()} else {format!("<chord:{name}>")})})
        };
        let shape = |base| {
            json!({"base_fret":base,"frets":["x",base,base+2,base+2,base+2,"x"],
                "fingers":[null,1,3,3,3,null],"barres":[]})
        };
        let diagram = |b: [i32; 4], name, base| {
            json!({"page":1,"part_id":"guitar","bbox":b,
                "parsed":{"kind":"chord_diagram","text":name,"diagram":shape(base)}})
        };
        let mut named = diagram([120, 155, 55, 70], "?", 7);
        named["chord_evidence"] = json!({"accepted":{"geometry":true,"full":true},
            "review_required":false});
        record_name_evidence(&mut named, vec![], Default::default());
        assert!(scorelib::chords::pending(&named));
        record_name_evidence(&mut named, vec![], ["E".to_owned()].into());
        assert!(!scorelib::chords::pending(&named));
        named["chord_evidence"]["review_required"] = json!(true);
        record_name_evidence(&mut named, vec![], ["E".to_owned()].into());
        assert!(scorelib::chords::pending(&named));
        let mut records = vec![
            measure(1, [100, 200, 160, 100], "E"),
            measure(2, [320, 200, 160, 100], "E"),
            measure(3, [100, 450, 160, 100], "G"),
            measure(4, [320, 450, 160, 100], ""),
        ];
        let mut predictions = vec![
            diagram([120, 155, 55, 70], "E", 7),
            diagram([350, 155, 55, 70], "E", 2),
            diagram([500, 20, 55, 70], "G", 3),
            // A margin alone does not establish a diagram's owning staff.
            diagram([210, 150, 40, 25], "F#", 9),
            // Complete geometry stays in its separate crop; the measure
            // must not widen rightwards to include neighbouring notes.
            diagram([240, 165, 45, 60], "E", 7),
            json!({"page":1,"part_id":"guitar","bbox":[135,150,35,25],
                "parsed":{"kind":"technique","text":"P.M."}}),
        ];
        prepare_regions(&mut records, &mut predictions, Some(output.path())).unwrap();
        assert_eq!(predictions[0]["context_measures"], json!([1]));
        assert_eq!(predictions[1]["context_measures"], json!([2]));
        assert_eq!(predictions[2]["context_measures"], json!([]));
        assert_eq!(predictions[3]["context_measures"], json!([]));
        assert_eq!(records[0]["content_bbox"], json!([100., 150., 160., 150.]));
        assert_eq!(records[0]["bbox"], json!([100, 200, 160, 100]));
        let crop = record_image(&records[0]).unwrap();
        let old =
            image_boundary::measure_crop_bounds(600, 700, [100., 200., 160., 100.], 1.5).unwrap();
        let new =
            image_boundary::measure_crop_bounds(600, 700, [100., 150., 160., 150.], 1.5).unwrap();
        assert_eq!((old[0], old[2]), (new[0], new[2]));
        for y in old[1]..old[3] {
            for x in old[0]..old[2] {
                assert_eq!(crop.get_pixel(x - new[0], y - new[1]), page.get_pixel(x, y));
            }
        }
        for (x, y) in [(120, 155), (174, 224), (140, 160), (210, 150)] {
            assert_eq!(crop.get_pixel(x - new[0], y - new[1]), page.get_pixel(x, y));
        }
        attach_chord_annotations(&mut records, &predictions).unwrap();
        assert!(text(&records[0], "target", "").contains(&diagram_effect(&shape(7)).unwrap()));
        assert!(text(&records[1], "target", "").contains(&diagram_effect(&shape(2)).unwrap()));
        assert!(!text(&records[2], "target", "").contains("diagram:"));
        assert!(!text(&records[3], "target", "").contains("chord:"));

        // A second local fingering makes the association unresolved; an old
        // automatically attached shape must not survive that ambiguity.
        let mut alternative = diagram([180, 155, 30, 70], "E", 2);
        alternative["context_measures"] = json!([1]);
        predictions.push(alternative);
        attach_chord_annotations(&mut records, &predictions).unwrap();
        assert!(!text(&records[0], "target", "").contains("diagram:"));
        assert_eq!(
            records[0]["chord_diagram_choices"][0]["diagrams"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(text(&records[1], "target", "").contains(&diagram_effect(&shape(2)).unwrap()));

        // A grid crossing two crops cannot acquire ownership from proximity.
        predictions.push(diagram([250, 165, 80, 60], "E", 7));
        prepare_regions(&mut records, &mut predictions, None).unwrap();
        assert_eq!(
            predictions.last().unwrap()["context_measures"],
            json!([1, 2])
        );
        let crossing = vec![predictions.last().unwrap().clone()];
        let mut owners = vec![
            measure(1, [100, 200, 160, 100], ""),
            measure(2, [320, 200, 160, 100], "E"),
        ];
        attach_chord_annotations(&mut owners, &crossing).unwrap();
        assert!(!text(&owners[0], "target", "").contains("diagram:"));
        assert!(text(&owners[1], "target", "").contains(&diagram_effect(&shape(7)).unwrap()));
        owners[0]["target"] = measure(1, [100, 200, 160, 100], "E")["target"].clone();
        attach_chord_annotations(&mut owners, &crossing).unwrap();
        assert!(owners
            .iter()
            .all(|row| !text(row, "target", "").contains("diagram:")));
        attach_chord_annotations(&mut records, &predictions).unwrap();
        assert!(!text(&records[1], "target", "").contains("diagram:"));
        assert_eq!(
            records[1]["chord_diagram_choices"][0]["diagrams"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        // An explicit local association overrides automatic context choices.
        let mut manual = diagram([500, 20, 55, 70], "E", 5);
        merge(
            &mut manual,
            &json!({"scope":"measure","scope_source":"manual","measure_number":1}),
        );
        predictions.push(manual);
        attach_chord_annotations(&mut records, &predictions).unwrap();
        assert!(text(&records[0], "target", "").contains(&diagram_effect(&shape(5)).unwrap()));
        assert_eq!(records[0]["chord_diagram_choices"], json!([]));

        // The upper crop's bottom margin touches the next row's diagram.
        // Its actual staff identity prevents an expansion into that row.
        let mut vertical = vec![
            measure(1, [100, 200, 160, 100], "E"),
            measure(2, [100, 440, 160, 100], "E"),
        ];
        let mut lower_diagram = vec![diagram([120, 310, 55, 150], "E", 7)];
        prepare_regions(&mut vertical, &mut lower_diagram, Some(output.path())).unwrap();
        assert_eq!(lower_diagram[0]["context_measures"], json!([2]));
        assert!(vertical[0]["content_bbox"].is_null());
        assert_eq!(vertical[1]["content_bbox"], json!([100., 310., 160., 230.]));
        vertical[0]["staff_band"] = json!([230., 295.]);
        prepare_regions(&mut vertical, &mut lower_diagram, None).unwrap();
        assert_eq!(lower_diagram[0]["context_measures"], json!([]));
        vertical[0]["bbox"] = json!([100, 200, 160, 160]);
        prepare_regions(&mut vertical, &mut lower_diagram, None).unwrap();
        assert_eq!(lower_diagram[0]["context_measures"], json!([]));

        // A tall chord area must neither split a technique within one system
        // nor make a neighbouring system appear to own a mark below the staff.
        let mut techniques = vec![
            measure(1, [100, 100, 160, 300], ""),
            measure(2, [260, 280, 160, 120], ""),
            measure(3, [100, 370, 160, 180], ""),
        ];
        for row in &mut techniques[..2] {
            row["system_index"] = json!(0);
            row["staff_band"] = json!([300., 360.]);
        }
        techniques[2]["system_index"] = json!(1);
        techniques[2]["staff_band"] = json!([470., 530.]);
        let marks = vec![
            json!({"page":1,"part_id":"guitar","bbox":[120,270,300,18],
                "parsed":{"kind":"technique","text":"P.M."}}),
            json!({"page":1,"part_id":"guitar","bbox":[120,375,30,15],
                "parsed":{"kind":"technique","text":"let ring"}}),
        ];
        attach_technique_regions(&mut techniques, &marks).unwrap();
        assert_eq!(list(&techniques[0], "technique_annotations").len(), 2);
        assert_eq!(list(&techniques[1], "technique_annotations").len(), 1);
        assert!(list(&techniques[2], "technique_annotations").is_empty());

        // Editing the library updates both stored copies and recomputes an
        // automatic beat attachment without replacing user-edited music.
        let sid = "11111111111111111111111111111111";
        let directory = output.path().join(sid);
        std::fs::create_dir(&directory).unwrap();
        let mut row = measure(1, [100, 200, 160, 100], "E");
        merge(
            &mut row,
            &json!({"source_id":"score","bar_index":0,"instrument":"guitar",
            "capo":0,"tuning_source":"manual","tuning_explicit":true,
            "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
                "voices":[{"voice":0,"events":[{"start":0,"duration":"q","status":"normal",
                    "notes":[{"string":1,"fret":0}],"effects":["chord:E"]}]}],"relations":[]}}),
        );
        let mut source = json!({"mode":"tab","instrument":"guitar","tuning_used":[64,59,55,50,45,40],
            "parts":[{"id":"guitar","document_metadata":{"score_annotations":[predictions[0]]}}],
            "document_metadata":{},"records":[row]});
        reassemble(&mut source).unwrap();
        let info_path = directory.join("info.json");
        crate::edit::json_file(&info_path, &source).unwrap();
        let manifest = crate::edit::save_recognition(&directory, &source).unwrap();
        let mut state = json!({"id":sid,"revision":0,"info":info_path,"recognition":manifest});
        crate::project::write_session_atomic(output.path(), sid, &state).unwrap();
        let body = json!({"part_id":"guitar","decision":"confirm","name":"E","diagram":shape(5)});
        crate::edit::chord_library::edit(output.path(), sid, 0, &body, Some("0")).unwrap();
        state = crate::project::load_session(output.path(), sid).unwrap();
        source = crate::view::read_json(Path::new(state["recognition"].as_str().unwrap())).unwrap();
        let info = crate::view::read_json(Path::new(state["info"].as_str().unwrap())).unwrap();
        assert_eq!(
            info["parts"][0]["document_metadata"],
            source["parts"][0]["document_metadata"]
        );
        assert!(
            text(&source["records"][0], "target", "").contains(&diagram_effect(&shape(5)).unwrap())
        );
        source["records"][0]["manually_edited"] = json!(true);
        let manual_target = source["records"][0]["target"].clone();
        state["recognition"] = json!(crate::edit::save_recognition(&directory, &source).unwrap());
        crate::project::write_session_atomic(output.path(), sid, &state).unwrap();
        let body = json!({"part_id":"guitar","decision":"confirm","name":"E","diagram":shape(9)});
        crate::edit::chord_library::edit(output.path(), sid, 0, &body, Some("1")).unwrap();
        state = crate::project::load_session(output.path(), sid).unwrap();
        source = crate::view::read_json(Path::new(state["recognition"].as_str().unwrap())).unwrap();
        assert_eq!(source["records"][0]["target"], manual_target);
        assert_eq!(
            source["parts"][0]["document_metadata"]["score_annotations"][0]["parsed"]["diagram"],
            shape(9)
        );
    }
}
