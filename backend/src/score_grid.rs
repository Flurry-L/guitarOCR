//! Align simultaneous measures and fuse paired notation/TAB rows without changing part identities.
use crate::image_boundary::{crop_measure, open_rgb, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
pub type Member = (Vec<Value>, usize, usize);
pub type AlignedMember = (Vec<(Value, usize)>, usize, usize);
fn bbox(row: &Value) -> Result<[f64; 4]> {
    let a = row["bbox"].as_array().ok_or("Measure needs bbox")?;
    if a.len() != 4 {
        return Err("Measure bbox must have four values".into());
    }
    let mut b = [0.; 4];
    for (i, v) in a.iter().enumerate() {
        b[i] = v
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("Measure bbox must be finite")?;
    }
    if b[2] <= 0. || b[3] <= 0. {
        return Err("Measure bbox must have positive width/height".into());
    }
    Ok(b)
}
fn score(r: &Value) -> f64 {
    r["score"].as_f64().filter(|v| v.is_finite()).unwrap_or(1.)
}
fn mark_structure_review(record: &mut Value, reasons: &[Value]) {
    let mut merged = record["structure_review_reasons"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for reason in reasons {
        if !merged.contains(reason) {
            merged.push(reason.clone());
        }
    }
    record["structure_needs_review"] = json!(true);
    record["structure_review_reasons"] = json!(merged);
    record["needs_review"] = json!(true);
}
fn save_crop(record: &mut Value, bounds: [f64; 4], path: &Path) -> Result<()> {
    let source = record["source_page"]
        .as_str()
        .ok_or("Resolved crop needs source_page")?;
    let image = open_rgb(Path::new(source))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    crop_measure(&image, bounds)?
        .save(path)
        .map_err(|e| e.to_string())?;
    record["bbox"] = json!(bounds);
    record["image"] = json!(path
        .canonicalize()
        .map_err(|e| e.to_string())?
        .to_string_lossy());
    Ok(())
}
pub fn align_system(members: &[Member]) -> Result<(Vec<AlignedMember>, usize)> {
    let mut rows: Vec<Vec<Value>> = members.iter().map(|(rows, _, _)| rows.clone()).collect();
    if rows.is_empty() || rows.iter().any(Vec::is_empty) {
        return Err("A score system must contain nonempty staff rows".into());
    }
    for row in &mut rows {
        for record in row.iter() {
            bbox(record)?;
        }
        row.sort_by(|a, b| {
            a["bbox"][0]
                .as_f64()
                .unwrap()
                .total_cmp(&b["bbox"][0].as_f64().unwrap())
        });
    }
    let reference = rows
        .iter()
        .max_by(|a, b| {
            a.len().cmp(&b.len()).then_with(|| {
                a.iter()
                    .map(score)
                    .sum::<f64>()
                    .total_cmp(&b.iter().map(score).sum::<f64>())
            })
        })
        .unwrap();
    let count = reference.len();
    let reasons = if rows.iter().any(|row| row.len() != count) {
        vec!["conflicting_measure_counts"]
    } else if !corresponding_columns(&rows)? {
        vec!["ambiguous_measure_columns"]
    } else {
        vec![]
    };
    let mut result = vec![];
    for (row, (_, part, staff)) in rows.iter().zip(members) {
        let columns = if row.len() == count {
            (0..count).collect()
        } else {
            ordered_match(row, reference)?
        };
        let mut records = vec![];
        for (original, column) in row.iter().zip(columns) {
            let mut record = original.clone();
            record["system_measure_index"] = json!(column);
            record["grid_column"] = json!(column);
            record["grid_columns"] = json!(count);
            if !reasons.is_empty() {
                record["alignment_needs_review"] = json!(true);
                record["needs_review"] = json!(true);
                record["alignment_review_reasons"] = json!(reasons);
                record["alignment_source"] = json!("monotone_bbox_candidate");
            }
            records.push((record, column));
        }
        result.push((records, *part, *staff));
    }
    Ok((result, count))
}

/// Equal counts do not establish column identity when different bars are missing.
/// Require a mutual unique positive horizontal IoU match for every paired column.
fn corresponding_columns(rows: &[Vec<Value>]) -> Result<bool> {
    fn unique_best(values: impl Iterator<Item = f64>, index: usize) -> bool {
        let values: Vec<_> = values.collect();
        let best = values.iter().copied().fold(0., f64::max);
        best > 0. && values[index] == best && values.iter().filter(|&&v| v == best).count() == 1
    }
    let bounds: Vec<Vec<_>> = rows
        .iter()
        .map(|row| row.iter().map(bbox).collect::<Result<_>>())
        .collect::<Result<_>>()?;
    for (index, row) in bounds.iter().enumerate() {
        for other in &bounds[index + 1..] {
            let scores: Vec<Vec<f64>> = row
                .iter()
                .map(|a| {
                    other
                        .iter()
                        .map(|b| {
                            let overlap = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.);
                            overlap / (a[2] + b[2] - overlap)
                        })
                        .collect()
                })
                .collect();
            for column in 0..row.len() {
                if !unique_best(scores[column].iter().copied(), column)
                    || !unique_best(scores.iter().map(|values| values[column]), column)
                {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// Associate original boxes monotonically; no crop is merged, split, or replaced.
fn ordered_match(row: &[Value], reference: &[Value]) -> Result<Vec<usize>> {
    let (count, width) = (row.len(), reference.len());
    let mut scores = vec![vec![f64::NEG_INFINITY; width + 1]; count + 1];
    let mut choices = vec![vec![false; width + 1]; count + 1];
    scores[0].fill(0.);
    for i in 1..=count {
        let a = bbox(&row[i - 1])?;
        for j in 1..=width {
            let b = bbox(&reference[j - 1])?;
            let overlap = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.);
            let similarity = overlap / (a[2] + b[2] - overlap).max(1e-12);
            let matched = scores[i - 1][j - 1] + similarity;
            let skipped = scores[i][j - 1];
            if matched > skipped {
                scores[i][j] = matched;
                choices[i][j] = true;
            } else {
                scores[i][j] = skipped;
            }
        }
    }
    let (mut i, mut j) = (count, width);
    let mut columns = vec![];
    while i > 0 {
        if choices[i][j] {
            columns.push(j - 1);
            i -= 1;
        }
        j -= 1;
    }
    columns.reverse();
    Ok(columns)
}

pub fn fuse_paired_staves(
    aligned: &[AlignedMember],
    output: &Path,
    page: usize,
    system: usize,
) -> Result<Vec<AlignedMember>> {
    // Preserve first appearance, matching Python's insertion-ordered dictionaries.
    let mut groups: Vec<((usize, usize), Vec<Vec<(Value, usize)>>)> = vec![];
    for (boxes, part, staff) in aligned {
        if let Some((_, rows)) = groups.iter_mut().find(|(key, _)| *key == (*part, *staff)) {
            rows.push(boxes.clone());
        } else {
            groups.push(((*part, *staff), vec![boxes.clone()]));
        }
    }
    let mut result = vec![];
    for ((part, staff), groups) in groups {
        if groups.len() == 1 {
            result.push((groups.into_iter().next().unwrap(), part, staff));
            continue;
        }
        if groups
            .iter()
            .flatten()
            .any(|(record, _)| record["alignment_needs_review"] == true)
        {
            result.extend(groups.into_iter().map(|group| (group, part, staff)));
            continue;
        }
        let mut columns: BTreeMap<usize, Vec<Value>> = BTreeMap::new();
        for group in groups {
            for (record, column) in group {
                columns.entry(column).or_default().push(record);
            }
        }
        let mut boxes = vec![];
        for (column, records) in columns {
            if records.len() == 1 {
                boxes.push((records.into_iter().next().unwrap(), column));
                continue;
            }
            let modes: BTreeSet<_> = records.iter().filter_map(|r| r["mode"].as_str()).collect();
            if records.len() != 2 || modes != BTreeSet::from(["notation", "tab"]) {
                return Err("Only complementary notation and TAB can share a staff".into());
            }
            let coords: Vec<_> = records.iter().map(bbox).collect::<Result<_>>()?;
            let left = coords.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
            let top = coords.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
            let right = coords
                .iter()
                .map(|b| b[0] + b[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let bottom = coords
                .iter()
                .map(|b| b[1] + b[3])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut record = records[0].clone();
            let path = output.join("resolved_crops").join(format!(
                "p{page}-s{system}-part{part}-staff{staff}-b{column}-both.png"
            ));
            save_crop(&mut record, [left, top, right - left, bottom - top], &path)?;
            let mut numbers = BTreeSet::new();
            for r in &records {
                if let Some(a) = r["source_measure_numbers"].as_array() {
                    for n in a {
                        numbers.insert(n.as_u64().ok_or("Invalid source measure number")?);
                    }
                } else {
                    numbers.insert(
                        r["measure_number"]
                            .as_u64()
                            .ok_or("Missing measure number")?,
                    );
                }
            }
            record["mode"] = json!("both");
            record["geometry_source"] = json!("notation_tab_pair");
            record["source_measure_numbers"] = json!(numbers);
            for source in &records {
                if source["structure_needs_review"] == true {
                    mark_structure_review(
                        &mut record,
                        source["structure_review_reasons"]
                            .as_array()
                            .map(Vec::as_slice)
                            .unwrap_or(&[]),
                    );
                }
            }
            boxes.push((record, column));
        }
        result.push((boxes, part, staff));
    }
    Ok(result)
}

/// Preserve resolved page profiles and per-row review evidence in stage diagnostics.
pub fn resolve_document(
    records: &[Value],
    predictions: &mut [Value],
    output: &Path,
) -> Result<(Vec<Value>, Vec<Value>)> {
    let rows = crate::score_structure::staff_rows(records)?;
    let mut grouped: BTreeMap<u64, Vec<Vec<Value>>> = BTreeMap::new();
    for row in rows {
        let page = row[0]["page"]
            .as_u64()
            .ok_or("Staff row needs a page number")?;
        grouped.entry(page).or_default().push(row);
    }
    let mut pages = vec![];
    for (page, mut rows) in grouped {
        let matching: Vec<_> = predictions
            .iter()
            .filter(|p| p["page"].as_u64() == Some(page))
            .collect();
        if matching.len() != 1 {
            return Err(format!(
                "Expected exactly one structure prediction for page {page}"
            ));
        }
        let structure = matching[0]["parsed"].clone();
        if !structure.is_object() {
            return Err(format!("Missing parsed structure for page {page}"));
        }
        let audit = &matching[0]["resolution_audit"];
        if let Some(indices) = audit["review_rows"].as_array() {
            // Relation indices address the ordered detector rows, not their
            // stored IDs. Mark before notation/TAB rows are fused.
            let reasons = audit["review_reasons"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            for (index, boxes) in rows.iter_mut().enumerate() {
                if indices.contains(&json!(index + 1)) {
                    for record in boxes {
                        mark_structure_review(record, reasons);
                    }
                }
            }
        }
        pages.push(crate::score_structure::StructurePage {
            page,
            structure,
            rows,
        });
    }
    let resolved = crate::score_structure::resolve_pages(&mut pages)?;
    for page in &pages {
        let prediction = predictions
            .iter_mut()
            .find(|p| p["page"].as_u64() == Some(page.page))
            .ok_or("Missing page diagnostics")?;
        prediction["parsed"] = page.structure.clone();
        let identities = page.structure["part_identity"].clone();
        if !prediction["resolution_audit"].is_object() {
            prediction["resolution_audit"] = json!({});
        }
        prediction["resolution_audit"]["part_identity"] = identities.clone();
        let mut reasons = prediction["resolution_audit"]["review_reasons"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for evidence in identities.as_array().into_iter().flatten() {
            if let Some(reason) = evidence["review_reason"].as_str() {
                if !reasons.contains(&json!(reason)) {
                    reasons.push(json!(reason));
                }
                prediction["structure_needs_review"] = json!(true);
                prediction["resolution_audit"]["needs_review"] = json!(true);
            }
        }
        prediction["resolution_audit"]["review_reasons"] = json!(reasons);
    }
    let mut result = vec![];
    let mut bar_offset = 0usize;
    for system in resolved.systems {
        let (aligned, count) = align_system(&system.members)?;
        let fused = fuse_paired_staves(&aligned, output, system.page as usize, system.system)?;
        for (boxes, part, staff) in fused {
            let profile = resolved
                .parts
                .get(part)
                .ok_or("Resolved system references a nonexistent part")?;
            for (mut row, column) in boxes {
                row["part_id"] = profile["id"].clone();
                row["staff_id"] = json!(format!("staff-{}", staff + 1));
                row["part_name"] = profile["name"].clone();
                row["instrument"] = profile["instrument"].clone();
                row["midi_program"] = profile["program"].clone();
                row["string_count"] = profile["strings"].clone();
                row["row_index"] = row
                    .get("row_index")
                    .cloned()
                    .unwrap_or_else(|| row["system_index"].clone());
                row["system_index"] = json!(system.system);
                row["bar_index"] = json!(bar_offset + column);
                row["measure_number"] = json!(result.len() + 1);
                if row["part_identity"]["review_reason"].is_string() {
                    let reason = row["part_identity"]["review_reason"].clone();
                    mark_structure_review(&mut row, &[reason]);
                }
                result.push(row);
            }
        }
        bar_offset = bar_offset
            .checked_add(count)
            .ok_or("Bar timeline overflow")?;
    }
    Ok((resolved.parts, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_review_survives_paired_rows_and_noncontiguous_row_ids() {
        let dir = tempfile::tempdir().unwrap();
        let page = dir.path().join("page.png");
        image::RgbImage::from_pixel(300, 400, image::Rgb([255; 3]))
            .save(&page)
            .unwrap();
        for ids in [[0, 1, 2, 3], [7, 11, 16, 31]] {
            let modes = ["notation", "tab", "notation", "tab"].map(str::to_owned);
            let (mut structure, audit) = crate::score_structure::relation_groups(
                &[0, 0, 1, 1],
                &[0.999, 0.49, 0.999],
                &[0.999, 0.001, 0.999],
                &modes,
                0.95,
                0.95,
            )
            .unwrap();
            for profile in structure["parts"].as_array_mut().unwrap() {
                profile["name"] = json!("Guitar");
                profile["name_source"] = json!("label_text");
            }
            let records: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    json!({
                        "page":1,"row_index":id,"mode":modes[index],"measure_number":index+1,
                        "bbox":[10,index*80+10,200,40],"source_page":page.to_string_lossy(),
                    })
                })
                .collect();
            let mut predictions = vec![json!({"page":1,"parsed":structure,
                "resolution_audit":audit,"structure_needs_review":true})];
            let (_, resolved) = resolve_document(&records, &mut predictions, dir.path()).unwrap();
            assert_eq!(resolved.len(), 2);
            for row in resolved {
                assert_eq!(row["mode"], "both");
                assert_eq!(row["structure_needs_review"], true);
                assert_eq!(
                    row["structure_review_reasons"],
                    json!(["uncertain_row_relation"])
                );
            }
        }
    }
}
