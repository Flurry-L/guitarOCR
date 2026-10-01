//! Simultaneous-staff consensus and notation/TAB fusion, ported from
//! layout/score_grid.py. Part and staff identities are never flattened.
use crate::image_boundary::{crop_measure, open_rgb, Result};
use image::RgbImage;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
pub type Member = (Vec<Value>, usize, usize);
pub type AlignedMember = (Vec<(Value, usize)>, usize, usize);
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    } else {
        v[n / 2]
    }
}
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
pub fn common_columns(rows: &[Vec<Value>]) -> Result<(Vec<f64>, f64)> {
    if rows.is_empty() || rows.iter().any(|r| r.is_empty()) {
        return Err("A score system must contain nonempty staff rows".into());
    }
    let coordinates: Vec<Vec<[f64; 4]>> = rows
        .iter()
        .map(|r| r.iter().map(bbox).collect())
        .collect::<Result<_>>()?;
    if rows.len() == 1 {
        return Ok((
            coordinates[0].iter().map(|b| b[0]).collect(),
            coordinates[0]
                .iter()
                .map(|b| b[0] + b[2])
                .fold(f64::NEG_INFINITY, f64::max),
        ));
    }
    let tolerance = 10f64.max(median(coordinates.iter().flatten().map(|b| b[2]).collect()) * 0.065);
    let mut points: Vec<(f64, usize, f64)> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter().enumerate().map({
                let c = &coordinates;
                move |(j, r)| (c[i][j][0], i, score(r))
            })
        })
        .collect();
    points.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
    });
    let mut clusters: Vec<Vec<(f64, usize, f64)>> = vec![];
    for p in points {
        if clusters
            .last()
            .map(|c| p.0 - median(c.iter().map(|p| p.0).collect()) <= tolerance)
            .unwrap_or(false)
        {
            clusters.last_mut().unwrap().push(p);
        } else {
            clusters.push(vec![p]);
        }
    }
    let support: Vec<usize> = clusters
        .iter()
        .map(|c| c.iter().map(|p| p.1).collect::<BTreeSet<_>>().len())
        .collect();
    let quorum = rows.len() / 2 + 1;
    let mut selected: Vec<f64> = clusters
        .iter()
        .zip(&support)
        .filter(|(_, votes)| **votes >= quorum)
        .map(|(c, _)| median(c.iter().map(|p| p.0).collect()))
        .collect();
    let typical = median(rows.iter().map(|r| r.len() as f64).collect());
    if (selected.len() as f64) < 1f64.max(typical * 0.6) {
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for r in rows {
            *counts.entry(r.len()).or_default() += 1;
        }
        let common = *counts
            .iter()
            .min_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)))
            .unwrap()
            .0;
        let mut reference = rows
            .iter()
            .enumerate()
            .find(|(_, r)| r.len() == common)
            .unwrap()
            .0;
        for (i, r) in rows.iter().enumerate() {
            if r.len() == common
                && r.iter().map(score).sum::<f64>() > rows[reference].iter().map(score).sum::<f64>()
            {
                reference = i;
            }
        }
        selected = coordinates[reference].iter().map(|r| r[0]).collect();
    } else {
        let first = median(coordinates.iter().map(|r| r[0][0]).collect());
        if selected[0] - first > tolerance * 2. {
            selected.insert(0, first);
        }
        for (c, &votes) in clusters.iter().zip(&support) {
            let x = median(c.iter().map(|p| p.0).collect());
            if votes < 2 || selected.iter().any(|s| (x - s).abs() <= tolerance) {
                continue;
            }
            let crossing = coordinates
                .iter()
                .filter(|row| {
                    row.iter()
                        .any(|b| b[0] + tolerance * 2. < x && x < b[0] + b[2] - tolerance * 2.)
                })
                .count();
            if crossing < votes && x > selected[0] {
                selected.push(x);
            }
        }
    }
    selected.sort_by(f64::total_cmp);
    let right = median(
        coordinates
            .iter()
            .map(|r| {
                r.iter()
                    .map(|b| b[0] + b[2])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect(),
    );
    let final_right = right.max(selected.last().unwrap() + tolerance * 2.);
    Ok((selected, final_right))
}

pub fn missing_bar_box(
    image: &RgbImage,
    row: &[Value],
    left: f64,
    right: f64,
) -> Result<Option<[f64; 4]>> {
    let modes: BTreeSet<&str> = row.iter().filter_map(|r| r["mode"].as_str()).collect();
    if modes.len() != 1 {
        return Ok(None);
    }
    let mode = *modes.first().unwrap();
    if mode != "tab" && mode != "notation" {
        return Ok(None);
    }
    let coords: Vec<_> = row.iter().map(bbox).collect::<Result<_>>()?;
    let top = coords.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
    let bottom = coords
        .iter()
        .map(|b| b[1] + b[3])
        .fold(f64::NEG_INFINITY, f64::max);
    let x0 = (left as i64 - 12).clamp(0, image.width() as i64) as u32;
    let x1 = (right as i64 + 12).clamp(0, image.width() as i64) as u32;
    let y0 = (top as i64).clamp(0, image.height() as i64) as u32;
    let y1 = (bottom as i64).clamp(0, image.height() as i64) as u32;
    if x1 <= x0 || y1 <= y0 {
        return Ok(None);
    }
    let crop = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
    let staffs = crate::staff_geometry::detect_staffs(&crop, Some(5.), false)?;
    if staffs.len() != 1
        || staffs[0].boundaries.len() != 2
        || (mode == "notation" && staffs[0].string_y.len() != 5)
    {
        return Ok(None);
    }
    let start = staffs[0].boundaries[0];
    let stop = staffs[0].boundaries[1];
    if start > 36f64.max((right - left) * 0.12) {
        return Ok(None);
    }
    let a = (x0 as f64).max(x0 as f64 + start - 10.);
    let b = (x1 as f64).min(x0 as f64 + stop + 10.);
    Ok(Some([a, y0 as f64, b - a, (y1 - y0) as f64]))
}

pub fn align_system(
    members: &[Member],
    output: &Path,
    page: usize,
    system: usize,
) -> Result<(Vec<AlignedMember>, usize)> {
    let mut rows: Vec<Vec<Value>> = members.iter().map(|(r, _, _)| r.clone()).collect();
    for row in &mut rows {
        for b in row.iter() {
            bbox(b)?;
        }
        row.sort_by(|a, b| {
            a["bbox"][0]
                .as_f64()
                .unwrap()
                .total_cmp(&b["bbox"][0].as_f64().unwrap())
        });
    }
    let (columns, right) = common_columns(&rows)?;
    let mut boundaries = columns.clone();
    boundaries.push(right);
    let mut result = vec![];
    for (row_index, (row, (_, part, staff))) in rows.iter().zip(members).enumerate() {
        let mut grouped: BTreeMap<usize, Vec<(Value, [f64; 4])>> = BTreeMap::new();
        for record in row {
            let [x, y, width, height] = bbox(record)?;
            let end = x + width;
            let margin = 12f64.max(width * 0.15);
            let mut edges = vec![x];
            edges.extend(
                columns
                    .iter()
                    .skip(1)
                    .filter(|&&b| x + margin < b && b < end - margin)
                    .copied(),
            );
            edges.push(end);
            for w in edges.windows(2) {
                let left = w[0];
                let stop = w[1];
                let mut column = (0..columns.len())
                    .min_by(|&a, &b| {
                        (left - columns[a])
                            .abs()
                            .total_cmp(&(left - columns[b]).abs())
                    })
                    .unwrap();
                if (left - columns[column]).abs()
                    > 18f64.max((boundaries[column + 1] - columns[column]) * 0.18)
                {
                    column = columns
                        .iter()
                        .position(|&c| c > left)
                        .map(|i| i.saturating_sub(1))
                        .unwrap_or(columns.len() - 1);
                }
                grouped
                    .entry(column)
                    .or_default()
                    .push((record.clone(), [left, y, stop - left, height]));
            }
        }
        let mut corrected = vec![];
        for (&column, fragments) in &grouped {
            let mut first = &fragments[0].0;
            for (r, _) in fragments {
                if score(r) > score(first) {
                    first = r;
                }
            }
            let left = fragments
                .iter()
                .map(|(_, b)| b[0])
                .fold(f64::INFINITY, f64::min);
            let top = fragments
                .iter()
                .map(|(_, b)| b[1])
                .fold(f64::INFINITY, f64::min);
            let end = fragments
                .iter()
                .map(|(_, b)| b[0] + b[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let bottom = fragments
                .iter()
                .map(|(_, b)| b[1] + b[3])
                .fold(f64::NEG_INFINITY, f64::max);
            let bounds = [left, top, end - left, bottom - top];
            let mut record = first.clone();
            let numbers: BTreeSet<u64> = fragments
                .iter()
                .map(|(r, _)| {
                    r["measure_number"]
                        .as_u64()
                        .ok_or("Measure number must be a nonnegative integer")
                })
                .collect::<std::result::Result<_, _>>()?;
            record["source_measure_numbers"] = json!(numbers);
            record["system_measure_index"] = json!(column);
            record["grid_column"] = json!(column);
            record["grid_columns"] = json!(columns.len());
            if fragments.len() != 1
                || bbox(first)?
                    .iter()
                    .zip(bounds)
                    .any(|(a, b)| (a - b).abs() > 1.)
            {
                let path = output
                    .join("resolved_crops")
                    .join(format!("p{page}-s{system}-r{row_index}-b{column}.png"));
                save_crop(&mut record, bounds, &path)?;
                record["geometry_source"] = json!("staff_consensus");
            }
            corrected.push((record, column));
        }
        let missing: Vec<usize> = (0..columns.len())
            .filter(|i| !grouped.contains_key(i))
            .collect();
        if !missing.is_empty() && rows.len() >= 3 && row[0]["source_page"].as_str().is_some() {
            let image = open_rgb(Path::new(row[0]["source_page"].as_str().unwrap()))?;
            for column in missing {
                let Some(bounds) =
                    missing_bar_box(&image, row, boundaries[column], boundaries[column + 1])?
                else {
                    continue;
                };
                let nearest = row
                    .iter()
                    .min_by(|a, b| {
                        (a["bbox"][0].as_f64().unwrap() - boundaries[column])
                            .abs()
                            .total_cmp(&(b["bbox"][0].as_f64().unwrap() - boundaries[column]).abs())
                    })
                    .unwrap();
                let mut record = nearest.clone();
                let path = output.join("resolved_crops").join(format!(
                    "p{page}-s{system}-r{row_index}-b{column}-recovered.png"
                ));
                save_crop(&mut record, bounds, &path)?;
                record["source_measure_numbers"] = json!([]);
                record["system_measure_index"] = json!(column);
                record["grid_column"] = json!(column);
                record["grid_columns"] = json!(columns.len());
                record["geometry_source"] = json!("staff_lines_and_consensus");
                corrected.push((record, column));
            }
        }
        corrected.sort_by_key(|(_, column)| *column);
        result.push((corrected, *part, *staff));
    }
    Ok((result, columns.len()))
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
            boxes.push((record, column));
        }
        result.push((boxes, part, staff));
    }
    Ok(result)
}

/// Resolve all structure predictions before OCR, keeping global part identities,
/// simultaneous bar columns, and distinct grand-staff lanes across page turns.
pub fn resolve_document(
    records: &[Value],
    predictions: &[Value],
    output: &Path,
) -> Result<(Vec<Value>, Vec<Value>)> {
    resolve_document_reconciled(records, &mut predictions.to_vec(), output)
}

/// Also preserve reconciled page profiles and model_parts in stage diagnostics.
pub fn resolve_document_reconciled(
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
    for (page, rows) in grouped {
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
    }
    let mut result = vec![];
    let mut bar_offset = 0usize;
    for system in resolved.systems {
        let (aligned, count) =
            align_system(&system.members, output, system.page as usize, system.system)?;
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
                result.push(row);
            }
        }
        bar_offset = bar_offset
            .checked_add(count)
            .ok_or("Bar timeline overflow")?;
    }
    Ok((resolved.parts, result))
}
