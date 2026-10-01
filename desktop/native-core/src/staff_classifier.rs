//! Raster layout classification and evidence-based TAB string-count voting.
//! Ported from layout/classifier.py; all analysis copies stay local and the
//! original image passed to the detector/OCR is never changed.
use crate::image_boundary::{check_dimensions, open_rgb, Result};
use crate::staff_geometry::{detect_paired_staffs, detect_staffs};
use image::{Rgb, RgbImage};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

fn gray(pixel: &Rgb<u8>) -> u8 {
    ((19595 * u32::from(pixel[0])
        + 38470 * u32::from(pixel[1])
        + 7471 * u32::from(pixel[2])
        + 32768)
        >> 16) as u8
}

fn classify_fixed_scale(image: &RgbImage) -> Result<Value> {
    let paired = detect_paired_staffs(image)?;
    if !paired.is_empty() {
        return Ok(json!({"layout":"score_tab","confidence":0.99,
            "systems":paired.len(),"method":"paired_five_and_tab_staffs"}));
    }
    let staffs = detect_staffs(image, Some(6.0), false)?;
    if staffs.is_empty() {
        return Ok(json!({"layout":"unknown","confidence":0.0,
            "systems":0,"method":"no_staffs"}));
    }
    let mut counts = BTreeMap::new();
    let mut score_like = 0;
    let mut tab_like = 0;
    for staff in &staffs {
        let count = staff.string_y.len();
        *counts.entry(count).or_insert(0usize) += 1;
        if count == 5 && staff.spacing < 15.0 {
            score_like += 1;
        }
        if (4..=8).contains(&count) && (count != 5 || staff.spacing >= 15.0) {
            tab_like += 1;
        }
    }
    let (layout, support) = if tab_like > score_like {
        ("tab_only", tab_like)
    } else {
        ("score_only", score_like)
    };
    let counts: Map<String, Value> = counts
        .into_iter()
        .map(|(n, c)| (n.to_string(), json!(c)))
        .collect();
    Ok(
        json!({"layout":layout,"confidence":support as f64/staffs.len() as f64,
        "systems":staffs.len(),"staff_line_counts":counts,
        "method":"unpaired_staff_spacing_and_line_count"}),
    )
}

/// Prefer paired five-line/TAB evidence, then supported fraction and system count.
/// Classification-only normalization uses image::Triangle (bilinear). Its
/// intermediate rounding can differ by an RGB8 level from Pillow BILINEAR;
/// exact pixel or threshold-boundary parity for normalized copies is not
/// asserted. Original
/// pixels and contrast thresholds follow the Python classifier unchanged.
pub fn classify_notation_layout(page: &RgbImage) -> Result<Value> {
    check_dimensions(page.width(), page.height())?;
    let longest = page.width().max(page.height());
    let scale = 2000.0 / f64::from(longest);
    let normalized = if longest >= 600 && (scale - 1.0).abs() >= 0.08 {
        let width = (f64::from(page.width()) * scale).round_ties_even().max(1.0) as u32;
        let height = (f64::from(page.height()) * scale)
            .round_ties_even()
            .max(1.0) as u32;
        Some(image::imageops::resize(
            page,
            width,
            height,
            image::imageops::FilterType::Triangle,
        ))
    } else {
        None
    };
    let mut candidates = vec![(1.0, page)];
    if let Some(image) = normalized.as_ref() {
        candidates.push((scale, image));
    }
    let mut results = Vec::new();
    for &(scale, candidate) in &candidates {
        let mut result = classify_fixed_scale(candidate)?;
        result["analysis_scale"] = json!(scale);
        results.push(result);
    }
    if !results.iter().any(|result| result["layout"] == "score_tab") {
        for &(scale, candidate) in &candidates {
            let contrast = RgbImage::from_fn(candidate.width(), candidate.height(), |x, y| {
                Rgb([if gray(candidate.get_pixel(x, y)) < 230 {
                    0
                } else {
                    255
                }; 3])
            });
            let mut result = classify_fixed_scale(&contrast)?;
            result["analysis_scale"] = json!(scale);
            result["contrast_threshold"] = json!(230);
            results.push(result);
        }
    }
    let mut best = 0;
    for i in 1..results.len() {
        let a = &results[i];
        let b = &results[best];
        let order = (a["layout"] == "score_tab")
            .cmp(&(b["layout"] == "score_tab"))
            .then(
                a["confidence"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["confidence"].as_f64().unwrap()),
            )
            .then(
                a["systems"]
                    .as_u64()
                    .unwrap()
                    .cmp(&b["systems"].as_u64().unwrap()),
            );
        // Python max() retains the first item when all evidence ties.
        if order.is_gt() {
            best = i;
        }
    }
    Ok(results.swap_remove(best))
}

/// NumPy's default linear percentile, using a bounded grayscale histogram.
fn percentile95(histogram: &[usize; 256], length: usize) -> f64 {
    let rank = (length - 1) as f64 * 0.95;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    let mut total = 0;
    let mut first = 0;
    let mut second = 0;
    for (value, &count) in histogram.iter().enumerate() {
        let end = total + count;
        if total <= low && low < end {
            first = value;
        }
        if total <= high && high < end {
            second = value;
            break;
        }
        total = end;
    }
    first as f64 + (second as f64 - first as f64) * (rank - low as f64)
}

/// Count only complete, strongly supported four-to-eight-line TAB grids.
/// A combined crop needs a second, tighter grid; a solitary staff cannot prove
/// which lines are TAB. At most one weak interior line may be recovered, and
/// only when the expected row has strong faint-ink evidence.
pub fn visible_tab_strings(image: &RgbImage, mode: &str) -> Result<Option<usize>> {
    if !matches!(mode, "tab" | "both") {
        return Ok(None);
    }
    check_dimensions(image.width(), image.height())?;
    if image.width() < 100 {
        return Ok(None);
    }
    let left = (f64::from(image.width()) * 0.13).round_ties_even() as u32;
    let right = (f64::from(image.width()) * 0.9).round_ties_even() as u32;
    let width = (right - left) as usize;
    let height = image.height() as usize;
    let mut values = Vec::with_capacity(width * height);
    let mut histogram = [0usize; 256];
    let mut ink = Vec::with_capacity(height);
    for y in 0..image.height() {
        let mut strong = 0;
        for x in left..right {
            let value = gray(image.get_pixel(x, y));
            histogram[value as usize] += 1;
            strong += usize::from(value < 210);
            values.push(value);
        }
        ink.push(strong as f64 / width as f64);
    }
    let faint_threshold = (percentile95(&histogram, values.len()) - 2.0).min(253.0);
    let faint: Vec<f64> = values
        .chunks_exact(width)
        .map(|row| {
            row.iter()
                .filter(|&&v| f64::from(v) < faint_threshold)
                .count() as f64
                / width as f64
        })
        .collect();
    let threshold = 0.35f64.max(ink.iter().copied().fold(0.0, f64::max) * 0.55);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (row, &strength) in ink.iter().enumerate() {
        if strength <= threshold {
            continue;
        }
        if groups
            .last()
            .is_some_and(|g| g.last().is_some_and(|&last| last + 1 == row))
        {
            groups.last_mut().unwrap().push(row);
        } else {
            groups.push(vec![row]);
        }
    }
    let ys: Vec<f64> = groups
        .iter()
        .map(|g| {
            g.iter().map(|&y| y as f64 * ink[y]).sum::<f64>()
                / g.iter().map(|&y| ink[y]).sum::<f64>()
        })
        .collect();
    let strengths: Vec<f64> = groups
        .iter()
        .map(|g| g.iter().map(|&y| ink[y]).fold(0.0, f64::max))
        .collect();
    let mut candidates = Vec::new();
    let mut i = 0;
    while i + 3 < ys.len() {
        let gap = ys[i + 1] - ys[i];
        if !(5.0..=45.0).contains(&gap) {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut count = 2;
        let mut missing = 0;
        while j + 1 < ys.len() {
            let distance = ys[j + 1] - ys[j];
            let steps = (distance / gap).round_ties_even() as i64;
            if !matches!(steps, 1 | 2)
                || missing + steps - 1 > 1
                || (distance - steps as f64 * gap).abs() >= 1.6f64.max(gap * 0.12) * steps as f64
            {
                break;
            }
            if steps == 2 {
                let expected = (ys[j] + gap).round_ties_even() as usize;
                let start = expected.saturating_sub(2).min(height);
                let end = (expected + 3).min(height);
                if faint[start..end].iter().copied().fold(0.0, f64::max) <= 0.6 {
                    break;
                }
            }
            count += steps as usize;
            missing += steps - 1;
            j += 1;
        }
        if (4..=8).contains(&count)
            && j - i + 1 >= 4
            && strengths[i..=j]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                > 0.45
        {
            candidates.push((count, gap));
            i = j + 1;
        } else {
            i += 1;
        }
    }
    if mode == "both" {
        let minimum = candidates
            .iter()
            .map(|(_, gap)| *gap)
            .reduce(f64::min)
            .unwrap_or(0.0);
        candidates.retain(|(_, gap)| *gap > minimum * 1.15);
    }
    let count = candidates.first().map(|(count, _)| *count);
    Ok(count.filter(|count| candidates.iter().all(|(n, _)| n == count)))
}

/// Sample up to twelve complete measures and require >=75% agreement, with
/// support from at least three source rows (or every row if fewer exist).
pub fn part_tab_strings(records: &[Value]) -> Result<Option<usize>> {
    let rows: Vec<_> = records
        .iter()
        .filter(|r| {
            matches!(r["mode"].as_str(), Some("tab" | "both"))
                && r["image"].as_str().is_some_and(|path| !path.is_empty())
        })
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    let samples = rows.len().min(12);
    let mut votes: Vec<(usize, usize)> = Vec::new();
    for i in 0..samples {
        let index = if samples == 1 {
            0
        } else {
            (i as u128 * (rows.len() - 1) as u128 / (samples - 1) as u128) as usize
        };
        let row = rows[index];
        let image = open_rgb(Path::new(row["image"].as_str().unwrap()))?;
        if let Some(count) = visible_tab_strings(&image, row["mode"].as_str().unwrap())? {
            if let Some((_, support)) = votes.iter_mut().find(|(n, _)| *n == count) {
                *support += 1;
            } else {
                votes.push((count, 1));
            }
        }
    }
    let Some(support) = votes.iter().map(|(_, support)| *support).max() else {
        return Ok(None);
    };
    let total: usize = votes.iter().map(|(_, support)| support).sum();
    Ok(
        if support >= 3.min(rows.len()) && support as f64 / total as f64 >= 0.75 {
            votes
                .iter()
                .find(|(_, n)| *n == support)
                .map(|(count, _)| *count)
        } else {
            None
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn staff_image(staves: &[(u32, u32, u32, u8)], width: u32, height: u32) -> RgbImage {
        let mut image = RgbImage::from_pixel(width, height, Rgb([255; 3]));
        for &(top, lines, spacing, ink) in staves {
            for n in 0..lines {
                for x in 20..=width - 20 {
                    image.put_pixel(x, top + n * spacing, Rgb([ink; 3]));
                }
            }
            for x in [20, width / 2, width - 20] {
                for y in top..=top + (lines - 1) * spacing {
                    image.put_pixel(x, y, Rgb([ink; 3]));
                }
            }
        }
        image
    }

    #[test]
    fn original_scale_layout_goldens_match_python_including_light_both() {
        // Full result contracts generated from layout.classifier on these
        // deterministic 500x250 images, using Pillow 12.3.0 / NumPy 2.5.3.
        let cases: Vec<(&str, Vec<(u32, u32, u32, u8)>, bool, Option<usize>)> = vec![
            ("score_only", vec![(50, 5, 8, 20)], false, Some(5)),
            ("tab_only", vec![(50, 6, 12, 20)], false, Some(6)),
            (
                "score_tab",
                vec![(50, 5, 8, 20), (140, 6, 12, 20)],
                false,
                None,
            ),
            ("score_only", vec![(50, 5, 8, 205)], true, Some(5)),
            ("tab_only", vec![(50, 6, 12, 205)], true, Some(6)),
            (
                "score_tab",
                vec![(50, 5, 8, 205), (140, 6, 12, 205)],
                true,
                None,
            ),
            (
                "score_tab",
                vec![(50, 5, 8, 205), (140, 6, 12, 20)],
                true,
                None,
            ),
            ("tab_only", vec![(50, 5, 16, 20)], false, Some(5)),
            ("unknown", vec![], false, None),
        ];
        for (layout, staves, contrast, lines) in cases {
            let image = staff_image(&staves, 500, 250);
            let original = image.clone();
            let mut expected = if layout == "score_tab" {
                json!({"layout":layout,"confidence":0.99,"systems":1,"method":"paired_five_and_tab_staffs","analysis_scale":1.0})
            } else if layout == "unknown" {
                json!({"layout":layout,"confidence":0.0,"systems":0,"method":"no_staffs","analysis_scale":1.0})
            } else {
                let mut counts = Map::new();
                counts.insert(lines.unwrap().to_string(), json!(1));
                json!({"layout":layout,"confidence":1.0,"systems":1,"staff_line_counts":counts,"method":"unpaired_staff_spacing_and_line_count","analysis_scale":1.0})
            };
            if contrast {
                expected["contrast_threshold"] = json!(230);
            }
            assert_eq!(
                classify_notation_layout(&image).unwrap(),
                expected,
                "{staves:?}"
            );
            assert_eq!(
                visible_tab_strings(&image, "tab").unwrap(),
                lines,
                "{staves:?}"
            );
            assert_eq!(
                visible_tab_strings(&image, "both").unwrap(),
                if layout == "score_tab" { Some(6) } else { None }
            );
            assert_eq!(image, original);
        }
    }

    #[test]
    fn visible_counts_require_supported_four_to_eight_line_grids() {
        for count in 4..=8 {
            let image = staff_image(&[(30, count, 12, 20)], 500, 250);
            assert_eq!(
                visible_tab_strings(&image, "tab").unwrap(),
                Some(count as usize)
            );
            assert_eq!(visible_tab_strings(&image, "both").unwrap(), None);
        }
        let conflict = staff_image(&[(25, 6, 12, 20), (150, 7, 12, 20)], 500, 250);
        assert_eq!(visible_tab_strings(&conflict, "tab").unwrap(), None);
        let same_spacing = staff_image(&[(25, 5, 12, 20), (140, 6, 12, 20)], 500, 250);
        assert_eq!(visible_tab_strings(&same_spacing, "both").unwrap(), None);
        assert_eq!(
            visible_tab_strings(&same_spacing, "notation").unwrap(),
            None
        );
        assert_eq!(
            visible_tab_strings(&RgbImage::from_pixel(99, 100, Rgb([0; 3])), "tab").unwrap(),
            None
        );
    }

    #[test]
    fn one_missing_line_needs_measured_faint_support() {
        let mut image = staff_image(&[(30, 6, 12, 20)], 500, 160);
        for x in 20..=480 {
            image.put_pixel(x, 66, Rgb([240; 3]));
        }
        assert_eq!(visible_tab_strings(&image, "tab").unwrap(), Some(6));
        for x in 20..=480 {
            image.put_pixel(x, 66, Rgb([255; 3]));
        }
        assert_eq!(visible_tab_strings(&image, "tab").unwrap(), None);
    }

    #[test]
    fn measure_votes_require_three_rows_and_seventy_five_percent() {
        let directory = tempfile::tempdir().unwrap();
        let six = directory.path().join("six.png");
        let seven = directory.path().join("seven.png");
        let blank = directory.path().join("blank.png");
        staff_image(&[(30, 6, 12, 20)], 500, 160)
            .save(&six)
            .unwrap();
        staff_image(&[(30, 7, 12, 20)], 500, 160)
            .save(&seven)
            .unwrap();
        RgbImage::from_pixel(500, 160, Rgb([255; 3]))
            .save(&blank)
            .unwrap();
        let records = |paths: &[&Path]| {
            paths
                .iter()
                .map(|p| json!({"mode":"tab","image":p}))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            part_tab_strings(&records(&[&six, &six, &six, &seven])).unwrap(),
            Some(6)
        );
        assert_eq!(
            part_tab_strings(&records(&[&six, &six, &seven, &seven])).unwrap(),
            None
        );
        assert_eq!(
            part_tab_strings(&records(&[&six, &six, &blank])).unwrap(),
            None
        );
        assert_eq!(part_tab_strings(&records(&[&six])).unwrap(), Some(6));
        assert_eq!(part_tab_strings(&[]).unwrap(), None);
        assert_eq!(
            part_tab_strings(&[json!({"mode":"notation","image":"not-opened.png"})]).unwrap(),
            None
        );
        assert!(
            part_tab_strings(&[json!({"mode":"tab","image":"missing-native-staff.png"})]).is_err()
        );
        // Exact endpoints and even coverage, using the same 12-index sample
        // rule as np.linspace(...,dtype=int). Non-sampled file paths stay closed.
        let sampled: Vec<usize> = (0..12).map(|i| i * 19 / 11).collect();
        let mut rows = Vec::new();
        for i in 0..20 {
            rows.push(json!({"mode":"tab","image":if sampled.contains(&i){six.as_path()}else{Path::new("not-sampled.png")}}));
        }
        assert_eq!(part_tab_strings(&rows).unwrap(), Some(6));
    }

    #[test]
    fn normalization_is_copy_only_and_invalid_dimensions_fail() {
        let image = RgbImage::from_pixel(600, 20, Rgb([255; 3]));
        let original = image.clone();
        assert_eq!(
            classify_notation_layout(&image).unwrap(),
            json!({"layout":"unknown","confidence":0.0,"systems":0,"method":"no_staffs","analysis_scale":1.0})
        );
        assert_eq!(image, original);
        assert!(classify_notation_layout(&RgbImage::new(0, 1)).is_err());
        assert!(visible_tab_strings(&RgbImage::new(100, 0), "tab").is_err());
        assert_eq!(
            percentile95(
                &{
                    let mut h = [0; 256];
                    h[10] = 1;
                    h[20] = 1;
                    h
                },
                2
            ),
            19.5
        );
    }
}
