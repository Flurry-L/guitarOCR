//! Raster staff/barline evidence shared by missing-bar recovery and explicit
//! image-geometry fallback. No synthetic music is inserted without this evidence.
use crate::image_boundary::{check_dimensions, round_even, Result};
use image::RgbImage;
use serde_json::{json, Value};
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
pub struct Staff {
    pub string_y: Vec<usize>,
    pub spacing: f64,
    pub boundaries: Vec<f64>,
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    } else {
        v[n / 2]
    }
}
fn groups(values: Vec<usize>, gap: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![];
    for v in values {
        if out
            .last()
            .and_then(|g| g.last())
            .map(|p| v > *p + gap)
            .unwrap_or(true)
        {
            out.push(vec![v]);
        } else {
            out.last_mut().unwrap().push(v);
        }
    }
    out
}
struct Ink {
    data: Vec<bool>,
    w: usize,
    h: usize,
}
impl Ink {
    fn new(image: &RgbImage) -> Self {
        Self {
            data: image
                .pixels()
                .map(|p| {
                    ((19595u32 * p[0] as u32
                        + 38470u32 * p[1] as u32
                        + 7471u32 * p[2] as u32
                        + 32768)
                        >> 16)
                        < 160
                })
                .collect(),
            w: image.width() as usize,
            h: image.height() as usize,
        }
    }
    fn at(&self, x: usize, y: usize) -> bool {
        self.data[y * self.w + x]
    }
    fn longest(&self, y: usize, left: usize, right: usize) -> usize {
        let (mut run, mut longest) = (0, 0);
        for x in left..right {
            if self.at(x, y) {
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
        longest
    }
    fn boundaries(&self, rows: &[usize], preserve: bool, min_density: f64) -> Vec<f64> {
        if rows.len() < 4 || *rows.last().unwrap() >= self.h {
            return vec![];
        }
        let spacing = median(rows.windows(2).map(|a| (a[1] - a[0]) as f64).collect());
        let top = rows[0];
        let bottom = *rows.last().unwrap();
        let bars = groups(
            (0..self.w)
                .filter(|&x| {
                    (top..=bottom).filter(|&y| self.at(x, y)).count() as f64
                        > (bottom - top + 1) as f64 * 0.9
                })
                .collect(),
            1,
        );
        if bars.len() < 2 {
            return vec![];
        }
        let tail = |x: usize| {
            let (mut above, mut below) = (0, 0);
            let low = round_even(top as f64 - 4. * spacing).max(-1.) as i64;
            for y in (low + 1..top as i64).rev() {
                if !self.at(x, y as usize) {
                    break;
                }
                above += 1;
            }
            let high = round_even(bottom as f64 + 4. * spacing).min(self.h as f64) as usize;
            for y in bottom + 1..high {
                if !self.at(x, y) {
                    break;
                }
                below += 1;
            }
            above.max(below)
        };
        let mut filtered = vec![bars[0].clone()];
        for g in &bars[1..bars.len() - 1] {
            let shortest = g.iter().map(|&x| tail(x)).min().unwrap() as f64;
            if shortest <= round_even(spacing * 0.19).max(2.)
                || (preserve && shortest >= round_even(spacing * 3.5))
            {
                filtered.push(g.clone());
            }
        }
        filtered.push(bars.last().unwrap().clone());
        let mut merged: Vec<Vec<usize>> = vec![];
        for g in filtered {
            if merged
                .last()
                .map(|m| (g[0] - m.last().unwrap()) as f64 <= 3f64.max(spacing * 0.75))
                .unwrap_or(false)
            {
                merged.last_mut().unwrap().extend(g);
            } else {
                merged.push(g);
            }
        }
        if merged.len() < 2 {
            return vec![];
        }
        let mut raw: Vec<usize> = merged.iter().map(|g| g[0]).collect();
        *raw.last_mut().unwrap() = *merged.last().unwrap().last().unwrap();
        let minimum = 80f64.max(spacing * 5.);
        let initial = minimum.max(spacing * 7.);
        let mut bounds = vec![raw[0] as f64];
        for (i, &x) in raw.iter().enumerate().take(raw.len() - 1).skip(1) {
            if x as f64 - bounds.last().unwrap() >= if i == 1 { initial } else { minimum } {
                bounds.push(x as f64);
            }
        }
        let final_x = *raw.last().unwrap() as f64;
        if final_x - bounds.last().unwrap() < minimum && bounds.len() > 1 {
            bounds.pop();
        }
        bounds.push(final_x);
        let left = raw[0];
        let right = (*raw.last().unwrap() + 1).min(self.w);
        let density = median(
            rows.iter()
                .map(|&y| {
                    (left..right).filter(|&x| self.at(x, y)).count() as f64 / (right - left) as f64
                })
                .collect(),
        );
        if density < min_density {
            vec![]
        } else {
            bounds
        }
    }
}
pub fn detect_staffs(
    image: &RgbImage,
    minimum_spacing: Option<f64>,
    preserve_cross_staff_barlines: bool,
) -> Result<Vec<Staff>> {
    check_dimensions(image.width(), image.height())?;
    let ink = Ink::new(image);
    let minimum_run = 60f64.max(round_even(ink.w as f64 * 0.055)) as usize;
    let min_ink = minimum_run.max(round_even(ink.w as f64 * 0.25) as usize);
    let lengths: Vec<usize> = (0..ink.h).map(|y| ink.longest(y, 0, ink.w)).collect();
    let row_ink: Vec<usize> = (0..ink.h)
        .map(|y| (0..ink.w).filter(|&x| ink.at(x, y)).count())
        .collect();
    let candidates: Vec<usize> = groups(
        (0..ink.h)
            .filter(|&y| lengths[y] >= minimum_run || row_ink[y] >= min_ink)
            .collect(),
        1,
    )
    .into_iter()
    .map(|g| {
        let mut best = g[0];
        for &y in &g {
            if lengths[y] > lengths[best] {
                best = y;
            }
        }
        best
    })
    .collect();
    let min_spacing = minimum_spacing.unwrap_or(8f64.max(ink.h as f64 * 0.006));
    let mut chains: Vec<Vec<usize>> = vec![];
    for (i, &first) in candidates.iter().enumerate() {
        for &second in &candidates[i + 1..] {
            let spacing = second - first;
            if (spacing as f64) < min_spacing {
                continue;
            }
            if spacing > 40 {
                break;
            }
            let tolerance = 2f64.max(spacing as f64 * 0.14);
            let mut chain = vec![first];
            let mut current = first;
            while chain.len() < 8 {
                let expected = current + spacing;
                let next = candidates
                    .iter()
                    .filter(|&&y| y > current && (y as f64 - expected as f64).abs() <= tolerance)
                    .min_by_key(|&&y| y.abs_diff(expected));
                let Some(&y) = next else {
                    break;
                };
                chain.push(y);
                current = y;
            }
            for n in 4..=chain.len() {
                chains.push(chain[..n].to_vec());
            }
        }
    }
    let quality = |chain: &Vec<usize>| {
        let med = median(chain.iter().map(|&y| row_ink[y] as f64).collect());
        (
            chain.iter().map(|&y| row_ink[y]).min().unwrap() as f64 / med.max(1.),
            med,
        )
    };
    chains.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then(quality(b).0.total_cmp(&quality(a).0))
            .then(quality(b).1.total_cmp(&quality(a).1))
            .then(a[0].cmp(&b[0]))
    });
    let mut selected: Vec<Vec<usize>> = vec![];
    for chain in chains {
        if quality(&chain).0 < 0.68
            || selected
                .iter()
                .any(|old| old.iter().any(|y| chain.contains(y)))
        {
            continue;
        }
        selected.push(chain);
    }
    selected.sort_by_key(|c| c[0]);
    if !selected.is_empty() {
        let mut counts = BTreeMap::new();
        for c in &selected {
            *counts.entry(c.len()).or_insert(0) += 1;
        }
        let dominant = *counts
            .iter()
            .max_by_key(|(n, count)| (**count, **n == 6, **n))
            .unwrap()
            .0;
        selected.retain(|c| c.len() == dominant);
    }
    Ok(selected
        .into_iter()
        .filter_map(|rows| {
            let boundaries = ink.boundaries(&rows, preserve_cross_staff_barlines, 0.62);
            if boundaries.is_empty() {
                None
            } else {
                Some(Staff {
                    spacing: median(rows.windows(2).map(|a| (a[1] - a[0]) as f64).collect()),
                    string_y: rows,
                    boundaries,
                })
            }
        })
        .collect())
}
#[derive(Clone, Debug)]
pub struct PairedStaff {
    pub score_y: Vec<f64>,
    pub score_spacing: f64,
    pub tab: Staff,
}
pub fn detect_paired_staffs(image: &RgbImage) -> Result<Vec<PairedStaff>> {
    let ink = Ink::new(image);
    let tabs = detect_staffs(image, None, true)?;
    let mut previous = -1.;
    let mut result = vec![];
    for tab in tabs {
        let left = tab.boundaries[0].max(0.) as usize;
        let right = (tab.boundaries.last().unwrap() + 1.).min(ink.w as f64) as usize;
        if right - left < 30 {
            continue;
        }
        let runs: Vec<usize> = (0..ink.h).map(|y| ink.longest(y, left, right)).collect();
        let search_top = (previous + 2. * tab.spacing).max(0.) as usize;
        let minimum = 25f64.max((right - left) as f64 * 0.045);
        let mut best: Option<(f64, Vec<f64>)> = None;
        let mut spacing = tab.spacing * 0.55;
        while spacing < tab.spacing * 0.96 {
            let search_bottom = (tab.string_y[0] as f64 - 2. * tab.spacing - 4. * spacing) as i64;
            for start in search_top..=search_top.max(search_bottom.max(0) as usize) {
                let mut positions = vec![];
                let mut strengths = vec![];
                for n in 0..5 {
                    let center = round_even(start as f64 + n as f64 * spacing) as i64;
                    let low = (center - 1).max(0) as usize;
                    let high = (center + 2).max(0).min(ink.h as i64) as usize;
                    let mut y = low;
                    for at in low..high {
                        if runs[at] > runs[y] {
                            y = at;
                        }
                    }
                    positions.push(y as f64);
                    strengths.push(if low < high { runs[y] as f64 } else { 0. });
                }
                let weak = strengths.iter().copied().fold(f64::INFINITY, f64::min);
                if weak < minimum {
                    continue;
                }
                let score = strengths.iter().sum::<f64>() + 2. * weak;
                if best.as_ref().map(|(v, _)| score > *v).unwrap_or(true) {
                    best = Some((score, positions));
                }
            }
            spacing += 0.25;
        }
        previous = *tab.string_y.last().unwrap() as f64;
        if let Some((_, score_y)) = best {
            result.push(PairedStaff {
                score_spacing: (score_y[4] - score_y[0]) / 4.,
                score_y,
                tab,
            });
        }
    }
    Ok(result)
}
fn clean_notation(mut boundaries: Vec<f64>) -> Vec<f64> {
    loop {
        if boundaries.len() < 4 {
            break;
        }
        let median = median(boundaries.windows(2).map(|w| w[1] - w[0]).collect());
        if median <= 0. {
            break;
        }
        let candidate = (1..boundaries.len() - 1)
            .filter_map(|i| {
                let l = boundaries[i] - boundaries[i - 1];
                let r = boundaries[i + 1] - boundaries[i];
                if l.min(r) < 0.5 * median && (0.7 * median..=1.45 * median).contains(&(l + r)) {
                    Some((l.min(r) / median, i))
                } else {
                    None
                }
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if let Some((_, i)) = candidate {
            boundaries.remove(i);
        } else {
            break;
        }
    }
    boundaries
}
pub fn measure_boxes(image: &RgbImage, mode: &str) -> Result<Vec<Value>> {
    let mut result = vec![];
    let mut append = |system: usize, top: f64, bottom: f64, boundaries: &[f64]| {
        for (index, w) in boundaries.windows(2).enumerate() {
            result.push(json!({"system_index":system,"system_measure_index":index,"bbox":[w[0],top,w[1]-w[0],bottom-top],"mode":mode,"geometry_source":"pixel_staff_fallback"}));
        }
    };
    if mode == "both" {
        for (system, pair) in detect_paired_staffs(image)?.iter().enumerate() {
            append(
                system,
                pair.score_y[0] - 4. * pair.score_spacing,
                *pair.tab.string_y.last().unwrap() as f64 + 3. * pair.tab.spacing,
                &pair.tab.boundaries,
            );
        }
    } else if mode == "tab" || mode == "notation" {
        for (system, staff) in detect_staffs(
            image,
            if mode == "notation" { Some(6.) } else { None },
            false,
        )?
        .iter()
        .enumerate()
        {
            if (mode == "notation" && staff.string_y.len() != 5)
                || (mode == "tab" && !(4..=8).contains(&staff.string_y.len()))
            {
                continue;
            }
            let boundaries = if mode == "notation" {
                clean_notation(staff.boundaries.clone())
            } else {
                staff.boundaries.clone()
            };
            append(
                system,
                staff.string_y[0] as f64 - if mode == "tab" { 4. } else { 5. } * staff.spacing,
                *staff.string_y.last().unwrap() as f64
                    + if mode == "tab" { 3. } else { 4. } * staff.spacing,
                &boundaries,
            );
        }
    } else {
        return Err("Raster geometry requires tab, notation or both mode".into());
    }
    Ok(result)
}

/// Read continuous system barlines from the full blue-R marked page before
/// compacting margins for the VLM. Ambiguous raster evidence returns None.
pub fn marked_systems(marked: &RgbImage, count: usize) -> Result<Option<Vec<usize>>> {
    check_dimensions(marked.width(), marked.height())?;
    if count == 0 {
        return Ok(None);
    }
    let (w, h) = (marked.width() as usize, marked.height() as usize);
    let blue_start = (w as f64 * 0.9) as usize;
    let blue_rows: Vec<usize> = (0..h)
        .filter(|&y| {
            (blue_start..w).any(|x| {
                let p = marked.get_pixel(x as u32, y as u32).0;
                (p[2] as i16 - p[0] as i16) > 40 && (p[2] as i16 - p[1] as i16) > 20
            })
        })
        .collect();
    let centers: Vec<f64> = groups(blue_rows, 2)
        .iter()
        .filter(|g| g.len() >= 2)
        .map(|g| (g[0] + g.last().unwrap()) as f64 / 2.)
        .collect();
    if centers.len() != count {
        return Ok(None);
    }
    let ink: Vec<bool> = marked
        .pixels()
        .map(|p| (p[0] as u16 + p[1] as u16 + p[2] as u16) < 630)
        .collect();
    let at = |x: usize, y: usize| ink[y * w + x];
    let left = (w as f64 * 0.22) as usize;
    let right = (w as f64 * 0.91) as usize;
    let local_left = (w as f64 * 0.07) as usize;
    let window = round_even(w as f64 * 0.12).max(8.) as usize;
    if right <= left || right - local_left < window {
        return Ok(None);
    }
    let mut lines = vec![];
    let mut short_lines = vec![];
    for y in 0..h {
        if (left..right).filter(|&x| at(x, y)).count() as f64 / (right - left) as f64 > 0.45 {
            lines.push(y);
        }
        let mut sum = (local_left..local_left + window)
            .filter(|&x| at(x, y))
            .count();
        let mut best = sum;
        for x in local_left + window..right {
            sum += usize::from(at(x, y));
            sum -= usize::from(at(x - window, y));
            best = best.max(sum);
        }
        if best as f64 / window as f64 > 0.9 {
            short_lines.push(y);
        }
    }
    let staff: Vec<f64> = centers
        .iter()
        .enumerate()
        .map(|(i, &center)| {
            let low = if i > 0 {
                (centers[i - 1] + center) / 2.
            } else {
                0.
            };
            let high = if i + 1 < count {
                (centers[i + 1] + center) / 2.
            } else {
                h as f64
            };
            let mut hits: Vec<f64> = lines
                .iter()
                .map(|&y| y as f64)
                .filter(|&y| y > low && y < high)
                .collect();
            if hits.len() < 4 {
                hits = short_lines
                    .iter()
                    .map(|&y| y as f64)
                    .filter(|&y| y > low && y < high)
                    .collect();
            }
            if hits.is_empty() {
                center
            } else {
                median(hits)
            }
        })
        .collect();
    let mut systems = vec![0];
    let margin = (w as f64 * 0.23) as usize;
    for pair in staff.windows(2) {
        let first = round_even(pair[0]).max(0.).min(h as f64) as usize;
        let second = round_even(pair[1]).max(0.).min(h as f64) as usize;
        if second <= first || margin == 0 {
            return Ok(None);
        }
        let mut best = 0;
        let mut column = 0;
        for x in 0..margin {
            let sum = (first..second).filter(|&y| at(x, y)).count();
            if sum > best {
                best = sum;
                column = x;
            }
        }
        let score = best as f64 / (second - first) as f64;
        if score > 0.7 && score < 0.9 {
            let gaps = groups((first..second).filter(|&y| !at(column, y)).collect(), 1);
            let longest = gaps.iter().map(Vec::len).max().unwrap_or(0);
            if (longest as f64) < 12f64.max((second - first) as f64 * 0.08) {
                return Ok(None);
            }
        }
        systems.push(systems.last().unwrap() + usize::from(score < 0.9));
    }
    Ok(Some(systems))
}
