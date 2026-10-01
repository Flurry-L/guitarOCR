//! Conservative chord-diagram raster refinement from research/inference/information/chord_geometry.py.
//! Grid/dot/barre evidence may correct fret positions, never chord names or
//! printed base/fingers. None means retain the model prediction unchanged.
use crate::image_boundary::{check_dimensions, open_rgb, round_even, Result};
use image::RgbImage;
use serde_json::{json, Value};
use std::{collections::VecDeque, path::Path};
struct Ink {
    data: Vec<u8>,
    w: usize,
    h: usize,
}
fn runs(values: impl IntoIterator<Item = usize>) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![];
    for v in values {
        if out
            .last()
            .and_then(|g| g.last())
            .map(|p| v > *p + 1)
            .unwrap_or(true)
        {
            out.push(vec![v]);
        } else {
            out.last_mut().unwrap().push(v);
        }
    }
    out
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}
impl Ink {
    fn new(image: &RgbImage) -> Self {
        Self {
            w: image.width() as usize,
            h: image.height() as usize,
            data: image
                .pixels()
                .map(|p| {
                    u8::from(
                        ((19595u32 * p[0] as u32
                            + 38470u32 * p[1] as u32
                            + 7471u32 * p[2] as u32
                            + 32768)
                            >> 16)
                            < 175,
                    )
                })
                .collect(),
        }
    }
    fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }
    // OpenCV morphologyEx OPEN: erode then dilate, default centered anchor
    // (floor(k/2), including even kernels). Outside erosion=1, dilation=0.
    fn opening(&self, kernel: usize, vertical: bool) -> Self {
        let (length, lanes) = if vertical {
            (self.h, self.w)
        } else {
            (self.w, self.h)
        };
        let anchor = kernel / 2;
        let mut result = vec![0; self.data.len()];
        for lane in 0..lanes {
            let at = |i: usize| {
                if vertical {
                    i * self.w + lane
                } else {
                    lane * self.w + i
                }
            };
            let mut summed = vec![0usize; length + 1];
            for i in 0..length {
                summed[i + 1] = summed[i] + self.data[at(i)] as usize;
            }
            let mut eroded = vec![0usize; length + 1];
            for i in 0..length {
                let start = i.saturating_sub(anchor);
                let stop = (i + kernel - anchor).min(length);
                eroded[i + 1] =
                    eroded[i] + usize::from(summed[stop] - summed[start] == stop - start);
            }
            for i in 0..length {
                let start = i.saturating_sub(anchor);
                let stop = (i + kernel - anchor).min(length);
                result[at(i)] = u8::from(eroded[stop] > eroded[start]);
            }
        }
        Self {
            data: result,
            w: self.w,
            h: self.h,
        }
    }
    fn patch(&self, left: i64, right: i64, upper: i64, lower: i64) -> (usize, usize, usize, usize) {
        let normalize = |v: i64, size: usize| {
            if v < 0 {
                (size as i64 + v).max(0) as usize
            } else {
                v.min(size as i64) as usize
            }
        };
        (
            normalize(left, self.w),
            normalize(right, self.w),
            normalize(upper, self.h),
            normalize(lower, self.h),
        )
    }
    fn mean(&self, left: i64, right: i64, upper: i64, lower: i64) -> Option<f64> {
        let (x0, x1, y0, y1) = self.patch(left, right, upper, lower);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some(
            (y0..y1)
                .map(|y| (x0..x1).map(|x| self.at(x, y) as usize).sum::<usize>())
                .sum::<usize>() as f64
                / ((x1 - x0) * (y1 - y0)) as f64,
        )
    }
    fn open_marker(&self, left: i64, right: i64, upper: i64, lower: i64) -> Option<bool> {
        let (x0, x1, y0, y1) = self.patch(left, right, upper, lower);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let (w, h) = (x1 - x0, y1 - y0);
        let total = (y0..y1)
            .map(|y| (x0..x1).map(|x| self.at(x, y) as usize).sum::<usize>())
            .sum::<usize>();
        if total < 4 {
            return None;
        }
        // RETR_CCOMP uses 8-connected foreground / 4-connected background. Each
        // enclosed background component produces a child contour of area >=1;
        // even a one-pixel hole has a 2-square-pixel diamond contour in OpenCV.
        let mut visited = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                if visited[y * w + x] || self.at(x + x0, y + y0) != 0 {
                    continue;
                }
                let mut queue = VecDeque::from([(x, y)]);
                visited[y * w + x] = true;
                let mut boundary = false;
                while let Some((x, y)) = queue.pop_front() {
                    boundary |= x == 0 || y == 0 || x + 1 == w || y + 1 == h;
                    for (nx, ny) in [
                        (x.wrapping_sub(1), y),
                        (x + 1, y),
                        (x, y.wrapping_sub(1)),
                        (x, y + 1),
                    ] {
                        if nx < w
                            && ny < h
                            && !visited[ny * w + nx]
                            && self.at(nx + x0, ny + y0) == 0
                        {
                            visited[ny * w + nx] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
                if !boundary {
                    return Some(true);
                }
            }
        }
        Some(false)
    }
}
fn rounded(v: f64) -> i64 {
    round_even(v) as i64
}
pub fn refine_diagram(path: &Path, diagram: &Value) -> Result<Option<Value>> {
    let eligible = diagram["frets"]
        .as_array()
        .is_some_and(|f| (3..=12).contains(&f.len()))
        && diagram["base_fret"]
            .as_i64()
            .is_some_and(|b| (1..=36).contains(&b));
    if !eligible {
        return Ok(None);
    }
    refine_diagram_image(&open_rgb(path)?, diagram)
}
/// Return corrected raw fields for the caller's shared normalize_diagram. Extra
/// fields, printed base_fret and fingers remain unchanged. Rejection is None.
pub fn refine_diagram_image(image: &RgbImage, diagram: &Value) -> Result<Option<Value>> {
    let Some(frets) = diagram["frets"].as_array() else {
        return Ok(None);
    };
    let Some(base) = diagram["base_fret"].as_i64() else {
        return Ok(None);
    };
    let count = frets.len();
    if !(1..=36).contains(&base) || !(3..=12).contains(&count) {
        return Ok(None);
    }
    check_dimensions(image.width(), image.height())?;
    let ink = Ink::new(image);
    let (h, w) = (ink.h, ink.w);
    let vertical = ink.opening(12.max(round_even(h as f64 * 0.23) as usize), true);
    let strength: Vec<usize> = (0..w)
        .map(|x| (0..h).map(|y| vertical.at(x, y) as usize).sum())
        .collect();
    let maximum = *strength.iter().max().unwrap();
    let columns = runs((0..w).filter(|&x| strength[x] as f64 >= 12f64.max(maximum as f64 * 0.7)));
    let xs: Vec<f64> = columns
        .iter()
        .map(|c| {
            c.iter()
                .map(|&x| x as f64 * strength[x] as f64)
                .sum::<f64>()
                / c.iter().map(|&x| strength[x]).sum::<usize>() as f64
        })
        .collect();
    if xs.len() != count {
        return Ok(None);
    }
    let gap = median(xs.windows(2).map(|a| a[1] - a[0]).collect());
    if xs
        .windows(2)
        .any(|a| ((a[1] - a[0]) - gap).abs() > 1.5f64.max(gap * 0.12))
    {
        return Ok(None);
    }
    let mut bands = vec![];
    for &x in &xs {
        let ys: Vec<_> = (0..h)
            .filter(|&y| vertical.at(rounded(x) as usize, y) != 0)
            .collect();
        if ys.is_empty() {
            return Ok(None);
        }
        bands.push((ys[0] as f64, *ys.last().unwrap() as f64));
    }
    let top = median(bands.iter().map(|b| b.0).collect());
    let bottom = median(bands.iter().map(|b| b.1).collect());
    if bands
        .iter()
        .any(|&(a, b)| (a - top).abs() > 2. || (b - bottom).abs() > 2.)
    {
        return Ok(None);
    }
    let horizontal = ink.opening(
        12.max(round_even((xs[count - 1] - xs[0]) * 0.7) as usize),
        false,
    );
    let strength: Vec<usize> = (0..h)
        .map(|y| {
            (rounded(xs[0]) as usize..=rounded(xs[count - 1]) as usize)
                .map(|x| horizontal.at(x, y) as usize)
                .sum()
        })
        .collect();
    let strokes: Vec<(f64, usize)> =
        runs((0..h).filter(|&y| strength[y] as f64 > (xs[count - 1] - xs[0]) * 0.7))
            .iter()
            .map(|c| {
                (
                    c.iter()
                        .map(|&y| y as f64 * strength[y] as f64)
                        .sum::<f64>()
                        / c.iter().map(|&y| strength[y]).sum::<usize>() as f64,
                    c.len(),
                )
            })
            .collect();
    let mut grid: Option<(usize, f64, f64)> = None;
    for fret_count in 3..8 {
        let mut best: Option<(f64, f64, f64)> = None;
        let mut origin = top - 1.;
        let end = top + 4f64.max((bottom - top) / fret_count as f64 * 0.3) + 0.25;
        while origin < end {
            let spacing = (bottom - origin) / fret_count as f64;
            if spacing >= 5. {
                let lines: Vec<f64> = strokes
                    .iter()
                    .filter(|(_, thickness)| {
                        *thickness <= 2.max(round_even(spacing * 0.15) as usize)
                    })
                    .map(|(y, _)| *y)
                    .collect();
                let errors: Vec<f64> = (1..=fret_count)
                    .map(|i| {
                        lines
                            .iter()
                            .map(|y| (y - (origin + i as f64 * spacing)).abs())
                            .fold(f64::INFINITY, f64::min)
                    })
                    .collect();
                if errors.iter().all(|&e| e <= 1.5f64.max(spacing * 0.1)) {
                    let candidate = (errors.iter().sum::<f64>(), origin, spacing);
                    if best
                        .as_ref()
                        .map(|b| {
                            candidate
                                .0
                                .total_cmp(&b.0)
                                .then(candidate.1.total_cmp(&b.1))
                                .then(candidate.2.total_cmp(&b.2))
                                .is_lt()
                        })
                        .unwrap_or(true)
                    {
                        best = Some(candidate);
                    }
                }
            }
            origin += 0.25;
        }
        if let Some((_, origin, spacing)) = best {
            grid = Some((fret_count, origin, spacing));
        }
    }
    let Some((fret_count, top, step)) = grid else {
        return Ok(None);
    };
    let mut positions = vec![];
    for &x in &xs {
        let mut dots = vec![];
        for i in 0..fret_count {
            let y = top + (i as f64 + 0.5) * step;
            let radius = gap.min(step) * 0.28;
            let (left, right, upper, lower) = (
                rounded(x - radius),
                rounded(x + radius) + 1,
                rounded(y - radius),
                rounded(y + radius) + 1,
            );
            if left.min(upper) < 0 || right > w as i64 || lower > h as i64 {
                return Ok(None);
            }
            if ink
                .mean(left, right, upper, lower)
                .is_some_and(|v| v > 0.48)
            {
                dots.push(base + i as i64);
            }
        }
        if let Some(&fret) = dots.last() {
            if fret > 36 {
                return Ok(None);
            }
            positions.push(json!(fret));
            continue;
        }
        let left = rounded(x - gap * 0.34).max(0);
        let right = rounded(x + gap * 0.34) + 1;
        let upper = rounded(top - step * 1.1).max(0);
        let lower = rounded(top - step * 0.18);
        let Some(open) = ink.open_marker(left, right, upper, lower) else {
            return Ok(None);
        };
        positions.push(if open { json!(0) } else { json!("x") });
    }
    let mut barres = vec![];
    for fret in 0..fret_count {
        let y = top + (fret as f64 + 0.5) * step;
        let mut bridges = vec![];
        for (index, pair) in xs.windows(2).enumerate() {
            let middle = (pair[0] + pair[1]) / 2.;
            let upper = rounded(y - step * 0.12);
            let lower = rounded(y + step * 0.12) + 1;
            let patch = ink.mean(
                rounded(middle - gap * 0.3),
                rounded(middle + gap * 0.3) + 1,
                upper,
                lower,
            );
            let center = ink.mean(
                rounded(middle - gap * 0.1),
                rounded(middle + gap * 0.1) + 1,
                upper,
                lower,
            );
            let compatible = positions[index..index + 2]
                .iter()
                .all(|p| p.as_i64().map(|p| p >= base + fret as i64).unwrap_or(true));
            if compatible && patch.is_some_and(|v| v > 0.7) && center.is_some_and(|v| v > 0.7) {
                bridges.push(index);
            }
        }
        for group in runs(bridges) {
            if base + fret as i64 > 36 {
                return Ok(None);
            }
            barres.push(json!([
                base + fret as i64,
                group[0],
                group.last().unwrap() + 1
            ]));
        }
    }
    let mut corrected = diagram.clone();
    corrected["frets"] = json!(positions);
    corrected["barres"] = json!(barres);
    Ok(Some(corrected))
}
