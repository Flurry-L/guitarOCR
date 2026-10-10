//! Bounded model views for dense pages. Only model proposals define measures.
use crate::image_boundary::{check_dimensions, round_even, Result};
use image::RgbImage;
use serde_json::{json, Value};

pub const MAX_VIEWS: usize = 24;
const SIDES: [u32; 2] = [1280, 1920];
const EDGE_GUARD: f64 = 0.05;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tile {
    pub side: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Tile {
    pub fn view(self) -> [u32; 5] {
        [self.side, self.x, self.y, self.width, self.height]
    }
}

fn origins(length: u32, side: u32) -> Vec<u32> {
    if length <= side {
        return vec![0];
    }
    let count = ((length - side) as f64 / (side as f64 * 0.5)).ceil() as u32 + 1;
    let mut result: Vec<u32> = (0..count)
        .map(|index| round_even((index * (length - side)) as f64 / (count - 1) as f64) as u32)
        .collect();
    result.dedup();
    result
}

/// At most 23 tile views plus one full page. Selection depends only on size.
pub fn tile_plan(width: u32, height: u32) -> Result<Vec<Tile>> {
    check_dimensions(width, height)?;
    if width > 20_000 || height > 20_000 {
        return Err("Layout page exceeds supported image dimensions".into());
    }
    if width.max(height) <= SIDES[1] {
        return Ok(vec![]);
    }
    let groups: Vec<Vec<Tile>> = SIDES
        .iter()
        .map(|&side| {
            let tw = width.min(side);
            let th = height.min(side);
            let mut views = Vec::new();
            for y in origins(height, th) {
                for x in origins(width, tw) {
                    views.push(Tile {
                        side,
                        x,
                        y,
                        width: tw,
                        height: th,
                    });
                }
            }
            views
        })
        .collect();
    let count: usize = groups.iter().map(Vec::len).sum();
    if count < MAX_VIEWS {
        return Ok(groups.into_iter().flatten().collect());
    }
    let quotas: Vec<f64> = groups
        .iter()
        .map(|group| ((MAX_VIEWS - 1) * group.len()) as f64 / count as f64)
        .collect();
    let mut allocation: Vec<usize> = quotas.iter().map(|quota| quota.floor() as usize).collect();
    let remainder = MAX_VIEWS - 1 - allocation.iter().sum::<usize>();
    let mut scale_order = [0usize, 1];
    scale_order.sort_by(|&a, &b| {
        (quotas[b] - allocation[b] as f64)
            .total_cmp(&(quotas[a] - allocation[a] as f64))
            .then_with(|| SIDES[a].cmp(&SIDES[b]))
    });
    for &index in scale_order.iter().take(remainder) {
        allocation[index] += 1;
    }
    let distance = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
    let mut chosen = Vec::new();
    for (group, quota) in groups.iter().zip(allocation) {
        let points: Vec<_> = group
            .iter()
            .map(|v| {
                (
                    (v.x as f64 + v.width as f64 / 2.) / width as f64,
                    (v.y as f64 + v.height as f64 / 2.) / height as f64,
                )
            })
            .collect();
        let mut selected = Vec::<usize>::new();
        for _ in 0..quota {
            let mut best = None;
            for index in 0..group.len() {
                if selected.contains(&index) {
                    continue;
                }
                let value = if selected.is_empty() {
                    -distance(points[index], (0.5, 0.5))
                } else {
                    selected
                        .iter()
                        .map(|&other| distance(points[index], points[other]))
                        .fold(f64::INFINITY, f64::min)
                };
                // Python min/max choose the first item on equal distance.
                if best.is_none_or(|(_, score)| value > score) {
                    best = Some((index, value));
                }
            }
            selected.push(best.expect("quota cannot exceed its candidate group").0);
        }
        selected.sort_unstable();
        chosen.extend(selected.into_iter().map(|index| group[index]));
    }
    Ok(chosen)
}

fn is_measure(box_: &Value) -> bool {
    matches!(
        box_["label"].as_str(),
        Some("measure" | "measure_tab" | "measure_notation" | "measure_both")
    )
}

fn coordinates(box_: &Value) -> Result<[f64; 4]> {
    let values = box_["coordinate"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or("Layout measure requires four coordinates")?;
    let mut coordinates = [0.; 4];
    for (index, value) in values.iter().enumerate() {
        coordinates[index] = value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("Layout measure coordinates must be finite")?;
    }
    if coordinates[2] <= coordinates[0] || coordinates[3] <= coordinates[1] {
        return Err("Layout measure must have positive area".into());
    }
    Ok(coordinates)
}

/// Reject internal-edge fragments and preserve model provenance in page pixels.
pub fn remap_measure(box_: &Value, tile: Tile, page_size: (u32, u32)) -> Result<Option<Value>> {
    if !is_measure(box_) {
        return Ok(None);
    }
    let [a, b, c, d] = coordinates(box_)?;
    let guard = tile.width.min(tile.height) as f64 * EDGE_GUARD;
    if (tile.x > 0 && a <= guard)
        || (tile.y > 0 && b <= guard)
        || (tile.x + tile.width < page_size.0 && c >= tile.width as f64 - guard)
        || (tile.y + tile.height < page_size.1 && d >= tile.height as f64 - guard)
    {
        return Ok(None);
    }
    let mut remapped = box_.clone();
    remapped["coordinate"] = json!([
        a + tile.x as f64,
        b + tile.y as f64,
        c + tile.x as f64,
        d + tile.y as f64
    ]);
    remapped["geometry_source"] = json!("model_tile");
    remapped["view"] = json!(tile.view());
    Ok(Some(remapped))
}

/// Stable score-ordered class-agnostic IoS NMS; non-measure regions are intact.
pub fn fuse_measures(boxes: &[Value]) -> Result<Vec<Value>> {
    let mut regions = Vec::new();
    let mut measures = Vec::new();
    for box_ in boxes {
        if is_measure(box_) {
            let score = box_
                .get("fusion_score")
                .unwrap_or(&box_["score"])
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("Layout measure score must be finite")?;
            measures.push((box_.clone(), coordinates(box_)?, score));
        } else {
            regions.push(box_.clone());
        }
    }
    measures.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut kept = Vec::<(Value, [f64; 4], f64)>::new();
    for (box_, coordinate, score) in measures {
        let [a, b, c, d] = coordinate;
        let area = (c - a) * (d - b);
        if kept.iter().any(|(_, other, _)| {
            let [x, y, r, s] = *other;
            let hit = (c.min(r) - a.max(x)).max(0.) * (d.min(s) - b.max(y)).max(0.);
            hit / area.min((r - x) * (s - y)).max(1e-9) >= 0.5
        }) {
            continue;
        }
        kept.push((box_, coordinate, score));
    }
    regions.extend(kept.into_iter().map(|(box_, _, _)| box_));
    Ok(regions)
}

/// Preserve every eligible proposal until the quality model ranks it.
pub fn collect_model_tiles(
    page: &RgbImage,
    full: Vec<Value>,
    cancelled: &dyn Fn() -> bool,
    mut predict: impl FnMut(&RgbImage) -> Result<Vec<Value>>,
) -> Result<(Vec<Value>, Vec<crate::layout_quality::View>)> {
    let views = tile_plan(page.width(), page.height())?;
    let mut executed = vec![[-1, 0, 0, page.width() as i64, page.height() as i64]];
    if cancelled() {
        return Err("Layout recognition cancelled".into());
    }
    if views.is_empty() {
        return Ok((full, executed));
    }
    let mut candidates = full;
    for box_ in &mut candidates {
        if is_measure(box_) {
            box_["view"] = json!([-1, 0, 0, page.width(), page.height()]);
        }
    }
    for view in views {
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        let tile =
            image::imageops::crop_imm(page, view.x, view.y, view.width, view.height).to_image();
        for box_ in predict(&tile)? {
            if let Some(remapped) = remap_measure(&box_, view, page.dimensions())? {
                candidates.push(remapped);
            }
        }
        executed.push(view.view().map(i64::from));
        // Tile pixels are released before the next model view is allocated.
        drop(tile);
    }
    if cancelled() {
        return Err("Layout recognition cancelled".into());
    }
    Ok((candidates, executed))
}
