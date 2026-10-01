//! Native inference for the exported PP-DocLayout and signature networks.
//! The layout graph already performs NMS and maps boxes to original pixels.
use crate::image_boundary::{check_dimensions, round_even, Result};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};
use std::path::Path;

pub const LAYOUT_LABELS: [&str; 6] = [
    "measure_tab",
    "measure_notation",
    "measure_both",
    "tempo_region",
    "clef_region",
    "annotation_region",
];
pub use guitarocr_engine::onnx::initialize_runtime;
use guitarocr_engine::onnx::session;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub struct LayoutDetector {
    session: Session,
}
impl LayoutDetector {
    /// Full detector stage including row geometry and large-page detail regions.
    pub fn detect_page(
        &mut self,
        page: &RgbImage,
        threshold: f32,
        include_tempo: bool,
    ) -> Result<Value> {
        self.detect_page_with_cancel(page, threshold, include_tempo, &|| false)
    }

    pub fn detect_page_with_cancel(
        &mut self,
        page: &RgbImage,
        threshold: f32,
        include_tempo: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Value> {
        if cancelled() {
            return Err("Layout recognition cancelled".into());
        }
        let mut boxes =
            crate::layout_postprocess::deduplicate_pitch_boxes(&self.predict(page, threshold)?)?;
        if include_tempo && (page.width() > 2400 || page.height() > 3400) {
            let (width, height) = page.dimensions();
            let tw = (width as f64 * 0.55).ceil() as u32;
            let th = (height as f64 * 0.55).ceil() as u32;
            let is_region = |b: &Value| {
                matches!(
                    b["label"].as_str(),
                    Some(
                        "clef_region"
                            | "annotation_region"
                            | "transposition_region"
                            | "tempo_region"
                    )
                )
            };
            let mut candidates: Vec<Value> =
                boxes.iter().filter(|b| is_region(b)).cloned().collect();
            for y in [0, height - th] {
                for x in [0, width - tw] {
                    if cancelled() {
                        return Err("Layout recognition cancelled".into());
                    }
                    let tile = image::imageops::crop_imm(page, x, y, tw, th).to_image();
                    for mut b in self.predict(&tile, threshold)? {
                        if !is_region(&b) {
                            continue;
                        }
                        let a = b["coordinate"][0].as_f64().unwrap();
                        let top = b["coordinate"][1].as_f64().unwrap();
                        let c = b["coordinate"][2].as_f64().unwrap();
                        let bottom = b["coordinate"][3].as_f64().unwrap();
                        if (x > 0 && a < 3.)
                            || (y > 0 && top < 3.)
                            || (x + tw < width && c > tw as f64 - 3.)
                            || (y + th < height && bottom > th as f64 - 3.)
                        {
                            continue;
                        }
                        b["coordinate"] = json!([
                            a + x as f64,
                            top + y as f64,
                            c + x as f64,
                            bottom + y as f64
                        ]);
                        b["geometry_source"] = json!("detail_crop");
                        candidates.push(b);
                    }
                }
            }
            candidates.sort_by(|a, b| {
                b["score"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&a["score"].as_f64().unwrap())
            });
            let mut kept: Vec<Value> = vec![];
            for b in candidates {
                let coords = |v: &Value| -> [f64; 4] {
                    std::array::from_fn(|i| v["coordinate"][i].as_f64().unwrap())
                };
                let [x, y, r, bottom] = coords(&b);
                let area = (r - x) * (bottom - y);
                if !kept.iter().any(|other| {
                    if other["label"] != b["label"] {
                        return false;
                    }
                    let [a, t, c, d] = coords(other);
                    let hit = (r.min(c) - x.max(a)).max(0.) * (bottom.min(d) - y.max(t)).max(0.);
                    hit / (area + (c - a) * (d - t) - hit).max(1.) > 0.4
                }) {
                    kept.push(b);
                }
            }
            boxes.retain(|b| !is_region(b));
            boxes.extend(kept);
        }
        let mut output = crate::layout_postprocess::process_page(page, &boxes, threshold as f64)?;
        if !include_tempo {
            output["tempo_regions"] = json!([]);
            output["pitch_regions"] = json!([]);
        }
        Ok(output)
    }

    pub fn new(model: &Path) -> Result<Self> {
        let session = session(model)?;
        for name in ["image", "im_shape", "scale_factor"] {
            if !session.inputs.iter().any(|v| v.name == name) {
                return Err(format!("Layout model missing input {name}"));
            }
        }
        if !session.outputs.iter().any(|v| v.name == "fetch_name_0") {
            return Err("Layout model missing fetch_name_0 output".into());
        }
        Ok(Self { session })
    }
    pub fn predict(&mut self, page: &RgbImage, threshold: f32) -> Result<Vec<Value>> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("Invalid layout threshold".into());
        }
        let input = layout_input(page)?;
        let tensor = Tensor::from_array(([1usize, 3, 800, 800], input)).map_err(err)?;
        let shape = Tensor::from_array(([1usize, 2], vec![800f32, 800.])).map_err(err)?;
        let scale = Tensor::from_array((
            [1usize, 2],
            vec![800f32 / page.height() as f32, 800f32 / page.width() as f32],
        ))
        .map_err(err)?;
        let outputs = self
            .session
            .run(ort::inputs!["image"=>tensor, "im_shape"=>shape, "scale_factor"=>scale])
            .map_err(err)?;
        let (shape, rows) = outputs["fetch_name_0"]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        if shape.len() != 2 || shape[1] < 7 {
            return Err(format!("Layout output must be [N,7+], got {shape:?}"));
        }
        decode_layout(
            rows,
            shape[1] as usize,
            page.width(),
            page.height(),
            threshold,
        )
    }
}

/// OpenCV INTER_CUBIC uses half-pixel centers and Keys A=-0.75, not
/// image::CatmullRom (A=-0.5). Keep the training resize kernel explicitly.
pub fn layout_input(page: &RgbImage) -> Result<Vec<f32>> {
    check_dimensions(page.width(), page.height())?;
    let resized = crate::image_transforms::resize_rgb_cubic(page, 800, 800);
    let mut result = vec![0f32; 3 * 800 * 800];
    for y in 0..800 {
        for x in 0..800 {
            for c in 0..3 {
                result[c * 800 * 800 + y * 800 + x] =
                    resized.get_pixel(x as u32, y as u32).0[c] as f32 / 255.;
            }
        }
    }
    Ok(result)
}

pub fn decode_layout(
    values: &[f32],
    columns: usize,
    width: u32,
    height: u32,
    threshold: f32,
) -> Result<Vec<Value>> {
    if columns < 7
        || values.len() % columns != 0
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
    {
        return Err("Invalid layout output shape or threshold".into());
    }
    let mut rows = Vec::new();
    for row in values.chunks_exact(columns) {
        if !row[..7].iter().all(|x| x.is_finite()) {
            return Err("Non-finite layout output".into());
        }
        if row[0] < 0. || row[1] <= threshold {
            continue;
        }
        if row[0].fract() != 0. || row[0] as usize >= LAYOUT_LABELS.len() {
            return Err(format!("Unknown layout class {}", row[0]));
        }
        let cls = row[0] as usize;
        let coord = [
            round_even(row[2] as f64).max(0.) as u32,
            round_even(row[3] as f64).max(0.) as u32,
            round_even(row[4] as f64).max(0.).min(width as f64) as u32,
            round_even(row[5] as f64).max(0.).min(height as f64) as u32,
        ];
        if coord[2] > coord[0] && coord[3] > coord[1] {
            rows.push((
                row[6],
                json!({"cls_id":cls,"label":LAYOUT_LABELS[cls],"score":row[1],"coordinate":coord}),
            ));
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(rows.into_iter().map(|(_, v)| v).collect())
}

pub struct SignatureReader {
    session: Session,
}
impl SignatureReader {
    pub fn new(model: &Path) -> Result<Self> {
        let session = session(model)?;
        if !session.inputs.iter().any(|v| v.name == "images") {
            return Err("Signature model missing images input".into());
        }
        for name in ["key", "numerator", "denominator"] {
            if !session.outputs.iter().any(|v| v.name == name) {
                return Err(format!("Signature model missing output {name}"));
            }
        }
        Ok(Self { session })
    }
    /// One bounded batch; callers retain cancellation control between batches.
    pub fn predict(&mut self, pages: &[RgbImage]) -> Result<Vec<Value>> {
        if pages.is_empty() {
            return Ok(vec![]);
        }
        if pages.len() > 64 {
            return Err("Signature batch exceeds 64 images".into());
        }
        let mut data = Vec::with_capacity(pages.len() * 2 * 3 * 192 * 512);
        for page in pages {
            data.extend(signature_input(page)?);
        }
        let tensor = Tensor::from_array(([pages.len(), 2, 3, 192, 512], data)).map_err(err)?;
        let output = self
            .session
            .run(ort::inputs!["images"=>tensor])
            .map_err(err)?;
        let mut labels = Vec::new();
        for (name, classes) in [("key", 16), ("numerator", 33), ("denominator", 8)] {
            let (shape, values) = output[name].try_extract_tensor::<f32>().map_err(err)?;
            if shape.as_ref() != [pages.len() as i64, classes as i64] {
                return Err(format!("Unexpected signature {name} shape: {shape:?}"));
            }
            labels.push(argmax_rows(values, classes)?);
        }
        Ok((0..pages.len())
            .map(|i| signature_value(labels[0][i], labels[1][i], labels[2][i]))
            .collect())
    }
}
fn argmax_rows(values: &[f32], classes: usize) -> Result<Vec<usize>> {
    if values.len() % classes != 0 || !values.iter().all(|v| v.is_finite()) {
        return Err("Malformed/non-finite signature logits".into());
    }
    Ok(values
        .chunks_exact(classes)
        .map(|row| {
            let mut best = 0;
            for i in 1..row.len() {
                if row[i] > row[best] {
                    best = i;
                }
            }
            best
        })
        .collect())
}
fn signature_value(key: usize, numerator: usize, denominator: usize) -> Value {
    let denominators = [0, 1, 2, 4, 8, 16, 32, 64];
    json!({"key":if key<15 { Some(key as i32-7) } else { None },
        "time":if numerator>0 && denominator>0 {Some(format!("{numerator}/{}",denominators[denominator]))}else{None}})
}

pub fn signature_prefix(page: &RgbImage) -> Result<RgbImage> {
    check_dimensions(page.width(), page.height())?;
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for y in 0..page.height() {
        let ink = (0..page.width())
            .filter(|&x| {
                let p = page.get_pixel(x, y).0;
                ((19595u32 * p[0] as u32 + 38470u32 * p[1] as u32 + 7471u32 * p[2] as u32 + 32768)
                    >> 16)
                    < 160
            })
            .count();
        if ink as f64 / page.width() as f64 > 0.45 {
            if groups
                .last()
                .and_then(|g| g.last())
                .map(|last| *last + 1 == y)
                .unwrap_or(false)
            {
                groups.last_mut().unwrap().push(y);
            } else {
                groups.push(vec![y]);
            }
        }
    }
    let lines: Vec<f64> = groups
        .iter()
        .map(|g| g.iter().map(|y| *y as f64).sum::<f64>() / g.len() as f64)
        .collect();
    for staff in lines.windows(5) {
        let mut gaps: Vec<f64> = staff.windows(2).map(|v| v[1] - v[0]).collect();
        gaps.sort_by(f64::total_cmp);
        if gaps[0] >= 3. && gaps[3] < gaps[0] * 1.4 {
            let gap = (gaps[1] + gaps[2]) / 2.;
            let top = staff[0];
            let y0 = (top - gap * 5.).max(0.) as u32;
            let x1 = page.width().min((gap * 28.) as u32);
            let y1 = page.height().min((top + gap * 10.) as u32);
            return Ok(image::imageops::crop_imm(page, 0, y0, x1, y1 - y0).to_image());
        }
    }
    Ok(image::imageops::crop_imm(
        page,
        0,
        0,
        page.width().min(160.max(page.height())),
        page.height(),
    )
    .to_image())
}

pub fn signature_input(page: &RgbImage) -> Result<Vec<f32>> {
    let prefix = signature_prefix(page)?;
    let mut data = vec![1f32; 2 * 3 * 192 * 512]; // white padding maps to +1
    for (index, view) in [page, &prefix].iter().enumerate() {
        let (w, h) = view.dimensions();
        let ratio = w as f64 / h as f64;
        let (nw, nh) = if ratio > 512. / 192. {
            (512, round_even(512. / ratio).max(1.) as u32)
        } else {
            (round_even(192. * ratio).max(1.) as u32, 192)
        };
        let resized = crate::image_transforms::resize_rgb_lanczos(view, nw, nh);
        let top = (192 - nh) / 2;
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..3 {
                    data[index * 3 * 192 * 512
                        + c * 192 * 512
                        + (y + top) as usize * 512
                        + x as usize] = (resized.get_pixel(x, y).0[c] as f32 / 255. - 0.5) / 0.5;
                }
            }
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    #[test]
    fn layout_coordinates_are_already_original_and_sorted() {
        let rows = [
            0., 0.9, -2., 2.5, 130., 99., 8., 2., 0.8, 1., 1., 20., 20., 1., -1., 0.99, 0., 0.,
            30., 30., 0.,
        ];
        let decoded = decode_layout(&rows, 7, 100, 60, 0.25).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0]["label"], "measure_both");
        assert_eq!(decoded[1]["coordinate"], json!([0, 2, 100, 60]));
        assert!(decode_layout(&[6., 1., 0., 0., 1., 1., 0.], 7, 10, 10, 0.).is_err());
    }
    #[test]
    fn signature_labels_and_padding() {
        assert_eq!(signature_value(7, 4, 3), json!({"key":0,"time":"4/4"}));
        assert_eq!(signature_value(15, 0, 0), json!({"key":null,"time":null}));
        assert_eq!(argmax_rows(&[1., 1., 0.], 3).unwrap(), vec![0]);
        let page = RgbImage::from_pixel(20, 20, Rgb([0; 3]));
        let values = signature_input(&page).unwrap();
        assert_eq!(values.len(), 2 * 3 * 192 * 512);
        assert_eq!(values[0], -1.);
        assert_eq!(values[511], 1.);
    }
    #[test]
    fn staff_prefix_geometry() {
        let mut page = RgbImage::from_pixel(600, 160, Rgb([255; 3]));
        for y in [50, 60, 70, 80, 90] {
            for x in 0..600 {
                page.put_pixel(x, y, Rgb([0; 3]));
            }
        }
        assert_eq!(signature_prefix(&page).unwrap().dimensions(), (280, 150));
    }
}
