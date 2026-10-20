//! Score image resolution, aspect-ratio padding, and VLM multimodal message encoding.

use crate::image_boundary;
use crate::recognition::Result;
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
use std::io::Cursor;

#[derive(Clone, Debug)]
pub struct ImagePolicy {
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub max_upscale: f64,
    pub patch_multiple: u32,
    pub page_max_pixels: u64,
}
impl Default for ImagePolicy {
    fn default() -> Self {
        Self {
            min_pixels: 256 * 256,
            max_pixels: 768 * 768,
            max_upscale: 2.5,
            patch_multiple: 28,
            page_max_pixels: 768 * 768,
        }
    }
}
impl ImagePolicy {
    pub fn from_value(v: &Value) -> Result<Self> {
        if v["version"].as_u64() != Some(2) {
            return Err("Unsupported score image policy version".into());
        }
        let mut p = Self::default();
        p.min_pixels = match v.get("min_pixels") {
            Some(value) => value.as_u64().ok_or("Invalid min_pixels")?,
            None => 256 * 256,
        };
        p.max_pixels = v["max_pixels"].as_u64().ok_or("Missing max_pixels")?;
        p.max_upscale = v["max_upscale"].as_f64().ok_or("Missing max_upscale")?;
        p.patch_multiple = v["patch_multiple"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("Invalid patch_multiple")?;
        p.page_max_pixels = match v.get("page_max_pixels") {
            Some(value) => value.as_u64().ok_or("Invalid page_max_pixels")?,
            None => p.max_pixels,
        };
        p.validate()?;
        Ok(p)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if self.patch_multiple == 0
            || self.patch_multiple > 256
            || self.max_pixels.min(self.page_max_pixels) < u64::from(self.patch_multiple).pow(2)
            || self.max_pixels < 112 * 112
            || self.page_max_pixels < 112 * 112
            || self.max_pixels > image_boundary::MAX_PAGE_PIXELS
            || self.page_max_pixels > image_boundary::MAX_PAGE_PIXELS
            || self.min_pixels < 112 * 112
            || self.min_pixels > self.max_pixels
            || !self.max_upscale.is_finite()
            || self.max_upscale <= 0.
            || self.max_upscale > 16.
        {
            return Err("Invalid image policy bounds".into());
        }
        Ok(())
    }
}

pub fn normalize_score_image(image: &RgbImage, policy: &ImagePolicy) -> Result<RgbImage> {
    image_boundary::check_dimensions(image.width(), image.height())?;
    policy.validate()?;
    // Preserve line weight and grid geometry while fitting the model's pixel budget.
    let area = f64::from(image.width()) * f64::from(image.height());
    let mut scale = policy
        .max_upscale
        .min((policy.min_pixels as f64 / area).sqrt().max(1.));
    let budget = if image.width() >= 600 && image.height() >= 900 {
        policy.page_max_pixels
    } else {
        policy.max_pixels
    };
    scale =
        scale.min((budget as f64 / (f64::from(image.width()) * f64::from(image.height()))).sqrt());
    let multiple = policy.patch_multiple;
    let (width, height, pw, ph) = loop {
        let w = (f64::from(image.width()) * scale).round_ties_even().max(1.) as u32;
        let h = (f64::from(image.height()) * scale)
            .round_ties_even()
            .max(1.) as u32;
        let pw = w.div_ceil(multiple) * multiple;
        let ph = h.div_ceil(multiple) * multiple;
        if u64::from(pw) * u64::from(ph) <= budget {
            break (w, h, pw, ph);
        }
        scale *= 0.98;
        if scale < 1e-12 {
            return Err("Image policy cannot fit a padded image".into());
        }
    };
    let mut pw = pw;
    while u64::from(pw) * u64::from(ph) < 112 * 112 {
        pw += multiple;
    }
    if u64::from(pw) * u64::from(ph) > budget {
        return Err("Image policy cannot fit minimum processor area".into());
    }
    image_boundary::check_dimensions(pw, ph)?;
    let resized = if image.dimensions() == (width, height) {
        image.clone()
    } else {
        crate::image_transforms::resize_rgb_lanczos(image, width, height)
    };
    let mut canvas = RgbImage::from_pixel(pw, ph, Rgb([255; 3]));
    image::imageops::replace(
        &mut canvas,
        &resized,
        i64::from((pw - width) / 2),
        i64::from((ph - height) / 2),
    );
    Ok(canvas)
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(ALPHABET[(n >> 18) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if c.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

pub fn image_message(image: &RgbImage, policy: Option<&ImagePolicy>) -> Result<Value> {
    let normalized = policy
        .map(|p| normalize_score_image(image, p))
        .transpose()?;
    let image = normalized.as_ref().unwrap_or(image);
    let mut data = Cursor::new(Vec::new());
    image
        .write_to(&mut data, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(
        json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}",base64(data.get_ref()))}}),
    )
}

pub fn image_messages(
    images: &[RgbImage],
    prompt: &str,
    policy: Option<&ImagePolicy>,
) -> Result<Value> {
    let mut content = images
        .iter()
        .map(|im| image_message(im, policy))
        .collect::<Result<Vec<_>>>()?;
    content.push(json!({"type":"text","text":prompt}));
    Ok(json!([{"role":"user","content":content}]))
}
