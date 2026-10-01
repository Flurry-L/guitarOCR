//! RGB resampling for the native ONNX input boundary.
//!
//! Layout uses OpenCV's cubic kernel (A=-0.75); signatures use Pillow's
//! antialiased Lanczos filter. Both return quantized RGB8 pixels, so tensor
//! normalization must happen *after* this step. Neither routine swaps channels.
//!
//! Provenance: source-informed Rust adaptation, not a clean-room implementation.
//! The coefficient formulas and fixed-point/quantization behavior were adapted
//! from OpenCV 4.13.0 `modules/imgproc/src/resize.cpp` and Pillow 12.3.0
//! `src/libImaging/Resample.c`:
//! https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp
//! https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
//! OpenCV's 4.x branch was consulted first; the relevant routines were then
//! checked against the fixed 4.13.0 tag used by the development goldens.
//!
//! Changes for GuitarOCR (2026): RGB8-only safe Rust storage and iteration,
//! a four-row cubic cache, i64 accumulators, and no C/C++/IPP/SIMD dependency.
//! OpenCV source copyrights: Intel Corporation (2000-2008, 2017),
//! Willow Garage Inc. (2009), and Itseez Inc. (2014-2015).
//! PIL/Pillow copyrights: Secret Labs AB (1997-2011), Fredrik Lundh and
//! contributors (1995-2011), Jeffrey 'Alex' Clark and contributors (2010).
//! Full upstream copyrights, conditions and disclaimers are retained in:
//! ../licenses/OpenCV-4.13.0-resize-NOTICE.txt (source-file license notice)
//! ../licenses/OpenCV-4.13.0-LICENSE.txt (upstream project Apache-2.0 license)
//! ../licenses/Pillow-12.3.0-LICENSE.txt (PIL/Pillow MIT-CMU license)
//! See ../licenses/image-transforms-SOURCES.txt for scope and provenance.
use image::RgbImage;

const CUBIC_BITS: u32 = 11;
const LANCZOS_BITS: u32 = 22;

fn assert_dimensions(input: &RgbImage, width: u32, height: u32) {
    assert!(
        input.width() > 0 && input.height() > 0,
        "empty resize input"
    );
    assert!(width > 0 && height > 0, "empty resize output");
}

struct CubicTap {
    pixels: [u32; 4],
    weights: [i32; 4],
}

fn cubic_taps(source: u32, target: u32) -> Vec<CubicTap> {
    // OpenCV computes the reciprocal of the explicit destination/source ratio.
    let scale = 1.0 / (f64::from(target) / f64::from(source));
    (0..target)
        .map(|i| {
            // Positions and polynomial arithmetic deliberately use float32,
            // before OpenCV's round-to-even 11-bit coefficient quantization.
            let position = ((f64::from(i) + 0.5) * scale - 0.5) as f32;
            let base = position.floor() as i64;
            let t = position - base as f32;
            let p = t + 1.0;
            let q = 1.0 - t;
            let a = -0.75f32;
            let left = ((a * p - 5.0 * a) * p + 8.0 * a) * p - 4.0 * a;
            let center = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
            let right = ((a + 2.0) * q - (a + 3.0)) * q * q + 1.0;
            let weights = [left, center, right, 1.0 - left - center - right]
                .map(|value| (value * (1 << CUBIC_BITS) as f32).round_ties_even() as i32);
            CubicTap {
                pixels: std::array::from_fn(|j| {
                    (base + j as i64 - 1).clamp(0, i64::from(source) - 1) as u32
                }),
                weights,
            }
        })
        .collect()
}

/// Resize RGB8 using OpenCV's generic integer INTER_CUBIC arithmetic.
///
/// Matches the half-pixel centers, replicated borders, float32 coefficients,
/// 11-bit weights and final rounded/saturated 22-bit shift in OpenCV 4.x:
/// https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp
/// OpenCV's IPP/SIMD paths can differ by one intensity level from this portable
/// scalar path; golden fixtures exercise both, without claiming cross-build
/// bit-exactness. This does not apply a downsampling antialias filter.
///
/// Panics if any source or destination dimension is zero. Callers must enforce
/// their image-size/allocation limits before reaching this low-level routine.
pub fn resize_rgb_cubic(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    assert_dimensions(input, width, height);
    if input.dimensions() == (width, height) {
        return input.clone();
    }
    let xs = cubic_taps(input.width(), width);
    let ys = cubic_taps(input.height(), height);
    let row_len = width as usize * 3;
    let mut output = RgbImage::new(width, height);
    // Only four horizontally filtered rows are needed at a time. Required
    // source rows are consecutive (or repeated at a border), so modulo four
    // is a collision-free cache for every vertical footprint.
    let mut rows: [(Option<u32>, Vec<i32>); 4] = std::array::from_fn(|_| (None, vec![0; row_len]));
    for (y, vertical) in ys.iter().enumerate() {
        for &sy in &vertical.pixels {
            let (cached_y, row) = &mut rows[sy as usize % 4];
            if *cached_y == Some(sy) {
                continue;
            }
            for (x, horizontal) in xs.iter().enumerate() {
                for channel in 0..3 {
                    row[x * 3 + channel] = horizontal
                        .pixels
                        .iter()
                        .zip(horizontal.weights)
                        .map(|(&sx, weight)| i32::from(input.get_pixel(sx, sy)[channel]) * weight)
                        .sum();
                }
            }
            *cached_y = Some(sy);
        }
        for (offset, dest) in output.as_mut()[y * row_len..(y + 1) * row_len]
            .iter_mut()
            .enumerate()
        {
            let sum: i64 = vertical
                .pixels
                .iter()
                .zip(vertical.weights)
                .map(|(&sy, weight)| i64::from(rows[sy as usize % 4].1[offset]) * i64::from(weight))
                .sum();
            *dest = ((sum + (1 << (2 * CUBIC_BITS - 1))) >> (2 * CUBIC_BITS)).clamp(0, 255) as u8;
        }
    }
    output
}

struct LanczosTap {
    start: u32,
    weights: Vec<i32>,
}

fn sinc(value: f64) -> f64 {
    if value == 0.0 {
        1.0
    } else {
        let angle = value * std::f64::consts::PI;
        angle.sin() / angle
    }
}

fn lanczos_taps(source: u32, target: u32) -> Vec<LanczosTap> {
    // Pillow's full-image box is stored as float32 before coefficient setup.
    let scale = f64::from(source as f32) / f64::from(target);
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    let inverse_scale = 1.0 / filter_scale;
    (0..target)
        .map(|i| {
            let center = (f64::from(i) + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as u32;
            let end = ((center + support + 0.5) as i64).min(i64::from(source)) as u32;
            let mut weights: Vec<f64> = (start..end)
                .map(|source_pixel| {
                    let distance = (f64::from(source_pixel) - center + 0.5) * inverse_scale;
                    if (-3.0..3.0).contains(&distance) {
                        sinc(distance) * sinc(distance / 3.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f64 = weights.iter().sum();
            if total != 0.0 {
                for weight in &mut weights {
                    *weight /= total;
                }
            }
            LanczosTap {
                start,
                // Pillow rounds signed coefficients away from zero at ties.
                weights: weights
                    .into_iter()
                    .map(|weight| (weight * f64::from(1 << LANCZOS_BITS)).round() as i32)
                    .collect(),
            }
        })
        .collect()
}

fn lanczos_pixel(sum: i64) -> u8 {
    ((sum + (1 << (LANCZOS_BITS - 1))) >> LANCZOS_BITS).clamp(0, 255) as u8
}

/// Resize RGB8 with Pillow's LANCZOS resampling contract.
///
/// Uses a radius-three sinc filter widened for downsampling, normalized edge
/// weights, 22-bit coefficients, and a rounded/clipped RGB8 horizontal pass
/// before the vertical pass. These details matter: image::Lanczos3's floating
/// intermediate and pass order are not equivalent to Pillow's RGB transform.
/// Algorithm reference (Pillow 12.3.0, src/libImaging/Resample.c):
/// https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
///
/// Panics if any dimension is zero. Same-size axes are copied, as in Pillow.
/// Input decoding, alpha handling, EXIF, contain geometry and normalization
/// belong to the calling image/tensor boundary, not this RGB-only primitive.
pub fn resize_rgb_lanczos(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    assert_dimensions(input, width, height);
    if input.dimensions() == (width, height) {
        return input.clone();
    }
    let horizontal = if width == input.width() {
        None
    } else {
        let taps = lanczos_taps(input.width(), width);
        let mut buffer = RgbImage::new(width, input.height());
        for y in 0..input.height() {
            for (x, tap) in taps.iter().enumerate() {
                let dest = buffer.get_pixel_mut(x as u32, y);
                for channel in 0..3 {
                    let sum = tap
                        .weights
                        .iter()
                        .enumerate()
                        .map(|(offset, &weight)| {
                            i64::from(input.get_pixel(tap.start + offset as u32, y)[channel])
                                * i64::from(weight)
                        })
                        .sum();
                    dest[channel] = lanczos_pixel(sum);
                }
            }
        }
        Some(buffer)
    };
    let intermediate = horizontal.as_ref().unwrap_or(input);
    if height == input.height() {
        return horizontal.unwrap_or_else(|| input.clone());
    }
    let taps = lanczos_taps(input.height(), height);
    let mut output = RgbImage::new(width, height);
    for (y, tap) in taps.iter().enumerate() {
        for x in 0..width {
            let dest = output.get_pixel_mut(x, y as u32);
            for channel in 0..3 {
                let sum = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(offset, &weight)| {
                        i64::from(intermediate.get_pixel(x, tap.start + offset as u32)[channel])
                            * i64::from(weight)
                    })
                    .sum();
                dest[channel] = lanczos_pixel(sum);
            }
        }
    }
    output
}
