//! Native RGB/image and PDF boundaries. PDFium is loaded explicitly from the
//! application's verified runtime bundle, never from PATH or a Python package.
use image::{ImageDecoder, ImageReader, Rgb, RgbImage};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_ulong, c_void},
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    sync::Mutex,
};

pub const MODEL_RENDER_DPI: u32 = 180;
pub const MAX_PAGE_PIXELS: u64 = 40_000_000;
pub const MAX_TOTAL_PIXELS: u64 = 200_000_000;
static PDFIUM_LOCK: Mutex<()> = Mutex::new(());
pub type Result<T> = std::result::Result<T, String>;

/// Decode with bounded allocations and composite transparent input onto white.
pub fn open_rgb(path: &Path) -> Result<RgbImage> {
    let mut images = Vec::new();
    for_each_image_page(path, 1, |image| {
        images.push(image);
        Ok(())
    })?;
    images.pop().ok_or_else(|| "Image contains no pages".into())
}

fn decode_rgb(bytes: &[u8], format: image::ImageFormat) -> Result<RgbImage> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(20_000);
    limits.max_image_height = Some(20_000);
    limits.max_alloc = Some(MAX_PAGE_PIXELS * 8);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(|e| e.to_string())?;
    let (width, height) = decoder.dimensions();
    check_dimensions(width, height)?;
    let orientation = decoder.orientation().map_err(|e| e.to_string())?;
    let mut decoded = image::DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    decoded.apply_orientation(orientation);
    check_dimensions(decoded.width(), decoded.height())?;
    let rgba = decoded.to_rgba8();
    Ok(RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y).0;
        Rgb(std::array::from_fn(|c| {
            ((p[c] as u32 * p[3] as u32 + 255 * (255 - p[3] as u32) + 127) / 255) as u8
        }))
    }))
}

/// Decode all TIFF IFD pages in original order, applying each frame's EXIF
/// orientation. Frames leave the decoder through a callback, never accumulating
/// a whole document in memory. Limit violations are explicit, never truncation.
pub fn for_each_image_page(
    path: &Path,
    max_pages: usize,
    mut consume: impl FnMut(RgbImage) -> Result<()>,
) -> Result<usize> {
    if max_pages == 0 {
        return Err("Image page limit must be positive".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(200 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 200 * 1024 * 1024 {
        return Err("Image file exceeds 200 MiB".into());
    }
    let format = image::guess_format(&bytes).map_err(|e| e.to_string())?;
    let mut total = 0u64;
    if format == image::ImageFormat::Tiff {
        let (pointer, size, offsets) = tiff_pages(&bytes, max_pages)?;
        for offset in &offsets {
            bytes[pointer..pointer + size].copy_from_slice(offset);
            let image = decode_rgb(&bytes, format)?;
            total += u64::from(image.width()) * u64::from(image.height());
            if total > MAX_TOTAL_PIXELS {
                return Err(
                    "Image document exceeds 200 megapixels; split it into smaller projects".into(),
                );
            }
            consume(image)?;
        }
        Ok(offsets.len())
    } else {
        consume(decode_rgb(&bytes, format)?)?;
        Ok(1)
    }
}

// Preserve encoded sample data and original absolute offsets: only the TIFF
// header's first-IFD pointer is changed before handing each frame to image's
// existing bounded TIFF decoder. Handles classic TIFF and BigTIFF, both endian.
fn tiff_pages(bytes: &[u8], max_pages: usize) -> Result<(usize, usize, Vec<Vec<u8>>)> {
    let little = match bytes.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Err("Invalid TIFF byte order".into()),
    };
    let number = |at: usize, size: usize| -> Result<u64> {
        let raw = bytes
            .get(at..at.checked_add(size).ok_or("TIFF offset overflow")?)
            .ok_or("Truncated TIFF directory")?;
        let mut n = 0u64;
        if little {
            for (i, &v) in raw.iter().enumerate() {
                n |= (v as u64) << (i * 8);
            }
        } else {
            for &v in raw {
                n = (n << 8) | v as u64;
            }
        }
        Ok(n)
    };
    let (pointer, size, count_size, entry_size) = match number(2, 2)? {
        42 => (4, 4, 2, 12),
        43 if number(4, 2)? == 8 && number(6, 2)? == 0 => (8, 8, 8, 20),
        _ => return Err("Unsupported TIFF header".into()),
    };
    let mut current = number(pointer, size)?;
    let mut seen = std::collections::BTreeSet::new();
    let mut offsets = Vec::new();
    while current != 0 {
        if offsets.len() >= max_pages {
            return Err(format!(
                "Image contains more than {max_pages} pages; use a larger document page limit"
            ));
        }
        if !seen.insert(current) {
            return Err("Cyclic TIFF page chain".into());
        }
        let at = usize::try_from(current).map_err(|_| "TIFF page offset exceeds platform limit")?;
        let count = usize::try_from(number(at, count_size)?)
            .map_err(|_| "TIFF directory count overflow")?;
        if count > 1_000_000 {
            return Err("Unreasonable TIFF directory size".into());
        }
        let next = at
            .checked_add(count_size)
            .and_then(|v| count.checked_mul(entry_size).and_then(|n| v.checked_add(n)))
            .ok_or("TIFF directory offset overflow")?;
        let encoded = if little {
            current.to_le_bytes()
        } else {
            current.to_be_bytes()
        };
        offsets.push(if little {
            encoded[..size].to_vec()
        } else {
            encoded[8 - size..].to_vec()
        });
        current = number(next, size)?;
    }
    if offsets.is_empty() {
        return Err("TIFF has no image directories".into());
    }
    Ok((pointer, size, offsets))
}

pub fn check_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PAGE_PIXELS {
        Err("Page dimensions are empty or exceed the 40 megapixel safety limit".into())
    } else {
        Ok(())
    }
}

/// Pillow RGB->L integer coefficients, expanded to RGB for native tensor APIs.
pub fn grayscale_rgb(image: &RgbImage) -> RgbImage {
    RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let p = image.get_pixel(x, y).0;
        let gray =
            ((19595u32 * p[0] as u32 + 38470u32 * p[1] as u32 + 7471u32 * p[2] as u32 + 32768)
                >> 16) as u8;
        Rgb([gray; 3])
    })
}

/// Python round()/numpy round() use ties-to-even, unlike f32::round().
pub fn round_even(value: f64) -> f64 {
    value.round_ties_even()
}

pub fn crop_measure(page: &RgbImage, bbox_xywh: [f64; 4]) -> Result<RgbImage> {
    let [left, top, width, height] = bbox_xywh;
    if !bbox_xywh.iter().all(|v| v.is_finite()) || width <= 0.0 || height <= 0.0 {
        return Err("Measure crop requires a finite, positive xywh rectangle".into());
    }
    let px = 1.5 * f64::from(MODEL_RENDER_DPI) / 25.4;
    let py = 4.0 * f64::from(MODEL_RENDER_DPI) / 25.4;
    let x0 = round_even(left - px).max(0.0).min(f64::from(page.width())) as u32;
    let y0 = round_even(top - py).max(0.0).min(f64::from(page.height())) as u32;
    let x1 = round_even(left + width + px)
        .max(0.0)
        .min(f64::from(page.width())) as u32;
    let y1 = round_even(top + height + py)
        .max(0.0)
        .min(f64::from(page.height())) as u32;
    if x1 <= x0 || y1 <= y0 {
        return Err("Measure crop does not intersect the page".into());
    }
    Ok(image::imageops::crop_imm(page, x0, y0, x1 - x0, y1 - y0).to_image())
}

#[derive(Debug, Clone)]
pub struct RenderedPage {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub page: usize,
}

/// PDFium's public C ABI. The global lock covers initialization, all native
/// handles, rendering, and destruction; PDFium is not thread-safe.
/// Text/path extraction is intentionally not claimed by this raster boundary.
pub fn render_pdf_pages(
    library_path: &Path,
    pdf: &Path,
    output: &Path,
    max_pages: usize,
) -> Result<Vec<RenderedPage>> {
    if max_pages == 0 {
        return Err("max_pages must be positive".into());
    }
    let _lock = PDFIUM_LOCK.lock().map_err(|_| "PDFium lock poisoned")?;
    if !library_path.is_absolute() || !library_path.is_file() {
        return Err("A verified absolute PDFium library path is required".into());
    }
    let length = fs::metadata(pdf).map_err(|e| e.to_string())?.len();
    if length == 0 || length > 200 * 1024 * 1024 {
        return Err("PDF must be between 1 byte and 200 MiB".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(pdf)
        .map_err(|e| e.to_string())?
        .take(200 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.is_empty() || bytes.len() > 200 * 1024 * 1024 {
        return Err("PDF must be between 1 byte and 200 MiB".into());
    }
    unsafe {
        let lib = Library::new(library_path).map_err(|e| format!("Cannot load PDFium: {e}"))?;
        macro_rules! api {
            ($name:literal, $ty:ty) => {
                *lib.get::<$ty>(concat!($name, "\0").as_bytes())
                    .map_err(|e| format!("PDFium ABI: {e}"))?
            };
        }
        let init = api!("FPDF_InitLibrary", unsafe extern "system" fn());
        let destroy = api!("FPDF_DestroyLibrary", unsafe extern "system" fn());
        let load = api!(
            "FPDF_LoadMemDocument64",
            unsafe extern "system" fn(*const c_void, usize, *const c_char) -> *mut c_void
        );
        let close = api!("FPDF_CloseDocument", unsafe extern "system" fn(*mut c_void));
        let last_error = api!("FPDF_GetLastError", unsafe extern "system" fn() -> c_ulong);
        let count = api!(
            "FPDF_GetPageCount",
            unsafe extern "system" fn(*mut c_void) -> c_int
        );
        let load_page = api!(
            "FPDF_LoadPage",
            unsafe extern "system" fn(*mut c_void, c_int) -> *mut c_void
        );
        let close_page = api!("FPDF_ClosePage", unsafe extern "system" fn(*mut c_void));
        let width = api!(
            "FPDF_GetPageWidth",
            unsafe extern "system" fn(*mut c_void) -> f64
        );
        let height = api!(
            "FPDF_GetPageHeight",
            unsafe extern "system" fn(*mut c_void) -> f64
        );
        let bitmap = api!(
            "FPDFBitmap_CreateEx",
            unsafe extern "system" fn(c_int, c_int, c_int, *mut c_void, c_int) -> *mut c_void
        );
        let close_bitmap = api!("FPDFBitmap_Destroy", unsafe extern "system" fn(*mut c_void));
        let render = api!(
            "FPDF_RenderPageBitmap",
            unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                c_int,
                c_int,
                c_int,
                c_int,
                c_int,
                c_int,
            )
        );
        struct Guard {
            handle: *mut c_void,
            close: unsafe extern "system" fn(*mut c_void),
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe {
                    (self.close)(self.handle);
                }
            }
        }
        struct Runtime(unsafe extern "system" fn());
        impl Drop for Runtime {
            fn drop(&mut self) {
                unsafe {
                    (self.0)();
                }
            }
        }
        init();
        let _runtime = Runtime(destroy);
        let document = load(bytes.as_ptr().cast(), bytes.len(), std::ptr::null());
        if document.is_null() {
            return Err(if last_error() == 4 {
                "Please remove PDF password protection first".into()
            } else {
                format!("Cannot open PDF (PDFium error {})", last_error())
            });
        }
        let _document = Guard {
            handle: document,
            close,
        };
        let pages = count(document);
        if pages <= 0 || pages as usize > max_pages {
            return Err(format!("PDF page count {pages} is outside 1..={max_pages}"));
        }
        // A dedicated new directory prevents stale/partial page sets from being
        // mistaken for a completed import. The caller may remove it on failure.
        fs::create_dir(output).map_err(|e| format!("Create new PDF page directory: {e}"))?;
        let mut result = Vec::with_capacity(pages as usize);
        let mut total_pixels = 0u64;
        for i in 0..pages {
            let page = load_page(document, i);
            if page.is_null() {
                return Err(format!("Cannot load PDF page {}", i + 1));
            }
            let _page = Guard {
                handle: page,
                close: close_page,
            };
            let w = (width(page) * f64::from(MODEL_RENDER_DPI) / 72.0).ceil();
            let h = (height(page) * f64::from(MODEL_RENDER_DPI) / 72.0).ceil();
            if !w.is_finite()
                || !h.is_finite()
                || w <= 0.0
                || h <= 0.0
                || w > 20_000.0
                || h > 20_000.0
            {
                return Err("Unsupported PDF page dimensions".into());
            }
            let (w, h) = (w as u32, h as u32);
            check_dimensions(w, h)?;
            total_pixels += u64::from(w) * u64::from(h);
            if total_pixels > MAX_TOTAL_PIXELS {
                return Err(
                    "PDF document exceeds 200 megapixels; split it into smaller projects".into(),
                );
            }
            let stride = (w as usize * 3 + 3) & !3;
            let mut bgr = vec![255u8; stride * h as usize];
            let bm = bitmap(
                w as c_int,
                h as c_int,
                2,
                bgr.as_mut_ptr().cast(),
                stride as c_int,
            ); // FPDFBitmap_BGR
            if bm.is_null() {
                return Err("PDFium bitmap allocation failed".into());
            }
            let _bitmap = Guard {
                handle: bm,
                close: close_bitmap,
            };
            render(bm, page, 0, 0, w as c_int, h as c_int, 0, 0x08); // FPDF_GRAYSCALE, no extra rotation
                                                                     // Grayscale rendering has equal B,G,R; emit L8 as the Python path did.
            let gray = image::GrayImage::from_fn(w, h, |x, y| {
                image::Luma([bgr[y as usize * stride + x as usize * 3]])
            });
            let path = output.join(format!("page_{:03}.png", i + 1));
            gray.save(&path).map_err(|e| e.to_string())?;
            result.push(RenderedPage {
                path,
                width: w,
                height: h,
                page: i as usize + 1,
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_and_crop() {
        assert_eq!(round_even(2.5), 2.0);
        assert_eq!(round_even(3.5), 4.0);
        let page = RgbImage::from_pixel(100, 100, Rgb([255; 3]));
        let crop = crop_measure(&page, [10.0, 10.0, 20.0, 20.0]).unwrap();
        assert_eq!(crop.dimensions(), (41, 58));
        assert!(crop_measure(&page, [500.0, 0.0, 1.0, 1.0]).is_err());
    }
}
