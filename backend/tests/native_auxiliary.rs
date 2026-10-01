//! Opt-in native library smoke checks. Never invoke Python or download assets.
use guitarocr_backend::auxiliary_onnx::{initialize_runtime, SignatureReader};
use guitarocr_backend::image_boundary::render_pdf_pages;
use std::path::PathBuf;

#[test]
#[ignore = "requires verified local ONNX Runtime and signature model paths"]
fn signature_native_smoke() {
    let runtime =
        PathBuf::from(std::env::var_os("GUITAROCR_TEST_ORT").expect("GUITAROCR_TEST_ORT"));
    let model = PathBuf::from(
        std::env::var_os("GUITAROCR_TEST_SIGNATURE").expect("GUITAROCR_TEST_SIGNATURE"),
    );
    initialize_runtime(&runtime).unwrap();
    let mut reader = SignatureReader::new(&model).unwrap();
    let image = image::RgbImage::from_pixel(160, 80, image::Rgb([255; 3]));
    let output = reader.predict(&[image]).unwrap();
    assert_eq!(output.len(), 1);
    assert!(output[0]["key"].is_null() || (-7..=7).contains(&output[0]["key"].as_i64().unwrap()));
    assert!(output[0]["time"].is_null() || output[0]["time"].as_str().unwrap().contains('/'));
}

fn blank_pdf() -> Vec<u8> {
    let mut data = b"%PDF-1.4\n".to_vec();
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 144] /Resources << >> >>",
    ];
    let mut offsets = vec![0usize];
    for (i, object) in objects.iter().enumerate() {
        offsets.push(data.len());
        data.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
    }
    let xref = data.len();
    data.extend_from_slice(b"xref\n0 4\n0000000000 65535 f \n");
    for offset in offsets.iter().skip(1) {
        data.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    data.extend_from_slice(
        format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    data
}

#[test]
#[ignore = "requires a verified local PDFium shared library"]
fn pdfium_native_smoke() {
    let library =
        PathBuf::from(std::env::var_os("GUITAROCR_TEST_PDFIUM").expect("GUITAROCR_TEST_PDFIUM"));
    let dir = tempfile::tempdir().unwrap();
    let pdf = dir.path().join("test.pdf");
    std::fs::write(&pdf, blank_pdf()).unwrap();
    let pages = render_pdf_pages(&library, &pdf, &dir.path().join("pages"), 10).unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!((pages[0].width, pages[0].height), (180, 360));
    let image = image::open(&pages[0].path).unwrap().into_luma8();
    assert!(image.pixels().all(|pixel| pixel.0 == [255]));
    assert!(render_pdf_pages(&library, &pdf, &dir.path().join("pages"), 10).is_err());
}
