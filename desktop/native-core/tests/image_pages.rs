use guitarocr_native_core::image_boundary::{for_each_image_page, open_rgb};
use std::path::PathBuf;
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
#[test]
fn tiff_frames_apply_orientation_and_never_silently_drop_pages() {
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/image_pages.json")).unwrap();
    let mut index = 0;
    let count = for_each_image_page(&fixture("pages_oriented.tiff"), 100, |image| {
        let reference = &expected[index];
        assert_eq!(
            (image.width(), image.height()),
            (
                reference["size"][0].as_u64().unwrap() as u32,
                reference["size"][1].as_u64().unwrap() as u32
            )
        );
        let pixels: Vec<u8> = reference["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(image.as_raw(), &pixels);
        index += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 2);
    assert_eq!(index, 2);
    assert!(open_rgb(&fixture("pages_oriented.tiff"))
        .unwrap_err()
        .contains("more than 1 pages"));
    let mut calls = 0;
    assert!(
        for_each_image_page(&fixture("pages_oriented.tiff"), 1, |_| {
            calls += 1;
            Ok(())
        })
        .is_err()
    );
    assert_eq!(calls, 0);
}
#[test]
fn jpeg_applies_exif_orientation() {
    let image = open_rgb(&fixture("oriented.jpg")).unwrap();
    assert_eq!(image.dimensions(), (2, 3));
}

#[test]
fn malformed_tiff_page_chain_is_rejected_before_decode() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("cycle.tiff");
    // Little-endian classic TIFF: empty IFD whose next pointer points to itself.
    std::fs::write(&file, [b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0]).unwrap();
    let error =
        for_each_image_page(&file, 100, |_| panic!("must not decode cyclic pages")).unwrap_err();
    assert!(error.contains("Cyclic"), "{error}");
}
