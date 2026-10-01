use guitarocr_backend::staff_geometry::{detect_staffs, measure_boxes};
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
#[test]
fn tab_and_paired_geometry_match_python_reference() {
    let cases: Value = serde_json::from_str(include_str!("fixtures/staff_geometry.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mut page = RgbImage::from_pixel(
            case["size"][0].as_u64().unwrap() as u32,
            case["size"][1].as_u64().unwrap() as u32,
            Rgb([255; 3]),
        );
        for line in case["lines"].as_array().unwrap() {
            let p: Vec<_> = line
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            for y in p[1]..=p[3] {
                for x in p[0]..=p[2] {
                    page.put_pixel(x, y, Rgb([0; 3]));
                }
            }
        }
        let staffs = detect_staffs(&page, None, false).unwrap();
        let actual:Vec<Value>=staffs.iter().map(|s|json!({"string_y":s.string_y.iter().map(|v|*v as f64).collect::<Vec<_>>(),"spacing":s.spacing,"boundaries":s.boundaries})).collect();
        assert_eq!(json!(actual), case["staffs"], "{}", case["name"]);
        let mut actual = measure_boxes(&page, case["mode"].as_str().unwrap()).unwrap();
        for b in &mut actual {
            b.as_object_mut().unwrap().remove("mode");
            b.as_object_mut().unwrap().remove("geometry_source");
        }
        assert_eq!(json!(actual), case["boxes"], "{}", case["name"]);
    }
}

#[test]
fn marked_systems_use_continuous_barlines_not_row_count() {
    use guitarocr_backend::staff_geometry::marked_systems;
    let mut image = RgbImage::from_pixel(400, 340, Rgb([255; 3]));
    for (center, start) in [(60, 40), (160, 140), (280, 260)] {
        for y in start..=start + 40 {
            if (y - start) % 10 == 0 {
                for x in 30..=340 {
                    image.put_pixel(x, y, Rgb([0; 3]));
                }
            }
        }
        for y in center - 4..=center + 4 {
            image.put_pixel(380, y, Rgb([0, 65, 220]));
        }
    }
    for y in 40..=180 {
        image.put_pixel(30, y, Rgb([0; 3]));
    }
    assert_eq!(marked_systems(&image, 3).unwrap(), Some(vec![0, 0, 1]));
    assert_eq!(marked_systems(&image, 2).unwrap(), None);
    // Scattered missing pixels are ambiguous, rather than enough to split a system.
    for y in (61..=159).step_by(5) {
        image.put_pixel(30, y, Rgb([255; 3]));
    }
    assert_eq!(marked_systems(&image, 3).unwrap(), None);
}
