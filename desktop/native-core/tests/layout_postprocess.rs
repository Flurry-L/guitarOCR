use guitarocr_native_core::layout_postprocess::process_page;
use image::{Rgb, RgbImage};
use serde_json::Value;
#[test]
fn matches_python_reference_geometric_cases() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/layout_postprocess.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let w = case["size"][0].as_u64().unwrap() as u32;
        let h = case["size"][1].as_u64().unwrap() as u32;
        let mut page = RgbImage::from_pixel(w, h, Rgb([255; 3]));
        for line in case["lines"].as_array().unwrap() {
            let p: Vec<u32> = line
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
        let actual = process_page(&page, case["boxes"].as_array().unwrap(), 0.25).unwrap();
        // Normalize JSON integer/float representation before comparison.
        fn close(a: &Value, b: &Value) -> bool {
            match (a, b) {
                (Value::Number(x), Value::Number(y)) => {
                    (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-9
                }
                (Value::Array(x), Value::Array(y)) => {
                    x.len() == y.len() && x.iter().zip(y).all(|(x, y)| close(x, y))
                }
                (Value::Object(x), Value::Object(y)) => {
                    x.len() == y.len()
                        && x.iter()
                            .all(|(k, v)| y.get(k).map(|o| close(v, o)).unwrap_or(false))
                }
                _ => a == b,
            }
        }
        assert!(
            close(&actual, &case["expected"]),
            "{}\nactual: {}\nexpected: {}",
            case["name"],
            actual,
            case["expected"]
        );
    }
}

#[test]
fn manual_rows_keep_editor_order_and_fields() {
    use serde_json::json;
    let mut rows = vec![
        json!({"page":1,"bbox":[110,80,100,30],"measure_number":1}),
        json!({"page":1,"bbox":[110,10,100,30],"measure_number":2}),
        json!({"page":1,"bbox":[0,10,100,30],"measure_number":3}),
        json!({"page":2,"bbox":[0,10,100,30],"measure_number":4}),
    ];
    guitarocr_native_core::layout_postprocess::assign_manual_rows(&mut rows).unwrap();
    assert_eq!(
        rows.iter()
            .map(|r| r["measure_number"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["row_index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 0, 0, 0]
    );
}
