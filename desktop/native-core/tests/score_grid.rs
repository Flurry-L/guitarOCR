use guitarocr_native_core::score_grid::{align_system, fuse_paired_staves};
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
#[test]
fn consensus_and_paired_staffs_match_reference() {
    let cases: Value = serde_json::from_str(include_str!("fixtures/score_grid.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("page.png");
        let mut image = RgbImage::from_pixel(
            case["size"][0].as_u64().unwrap() as u32,
            case["size"][1].as_u64().unwrap() as u32,
            Rgb([255; 3]),
        );
        for line in case["lines"].as_array().unwrap() {
            let p: Vec<u32> = line
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            for y in p[1]..=p[3] {
                for x in p[0]..=p[2] {
                    image.put_pixel(x, y, Rgb([0; 3]));
                }
            }
        }
        image.save(&path).unwrap();
        let members: Vec<_> = case["members"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let mut rows = m[0].as_array().unwrap().clone();
                for r in &mut rows {
                    r["source_page"] = json!(path.to_string_lossy());
                }
                (
                    rows,
                    m[1].as_u64().unwrap() as usize,
                    m[2].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let (mut aligned, count) = align_system(&members, root.path(), 1, 0).unwrap();
        assert_eq!(count, case["columns"].as_u64().unwrap() as usize);
        if case["fuse"].as_bool().unwrap_or(false) {
            aligned = fuse_paired_staves(&aligned, root.path(), 1, 0).unwrap();
        }
        for (row, _, _) in &mut aligned {
            for (r, _) in row {
                let object = r.as_object_mut().unwrap();
                object.remove("source_page");
                let image = object.remove("image").unwrap();
                let changed = image != "OLD";
                if changed {
                    assert!(std::path::Path::new(image.as_str().unwrap()).is_file());
                }
                object.insert("image_changed".into(), json!(changed));
            }
        }
        let actual = json!(aligned);
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
                            .all(|(k, v)| y.get(k).map(|w| close(v, w)).unwrap_or(false))
                }
                _ => a == b,
            }
        }
        assert!(
            close(&actual, &case["expected"]),
            "{} actual={} expected={}",
            case["name"],
            actual,
            case["expected"]
        );
    }
}

#[test]
fn resolved_document_preserves_simultaneous_timeline_and_global_parts() {
    let root = tempfile::tempdir().unwrap();
    let mut records = vec![];
    let mut predictions = vec![];
    for page in 1..=2 {
        for row in 0..2 {
            for bar in 0..2 {
                records.push(json!({"page":page,"system_index":row,"system_measure_index":bar,"measure_number":records.len()+1,"mode":"tab","bbox":[10+bar*100,30+row*100,100,70],"score":0.9,"image":"original"}));
            }
        }
        predictions.push(json!({"page":page,"parsed":{"parts":[{"name":"Guitar","instrument":"guitar","strings":6,"program":25},{"name":"Bass","instrument":"bass","strings":4,"program":33}],"rows":[[0,0,0],[0,1,0]]}}));
    }
    let (parts, resolved) = guitarocr_native_core::score_grid::resolve_document_reconciled(
        &records,
        &mut predictions,
        root.path(),
    )
    .unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(resolved.len(), 8);
    assert_eq!(
        resolved
            .iter()
            .map(|r| r["bar_index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 0, 1, 2, 3, 2, 3]
    );
    assert_eq!(
        resolved
            .iter()
            .map(|r| r["part_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["part-1", "part-1", "part-2", "part-2", "part-1", "part-1", "part-2", "part-2"]
    );
    assert_eq!(resolved[6]["string_count"], 4);
    assert_eq!(resolved[7]["measure_number"], 8);
}
