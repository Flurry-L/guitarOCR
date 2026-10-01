use guitarocr_native_core::{
    image_boundary::open_rgb,
    ottava_geometry::{
        dashed_line, link_octave_continuations, link_octave_continuations_with_images,
    },
    pixel_refinement::refine_diagram_image,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pixel-geometry")
}
#[test]
fn chord_pixel_refinement_matches_python_reference() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/pixel-geometry/chords.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let image = open_rgb(&root().join(case["image"].as_str().unwrap())).unwrap();
        let actual = refine_diagram_image(&image, &case["diagram"])
            .unwrap()
            .unwrap_or(Value::Null);
        assert_eq!(actual, case["expected"], "{}", case["name"]);
    }
}
#[test]
fn dashed_line_properties_match_python_reference() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/pixel-geometry/lines.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let actual=dashed_line(Some(&root().join(case["image"].as_str().unwrap()))).unwrap().map(|line|json!({"y":line.y,"left":line.left,"right":line.right,"hook":line.hook,"bare":line.bare})).unwrap_or(Value::Null);
        assert_eq!(actual, case["expected"], "{}", case["name"]);
    }
    assert_eq!(dashed_line(None).unwrap(), None);
    assert_eq!(
        dashed_line(Some(Path::new("/nonexistent/line.png"))).unwrap(),
        None
    );
}
#[test]
fn octave_continuations_match_python_reference_without_changing_input() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/pixel-geometry/octaves.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mut predictions = case["predictions"].as_array().unwrap().clone();
        for p in &mut predictions {
            p["image"] = json!(root().join(p["image"].as_str().unwrap()).to_string_lossy());
        }
        let before = predictions.clone();
        let mut actual =
            link_octave_continuations(case["records"].as_array().unwrap(), &predictions).unwrap();
        assert_eq!(predictions, before);
        for p in &mut actual {
            p["image"] = json!(Path::new(p["image"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy());
        }
        assert_eq!(json!(actual), case["expected"], "{}", case["name"]);
    }
}
#[test]
fn native_image_resolver_preserves_same_continuation_semantics() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/pixel-geometry/octaves.json")).unwrap();
    let case = &cases[0];
    let mut predictions = case["predictions"].as_array().unwrap().clone();
    for p in &mut predictions {
        p["crop_fixture"] = p["image"].clone();
        p.as_object_mut().unwrap().remove("image");
    }
    let actual = link_octave_continuations_with_images(
        case["records"].as_array().unwrap(),
        &predictions,
        &mut |p| open_rgb(&root().join(p["crop_fixture"].as_str().unwrap())).map(Some),
    )
    .unwrap();
    assert_eq!(actual[1]["parsed"]["semitones"], 12);
    assert_eq!(actual[2]["parsed"]["semitones"], 12);
    assert_eq!(actual[2]["octave_continuation"]["page"], 1);
    assert_eq!(actual[2]["model_parsed"], predictions[2]["parsed"]);
}

#[test]
fn ineligible_diagrams_do_not_read_an_image_or_invent_positions() {
    use guitarocr_native_core::pixel_refinement::refine_diagram;
    for diagram in [
        Value::Null,
        json!({"base_fret":1,"frets":[0,0]}),
        json!({"base_fret":0,"frets":[1,1,1]}),
        json!({"base_fret":1.5,"frets":[1,1,1]}),
    ] {
        assert_eq!(
            refine_diagram(Path::new("/nonexistent/chord.png"), &diagram).unwrap(),
            None
        );
    }
}
