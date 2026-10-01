//! Checked-in outputs generated from scorelib/python/scorelib/m2.py and scorelib/python/scorelib/constraints.py.
//! These tests need no Python interpreter, model, server or network.
use scorelib::score::*;
use serde_json::{json, Value};

#[test]
fn python_golden_parse_format_and_validation() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/score-golden.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let target = case["target"].as_str().unwrap();
        let mode = case["mode"].as_str().unwrap();
        let tuning: Option<Vec<i64>> = case["tuning"]
            .as_array()
            .map(|t| t.iter().map(|p| p.as_i64().unwrap()).collect());
        let string_count = case["string_count"].as_u64().map(|n| n as usize);
        let parsed =
            parse_measure_target(target).unwrap_or_else(|error| panic!("{target}: {error}"));
        assert_eq!(parsed, case["parsed"], "parse {target}");
        let (validated, errors) =
            validate_measure_target(target, mode, tuning.as_deref(), string_count);
        assert_eq!(validated.as_ref(), Some(&parsed));
        assert_eq!(json!(errors), case["errors"], "validate {target}");
        for format in case["formats"].as_array().unwrap() {
            let result = format_measure_target(
                &parsed,
                format["mode"].as_str().unwrap(),
                format["preserve_playback"].as_bool().unwrap(),
            );
            assert_eq!(
                result.as_deref(),
                Ok(format["target"].as_str().unwrap()),
                "format {target}: {format}"
            );
        }
    }
    for target in fixture["invalid"].as_array().unwrap() {
        assert!(
            parse_measure_target(target.as_str().unwrap()).is_err(),
            "invalid {target}"
        );
    }
}

#[test]
fn exact_rational_duration_golden() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/score-golden.json")).unwrap();
    for case in fixture["durations"].as_array().unwrap() {
        let duration = parse_duration_token(case["token"].as_str().unwrap()).unwrap();
        let ticks = duration_ticks(&duration).unwrap();
        assert_eq!(ticks.numerator, case["numerator"].as_i64().unwrap() as i128);
        assert_eq!(
            ticks.denominator,
            case["denominator"].as_i64().unwrap() as i128
        );
    }
    assert_eq!(
        duration_ticks(&json!({"value": 4})).unwrap(),
        Rational::new(960, 1).unwrap()
    );
    assert!(duration_ticks(&json!({"value":0})).is_err());
    assert!(duration_ticks(&json!({"value":1,"tuplet_enters":0})).is_err());
    assert!(duration_ticks(
        &json!({"value":i64::MAX,"tuplet_enters":i64::MAX,"double_dotted":true})
    )
    .is_err());
}

#[test]
fn rejects_unsupported_or_lossy_inputs() {
    for text in [
        "M2 section=%FF | V0{@0:q:r}",
        "M2 | V0{@0:q:}",
        "M2 | V0{@0:q:p60,}",
        "M2 | V0{@0:q:,p60}",
        "M2 | V0{@0:q:p60(foo[)}",
        "M2 | V0{@0:q:p60(ghost,)}",
        "M2 | V0{@0:q:p60<,fade>}",
        "M2 | V0{@9223372036854775808:q:p60}",
    ] {
        assert!(parse_measure_target(text).is_err(), "{text}");
    }
    let original = parse_measure_target("M2 | V0{@0:w:p60}").unwrap();
    let mut measure = original.clone();
    measure["unknown"] = json!("must not disappear");
    assert!(format_measure_target(&measure, "notation", true).is_err());
    measure = original.clone();
    measure["voices"][0]["events"][0]["notes"][0]["new_note_field"] = json!(12);
    assert!(format_measure_target(&measure, "notation", true).is_err());
    measure = original.clone();
    measure["voices"][0]["events"][0]["effects"] = json!(["text:a,b"]);
    assert!(format_measure_target(&measure, "notation", true).is_err());
    measure = original.clone();
    measure["voices"][0]["events"][0]["notes"][0]["effects"] = json!(["vel:60"]);
    assert!(format_measure_target(&measure, "notation", true).is_err());
    measure = original;
    measure["voices"][0]["events"][0]["status"] = json!("rest");
    assert!(format_measure_target(&measure, "notation", true).is_err());
}

#[test]
fn duplicate_payload_clones_every_playback_field_and_resets_per_voice() {
    let parsed = parse_measure_target(
        "M2 | V0{@0:q:p36(vel:0,accswap)<dyn:127,ottava:12> @960:e[7:4]:^} || V1{@0:w:p42}",
    )
    .unwrap();
    let mut expected = parsed["voices"][0]["events"][0].clone();
    expected["start"] = json!(960);
    expected["duration"] = parse_duration_token("e[7:4]").unwrap();
    assert_eq!(parsed["voices"][0]["events"][1], expected);
    assert!(parse_measure_target("M2 | V0{@0:w:p60} || V1{@0:w:^}").is_err());
}

#[test]
fn percussion_and_unusual_voice_slots_are_not_guitar_coerced() {
    let target = "M2 time=42/8 | V15{@0:q[7:4]:p36(vel:100),p42(vel:0) @549:w:p49(tie)}";
    let (measure, errors) =
        validate_measure_target(target, "notation", Some(&[64, 59, 55, 50, 45, 40]), None);
    assert!(errors.is_empty());
    let output = format_measure_target(&measure.unwrap(), "notation", true).unwrap();
    assert!(output.contains("p42(vel:0),p36(vel:100)"));
    assert!(!output.contains("s1f"));
    assert!(output.contains("V15{"));
}

#[test]
fn readable_headers_do_not_change_annotations() {
    let original = "  M2 | V0{@0:w:p60<text:M2>}\n\tC2 | V0{}\nM20 | nope\nM2\n";
    let display = "  MEASURE | V0{@0:w:p60<text:M2>}\n\tCONTEXT | V0{}\nM20 | nope\nMEASURE\n";
    assert_eq!(display_score_text(original), display);
    assert_eq!(model_score_text(display), original);
    assert_eq!(
        display_error("Invalid M2; C2 context; M20; XM2"),
        "Invalid MEASURE; CONTEXT context; M20; XM2"
    );
}

#[test]
fn python_score_document_golden() {
    let fixtures: Value =
        serde_json::from_str(include_str!("fixtures/score-document-golden.json")).unwrap();
    for fixture in fixtures.as_array().unwrap() {
        let source = fixture["result"].clone();
        assert_eq!(score_document(&source).unwrap(), fixture["score"]);
        assert_eq!(
            source, fixture["result"],
            "projection must not mutate original records"
        );
    }
}

#[test]
fn score_document_rejects_conflicts_and_unbounded_timeline() {
    let baseline = json!({"mode":"notation", "records":[
        {"measure_number":1,"target":"M2 | V0{@0:w:p60}","bar_index":0},
        {"measure_number":2,"target":"M2 | V0{@0:w:p62}","bar_index":1}
    ]});
    let mut result = baseline.clone();
    result["records"][1]["bar_index"] = json!(0);
    assert!(score_document(&result)
        .unwrap_err()
        .contains("Duplicate bar"));
    result = baseline.clone();
    result["records"][1]["instrument"] = json!("drums");
    assert!(score_document(&result)
        .unwrap_err()
        .contains("Conflicting instrument"));
    result = baseline.clone();
    result["records"][1]["tuning"] = json!([64, 59, 55]);
    assert!(score_document(&result)
        .unwrap_err()
        .contains("Conflicting instrument or tuning"));
    result = baseline.clone();
    result["records"][0]["bar_index"] = json!(i64::MAX);
    assert!(score_document(&result).unwrap_err().contains("bar index"));
    result = baseline;
    result["records"][0]["instrument"] = json!("flute");
    assert!(score_document(&result)
        .unwrap_err()
        .contains("Unsupported instrument"));
}
