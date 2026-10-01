use scorelib::music_exports::{score_musicxml, write_gp5};
use scorelib::score::score_document;
use serde_json::{json, Value};
fn document(targets: &[&str], instrument: &str, tuning: Value) -> Value {
    score_document(&json!({"title":"原创 🎸","artist":"测试","instrument":instrument,"mode":if instrument=="guitar"{"both"}else{"notation"},"tuning_used":tuning,"records":targets.iter().enumerate().map(|(i,t)|json!({"measure_number":i+1,"target":t})).collect::<Vec<_>>()})).unwrap()
}
#[test]
fn xml_multivoice_ties_and_escaping() {
    let mut score = document(
        &[
            "M2 time=4/4 | V0{@0:w:s1f3} || V1{@0:w:s7f0}",
            "M2 time=4/4 | V0{@0:w:s1f3(tie)} || V1{@0:w:r}",
        ],
        "guitar",
        json!([64, 59, 55, 50, 45, 40, 35]),
    );
    score["title"] = json!("A&B <song>");
    let xml = String::from_utf8(score_musicxml(&score).unwrap()).unwrap();
    assert!(xml.contains("A&amp;B &lt;song&gt;"));
    assert!(xml.contains("<tie type=\"start\"/>"));
    assert!(xml.contains("<tie type=\"stop\"/>"));
    assert!(xml.contains("<backup><duration>3840</duration></backup>"));
    assert!(xml.contains("<string>7</string>"));
}
#[test]
fn xml_tuplet_ticks_follow_legacy_projection() {
    let score = document(
        &["M2 time=4/4 | V0{@0:q[7:4]:s1f3}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let xml = String::from_utf8(score_musicxml(&score).unwrap()).unwrap();
    assert!(xml.contains("<divisions>960</divisions>"));
    assert!(xml.contains("<duration>548</duration>"));
    assert!(xml.contains("<actual-notes>7</actual-notes>"));
}
#[test]
fn gp5_splits_strings_and_voices_and_reports_encoding() {
    let score = document(
        &[
            "M2 time=4/4 | V0{@0:h:s1f3,s8f2 @1920:h:r} || V1{@0:w:s7f0} || V2{@0:w:s2f1}",
            "M2 time=4/4 | V0{@0:w:s1f3(tie)} || V1{@0:w:s7f0(tie)} || V2{@0:w:r}",
        ],
        "guitar",
        json!([64, 59, 55, 50, 45, 40, 35, 30]),
    );
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("score.gp5");
    let report = write_gp5(&score, &path).unwrap();
    assert_eq!(report["report"]["tracks"], 4);
    assert_eq!(report["report"]["readback_verified"], true);
    assert_eq!(report["report"]["replacements"][0]["written"], "原创 ?");
    assert!(path.is_file());
    assert!(path.with_file_name("score.gp5.encoding.json").is_file());
    assert_eq!(
        &std::fs::read(path).unwrap()[1..25],
        b"FICHIER GUITAR PRO v5.10"
    );
}
#[test]
fn gp5_failure_does_not_replace_existing_file() {
    let score = document(
        &["M2 time=4/4 | V0{@0:q:s1f3<bizarre>}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("score.gp5");
    std::fs::write(&path, b"prior").unwrap();
    assert!(write_gp5(&score, &path).is_err());
    assert_eq!(std::fs::read(path).unwrap(), b"prior");
}
#[test]
fn gp5_pitch_only_and_drum_notes_have_native_slots() {
    for instrument in ["pitched", "drums"] {
        let score = document(
            &["M2 time=4/4 | V0{@0:h:p36,p60,p67 @1920:h:p36(tie),p72}"],
            instrument,
            json!([]),
        );
        let temp = tempfile::tempdir().unwrap();
        let report = write_gp5(&score, &temp.path().join("score.gp5")).unwrap();
        assert_eq!(report["report"]["readback_verified"], true);
    }
}
#[test]
fn gp5_excerpt_tie_is_reported() {
    let score = document(
        &["M2 time=4/4 | V0{@0:w:s1f3(tie)}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let temp = tempfile::tempdir().unwrap();
    let report = write_gp5(&score, &temp.path().join("score.gp5")).unwrap();
    assert_eq!(
        report["report"]["detached_ties"].as_array().unwrap().len(),
        1
    );
}
#[test]
fn gp5_gap_materialized_and_conflicting_fret_rejected() {
    let score = document(
        &["M2 time=4/4 | V0{@960:q:s1f3}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let temp = tempfile::tempdir().unwrap();
    assert!(write_gp5(&score, &temp.path().join("score.gp5")).is_ok());
    let score = document(
        &["M2 time=4/4 | V0{@0:w:s1f3p70}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    assert!(write_gp5(&score, &temp.path().join("bad.gp5"))
        .unwrap_err()
        .contains("conflicts"));
}
#[test]
fn xml_drums_chord_frames_and_transposition() {
    let mut score=document(&["M2 time=4/4 key=BMajorFlat | V0{@0:w:p60<chord:C%23m7%2FG%23,diagram:1:x/3/2/0/1/0:-/3/2/0/1/0:1/1/4,ottava:12>}"],"pitched",json!([]));
    score["parts"][0]["staves"][0]["measures"][0]["pitch_context"] =
        json!({"instrument_transpose":-2,"clef":"G2"});
    let xml = String::from_utf8(score_musicxml(&score).unwrap()).unwrap();
    assert!(xml.contains("<root-step>C</root-step><root-alter>1</root-alter>"));
    assert!(xml.contains("<frame-strings>6</frame-strings>"));
    assert!(xml.contains("<chromatic>-2</chromatic>"));
    assert!(xml.contains("<pitch><step>D</step><octave>4</octave></pitch>"));
    assert!(xml.contains("<octave-shift type=\"down\""));
    let score = document(&["M2 time=4/4 | V0{@0:w:p36,p42}"], "drums", json!([]));
    let xml = String::from_utf8(score_musicxml(&score).unwrap()).unwrap();
    assert!(xml.contains("<midi-unpitched>37</midi-unpitched>"));
    assert!(xml.contains("<unpitched>"));
    assert!(xml.contains("<sign>percussion</sign>"));
}
#[test]
fn gp5_all_oracle_fixtures_verify_without_python() {
    for s in [
        include_str!("fixtures/music-exports/chord.json"),
        include_str!("fixtures/music-exports/drums.json"),
        include_str!("fixtures/music-exports/piano.json"),
        include_str!("fixtures/music-exports/split.json"),
        include_str!("fixtures/music-exports/techniques.json"),
        include_str!("fixtures/music-exports/tempo.json"),
        include_str!("fixtures/music-exports/ties.json"),
    ] {
        let score = serde_json::from_str(s).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let result = write_gp5(&score, &temp.path().join("score.gp5")).unwrap();
        assert_eq!(result["report"]["readback_verified"], true);
        assert!(score_musicxml(&score).is_ok());
    }
}

#[test]
fn gp5_velocity_quantization_is_reported() {
    let score = document(
        &["M2 time=4/4 | V0{@0:w:s1f3<dyn:88>}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let temp = tempfile::tempdir().unwrap();
    let result = write_gp5(&score, &temp.path().join("score.gp5")).unwrap();
    assert_eq!(result["report"]["velocity_quantization"][0]["original"], 88);
    assert_eq!(result["report"]["velocity_quantization"][0]["written"], 79);
}
#[test]
fn duplicate_voices_are_rejected_before_projection() {
    let mut score = document(
        &["M2 time=4/4 | V0{@0:w:s1f3}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let voices = score["parts"][0]["staves"][0]["measures"][0]["voices"]
        .as_array_mut()
        .unwrap();
    voices.push(voices[0].clone());
    assert!(score_musicxml(&score).unwrap_err().contains("Duplicate"));
}
#[test]
fn gp5_encoding_is_cp936_not_extended_gbk() {
    let mut score = document(
        &["M2 time=4/4 | V0{@0:w:s1f3}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    score["title"] = json!("中文 € ǹ \u{e000}");
    let temp = tempfile::tempdir().unwrap();
    let result = write_gp5(&score, &temp.path().join("score.gp5")).unwrap();
    assert_eq!(result["report"]["replacements"][0]["written"], "中文 ? ? ?");
}
#[test]
fn legacy_projection_boundaries_remain_exportable() {
    for source in [
        include_str!("fixtures/music-exports/overfull.json"),
        include_str!("fixtures/music-exports/tab_hidden.json"),
        include_str!("fixtures/music-exports/fractional_tuplet.json"),
        include_str!("fixtures/music-exports/double_dot.json"),
        include_str!("fixtures/music-exports/advanced.json"),
        include_str!("fixtures/music-exports/diagram.json"),
        include_str!("fixtures/music-exports/navigation.json"),
        include_str!("fixtures/music-exports/empty_only.json"),
        include_str!("fixtures/music-exports/zero_notes.json"),
    ] {
        let score = serde_json::from_str(source).unwrap();
        let dir = tempfile::tempdir().unwrap();
        assert!(write_gp5(&score, &dir.path().join("score.gp5")).is_ok());
        assert!(score_musicxml(&score).is_ok());
    }
}
#[test]
fn double_dot_keeps_legacy_projection_without_mutating_ir() {
    let score: Value =
        serde_json::from_str(include_str!("fixtures/music-exports/double_dot.json")).unwrap();
    let before = score.clone();
    let dir = tempfile::tempdir().unwrap();
    let report = write_gp5(&score, &dir.path().join("score.gp5")).unwrap();
    assert_eq!(
        report["report"]["duration_projections"][0]["written_ticks"],
        960
    );
    assert_eq!(score, before);
    let xml = String::from_utf8(score_musicxml(&score).unwrap()).unwrap();
    assert!(xml.contains("<duration>1680</duration>"));
    assert!(xml.contains("<dot/><dot/>"));
}
#[test]
fn musicxml_does_not_add_new_legacy_invisible_marks() {
    let score: Value =
        serde_json::from_str(include_str!("fixtures/music-exports/navigation.json")).unwrap();
    let xml = String::from_utf8(score_musicxml(&score).unwrap()).unwrap();
    assert!(!xml.contains("<ending"));
    assert!(!xml.contains("<swing"));
    assert!(!xml.contains("dalsegno="));
    assert!(xml.contains("<repeat direction=\"backward\""));
}
#[test]
fn fractional_gp5_following_onset_matches_legacy_boundary() {
    let score = document(
        &["M2 time=4/4 | V0{@0:q[7:4]:s1f3 @548:q:s2f1}"],
        "guitar",
        json!([64, 59, 55, 50, 45, 40]),
    );
    let dir = tempfile::tempdir().unwrap();
    assert!(write_gp5(&score, &dir.path().join("score.gp5")).is_err());
}
