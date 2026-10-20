use super::*;

#[test]
fn export_retains_known_chord_tones_without_completing_the_observation() {
    let source = json!({"instrument":"guitar","parts":[],"document_metadata":{},
        "records":[{"source_id":"score","part_id":"guitar","staff_id":"tab",
            "bar_index":0,"measure_number":1,"mode":"tab","instrument":"guitar",
            "capo":0,"tuning":[64,59,55,50,45,40],"tuning_source":"manual",
            "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
                "voices":[{"voice":0,"events":[{"start":960,"duration":"h","status":"normal",
                    "notes":[{"string":1,"fret":5},{"string":2,"fret":null}],
                    "effects":["chord:F%230","chord:hammer"]}]}],
                "relations":[]}}]});
    let original = source.clone();
    let (document, summary, warnings) = export_score(&source).unwrap();
    assert_eq!(source, original);
    assert_eq!(summary["partial_measures"], 1);
    assert_eq!(summary["omitted_notes"], 1);
    let measure = &document["parts"][0]["staves"][0]["measures"][0];
    let event = measure["voices"][0]["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["status"] == "normal")
        .unwrap();
    assert_eq!(event["notes"].as_array().unwrap().len(), 1);
    assert_eq!(event["notes"][0]["fret"], 5);
    assert_eq!(event["start"], 960);
    assert!(event["effects"]
        .as_array()
        .unwrap()
        .iter()
        .any(|effect| effect.as_str().unwrap().starts_with("text:")));
    assert!(!warnings.is_empty());
    let output = tempfile::tempdir().unwrap();
    let gp5 =
        scorelib::music_exports::write_gp5(&document, &output.path().join("partial.gp5")).unwrap();
    assert_eq!(gp5["report"]["readback_verified"], true);
    scorelib::music_exports::write_musicxml(&document, &output.path().join("partial.musicxml"))
        .unwrap();
}

#[test]
fn export_does_not_complete_a_duplicate_unknown_string_from_a_notation_tie() {
    // Spring, bar 135: two visible TAB notes and a spurious unreadable
    // third item on string 1 must not become two simultaneous 1:14 ties.
    let source = json!({"instrument":"guitar","parts":[],"document_metadata":{},
        "records":[{"source_id":"score","part_id":"guitar","staff_id":"both",
            "bar_index":0,"measure_number":1,"mode":"both","instrument":"guitar",
            "capo":0,"tuning":[64,59,55,50,45,40],"tuning_source":"manual",
            "pitch_context":{"instrument_transpose":-12,"instrument_transpose_base":-12},
            "observation":{"state":{"time":{"status":"observed","value":"6/8"},
                "key":{"status":"observed","value":0},
                "clef":{"status":"observed","value":"G2"},
                "clef_octave":{"status":"observed","value":0}},
                "voices":[{"voice":0,"events":[
                    {"start":0,"duration":"e","status":"normal",
                     "notes":[{"string":1,"fret":14},{"string":3,"fret":11}],
                     "notation_notes":[{"staff_position":15,"accidental":1},
                         {"staff_position":8,"accidental":1},{"staff_position":8,"accidental":null}]},
                    {"start":480,"duration":"q","status":"normal",
                     "notes":[{"string":1,"fret":14,"effects":["sod"]},
                         {"string":3,"fret":11},{"string":1,"effects":["trem:32"]}],
                     "notation_notes":[{"staff_position":15,"accidental":null},
                         {"staff_position":8,"accidental":null}]}]}],
                "relations":[{"kind":"tie","start":[0,0,0],"end":[0,1,0],"representation":"notation"},
                    {"kind":"tie","start":[0,0,1],"end":[0,1,3],"representation":"notation"}]}}]});
    let original = source.clone();
    let (document, summary, warnings) = export_score(&source).unwrap();
    assert_eq!(source, original);
    assert_eq!(summary["omitted_notes"], 1);
    assert!(warnings.iter().any(|warning| warning.contains("延音")));
    let events = document["parts"][0]["staves"][0]["measures"][0]["voices"][0]["events"]
        .as_array()
        .unwrap();
    let known: Vec<_> = events
        .iter()
        .filter(|event| event["status"] == "normal")
        .collect();
    assert_eq!(known.len(), 2);
    assert_eq!(known[1]["start"], 480);
    assert_eq!(known[1]["notes"].as_array().unwrap().len(), 2);
    assert_eq!(known[1]["notes"][0]["fret"], 14);
    assert!(known[1]["notes"][0]["effects"]
        .as_array()
        .unwrap()
        .contains(&json!("tie")));
    let output = tempfile::tempdir().unwrap();
    let report =
        scorelib::music_exports::write_gp5(&document, &output.path().join("spring.gp5")).unwrap();
    assert_eq!(report["report"]["readback_verified"], true);
    scorelib::music_exports::write_musicxml(&document, &output.path().join("spring.musicxml"))
        .unwrap();
}

#[test]
fn export_preserves_polyphonic_attacks_with_explicitly_estimated_durations() {
    // Native TAB can share one visible stem between different source
    // voices. Keep the unreadable lower durations unknown in the project.
    let source = json!({"instrument":"guitar","parts":[],"document_metadata":{},
        "records":[{"source_id":"score","part_id":"guitar","staff_id":"tab",
            "bar_index":0,"measure_number":1,"mode":"tab","instrument":"guitar",
            "capo":0,"tuning":[64,59,55,50,45,40],"tuning_source":"manual",
            "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
                "voices":[{"voice":0,"events":[
                    {"start":0,"duration":"e[3:2]","status":"normal","notes":[{"string":4,"fret":4}]},
                    {"start":320,"duration":"e[3:2]","status":"normal","notes":[{"string":4,"fret":9}]},
                    {"start":640,"duration":"e[3:2]","status":"rest","notes":[]},
                    {"start":960,"duration":"q.","status":"normal","notes":[{"string":4,"fret":5}]},
                    {"start":2400,"duration":"e","status":"rest","notes":[]},
                    {"start":2880,"duration":"q","status":"normal","notes":[{"string":4,"fret":4}]}]},
                    {"voice":1,"events":[
                    {"start":0,"duration":null,"status":"normal","notes":[{"string":6,"fret":2}]},
                    {"start":960,"duration":null,"status":"normal","notes":[{"string":6,"fret":10}]},
                    {"start":null,"duration":null,"status":"normal","notes":[{"string":6,"fret":2}]},
                    {"start":2880,"duration":null,"status":"normal","notes":[{"string":6,"fret":4}]}]}],
                "relations":[]}}]});
    let original = source.clone();
    let (document, summary, warnings) = export_score(&source).unwrap();
    assert_eq!(source, original);
    assert_eq!(summary["estimated_durations"], 3);
    assert_eq!(summary["omitted_events"], 1);
    assert_eq!(summary["omitted_notes"], 1);
    assert!(warnings.iter().any(|warning| warning.contains("临时")));
    let lower = &document["parts"][0]["staves"][0]["measures"][0]["voices"][1]["events"];
    assert_eq!(
        lower
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["start"].clone())
            .collect::<Vec<_>>(),
        vec![json!(0), json!(960), json!(2880)]
    );
    assert_eq!(lower[1]["notes"][0]["fret"], 10);
    assert_eq!(lower[2]["duration"]["value"], 4);
    assert!(lower
        .as_array()
        .unwrap()
        .iter()
        .all(|e| e.get("export_estimated_end").is_none()));
    let output = tempfile::tempdir().unwrap();
    let report =
        scorelib::music_exports::write_gp5(&document, &output.path().join("polyphonic.gp5"))
            .unwrap();
    assert_eq!(report["report"]["readback_verified"], true);
    let xml =
        String::from_utf8(scorelib::music_exports::score_musicxml(&document).unwrap()).unwrap();
    assert_eq!(xml.matches("<pitch>").count(), 7);
    assert!(xml.contains("<voice>2</voice>"));

    // A nonstandard gap uses the existing tied fragments. An unknown
    // rest bounds the estimate but contributes no invented note; without
    // a later boundary, an unknown duration beyond the bar stays omitted.
    let record = json!({"mode":"tab","score_state":{"time":"4/4"},
        "resolved":{"voices":[{"voice":0,"events":[
            {"start":0,"duration":null,"status":"normal","notes":[{"string":1,"fret":5},{"string":2,"fret":"x"}]},
            {"start":600,"duration":null,"status":"rest","notes":[]},
            {"start":4320,"duration":null,"status":"normal","notes":[{"string":1,"fret":9}]}]}]}});
    let partial = scorelib::observation::export_measure(&record).unwrap();
    assert_eq!(partial["omitted_events"], 2);
    assert_eq!(partial["omitted_notes"], 1);
    let projected = scorelib::observation::export_rhythm(&partial["measure"], "4/4").unwrap();
    assert_eq!(projected["estimated_durations"], 1);
    assert_eq!(projected["split_events"], 1);
    let events = &projected["measure"]["voices"][0]["events"];
    assert_eq!(events[1]["start"], 480);
    assert_eq!(events[1]["notes"].as_array().unwrap().len(), 1);
    assert_eq!(events[1]["notes"][0]["effects"], json!(["tie"]));
    assert_eq!(events[2]["start"], 600);
    assert_eq!(events[2]["status"], "rest");
}

#[test]
fn partial_metadata_edit_preserves_existing_values_and_unknown_tempo() {
    let previous = json!({"title":"Before","artist":"Composer","instrument":"guitar",
        "tuning_used":[64,59,55,50,45,40],"tuning_source":"manual","capo":3,
        "transpose":-12,"document_metadata":{"tempo_quarter":null},
        "measure_pitch_contexts":[{"measure_number":1,"pitch_context":{"capo":3}}]});
    let layout = json!({"records":[]});
    let edited = edited_single(&layout, &previous, &json!({"title":"After"})).unwrap();
    assert_eq!(edited["title"], "After");
    for key in [
        "artist",
        "instrument",
        "tuning_used",
        "capo",
        "transpose",
        "measure_pitch_contexts",
    ] {
        assert_eq!(edited[key], previous[key], "{key}");
    }
    assert!(edited["document_metadata"]["tempo_quarter"].is_null());
    let mut previous = previous.clone();
    previous["document_metadata"]["tempo_quarter"] = json!(96);
    let edited = edited_single(&layout, &previous, &json!({"title":"After"})).unwrap();
    assert_eq!(edited["document_metadata"]["tempo_quarter"], 96);
    let edited = edited_single(&layout, &previous, &json!({"tempo_quarter":null})).unwrap();
    assert!(edited["document_metadata"]["tempo_quarter"].is_null());
    assert_eq!(edited["document_metadata"]["tempo_source"], "manual");
    let edited = edited_single(&layout, &previous, &json!({"tempo_quarter":476})).unwrap();
    assert_eq!(edited["document_metadata"]["tempo_quarter"], 476);
    assert_eq!(edited["document_metadata"]["tempo_source"], "manual");
    assert!(edited_single(&layout, &json!({}), &json!({"title":"Untitled"})).is_err());
    assert!(edited_single(&layout, &json!({}), &json!({"instrument":"guitar"})).is_err());
}

#[test]
fn rereading_a_staff_keeps_its_confirmed_instrument() {
    let previous = json!({"parts":[{"id":"part-1","instrument":"pitched",
        "midi_program":42,"capo":0,"transpose":null,
        "document_metadata":{"instrument_source":"manual"}}],
        "measure_profiles":[{"part_id":"part-1","measure_number":1,"mode":"notation"}]});
    let detected = json!({"parts":[{"id":"part-1","name":"","instrument":null,
        "document_metadata":{"instrument_needs_review":true}}],
        "measure_profiles":[{"part_id":"part-1","measure_number":1,"mode":"notation"}],
        "measure_pitch_contexts":[],"document_metadata":{}});
    let layout = json!({"records":[]});
    let mut current = detected.clone();
    preserve_manual_instruments(&layout, &previous, &mut current).unwrap();
    assert_eq!(current["instrument"], "pitched");
    assert_eq!(current["parts"][0]["midi_program"], 42);
    assert_eq!(current["measure_profiles"][0]["instrument"], "pitched");
    assert_eq!(current["document_metadata"]["instrument_source"], "manual");
    let mut regrouped = detected.clone();
    regrouped["measure_profiles"][0]["measure_number"] = json!(2);
    preserve_manual_instruments(&layout, &previous, &mut regrouped).unwrap();
    assert!(regrouped["parts"][0]["instrument"].is_null());
}

#[test]
fn incomplete_measure_survives_save_and_becomes_exportable_after_edit() {
    let output = tempfile::tempdir().unwrap();
    let mut source = json!({"mode":"tab","instrument":"guitar","tuning_used":[64,59,55,50,45,40],
        "records":[{"source_id":"score","part_id":"guitar","staff_id":"tab","bar_index":0,
            "measure_number":1,"mode":"tab","instrument":"guitar","capo":0,"tuning":[64,59,55,50,45,40],
            "observation":{"state":{"time":{"status":"observed","value":"4/4"}},
                "voices":[{"voice":0,"events":[{"start":0,"duration":null,"status":"normal",
                    "notes":[{"string":1,"fret":5}]}]}],"relations":[]}}]});
    crate::recognition::reassemble(&mut source).unwrap();
    let manifest = save_recognition(output.path(), &source).unwrap();
    let mut saved: Value = serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
    assert!(saved["records"][0]["target"].is_null());
    assert!(saved["score_text"].is_null());
    assert!(saved["musicxml"].is_null());
    let document = score_document(&saved).unwrap();
    let measure = &document["parts"][0]["staves"][0]["measures"][0];
    assert_eq!(measure["incomplete"], true);
    assert_eq!(measure["voices"], json!([]));
    assert_eq!(
        measure["observation"]["voices"][0]["events"][0]["notes"][0]["fret"],
        5
    );
    assert!(scorelib::music_exports::score_document(&document).is_err());
    assert!(correct_record(&mut saved, 1, &json!({"reviewed":true})).is_err());

    let mut observed = saved["records"][0]["observation"].clone();
    observed["voices"][0]["events"][0]["duration"] = json!("w");
    correct_record(&mut saved, 1, &json!({"observation":observed})).unwrap();
    crate::recognition::reassemble(&mut saved).unwrap();
    assert!(saved["records"][0]["target"]
        .as_str()
        .unwrap()
        .contains("@0:w:s1f5"));
    assert_eq!(saved["records"][0]["needs_review"], false);
    let manifest = save_recognition(output.path(), &saved).unwrap();
    let saved: Value = serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
    assert!(saved["musicxml"].is_string());
    assert!(saved["score_text"].is_string());
}
