use scorelib::{information, pitch, score};
use serde_json::json;

#[test]
fn pitch_conversion_keeps_the_written_source_and_transposes_ornaments_once() {
    let target = "M key=CMajor | V0{@0:q:p60(grace:p59:16:none:0:1,trill:p62:16)<ottava:12> @960:q:p0(dead)}";
    let context = json!({"instrument_transpose":-2,"clef_octave":-12,"capo":3});
    let sounding = pitch::convert_pitch(target, &context, "notation", false).unwrap();
    let measure = score::parse_measure_target(&sounding).unwrap();
    assert_eq!(measure["key_signature"], "BMajorFlat");
    let events = &measure["voices"][0]["events"];
    assert_eq!(events[0]["notes"][0]["pitch"], 55);
    assert_eq!(
        events[0]["notes"][0]["effects"],
        json!(["grace:p54:16:none:0:1", "trill:p57:16"])
    );
    assert_eq!(events[1]["notes"][0]["pitch"], 0);
    let written = pitch::convert_pitch(&sounding, &context, "notation", true).unwrap();
    assert_eq!(
        score::parse_measure_target(&written).unwrap(),
        score::parse_measure_target(target).unwrap()
    );
}

#[test]
fn partial_tab_positions_survive_without_inventing_a_notation_pitch() {
    let mut measure = score::parse_measure_target("M | V0{@0:q:s1f3}").unwrap();
    let original = measure.clone();
    pitch::convert_measure_pitch(&mut measure, &json!({"instrument_transpose":-12}), 1).unwrap();
    assert_eq!(measure, original);
    assert!(pitch::convert_pitch(
        "M | V0{@0:q:p1}",
        &json!({"instrument_transpose":-12}),
        "notation",
        false
    )
    .is_err());
}

#[test]
fn key_names_accept_gp_and_display_spellings_without_guessing_unknown_keys() {
    for name in ["BMajorFlat", "BFlatMajor", "B♭Major", "BbMajor", "-2:major"] {
        assert_eq!(pitch::key_parts(name).unwrap(), (-2, false));
        assert_eq!(pitch::transpose_key(name, 2).unwrap(), "CMajor");
    }
    assert_eq!(pitch::transpose_key("CMinor", 2).unwrap(), "DMinor");
    assert_eq!(pitch::transpose_fifths(-8, 0), 4);
    assert_eq!(pitch::transpose_fifths(8, 0), -4);
    assert!(pitch::key_parts("unknown").is_err());
}

#[test]
fn information_interprets_visible_pitch_instructions_and_preserves_other_text() {
    for (kind, text, field, value) in [
        ("instrument", "Bass clarinet in B♭", "semitones", -14),
        ("ottava", "8va bassa", "semitones", -12),
        ("ottava", "15ma alta", "semitones", 24),
        ("capo", "Capo at fret IV", "capo", 4),
        ("capo", "变调夹 24 品", "capo", 24),
    ] {
        let raw = json!({"kind":kind,"text":text}).to_string();
        let parsed = information::parse_response(&raw, "annotation");
        assert_eq!(parsed[field], value, "{text}");
        assert_eq!(parsed["text"], text);
    }
    let unknown = information::parse_response(r#"{"kind":"capo","text":"Capo XXV"}"#, "annotation");
    assert!(unknown["capo"].is_null());
    assert_eq!(unknown["text"], "Capo XXV");
    let chord = information::parse_response(
        r#"{"kind":"chord","text":"B♭","semitones":-2}"#,
        "annotation",
    );
    assert_eq!(chord["kind"], "chord");
    assert!(chord["semitones"].is_null());
    assert_eq!(
        information::parse_response("```json\n{\"text\":\"  标题  \"}\n```", "title"),
        json!({"title":"标题"})
    );
}
