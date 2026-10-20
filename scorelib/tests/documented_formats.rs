use scorelib::score::{format_measure_target, parse_measure_target, validate_measure_target};

#[test]
fn documented_m_examples_parse_and_round_trip() {
    let guide = include_str!("../../docs/ir-and-m-spec.md");
    let mut count = 0;
    for block in guide.split("```text\n").skip(1) {
        let text = block.split("```").next().unwrap().trim();
        let mode = if text.contains("s1f") {
            "tab"
        } else {
            "notation"
        };
        let (measure, errors) = validate_measure_target(text, mode, None, None);
        assert!(errors.is_empty(), "{text}: {errors:?}");
        let measure = measure.unwrap();
        let formatted = format_measure_target(&measure, mode, true).unwrap();
        assert_eq!(parse_measure_target(&formatted).unwrap(), measure);
        count += 1;
    }
    assert!(count > 0, "The guide should contain runnable M examples");
}
