//! Python bindings use JSON only at the language boundary; Rust owns semantics.
use crate::{chords, information, music_exports, observation, pitch, score};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::Value;
fn value(text: &str) -> PyResult<Value> {
    serde_json::from_str(text).map_err(error)
}
fn error(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn result(v: Result<Value, String>) -> PyResult<String> {
    v.map(|v| v.to_string()).map_err(error)
}

#[pyfunction]
fn parse_measure_target(target: &str) -> PyResult<String> {
    result(score::parse_measure_target(target))
}
#[pyfunction]
fn information_prompt(kind: &str) -> PyResult<&'static str> {
    information::prompt(kind).map_err(error)
}
#[pyfunction]
fn information_schema(kind: &str) -> PyResult<String> {
    result(information::prompts::information_schema(kind))
}
#[pyfunction]
fn parse_information(raw: &str, kind: &str) -> String {
    information::parse_response(raw, kind).to_string()
}
#[pyfunction]
fn key_signature(name: &str) -> PyResult<(i64, bool)> {
    pitch::key_parts(name).map_err(error)
}
#[pyfunction]
fn transpose_key(name: &str, semitones: i64) -> PyResult<String> {
    pitch::transpose_key(name, semitones).map_err(error)
}
#[pyfunction]
fn convert_pitch_target(
    target: &str,
    context: &str,
    mode: &str,
    to_written: bool,
) -> PyResult<String> {
    pitch::convert_pitch(target, &value(context)?, mode, to_written).map_err(error)
}
#[pyfunction]
fn ornament_position(token: &str) -> PyResult<(Option<i64>, Option<i64>)> {
    score::ornament_position(token).map_err(error)
}
#[pyfunction]
fn duration_ticks(duration: &str) -> PyResult<(i128, i128)> {
    let ticks = score::duration_ticks(&value(duration)?).map_err(error)?;
    Ok((ticks.numerator, ticks.denominator))
}
#[pyfunction]
fn duration_token(duration: &str) -> PyResult<String> {
    score::duration_token(&value(duration)?).map_err(error)
}
#[pyfunction]
fn format_measure_target(measure: &str, mode: &str, preserve_playback: bool) -> PyResult<String> {
    score::format_measure_target(&value(measure)?, mode, preserve_playback).map_err(error)
}
#[pyfunction]
#[pyo3(signature = (target, mode, tuning=None, string_count=None))]
fn validate_measure_target(
    target: &str,
    mode: &str,
    tuning: Option<Vec<i64>>,
    string_count: Option<usize>,
) -> String {
    let (parsed, errors) =
        score::validate_measure_target(target, mode, tuning.as_deref(), string_count);
    serde_json::json!([parsed, errors]).to_string()
}
#[pyfunction]
fn score_document(input: &str) -> PyResult<String> {
    // Editable IR may contain unresolved evidence. Export validation belongs
    // to the file writers, not to project persistence or this shared binding.
    result(score::score_document(&value(input)?))
}
#[pyfunction]
fn score_musicxml(py: Python<'_>, input: &str) -> PyResult<String> {
    let input = value(input)?;
    let bytes = py
        .allow_threads(|| music_exports::score_musicxml(&input))
        .map_err(error)?;
    String::from_utf8(bytes).map_err(error)
}
#[pyfunction]
fn write_gp5(py: Python<'_>, input: &str, output: &str) -> PyResult<String> {
    let input = value(input)?;
    result(py.allow_threads(|| music_exports::write_gp5(&input, std::path::Path::new(output))))
}
#[pyfunction]
fn write_musicxml(py: Python<'_>, input: &str, output: &str) -> PyResult<String> {
    let input = value(input)?;
    result(py.allow_threads(|| music_exports::write_musicxml(&input, std::path::Path::new(output))))
}
#[pyfunction]
fn normalize_diagram(input: &str) -> PyResult<String> {
    Ok(chords::normalize_diagram(&value(input)?).to_string())
}
#[pyfunction]
fn diagram_effect(input: &str) -> PyResult<String> {
    chords::diagram_effect(&value(input)?).map_err(error)
}
#[pyfunction]
fn parse_diagram_effect(effect: &str) -> String {
    effect
        .strip_prefix("diagram:")
        .and_then(|text| chords::parse_diagram(text).ok())
        .map(|diagram| diagram.value())
        .unwrap_or(Value::Null)
        .to_string()
}
#[pyfunction(signature = (input, mode=None))]
fn parse_observation(input: &str, mode: Option<&str>) -> PyResult<String> {
    let input = value(input)?;
    result(match mode {
        Some(mode) => observation::parse_measure_observation(&input, mode),
        None => observation::parse_observation(&input),
    })
}
#[pyfunction(signature = (mode, instrument=None))]
fn observation_schema(mode: &str, instrument: Option<&str>) -> PyResult<String> {
    result(observation::observation_schema(mode, instrument))
}
#[pyfunction]
fn observation_prompt(mode: &str) -> PyResult<String> {
    observation::observation_prompt(mode).map_err(error)
}
#[pyfunction]
fn range_relation(input: &str) -> PyResult<bool> {
    Ok(observation::range_relation(&value(input)?))
}
#[pyfunction]
fn relation_voice(input: &str) -> PyResult<Option<i64>> {
    Ok(observation::relation_voice(&value(input)?))
}
#[pyfunction]
fn relation_problem(input: &str, relation: &str) -> PyResult<Option<String>> {
    Ok(observation::relation_problem(
        &value(input)?,
        &value(relation)?,
    ))
}
#[pyfunction]
#[pyo3(signature = (records, auxiliary=None))]
fn assemble_observations(
    py: Python<'_>,
    records: &str,
    auxiliary: Option<&str>,
) -> PyResult<String> {
    let records = value(records)?;
    let records = records
        .as_array()
        .ok_or_else(|| error("Records must be an array"))?;
    let auxiliary = auxiliary.map(value).transpose()?;
    let auxiliary = auxiliary
        .as_ref()
        .map(|v| {
            v.as_array()
                .ok_or_else(|| error("Auxiliary observations must be an array"))
        })
        .transpose()?;
    result(py.allow_threads(|| observation::assemble(records, auxiliary.map(Vec::as_slice))))
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(information_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(information_schema, m)?)?;
    m.add_function(wrap_pyfunction!(parse_information, m)?)?;
    m.add_function(wrap_pyfunction!(key_signature, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_key, m)?)?;
    m.add_function(wrap_pyfunction!(convert_pitch_target, m)?)?;
    m.add_function(wrap_pyfunction!(ornament_position, m)?)?;
    m.add_function(wrap_pyfunction!(parse_observation, m)?)?;
    m.add_function(wrap_pyfunction!(observation_schema, m)?)?;
    m.add_function(wrap_pyfunction!(observation_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(range_relation, m)?)?;
    m.add_function(wrap_pyfunction!(relation_voice, m)?)?;
    m.add_function(wrap_pyfunction!(relation_problem, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_observations, m)?)?;
    m.add_function(wrap_pyfunction!(parse_measure_target, m)?)?;
    m.add_function(wrap_pyfunction!(duration_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(duration_token, m)?)?;
    m.add_function(wrap_pyfunction!(format_measure_target, m)?)?;
    m.add_function(wrap_pyfunction!(validate_measure_target, m)?)?;
    m.add_function(wrap_pyfunction!(score_document, m)?)?;
    m.add_function(wrap_pyfunction!(score_musicxml, m)?)?;
    m.add_function(wrap_pyfunction!(write_gp5, m)?)?;
    m.add_function(wrap_pyfunction!(write_musicxml, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_diagram, m)?)?;
    m.add_function(wrap_pyfunction!(diagram_effect, m)?)?;
    m.add_function(wrap_pyfunction!(parse_diagram_effect, m)?)?;
    Ok(())
}
