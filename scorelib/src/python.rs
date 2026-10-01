//! Python bindings use JSON only at the language boundary; Rust owns semantics.
use crate::{gp5_constraints, music_exports, score};
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
fn parse_duration_token(token: &str) -> PyResult<String> {
    result(score::parse_duration_token(token))
}
#[pyfunction]
fn duration_ticks(duration: &str) -> PyResult<(i128, i128)> {
    let ticks = score::duration_ticks(&value(duration)?).map_err(error)?;
    Ok((ticks.numerator, ticks.denominator))
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
    result(music_exports::score_document(&value(input)?))
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
fn timing_errors(target: &str) -> PyResult<Vec<String>> {
    gp5_constraints::timing_errors(target).map_err(error)
}
#[pyfunction]
fn notation_fingering_errors(target: &str, tuning: Vec<i64>) -> PyResult<Vec<String>> {
    gp5_constraints::notation_fingering_errors(&value(target)?, &tuning).map_err(error)
}
#[pyfunction]
fn display_score_text(text: &str) -> String {
    score::display_score_text(text)
}
#[pyfunction]
fn model_score_text(text: &str) -> String {
    score::model_score_text(text)
}
#[pyfunction]
fn display_error(text: &str) -> String {
    score::display_error(text)
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_measure_target, m)?)?;
    m.add_function(wrap_pyfunction!(parse_duration_token, m)?)?;
    m.add_function(wrap_pyfunction!(duration_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(format_measure_target, m)?)?;
    m.add_function(wrap_pyfunction!(validate_measure_target, m)?)?;
    m.add_function(wrap_pyfunction!(score_document, m)?)?;
    m.add_function(wrap_pyfunction!(score_musicxml, m)?)?;
    m.add_function(wrap_pyfunction!(write_gp5, m)?)?;
    m.add_function(wrap_pyfunction!(write_musicxml, m)?)?;
    m.add_function(wrap_pyfunction!(timing_errors, m)?)?;
    m.add_function(wrap_pyfunction!(notation_fingering_errors, m)?)?;
    m.add_function(wrap_pyfunction!(display_score_text, m)?)?;
    m.add_function(wrap_pyfunction!(model_score_text, m)?)?;
    m.add_function(wrap_pyfunction!(display_error, m)?)?;
    Ok(())
}
