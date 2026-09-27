"""Fields exchanged by stages and editing sessions."""

from typing import Literal, TypedDict

Stage = Literal["layout", "document_info", "measure_ocr", "gp5_export"]
Mode = Literal["tab", "notation", "both"]
Instrument = Literal["guitar", "bass", "pitched", "drums"]
STAGE_SCHEMA_VERSION = "1.0"


class MeasureRecord(TypedDict, total=False):
    measure_number: int
    part_id: str
    part_name: str
    staff_id: str
    bar_index: int
    instrument: Instrument
    midi_program: int
    tuning: list[int]
    capo: int
    pitch_context: dict
    pitch_reference: Literal["sounding", "before_capo", "percussion_key"]
    pitch_needs_review: bool
    timing_errors: list[str]
    mode: Mode
    mode_source: str
    detected_mode: Mode
    score: float
    page: int
    bbox: list[float]
    image: str
    target: str
    previous_context: str
    needs_review: bool
    fallback_reason: list[str]
    manually_edited: bool


class StageResult(TypedDict, total=False):
    schema_version: str
    stage: Stage
    mode: Mode | Literal["auto"]
    status: str
    layout: str
    info: str
    recognition: str
    records: list[MeasureRecord]
    regions: list[dict]
    pages: list[dict]
    inputs: list[str]
    document_metadata: dict
    title: str
    artist: str
    instrument: Instrument
    midi_program: int
    tuning_used: list[int]
    capo: int
    transpose: int | None
    measure_pitch_contexts: list[dict]
    measures: int
    m2: str
    score_text: str
    score_document: str
    recognition_log: str
    review_measures: list[int]
    gp5: str
    encoding_report: str
