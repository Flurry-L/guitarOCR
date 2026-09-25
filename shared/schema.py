"""Versioned fields exchanged by stages and editing sessions."""

from typing import Literal, TypedDict

Stage = Literal["layout", "document_info", "measure_ocr", "gp5_export"]
Mode = Literal["tab", "notation", "both"]
STAGE_SCHEMA_VERSION = "1.0"


class MeasureRecord(TypedDict, total=False):
    measure_number: int
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
    mode: Mode
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
    tuning_used: list[int]
    capo: int
    measures: int
    m2: str
    recognition_log: str
    review_measures: list[int]
    gp5: str
    encoding_report: str
