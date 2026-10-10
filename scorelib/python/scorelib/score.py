"""Score IR parsing, validation and native format exports."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from scorelib import _native

DURATION_NAMES = {
    1: "w",
    2: "h",
    4: "q",
    8: "e",
    16: "s",
    32: "t",
    64: "f",
}


def parse_measure_target(text: str) -> dict:
    return json.loads(_native.parse_measure_target(text))


def duration_ticks(duration: dict) -> Fraction:
    return Fraction(*_native.duration_ticks(json.dumps(duration)))


def format_measure_target(
    measure: dict, mode: str, *, preserve_playback: bool = False
) -> str:
    return _native.format_measure_target(json.dumps(measure), mode, preserve_playback)


def _duration_token(duration: dict[str, Any]) -> str:
    value = int(duration["value"])
    token = DURATION_NAMES.get(value, f"d{value}")
    if duration.get("double_dotted"):
        token += ".."
    elif duration.get("dotted"):
        token += "."
    enters = int(duration.get("tuplet_enters", 1))
    times = int(duration.get("tuplet_times", 1))
    if enters != 1 or times != 1:
        token += f"[{enters}:{times}]"
    return token


def full_measure_rest_target(time_signature: tuple[int, int]) -> str:
    numerator, denominator = time_signature
    total_ticks = max(1, round(3840 * numerator / denominator))
    return _native.rest_target(total_ticks)


def validate_measure_target(target, mode, *, tuning=None, string_count=None):
    parsed, errors = json.loads(
        _native.validate_measure_target(target, mode, tuning, string_count)
    )
    return parsed, errors


def score_document(result: dict) -> dict:
    return json.loads(_native.score_document(json.dumps(result)))


def score_musicxml(score):
    return _native.score_musicxml(json.dumps(score)).encode("utf-8")


def write_musicxml(score, output):
    output = Path(output)
    _native.write_musicxml(json.dumps(score), str(output))
    return output


def write_score_gp5(result, output):
    output = Path(output)
    _native.write_gp5(json.dumps(result), str(output))
    return output
