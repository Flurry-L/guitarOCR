"""Python interface to the shared Rust M2 implementation."""
from __future__ import annotations
import json
from fractions import Fraction
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


def parse_duration_token(token: str) -> dict:
    return json.loads(_native.parse_duration_token(token))


def duration_ticks(duration: dict) -> Fraction:
    return Fraction(*_native.duration_ticks(json.dumps(duration)))


def format_measure_target(measure: dict, mode: str, *, preserve_playback: bool = False) -> str:
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


def format_previous_measure_context(target: str, mode: str, *, active_metadata: dict[str, Any] | None = None) -> str:
    """Compact a previous M2 measure for autoregressive document recognition.

    Only the last event of each voice and printed continuity metadata are kept.
    This gives the decoder enough evidence for ties and voice continuity without
    doubling the sequence length with an entire preceding measure.
    """
    measure = parse_measure_target(target)
    # Clefs/key signatures are often printed only at the start of a system;
    # later crops still need the prevailing signature to resolve pitch.
    for field in ("time_signature", "key_signature"):
        if not measure.get(field) and active_metadata and active_metadata.get(field):
            measure[field] = active_metadata[field]
            measure["print_" + field] = True
    context = {
        "time_signature": measure.get("time_signature"),
        "print_time_signature": bool(measure.get("print_time_signature")),
        "tempo_quarter": measure.get("tempo_quarter"),
        "key_signature": measure.get("key_signature"),
        "print_key_signature": bool(measure.get("print_key_signature")),
        "triplet_feel": measure.get("triplet_feel"),
        "bars": [],
        "repeat_count": None,
        "alternate_endings": None,
        "section": None,
        "direction": None,
        "from_direction": None,
        "voices": [],
    }
    for voice in measure.get("voices", []):
        events = voice.get("events") or []
        if not events:
            continue
        last = max(events, key=lambda event: int(event["start"]))
        context["voices"].append({"voice": int(voice["voice"]), "events": [last]})
    return format_measure_target(context, mode).replace("M2", "C2", 1)


def format_history_context(targets: list[str], mode: str) -> str:
    if not targets:
        return "START"
    active = {}
    for target in reversed(targets):
        measure = parse_measure_target(target)
        for field in ("time_signature", "key_signature"):
            if field not in active and measure.get(field):
                active[field] = measure[field]
        if len(active) == 2:
            break
    return format_previous_measure_context(targets[-1], mode, active_metadata=active)


def full_measure_rest_target(time_signature: tuple[int, int]) -> str:
    numerator, denominator = time_signature
    total_ticks = max(1, round(3840 * numerator / denominator))
    durations = (
        (3840, "w"),
        (2880, "h."),
        (1920, "h"),
        (1440, "q."),
        (960, "q"),
        (720, "e."),
        (480, "e"),
        (360, "s."),
        (240, "s"),
        (180, "t."),
        (120, "t"),
        (60, "f"),
        (40, "f[3:2]"),
        (30, "d128"),
        (20, "d128[3:2]"),
    )
    events = []
    start = 0
    remaining = total_ticks
    while remaining > 0:
        choice = next(((ticks, token) for ticks, token in durations
                       if ticks <= remaining and remaining - ticks != 10), None)
        if choice is None:
            raise ValueError(f"Rest duration {total_ticks} ticks cannot be represented exactly")
        ticks, token = choice
        events.append(f"@{start}:{token}:r")
        start += ticks
        remaining -= ticks
    return "M2 | V0{" + " ".join(events) + "}"
