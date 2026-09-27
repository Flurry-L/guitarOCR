from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
from urllib.parse import quote, unquote
from shared.techniques import visible_effect


DURATION_NAMES = {
    1: "w",
    2: "h",
    4: "q",
    8: "e",
    16: "s",
    32: "t",
    64: "f",
}

_DURATION_PATTERN = re.compile(
    r"^(?P<base>w|h|q|e|s|t|f|d\d+)(?P<dots>\.\.|\.)?"
    r"(?:\[(?P<enters>\d+):(?P<times>\d+)\])?$"
)
_NOTE_PATTERN = re.compile(
    r"^(?:s(?P<string>\d+)f(?P<fret>x|-?\d+))?"
    r"(?:p(?P<pitch>-?\d+))?(?:\((?P<effects>.*)\))?$"
)


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


def _note_token(note: dict[str, Any], mode: str, *, preserve_playback: bool = False) -> str:
    if mode == "notation":
        token = f"p{int(note['pitch'])}"
    elif mode == "both":
        token = f"s{int(note['string'])}f{note['fret']}p{int(note['pitch'])}"
    else:
        token = f"s{int(note['string'])}f{note['fret']}"
    effects = [visible_effect(effect, mode, preserve_playback=preserve_playback) for effect in note.get("effects") or []]
    velocity = int(note.get("velocity", 95))
    if velocity != 95:
        effects = [*effects, f"vel:{velocity}"]
    # Accidental spelling is visible on a notation staff, but a TAB-only
    # staff prints only the fret value. Do not leak hidden source spelling
    # state into TAB supervision.
    if mode != "tab" and note.get("swap_accidentals"):
        effects = [*effects, "accswap"]
    if effects:
        token += "(" + ",".join(str(effect) for effect in effects) + ")"
    return token


def format_measure_target(measure: dict[str, Any], mode: str, *, preserve_playback: bool = False) -> str:
    if mode not in {"tab", "notation", "both"}:
        raise ValueError(f"Unsupported display mode: {mode}")
    metadata = []
    if measure.get("print_time_signature"):
        metadata.append("time=" + str(measure["time_signature"]))
    if measure.get("tempo_quarter"):
        metadata.append("tempo=" + str(measure["tempo_quarter"]))
    if mode in {"notation", "both"} and measure.get("print_key_signature"):
        metadata.append("key=" + str(measure["key_signature"]))
    if measure.get("triplet_feel"):
        metadata.append("feel=" + str(measure["triplet_feel"]))
    if measure.get("bars"):
        metadata.append("bar=" + ",".join(str(item) for item in measure["bars"]))
    if measure.get("repeat_count"):
        metadata.append("rep=" + str(measure["repeat_count"]))
    if measure.get("alternate_endings"):
        metadata.append("alt=" + str(measure["alternate_endings"]))
    if measure.get("section"):
        metadata.append("section=" + quote(str(measure["section"]), safe=""))
    if str(measure.get("direction", "none")).lower() not in {"none", "null"}:
        metadata.append("dir=" + quote(str(measure["direction"]), safe=""))
    if str(measure.get("from_direction", "none")).lower() not in {"none", "null"}:
        metadata.append("from=" + quote(str(measure["from_direction"]), safe=""))
    voice_tokens = []
    for voice in measure.get("voices", []):
        events = []
        previous_payload: str | None = None
        for event in voice.get("events", []):
            duration = _duration_token(event["duration"])
            status = str(event.get("status", "normal"))
            if status == "empty":
                payload = "e"
            elif status == "rest":
                payload = "r"
            else:
                notes = list(event.get("notes", []))
                # The source string assignment is invisible in notation-only
                # output.  Canonical pitch order removes an impossible target
                # dependency and makes chord equality independent of the GP
                # parser's internal note ordering.
                if mode == "notation":
                    notes.sort(key=lambda note: (
                        -int(note["pitch"]),
                        tuple(sorted(str(effect) for effect in note.get("effects", []))),
                        int(note.get("velocity", 95)),
                        bool(note.get("swap_accidentals")),
                    ))
                else:
                    notes.sort(key=lambda note: (int(note["string"]), repr(note)))
                payload = ",".join(_note_token(note, mode, preserve_playback=preserve_playback) for note in notes) or "z"
            beat_effects = [
                value
                for value in (event.get("effects") or [])
                # A source velocity on every beat does not identify where a
                # dynamic mark is actually printed. OCR dynamics supervision
                # is not implemented; keep manual/legacy playback support.
                if preserve_playback or not str(value).startswith("dyn:")
            ]
            if beat_effects:
                payload += "<" + ",".join(str(value) for value in beat_effects) + ">"
            expanded_payload = payload
            if previous_payload is not None and expanded_payload == previous_payload:
                payload = "^"
            events.append(f"@{int(event['start'])}:{duration}:{payload}")
            previous_payload = expanded_payload
        voice_tokens.append(f"V{int(voice['voice'])}" + "{" + " ".join(events) + "}")
    prefix = "M2"
    if metadata:
        prefix += " " + " ".join(metadata)
    return prefix + " | " + " || ".join(voice_tokens)


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


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    values = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth = max(0, depth - 1)
        elif character == delimiter and depth == 0:
            values.append(text[start:index])
            start = index + 1
    values.append(text[start:])
    return [value for value in values if value]


def parse_duration_token(token: str) -> dict[str, Any]:
    match = _DURATION_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"Invalid M2 duration token: {token!r}")
    base = match.group("base")
    reverse = {value: key for key, value in DURATION_NAMES.items()}
    value = int(base[1:]) if base.startswith("d") else reverse[base]
    dots = match.group("dots") or ""
    return {
        "value": value,
        "dotted": dots == ".",
        "double_dotted": dots == "..",
        "tuplet_enters": int(match.group("enters") or 1),
        "tuplet_times": int(match.group("times") or 1),
    }


def _parse_note_token(token: str) -> dict[str, Any]:
    match = _NOTE_PATTERN.fullmatch(token)
    if match is None or not (match.group("string") or match.group("pitch")):
        raise ValueError(f"Invalid M2 note token: {token!r}")
    effects = _split_top_level(match.group("effects") or "")
    velocity = 95
    swap_accidentals = False
    musical_effects = []
    for effect in effects:
        if effect.startswith("vel:"):
            velocity = int(effect.partition(":")[2])
        elif effect == "accswap":
            swap_accidentals = True
        else:
            musical_effects.append(effect)
    fret_text = match.group("fret")
    result: dict[str, Any] = {
        "effects": musical_effects,
        "velocity": velocity,
        "swap_accidentals": swap_accidentals,
    }
    if match.group("string"):
        result["string"] = int(match.group("string"))
        result["fret"] = "x" if fret_text == "x" else int(fret_text)
    if match.group("pitch"):
        result["pitch"] = int(match.group("pitch"))
    return result


def parse_measure_target(text: str) -> dict[str, Any]:
    """Parse one strict M2 target into the canonical measure dictionary."""

    prefix, separator, voice_text = text.strip().partition("|")
    if not separator or not prefix.split() or prefix.split()[0] != "M2":
        raise ValueError("M2 target must start with 'M2' and contain '|'")
    metadata: dict[str, str] = {}
    for token in prefix.strip().split()[1:]:
        key, equals, value = token.partition("=")
        if not equals:
            raise ValueError(f"Invalid M2 metadata token: {token!r}")
        if key not in {"time", "tempo", "key", "feel", "bar", "rep", "alt", "section", "dir", "from"}:
            raise ValueError(f"Unknown M2 metadata: {key}")
        if key in metadata:
            raise ValueError(f"Duplicate M2 metadata: {key}")
        metadata[key] = value
    measure: dict[str, Any] = {
        "time_signature": metadata.get("time"),
        "print_time_signature": "time" in metadata,
        "tempo_quarter": int(metadata["tempo"]) if "tempo" in metadata else None,
        "key_signature": metadata.get("key"),
        "print_key_signature": "key" in metadata,
        "triplet_feel": metadata.get("feel"),
        "bars": metadata.get("bar", "").split(",") if metadata.get("bar") else [],
        "repeat_count": int(metadata["rep"]) if "rep" in metadata else None,
        "alternate_endings": int(metadata["alt"]) if "alt" in metadata else None,
        "section": unquote(metadata["section"]) if "section" in metadata else None,
        "direction": unquote(metadata["dir"]) if "dir" in metadata else None,
        "from_direction": unquote(metadata["from"]) if "from" in metadata else None,
        "voices": [],
    }
    voice_parts = [part.strip() for part in voice_text.split("||") if part.strip()]
    for voice_part in voice_parts:
        match = re.fullmatch(r"V(?P<voice>\d+)\{(?P<events>.*)\}", voice_part)
        if match is None:
            raise ValueError(f"Invalid M2 voice: {voice_part!r}")
        events = []
        for event_text in match.group("events").split():
            if not event_text.startswith("@"):
                raise ValueError(f"Invalid M2 event: {event_text!r}")
            start_text, separator, remainder = event_text[1:].partition(":")
            if not separator:
                raise ValueError(f"Invalid M2 event start: {event_text!r}")
            bracket_end = remainder.find("]")
            duration_end = bracket_end + 1 if bracket_end >= 0 else remainder.find(":")
            if duration_end <= 0 or duration_end >= len(remainder) or remainder[duration_end] != ":":
                raise ValueError(f"Invalid M2 event duration: {event_text!r}")
            duration_text = remainder[:duration_end]
            payload = remainder[duration_end + 1:]
            if payload == "^":
                if not events:
                    raise ValueError("M2 repeated payload '^' cannot be the first event")
                previous = deepcopy(events[-1])
                previous["start"] = int(start_text)
                previous["duration"] = parse_duration_token(duration_text)
                events.append(previous)
                continue
            beat_effects = []
            if payload.endswith(">") and "<" in payload:
                payload, _opening, effect_text = payload.rpartition("<")
                beat_effects = _split_top_level(effect_text[:-1])
            if payload == "e":
                status = "empty"
                notes = []
            elif payload == "r":
                status = "rest"
                notes = []
            elif payload == "z":
                status = "normal"
                notes = []
            else:
                status = "normal"
                notes = [_parse_note_token(note) for note in _split_top_level(payload)]
            events.append({
                "start": int(start_text),
                "duration": parse_duration_token(duration_text),
                "status": status,
                "notes": notes,
                "effects": beat_effects,
            })
        measure["voices"].append({"voice": int(match.group("voice")), "events": events})
    return measure


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
    )
    events = []
    start = 0
    remaining = total_ticks
    while remaining > 0:
        ticks, token = next(
            ((ticks, token) for ticks, token in durations if ticks <= remaining),
            (remaining, f"d{max(1, round(3840 / remaining))}"),
        )
        events.append(f"@{start}:{token}:r")
        start += ticks
        remaining -= ticks
    return "M2 | V0{" + " ".join(events) + "}"
