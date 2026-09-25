from __future__ import annotations

import re
from typing import Any

from shared.m2 import parse_measure_target


_NOTE_EFFECTS = {
    "accent",
    "dead",
    "ghost",
    "grace",
    "hammer",
    "heavy",
    "let",
    "pm",
    "sia",
    "sib",
    "sl",
    "sod",
    "sou",
    "ss",
    "stacc",
    "tap",
    "tie",
    "trem",
    "trill",
    "vib",
}
_BEAT_EFFECTS = {
    "fade",
    "pick_down",
    "pick_up",
    "rasg",
    "stroke_down",
    "stroke_up",
}
_DURATION_VALUES = {1, 2, 4, 8, 16, 32, 64}


def _valid_note_effect(effect: str) -> bool:
    if effect in _NOTE_EFFECTS:
        return True
    if effect.startswith(("harm:", "slide:")):
        return bool(effect.partition(":")[2])
    if effect.startswith("bend:"):
        return re.fullmatch(r"bend:[^:]+:-?\d+", effect) is not None
    return False


def _valid_beat_effect(effect: str) -> bool:
    if effect in _BEAT_EFFECTS:
        return True
    if effect.startswith(("chord:", "slap:", "text:")):
        return bool(effect.partition(":")[2])
    if effect.startswith("tempo:"):
        value = effect.partition(":")[2]
        return value.isdigit() and 1 <= int(value) <= 999
    if effect.startswith("dyn:"):
        value = effect.partition(":")[2]
        return value.isdigit() and 0 <= int(value) <= 127
    return False


def validate_measure_target(
    target: str,
    mode: str,
    *,
    tuning: list[int] | None = None,
    string_count: int | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate syntax plus invariants that every legal M2 measure must obey.

    Deliberately not checked: whether durations fill the time signature.  Real
    GP3/4/5 files contain pickup bars and a small number of overfull legacy
    measures, so treating that relationship as a hard rule would corrupt valid
    source semantics.
    """
    if mode not in {"tab", "notation", "both"}:
        return None, [f"unsupported_mode:{mode}"]
    try:
        measure = parse_measure_target(target)
    except Exception as error:  # error text is fed back to the decoder on retry
        return None, [f"syntax:{error}"]

    errors: list[str] = []
    signature = measure.get("time_signature")
    if signature:
        match = re.fullmatch(r"(\d+)/(\d+)", str(signature))
        if match is None:
            errors.append("invalid_time_signature")
        else:
            numerator, denominator = map(int, match.groups())
            if not 1 <= numerator <= 32 or denominator not in _DURATION_VALUES:
                errors.append("invalid_time_signature")
    tempo = measure.get("tempo_quarter")
    if tempo is not None and not 1 <= int(tempo) <= 999:
        errors.append("invalid_tempo")
    if not measure.get("voices"):
        errors.append("missing_voice")

    voice_ids: set[int] = set()
    maximum_string = string_count or (len(tuning) if tuning else 8)
    for voice in measure.get("voices", []):
        voice_id = int(voice["voice"])
        if voice_id in voice_ids:
            errors.append(f"duplicate_voice:V{voice_id}")
        voice_ids.add(voice_id)
        events = voice.get("events") or []
        starts = [int(event["start"]) for event in events]
        if any(start < 0 for start in starts):
            errors.append(f"negative_event_start:V{voice_id}")
        if starts != sorted(starts) or len(starts) != len(set(starts)):
            errors.append(f"non_increasing_event_starts:V{voice_id}")
        for event_index, event in enumerate(events):
            duration = event.get("duration") or {}
            value = int(duration.get("value", 0) or 0)
            enters = int(duration.get("tuplet_enters", 0) or 0)
            times = int(duration.get("tuplet_times", 0) or 0)
            if value not in _DURATION_VALUES or not 1 <= enters <= 32 or not 1 <= times <= 32:
                errors.append(f"invalid_duration:V{voice_id}:E{event_index}")
            for effect in event.get("effects", []):
                if not _valid_beat_effect(str(effect)):
                    errors.append(f"unknown_beat_effect:{effect}")

            strings: list[int] = []
            for note_index, note in enumerate(event.get("notes") or []):
                location = f"V{voice_id}:E{event_index}:N{note_index}"
                has_string = "string" in note and "fret" in note
                has_pitch = "pitch" in note
                if mode == "tab" and (not has_string or has_pitch):
                    errors.append(f"tab_note_fields:{location}")
                elif mode == "notation" and (not has_pitch or has_string):
                    errors.append(f"notation_note_fields:{location}")
                elif mode == "both" and (not has_string or not has_pitch):
                    errors.append(f"both_note_fields:{location}")

                if has_string:
                    string = int(note["string"])
                    strings.append(string)
                    if not 1 <= string <= maximum_string:
                        errors.append(f"invalid_string:{location}")
                    fret = note["fret"]
                    if fret != "x" and not 0 <= int(fret) <= 36:
                        errors.append(f"invalid_fret:{location}")
                    if (
                        mode == "both"
                        and tuning
                        and fret != "x"
                        and 1 <= string <= len(tuning)
                        and int(note["pitch"]) != int(tuning[string - 1]) + int(fret)
                    ):
                        errors.append(f"pitch_string_fret_conflict:{location}")
                if has_pitch and not 0 <= int(note["pitch"]) <= 127:
                    errors.append(f"invalid_pitch:{location}")
                if not 0 <= int(note.get("velocity", 95)) <= 127:
                    errors.append(f"invalid_velocity:{location}")
                for effect in note.get("effects", []):
                    if not _valid_note_effect(str(effect)):
                        errors.append(f"unknown_note_effect:{effect}")
            if len(strings) != len(set(strings)):
                errors.append(f"duplicate_string_in_chord:V{voice_id}:E{event_index}")
    return measure, list(dict.fromkeys(errors))
