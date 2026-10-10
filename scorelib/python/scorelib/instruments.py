"""Instrument identity, string tunings and visible percussion keys."""

from __future__ import annotations

import re

INSTRUMENTS = ("guitar", "bass", "pitched", "drums")
DEFAULT_PROGRAMS = {"guitar": 25, "bass": 33, "pitched": 0, "drums": 0}
DEFAULT_TUNINGS = {
    "guitar": [64, 59, 55, 50, 45, 40],
    "bass": [43, 38, 33, 28],
    "pitched": [],
    "drums": [],
}

STANDARD_TUNINGS = {
    ("guitar", 4): DEFAULT_TUNINGS["guitar"][:4],
    ("guitar", 5): DEFAULT_TUNINGS["guitar"][:5],
    ("guitar", 6): DEFAULT_TUNINGS["guitar"],
    ("guitar", 7): [64, 59, 55, 50, 45, 40, 35],
    ("guitar", 8): [64, 59, 55, 50, 45, 40, 35, 30],
    ("bass", 4): DEFAULT_TUNINGS["bass"],
    ("bass", 5): [43, 38, 33, 28, 23],
    ("bass", 6): [48, 43, 38, 33, 28, 23],
    ("bass", 7): [53, 48, 43, 38, 33, 28, 23],
}


def pitch_reference(instrument: str) -> str:
    if instrument == "drums":
        return "percussion_key"
    return "before_capo" if instrument in {"guitar", "bass"} else "sounding"


def standard_tuning(instrument: str, string_count: int | None = None) -> list[int]:
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unsupported instrument: {instrument}")
    if instrument in {"pitched", "drums"}:
        return []
    count = string_count or len(DEFAULT_TUNINGS[instrument])
    if (instrument, count) not in STANDARD_TUNINGS:
        name = "吉他" if instrument == "guitar" else "贝斯"
        raise ValueError(
            f"无法自动确定 {count} 弦{name}的调弦，请核对乐器并填写各弦音高"
        )
    return list(STANDARD_TUNINGS[instrument, count])


def track_instrument(track):
    if track.isPercussionTrack:
        return "drums"
    program = int(track.channel.instrument)
    if 24 <= program <= 31:
        return "guitar"
    if 32 <= program <= 39:
        return "bass"
    return "pitched"


DEFAULT_TUNING = (64, 59, 55, 50, 45, 40)


TUNING_NAMES = {
    "standard": list(DEFAULT_TUNING),
    "standard tuning": list(DEFAULT_TUNING),
    "dropped d": [64, 59, 55, 50, 45, 38],
    "drop d": [64, 59, 55, 50, 45, 38],
    "dropped c": [62, 57, 53, 48, 43, 36],
    "drop c": [62, 57, 53, 48, 43, 36],
    "dropped b": [61, 56, 52, 47, 42, 35],
    "drop b": [61, 56, 52, 47, 42, 35],
    "dropped d tune down 1/2 step": [63, 58, 54, 49, 44, 37],
    "tune down 1/2 step": [63, 58, 54, 49, 44, 39],
    "tune down 1 step": [62, 57, 53, 48, 43, 38],
    "tune down 2 step": [60, 55, 51, 46, 41, 36],
    "dadgad": [62, 57, 55, 50, 45, 38],
    "open d": [62, 57, 54, 50, 45, 38],
    "open g": [62, 59, 55, 50, 43, 38],
    "open e": [64, 59, 56, 52, 47, 40],
    "open a": [64, 61, 57, 52, 45, 40],
    "double drop d": [62, 59, 55, 50, 45, 38],
}

ALIASES = {
    "half step down": "tune down 1/2 step",
    "1/2 step down": "tune down 1/2 step",
    "eb standard": "tune down 1/2 step",
    "e flat standard": "tune down 1/2 step",
    "whole step down": "tune down 1 step",
    "full step down": "tune down 1 step",
    "one step down": "tune down 1 step",
    "d standard": "tune down 1 step",
    "two steps down": "tune down 2 step",
    "c standard": "tune down 2 step",
}


def tuning_from_name(
    name: str | None, instrument: str = "guitar", string_count: int | None = None
) -> list[int] | None:
    if not name:
        return None
    label = re.sub(
        r"[\s\-–]+", " ", name.strip().casefold().replace("½", "1/2").replace("♭", "b")
    )
    # Printed labels commonly include the instrument before the tuning name.
    # Only remove a matching, explicit prefix; an incompatible instrument or
    # an unknown tuning must not become a guessed standard tuning.
    for prefix in ("guitar", "bass"):
        if label.startswith(prefix + " "):
            if prefix != instrument:
                return None
            label = label[len(prefix) + 1 :]
            break
    label = re.sub(r"\s+tuning$", "", label)
    label = ALIASES.get(label, label)
    if instrument == "guitar" and string_count in {None, 6}:
        return TUNING_NAMES.get(label)

    if instrument not in {"guitar", "bass"}:
        return None
    if instrument == "guitar" and string_count in {4, 5}:
        # A shortened guitar TAB has no unique "standard" tuning: it may
        # represent a tenor guitar, a low-tuned track, or omitted strings.
        # Keep the name as evidence, but do not turn it into explicit pitches.
        return None
    try:
        tuning = standard_tuning(instrument, string_count)
    except ValueError:
        return None
    if label in {"standard", "standard tuning"}:
        return tuning
    shifts = {"tune down 1/2 step": -1, "tune down 1 step": -2, "tune down 2 step": -4}
    if label in shifts:
        return [pitch + shifts[label] for pitch in tuning]
    if instrument == "bass" and len(tuning) == 4:
        drop_shifts = {
            "drop d": 0,
            "dropped d": 0,
            "drop c": -2,
            "dropped c": -2,
            "drop b": -3,
            "dropped b": -3,
            "dropped d tune down 1/2 step": -1,
        }
        if label in drop_shifts:
            return [pitch + drop_shifts[label] for pitch in [*tuning[:-1], 26]]
    return None


VISIBLE_DRUM_KEYS = frozenset(range(29, 88)) - {32}
DRUM_KEY_ALIASES = {33: 37, 34: 38, 39: 38, 40: 38, 41: 45}


def visible_drum_key(key: int) -> int:
    if key not in VISIBLE_DRUM_KEYS:
        raise ValueError(f"Drum key {key} has no verified visible GP8 notehead")
    return DRUM_KEY_ALIASES.get(key, key)
