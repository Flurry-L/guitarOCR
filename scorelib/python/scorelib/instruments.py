"""Instrument identity, string tunings and visible percussion keys."""

from __future__ import annotations

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


VISIBLE_DRUM_KEYS = frozenset(range(29, 88)) - {32}
DRUM_KEY_ALIASES = {33: 37, 34: 38, 39: 38, 40: 38, 41: 45}


def visible_drum_key(key: int) -> int:
    if key not in VISIBLE_DRUM_KEYS:
        raise ValueError(f"Drum key {key} has no verified visible GP8 notehead")
    return DRUM_KEY_ALIASES.get(key, key)
