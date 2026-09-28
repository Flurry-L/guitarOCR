"""Instrument identity is independent of the way a staff is displayed."""

import re

INSTRUMENTS = ("guitar", "bass", "pitched", "drums")
DEFAULT_PROGRAMS = {"guitar": 25, "bass": 33, "pitched": 0, "drums": 0}
PROGRAM_NAMES = {0: "Piano", 56: "Trumpet", 60: "Horn", 64: "Soprano Saxophone",
                 65: "Alto Saxophone", 66: "Tenor Saxophone", 67: "Baritone Saxophone", 69: "English Horn"}
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


def program_from_visible_name(text: str | None) -> int | None:
    """Choose a GP5 sound from an explicit printed name, without inferring pitch."""
    if not text:
        return None
    text = " ".join(re.findall(r"[a-z]+", text.lower()))
    names = (
        (r"\bpiano\b", 0),
        (r"\bviolin\b", 40),
        (r"\bviola\b", 41),
        (r"\b(?:cello|violoncello)\b", 42),
        (r"\bcontrabass\b", 43),
        (r"\bflute\b", 73),
        (r"\boboe\b", 68),
        (r"\bclarinet\b", 71),
        (r"\bbassoon\b", 70),
        (r"\b(?:english horn|cor anglais)\b", 69),
        (r"\bsoprano sax(?:ophone)?\b", 64),
        (r"\balto sax(?:ophone)?\b", 65),
        (r"\btenor sax(?:ophone)?\b", 66),
        (r"\bbaritone sax(?:ophone)?\b", 67),
        (r"\btrumpet\b", 56),
        (r"\b(?:french horn|horn in f|f horn)\b", 60),
    )
    return next((program for pattern, program in names if re.search(pattern, text)), None)


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
        raise ValueError(f"无法自动确定 {count} 弦{name}的调弦，请核对乐器并填写各弦音高")
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


def instrument_modes(instrument):
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unsupported instrument: {instrument}")
    return (
        ("tab", "notation", "both")
        if instrument in {"guitar", "bass"}
        else ("notation",)
    )
