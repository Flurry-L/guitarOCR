"""Visible-information prompts used to produce OCR training samples."""

import re

from scorelib.information import prompt

HEADER_PROMPT = prompt("header")
TEMPO_PROMPT = prompt("tempo")
STAFF_PROMPT = prompt("staff")
CLEF_PROMPT = prompt("clef")
ANNOTATION_PROMPT = prompt("annotation")
TRANSPOSITION_PROMPT = ANNOTATION_PROMPT


SIGNATURE_PROMPT = (
    "Read only the printed time and key signatures in this measure image. "
    "Return S2 time=N/D key=K. Use - for a signature that is not printed. "
    "K is the signed number of key-signature accidentals: sharps positive, flats negative, "
    "cancellation to C major 0. Do not count accidentals attached to individual notes."
)


def signature_target(time_signature=None, key=None):
    return f"S2 time={time_signature or '-'} key={key if key is not None else '-'}"


def parse_signature(text):
    match = re.search(r"S2\s+time=(-|\d{1,2}/\d{1,2})\s+key=([+-]?\d|-)(?!\d)", text)
    if not match:
        raise ValueError("Invalid printed signature output")
    time, key = match.groups()
    if time != "-":
        n, d = map(int, time.split("/"))
        if not 1 <= n <= 32 or d not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("Invalid time signature")
    if key != "-" and not -7 <= int(key) <= 7:
        raise ValueError("Invalid key signature")
    return {"time": None if time == "-" else time, "key": None if key == "-" else int(key)}


def key_fifths(name):
    from scorelib.pitch_context import key_signature

    fifths = key_signature(str(name)).value[0]
    # Native engraving spells the theoretical eight-accidental keys using
    # their ordinary enharmonic equivalents (F-flat -> E, G-sharp -> A-flat).
    return fifths + 12 if fifths < -7 else fifths - 12 if fifths > 7 else fifths


def observation_prompt(mode):
    """Use the exact visible-observation prompt shared with product inference."""
    from scorelib import _native

    return _native.observation_prompt(mode)
