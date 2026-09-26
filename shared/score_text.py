"""Readable score text at the UI/file boundary; the trained protocol is unchanged."""

import re


_DISPLAY_NAMES = {"M2": "MEASURE", "C2": "CONTEXT"}
_MODEL_NAMES = {value: key for key, value in _DISPLAY_NAMES.items()}


def _rename_headers(text: str, names: dict[str, str]) -> str:
    # Change only complete line-leading markers, never annotations, note values,
    # or a similar-looking invalid prefix such as M20 / MEASURE2.
    pattern = r"^([ \t]*)(" + "|".join(names) + r")(?=[ \t|]|$)"
    return re.sub(pattern, lambda m: m[1] + names[m[2]], text, flags=re.MULTILINE)


def display_score_text(text: str) -> str:
    return _rename_headers(text, _DISPLAY_NAMES)


def model_score_text(text: str) -> str:
    """Accept readable text or legacy input without changing musical content."""
    return _rename_headers(text, _MODEL_NAMES)


def display_error(text: str) -> str:
    """Use readable names in validation/job messages, preserving technical logs."""
    return re.sub(r"\b(M2|C2)\b", lambda m: _DISPLAY_NAMES[m[0]], text)
