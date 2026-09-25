from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TUNING_PRESETS = {
    "standard tuning": [64, 59, 55, 50, 45, 40],
    "dropped d": [64, 59, 55, 50, 45, 38],
    "drop d": [64, 59, 55, 50, 45, 38],
    "open d": [62, 57, 54, 50, 45, 38],
    "open dsus4": [62, 57, 55, 50, 45, 38],
    "dadgad": [62, 57, 55, 50, 45, 38],
    "open d6": [62, 59, 54, 50, 45, 38],
}


def extract_pdf_vector_metadata(pdf: str | Path) -> dict[str, Any]:
    """Read reliable GP8 header text without pretending hidden data is visible."""

    import pymupdf

    with pymupdf.open(Path(pdf)) as document:
        lines = [
            line.strip()
            for line in document[0].get_text("text").splitlines()
            if line.strip()
        ]
    # Chord-diagram names can precede the tuning/tempo block and easily push
    # it beyond the first forty extracted lines.
    header = lines[:300]
    tempo = next(
        (
            int(match.group(1))
            for line in header
            if (match := re.fullmatch(r"=\s*(\d{2,3})", line))
        ),
        None,
    )
    tuning_index = next(
        (
            index for index, line in enumerate(header)
            if line.lower() in TUNING_PRESETS
            or line.lower() == "custom"
            or "tuning" in line.lower()
            or line.lower().startswith(("open ", "drop ", "dropped "))
        ),
        None,
    )
    tempo_index = next(
        (
            index for index, line in enumerate(header)
            if re.fullmatch(r"=\s*\d{2,3}", line)
        ),
        len(header),
    )
    title = header[0] if header else None
    if title and (
        re.fullmatch(r"=\s*\d{2,3}", title)
        or title.lower() in TUNING_PRESETS
        or title.lower() == "custom"
    ):
        title = None
    artist = None
    metadata_end = min(
        value for value in (tuning_index, tempo_index) if value is not None
    )
    if metadata_end >= 2:
        second = header[1]
        if not second.lower().startswith(("words ", "music ", "copyright")):
            artist = second

    tuning_name = header[tuning_index] if tuning_index is not None else None
    preset = TUNING_PRESETS.get((tuning_name or "").lower())
    warnings = []
    if tuning_name and preset is None:
        warnings.append(
            "PDF prints a custom or unsupported tuning; pass --tuning for exact playback."
        )
    return {
        "title": title,
        "artist": artist,
        "tempo_quarter": tempo,
        "tuning_name": tuning_name,
        "tuning_midi_high_to_low": preset,
        "capo": None,
        "warnings": warnings,
        "source": "pdf_vector_header_text",
    }
