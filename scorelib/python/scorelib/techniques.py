"""Canonical visible ornaments and chord marks."""

from __future__ import annotations

import re
from collections import Counter

POSITION = re.compile(r"^(?:f(?P<fret>\d+))?(?:p(?P<pitch>\d+))?$")


def canonical_chord_marks(measure):
    """Anchor chord-wide printed marks to descending pitches, retaining counts.

    Engravers place these symbols on the chord regardless of which source
    note owns them. Ties and note-specific techniques retain their anchors.
    """
    common = {"accent", "heavy", "stacc"}
    for voice in measure["voices"]:
        for event in voice["events"]:
            notes = sorted(event.get("notes", []), key=lambda n: -n["pitch"])
            if len(notes) < 2:
                continue
            marks = Counter(
                effect
                for note in notes
                for effect in note.get("effects", [])
                if effect in common
            )
            for note in notes:
                note["effects"] = [
                    e for e in note.get("effects", []) if e not in common
                ]
            for effect, count in sorted(marks.items()):
                for i in range(count):
                    notes[i % len(notes)]["effects"].append(effect)
    return measure


def ornament_position(token: str) -> tuple[int | None, int | None]:
    if token == "x":
        return None, None
    match = POSITION.fullmatch(token)
    if not match or not token:
        raise ValueError(f"Invalid ornament position: {token}")
    return tuple(
        int(match.group(k)) if match.group(k) is not None else None
        for k in ("fret", "pitch")
    )
