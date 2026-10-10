"""Canonical visible ornaments and event-level technique coverage."""

from __future__ import annotations

import re
from collections import Counter

from scorelib.score import parse_measure_target

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


EFFECTS = {
    "none": frozenset(),
    "pm": frozenset({"pm"}),
    "let": frozenset({"let"}),
    "both": frozenset({"pm", "let"}),
}


def _measure(value):
    return parse_measure_target(value) if isinstance(value, str) else value


def sounding_events(value):
    """Return unique anchors, rejecting inputs that cannot be joined safely."""
    events = {}
    for voice in _measure(value).get("voices", []):
        for event in voice.get("events", []):
            if not event.get("notes"):
                continue
            key = voice["voice"], event["start"]
            if key in events:
                raise ValueError("Duplicate sounding event anchor")
            events[key] = event
    return events


def visible_event_effects(value):
    result = {}
    for key, event in sounding_events(value).items():
        effects = set(event.get("effects", []))
        for note in event["notes"]:
            effects.update(note.get("effects", []))
        result[key] = frozenset(effects & {"pm", "let"})
    return result
