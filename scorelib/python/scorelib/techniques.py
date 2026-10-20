"""Canonical visible ornaments and chord marks."""

from __future__ import annotations

from collections import Counter

from scorelib._native import ornament_position as ornament_position


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
