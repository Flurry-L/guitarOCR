"""Visible ornament positions; notation targets never require hidden frets."""

import re
from collections import Counter

POSITION = re.compile(r"^(?:f(?P<fret>\d+))?(?:p(?P<pitch>\d+))?$")


def canonical_chord_marks(measure):
    """Anchor chord-wide printed marks to descending pitches, retaining counts.

    Engravers place these symbols on the chord regardless of which source
    note owns them. Ties and note-specific techniques retain their anchors.
    """
    common = {'accent', 'heavy', 'stacc'}
    for voice in measure['voices']:
        for event in voice['events']:
            notes = sorted(event.get('notes', []), key=lambda n: -n['pitch'])
            if len(notes) < 2:
                continue
            marks = Counter(effect for note in notes for effect in note.get('effects', []) if effect in common)
            for note in notes:
                note['effects'] = [e for e in note.get('effects', []) if e not in common]
            for effect, count in sorted(marks.items()):
                for i in range(count):
                    notes[i % len(notes)]['effects'].append(effect)
    return measure


def ornament_position(token: str) -> tuple[int | None, int | None]:
    if token == "x":
        return None, None
    match = POSITION.fullmatch(token)
    if not match or not token:
        raise ValueError(f"Invalid ornament position: {token}")
    return tuple(int(match.group(k)) if match.group(k) is not None else None for k in ("fret", "pitch"))


def visible_effect(effect: str, mode: str, *, preserve_playback: bool = False) -> str:
    parts = effect.split(":")
    if parts[0] in {"grace", "trill"} and len(parts) > 1:
        if parts[0] == "trill" and mode == "notation" and not preserve_playback:
            # GP8 prints tr, but neither the auxiliary pitch nor playback rate.
            return "trill"
        fret, pitch = ornament_position(parts[1])
        if mode == "notation" and pitch is not None:
            parts[1] = f"p{pitch}"
        elif mode == "tab" and fret is not None:
            parts[1] = f"f{fret}"
        if not preserve_playback:
            if parts[0] == "trill":
                parts = parts[:2]
            else:
                if len(parts) == 6:
                    del parts[2]  # Playback duration does not change the glyph.
                if mode == "tab":
                    parts[3] = "-"  # On-beat/slashed distinction needs notation.
                if parts[4] == "1":
                    parts[1] = "x"  # A dead grace note has no visible fret/pitch.
    return ":".join(parts)


def ornament_pitches(note: dict) -> list[int]:
    """Pitches that must fit the export slot, including default trill playback."""
    pitches = []
    for effect in note.get("effects", []):
        if effect == "trill" and "pitch" in note:
            # Notation prints no auxiliary pitch. The GP5 exporter uses the
            # next semitone, which still needs room in its 0..30 fret range.
            pitches.append(int(note["pitch"]) + 1)
        elif effect.startswith(("grace:", "trill:")):
            _, pitch = ornament_position(effect.split(":")[1])
            if pitch is not None:
                pitches.append(pitch)
    return pitches
