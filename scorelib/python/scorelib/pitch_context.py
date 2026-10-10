"""Convert written and sounding pitches using explicit musical context."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=128)
def key_signature(name):
    from guitarpro.models import KeySignature

    def identity(value):
        mode = "minor" if "Minor" in value else "major"
        tonic = (
            value.replace("Major", "")
            .replace("Minor", "")
            .replace("Sharp", "#")
            .replace("Flat", "b")
            .replace("♯", "#")
            .replace("♭", "b")
        )
        return tonic, mode

    wanted = identity(str(name))
    for candidate in KeySignature:
        if identity(candidate.name) == wanted:
            return candidate
    raise KeyError(name)


def transpose_key(name, semitones):
    """Transpose a GP key name, choosing the spelling with fewest accidentals."""
    if not semitones % 12:
        return name
    from guitarpro.models import KeySignature

    key = key_signature(name)
    fifths, minor = key.value
    tonic = (7 * fifths + (9 if minor else 0) + semitones) % 12
    choices = [
        k
        for k in KeySignature
        if k.value[1] == minor and (7 * k.value[0] + (9 if minor else 0)) % 12 == tonic
    ]
    return min(choices, key=lambda k: (abs(k.value[0]), abs(k.value[0] - fifths))).name


def convert_pitch_target(target, context, *, to_written=False, mode="notation"):
    """Convert notation pitches once, including pitched grace/trill notes.

    Source generation chooses the direction explicitly.
    Local octave effects name the affected events; their pitches are converted
    while the effects remain available for GP5 engraving.
    """
    from scorelib.score import format_measure_target, parse_measure_target
    from scorelib.techniques import ornament_position

    measure = parse_measure_target(target)
    direction = -1 if to_written else 1
    instrument_shift = context.get("instrument_transpose") or 0
    base_shift = (
        instrument_shift
        + (context.get("clef_octave") or 0)
        - (context.get("capo") or 0)
    )
    if measure.get("key_signature"):
        measure["key_signature"] = transpose_key(
            measure["key_signature"], direction * instrument_shift
        )
    for voice in measure["voices"]:
        for event in voice["events"]:
            octaves = [
                int(e.partition(":")[2])
                for e in event.get("effects", [])
                if e.startswith("ottava:")
            ]
            if len(octaves) > 1 or any(o not in {-24, -12, 12, 24} for o in octaves):
                raise ValueError("Invalid local octave instruction")
            shift = direction * (base_shift + (octaves[0] if octaves else 0))
            for note in event.get("notes", []):
                if "dead" in note.get("effects", []):
                    continue
                note["pitch"] += shift
                if not 0 <= note["pitch"] <= 127:
                    raise ValueError("Transposed pitch is outside MIDI range")
                effects = []
                for effect in note.get("effects", []):
                    parts = effect.split(":")
                    if parts[0] in {"grace", "trill"} and len(parts) > 1:
                        fret, pitch = ornament_position(parts[1])
                        if pitch is not None:
                            if not 0 <= pitch + shift <= 127:
                                raise ValueError(
                                    "Transposed ornament is outside MIDI range"
                                )
                            parts[1] = (
                                f"f{fret}" if fret is not None else ""
                            ) + f"p{pitch + shift}"
                    effects.append(":".join(parts))
                note["effects"] = effects
    return format_measure_target(measure, mode, preserve_playback=True)
