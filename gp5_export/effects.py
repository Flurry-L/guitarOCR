from __future__ import annotations

from typing import Any
from urllib.parse import unquote
from shared.techniques import ornament_position


def _duration(value: dict[str, Any], gm: Any) -> Any:
    duration = gm.Duration(value=int(value["value"]))
    duration.isDotted = bool(value.get("dotted"))
    duration.isDoubleDotted = bool(value.get("double_dotted"))
    enters = int(value.get("tuplet_enters", 1))
    times = int(value.get("tuplet_times", 1))
    duration.tuplet = gm.Tuplet(enters=enters, times=times)
    return duration


def _enum_member(enum_type: Any, name: str | None, default: Any) -> Any:
    if not name:
        return default
    normalized = str(name).replace(" ", "").lower()
    for key, value in vars(enum_type).items():
        if key.startswith("_"):
            continue
        value_name = str(getattr(value, "name", key)).replace(" ", "").lower()
        if key.replace(" ", "").lower() == normalized or value_name == normalized:
            return value
    return default


def _bend(effect_text: str, gm: Any) -> Any:
    _prefix, _separator, tail = effect_text.partition(":")
    kind_text, _separator, value_text = tail.partition(":")
    value = int(value_text or 100)
    kind = _enum_member(gm.BendType, kind_text, gm.BendType.bend)
    point_value = max(0, round(value / 25))
    contours = {
        "bend": [(0, 0), (6, point_value), (12, point_value)],
        "bendRelease": [(0, 0), (6, point_value), (12, 0)],
        "bendReleaseBend": [(0, 0), (4, point_value), (8, 0), (12, point_value)],
        "prebend": [(0, point_value), (12, point_value)],
        "prebendRelease": [(0, point_value), (6, point_value), (12, 0)],
    }
    points = [gm.BendPoint(position=x, value=y) for x, y in contours.get(kind.name, contours["bend"])]
    return gm.BendEffect(type=kind, value=value, points=points)


def _apply_note_effects(note: Any, effects: list[str], gm: Any) -> None:
    slide_map = {
        "sl": "legatoSlideTo",
        "ss": "shiftSlideTo",
        "sib": "intoFromBelow",
        "sia": "intoFromAbove",
        "sod": "outDownwards",
        "sou": "outUpwards",
    }
    for effect in effects:
        if effect == "tie":
            note.type = gm.NoteType.tie
        elif effect == "dead":
            note.type = gm.NoteType.dead
        elif effect == "vib":
            note.effect.vibrato = True
        elif effect == "hammer":
            note.effect.hammer = True
        elif effect == "ghost":
            note.effect.ghostNote = True
        elif effect == "pm":
            note.effect.palmMute = True
        elif effect == "stacc":
            note.effect.staccato = True
        elif effect == "let":
            note.effect.letRing = True
        elif effect == "tap":
            # Legacy M2 used a note-level alias. GP3-5 stores tapping as a
            # beat slap-effect, which the score format serializes as
            # ``slap:tapping``.
            note.beat.effect.slapEffect = gm.SlapEffect.tapping
        elif effect == "accent":
            note.effect.accentuatedNote = True
        elif effect == "heavy":
            note.effect.heavyAccentuatedNote = True
        elif effect in slide_map:
            slide = _enum_member(gm.SlideType, slide_map[effect], None)
            if slide is not None:
                note.effect.slides.append(slide)
        elif effect.startswith("slide:"):
            slide = _enum_member(gm.SlideType, effect.partition(":")[2], None)
            if slide is not None:
                note.effect.slides.append(slide)
        elif effect.startswith("bend:"):
            note.effect.bend = _bend(effect, gm)
        elif effect == "harm:natural":
            note.effect.harmonic = gm.NaturalHarmonic()
        elif effect.startswith("harm:"):
            harmonic_parts = effect.split(":")
            harmonic_name = harmonic_parts[1]
            harmonic_types = {
                "artificial": gm.ArtificialHarmonic,
                "pinch": gm.PinchHarmonic,
                "tapped": gm.TappedHarmonic,
                "semi": gm.SemiHarmonic,
            }
            harmonic_type = harmonic_types.get(harmonic_name)
            if harmonic_type is not None:
                if harmonic_name == "tapped":
                    # GP5 requires this byte. Older M2 used only
                    # ``harm:tapped``; a musically conventional +12 fret
                    # fallback keeps those targets writable.
                    harmonic_fret = (
                        int(harmonic_parts[2])
                        if len(harmonic_parts) >= 3
                        else min(255, max(0, int(note.value) + 12))
                    )
                    note.effect.harmonic = gm.TappedHarmonic(fret=harmonic_fret)
                elif harmonic_name == "artificial" and len(harmonic_parts) >= 5:
                    octave_value = int(harmonic_parts[4])
                    octave = next(
                        (
                            value for key, value in vars(gm.Octave).items()
                            if not key.startswith("_")
                            and getattr(value, "value", None) == octave_value
                        ),
                        gm.Octave.ottava,
                    )
                    note.effect.harmonic = gm.ArtificialHarmonic(
                        pitch=gm.PitchClass(
                            int(harmonic_parts[2]), int(harmonic_parts[3])
                        ),
                        octave=octave,
                    )
                else:
                    note.effect.harmonic = harmonic_type()
        elif effect == "grace":
            note.effect.grace = gm.GraceEffect(fret=max(0, int(note.value)))
        elif effect == "trill":
            note.effect.trill = gm.TrillEffect(
                fret=max(0, int(note.value) + 1), duration=gm.Duration(value=16)
            )
        elif effect == "trem":
            note.effect.tremoloPicking = gm.TremoloPickingEffect(
                duration=gm.Duration(value=16)
            )
        elif effect.startswith("trem:"):
            note.effect.tremoloPicking = gm.TremoloPickingEffect(duration=gm.Duration(value=int(effect.split(":")[1])))
        elif effect.startswith(("grace:", "trill:")):
            parts = effect.split(":")
            fret, pitch = ornament_position(parts[1])
            if parts[1] == "x":
                fret = max(0, int(note.value))
            elif fret is None:
                tuning = note.beat.voice.measure.track.strings
                fret = pitch - next(s.value for s in tuning if s.number == note.string)
            if not 0 <= fret <= 36:
                raise ValueError("Ornament pitch cannot be played on the assigned string")
            if parts[0] == "trill":
                note.effect.trill = gm.TrillEffect(fret=fret, duration=gm.Duration(value=int(parts[2]) if len(parts) == 3 else 32))
            else:
                if len(parts) == 5:
                    parts.insert(2, "32")
                note.effect.grace = gm.GraceEffect(fret=fret, duration=int(parts[2]),
                    transition=_enum_member(gm.GraceEffectTransition, parts[3], gm.GraceEffectTransition.none),
                    isOnBeat=parts[4] == "1", isDead=parts[5] == "1")


def _note(
    note_data: dict[str, Any], beat: Any, tuning: list[int],
    position: tuple[int, int], gm: Any, default_velocity: int = 95,
) -> Any:
    string, fret = position
    note = gm.Note(
        beat=beat,
        string=string,
        value=fret,
        velocity=(
            int(default_velocity)
            if int(note_data.get("velocity", 95)) == 95
            else int(note_data["velocity"])
        ),
        swapAccidentals=bool(note_data.get("swap_accidentals")),
        type=gm.NoteType.normal,
    )
    effects = list(note_data.get("effects") or [])
    if note_data.get("fret") == "x" and "dead" not in effects:
        effects.append("dead")
    _apply_note_effects(note, effects, gm)
    return note


def _apply_beat_effects(beat: Any, effects: list[str], string_count: int, gm: Any) -> None:
    for effect in effects:
        if effect == "pick_up":
            beat.effect.pickStroke = gm.BeatStrokeDirection.up
        elif effect == "pick_down":
            beat.effect.pickStroke = gm.BeatStrokeDirection.down
        elif effect == "stroke_up":
            beat.effect.stroke = gm.BeatStroke(direction=gm.BeatStrokeDirection.up, value=8)
        elif effect == "stroke_down":
            beat.effect.stroke = gm.BeatStroke(direction=gm.BeatStrokeDirection.down, value=8)
        elif effect.startswith("slap:"):
            beat.effect.slapEffect = _enum_member(
                gm.SlapEffect, effect.partition(":")[2], gm.SlapEffect.none
            )
        elif effect == "fade":
            beat.effect.fadeIn = True
        elif effect == "rasg":
            beat.effect.hasRasgueado = True
        elif effect.startswith("chord:"):
            name = unquote(effect.partition(":")[2])
            beat.effect.chord = gm.Chord(
                length=string_count,
                name=name,
                firstFret=1,
                strings=[-1] * string_count,
                omissions=[False] * 7,
                show=True,
                newFormat=True,
            )
        elif effect.startswith("tempo:"):
            tempo = int(effect.partition(":")[2])
            beat.effect.mixTableChange = gm.MixTableChange(
                tempo=gm.MixTableItem(value=tempo, duration=0), hideTempo=False
            )
        elif effect.startswith("text:"):
            beat.text = unquote(effect.partition(":")[2])
