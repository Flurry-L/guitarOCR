"""PyGuitarPro source adapter to scorelib's shared music representation."""
from __future__ import annotations
from typing import Any
from urllib.parse import quote
from scorelib.m2 import parse_measure_target, full_measure_rest_target

SLIDE_NAMES = {
    "legatoSlideTo": "sl",
    "shiftSlideTo": "ss",
    "intoFromBelow": "sib",
    "intoFromAbove": "sia",
    "outDownwards": "sod",
    "outUpwards": "sou",
}


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _truthy_effect(effect: Any, *names: str) -> bool:
    return any(bool(getattr(effect, name, False)) for name in names)


def _duration_dict(duration: Any) -> dict[str, Any]:
    tuplet = getattr(duration, "tuplet", None)
    enters = int(getattr(tuplet, "enters", 1) or 1)
    times = int(getattr(tuplet, "times", 1) or 1)
    return {
        "value": int(getattr(duration, "value", 4) or 4),
        "dotted": bool(getattr(duration, "isDotted", False)),
        "double_dotted": bool(getattr(duration, "isDoubleDotted", False)),
        "tuplet_enters": enters,
        "tuplet_times": times,
    }


def _bend_token(bend: Any) -> str | None:
    if bend is None:
        return None
    value = int(getattr(bend, "value", 0) or 0)
    if not value:
        points = list(getattr(bend, "points", []) or [])
        if points:
            value = max(int(getattr(point, "value", 0) or 0) for point in points) * 25
    bend_type = _enum_name(getattr(bend, "type", "bend"))
    return f"bend:{bend_type}:{value}"


def _harmonic_token(harmonic: Any) -> str | None:
    if harmonic is None:
        return None
    name = harmonic.__class__.__name__.replace("Harmonic", "").replace("Effect", "")
    kind = name.lower() or "natural"
    if kind == "tapped":
        fret = getattr(harmonic, "fret", None)
        if fret is not None:
            return f"harm:tapped:{int(fret)}"
    if kind == "artificial":
        pitch = getattr(harmonic, "pitch", None)
        octave = getattr(harmonic, "octave", None)
        if pitch is not None and octave is not None:
            return (
                f"harm:artificial:{int(pitch.just)}:"
                f"{int(pitch.accidental)}:{int(octave.value)}"
            )
    return "harm:" + kind


def encode_note(note: Any) -> dict[str, Any]:
    effect = getattr(note, "effect", None)
    note_type = _enum_name(getattr(note, "type", "normal"))
    note_effects: list[str] = []
    if note_type == "tie":
        note_effects.append("tie")
    if note_type == "dead":
        note_effects.append("dead")
    if effect is not None:
        boolean_effects = (
            (("vibrato",), "vib"),
            (("hammer",), "hammer"),
            (("ghostNote", "ghost"), "ghost"),
            (("palmMute", "palm_mute"), "pm"),
            (("staccato",), "stacc"),
            (("letRing", "let_ring"), "let"),
            (("leftHandTapped",), "tap"),
            (("accentuatedNote",), "accent"),
            (("heavyAccentuatedNote",), "heavy"),
        )
        for names, token in boolean_effects:
            if _truthy_effect(effect, *names):
                note_effects.append(token)
        for slide in list(getattr(effect, "slides", []) or []):
            slide_name = _enum_name(slide)
            note_effects.append(SLIDE_NAMES.get(slide_name, "slide:" + slide_name))
        bend = _bend_token(getattr(effect, "bend", None))
        if bend:
            note_effects.append(bend)
        harmonic = _harmonic_token(getattr(effect, "harmonic", None))
        if harmonic:
            note_effects.append(harmonic)
        open_pitch = int(note.realValue) - int(note.value)
        grace = getattr(effect, "grace", None)
        if grace is not None:
            note_effects.append("grace" if int(grace.fret) < 0 else (
                f"grace:f{int(grace.fret)}p{open_pitch + int(grace.fret)}:{int(grace.duration)}:"
                f"{_enum_name(grace.transition)}:{int(grace.isOnBeat)}:{int(grace.isDead)}"
            ))
        trill = getattr(effect, "trill", None)
        if trill is not None:
            note_effects.append("trill" if int(trill.fret) < 0 else f"trill:f{int(trill.fret)}p{open_pitch + int(trill.fret)}:{int(trill.duration.value)}")
        tremolo = getattr(effect, "tremoloPicking", None)
        if tremolo is not None:
            note_effects.append(f"trem:{int(tremolo.duration.value)}")
    fret: int | str = int(getattr(note, "value", 0) or 0)
    if note_type == "dead":
        fret = "x"
    velocity = int(getattr(note, "velocity", 95) or 95)
    # Some legacy GP3/GP4 files use -1 as an unspecified-dynamic sentinel.
    # It is not a playable MIDI velocity and is not a distinct printed symbol.
    if not 0 <= velocity <= 127:
        velocity = 95
    return {
        "string": int(getattr(note, "string", 1) or 1),
        "fret": fret,
        "pitch": int(getattr(note, "realValue", 0) or 0),
        "velocity": velocity,
        "swap_accidentals": bool(getattr(note, "swapAccidentals", False)),
        "effects": list(dict.fromkeys(note_effects)),
    }


def _beat_effects(beat: Any) -> list[str]:
    effect = getattr(beat, "effect", None)
    if effect is None:
        return []
    values: list[str] = []
    octave = {"ottava": 12, "ottavaBassa": -12,
              "quindicesima": 24, "quindicesimaBassa": -24}.get(
                  _enum_name(getattr(beat, "octave", None)))
    if octave:
        values.append(f"ottava:{octave}")
    pick = _enum_name(getattr(effect, "pickStroke", "none"))
    if pick == "up":
        values.append("pick_up")
    elif pick == "down":
        values.append("pick_down")
    stroke = getattr(effect, "stroke", None)
    stroke_direction = _enum_name(getattr(stroke, "direction", "none"))
    if stroke_direction == "up":
        values.append("stroke_up")
    elif stroke_direction == "down":
        values.append("stroke_down")
    slap = _enum_name(getattr(effect, "slapEffect", "none"))
    if slap not in {"none", "0"}:
        values.append("slap:" + slap)
    if bool(getattr(effect, "fadeIn", False)):
        values.append("fade")
    if bool(getattr(effect, "hasRasgueado", False)):
        values.append("rasg")
    chord = getattr(effect, "chord", None)
    chord_name = str(getattr(chord, "name", "") or "").strip()
    if chord is not None and chord_name:
        values.append("chord:" + quote(chord_name, safe=""))
    mix = getattr(effect, "mixTableChange", None)
    tempo = getattr(mix, "tempo", None)
    tempo_value = int(getattr(tempo, "value", 0) or 0)
    if tempo_value and not bool(getattr(mix, "hideTempo", False)):
        values.append(f"tempo:{tempo_value}")
    text = getattr(beat, "text", None)
    text_value = str(getattr(text, "value", text) or "").strip()
    if text_value:
        values.append("text:" + quote(text_value, safe=""))
    return values


def encode_measure(
    measure: Any,
    previous_time_signature: str | None,
    previous_key_signature: str | None,
    *,
    initial_tempo: int | None = None,
) -> dict[str, Any]:
    header = measure.header
    numerator = int(header.timeSignature.numerator)
    denominator = int(header.timeSignature.denominator.value)
    time_signature = f"{numerator}/{denominator}"
    key_signature = _enum_name(getattr(header, "keySignature", "CMajor"))
    triplet_feel = _enum_name(getattr(header, "tripletFeel", "none"))
    measure_start = int(getattr(measure, "start", getattr(header, "start", 0)) or 0)
    voices = []
    for voice_index, voice in enumerate(measure.voices):
        events = []
        for beat in voice.beats:
            status = _enum_name(getattr(beat, "status", "normal"))
            event = {
                "start": int(getattr(beat, "start", measure_start) or measure_start) - measure_start,
                "duration": _duration_dict(beat.duration),
                "status": status,
                "notes": [encode_note(note) for note in beat.notes],
                "effects": _beat_effects(beat),
            }
            events.append(event)
        # GP8 can render an empty primary voice as a measure rest. Secondary
        # voices containing only placeholder empty beats are not visible and
        # must not become impossible OCR targets.
        if events and (voice_index == 0 or any(event["status"] != "empty" for event in events)):
            voices.append({"voice": voice_index, "events": events})
    if not voices:
        # An entirely empty source bar is printed as a full-measure rest by
        # Guitar Pro. An empty "M2 |" target violates the OCR output contract.
        voices = parse_measure_target(full_measure_rest_target((numerator, denominator)))["voices"]
    bars = []
    if bool(getattr(header, "isRepeatOpen", False)):
        bars.append("repeat_open")
    repeat_close = int(getattr(header, "repeatClose", 0) or 0)
    if repeat_close > 0:
        bars.append("repeat_close")
    if bool(getattr(header, "hasDoubleBar", False)):
        bars.append("double")
    visible_initial_tempo = initial_tempo
    if initial_tempo is not None:
        promoted_tempo = None
        for voice in voices:
            for event in voice["events"]:
                if int(event["start"]) != 0:
                    continue
                for effect in event.get("effects", []):
                    if str(effect).startswith("tempo:"):
                        promoted_tempo = int(str(effect).partition(":")[2])
                        break
                if promoted_tempo is not None:
                    break
            if promoted_tempo is not None:
                break
        if promoted_tempo is not None:
            visible_initial_tempo = promoted_tempo
            for voice in voices:
                for event in voice["events"]:
                    if int(event["start"]) == 0:
                        event["effects"] = [
                            effect for effect in event.get("effects", [])
                            if not str(effect).startswith("tempo:")
                        ]
    return {
        "index": int(measure.number) - 1,
        "number": int(measure.number),
        "time_signature": time_signature,
        "print_time_signature": previous_time_signature is None or time_signature != previous_time_signature,
        "key_signature": key_signature,
        "print_key_signature": previous_key_signature is None or key_signature != previous_key_signature,
        "tempo_quarter": visible_initial_tempo,
        "triplet_feel": triplet_feel if triplet_feel != "none" else None,
        "bars": bars,
        "repeat_count": repeat_close + 1 if repeat_close > 0 else None,
        "alternate_endings": int(getattr(header, "repeatAlternative", 0) or 0) or None,
        "section": str(getattr(getattr(header, "marker", None), "title", "") or "").strip() or None,
        "direction": _enum_name(getattr(header, "direction", None)),
        "from_direction": _enum_name(getattr(header, "fromDirection", None)),
        "voices": voices,
    }
