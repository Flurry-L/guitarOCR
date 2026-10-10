"""Read and construct Guitar Pro files for source data generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType
from typing import Any
from urllib.parse import quote

from scorelib.instruments import DEFAULT_PROGRAMS, DEFAULT_TUNINGS
from scorelib.score import write_score_gp5

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


def _duration_dict(duration: Any) -> dict[str, Any]:
    return {
        "value": int(duration.value),
        "dotted": duration.isDotted,
        "double_dotted": False,
        "tuplet_enters": int(duration.tuplet.enters),
        "tuplet_times": int(duration.tuplet.times),
    }


def _bend_token(bend: Any) -> str | None:
    if bend is None:
        return None
    value = int(bend.value)
    if not value:
        points = bend.points
        if points:
            value = max(int(point.value) for point in points) * 25
    bend_type = bend.type.name
    return f"bend:{bend_type}:{value}"


def _harmonic_token(harmonic: Any) -> str | None:
    if harmonic is None:
        return None
    name = harmonic.__class__.__name__.replace("Harmonic", "").replace("Effect", "")
    kind = name.lower()
    if kind == "tapped":
        fret = harmonic.fret
        if fret is not None:
            return f"harm:tapped:{int(fret)}"
    if kind == "artificial":
        pitch = harmonic.pitch
        octave = harmonic.octave
        if pitch is not None and octave is not None:
            return (
                f"harm:artificial:{int(pitch.just)}:"
                f"{int(pitch.accidental)}:{int(octave.value)}"
            )
    return "harm:" + kind


def encode_note(note: Any) -> dict[str, Any]:
    effect = note.effect
    note_type = note.type.name
    note_effects: list[str] = []
    if note_type == "tie":
        note_effects.append("tie")
    if note_type == "dead":
        note_effects.append("dead")
    if effect is not None:
        boolean_effects = (
            ("vibrato", "vib"),
            ("hammer", "hammer"),
            ("ghostNote", "ghost"),
            ("palmMute", "pm"),
            ("staccato", "stacc"),
            ("letRing", "let"),
            ("accentuatedNote", "accent"),
            ("heavyAccentuatedNote", "heavy"),
        )
        for name, token in boolean_effects:
            if getattr(effect, name):
                note_effects.append(token)
        for slide in effect.slides:
            note_effects.append(SLIDE_NAMES[slide.name])
        bend = _bend_token(effect.bend)
        if bend:
            note_effects.append(bend)
        harmonic = _harmonic_token(effect.harmonic)
        if harmonic:
            note_effects.append(harmonic)
        open_pitch = int(note.realValue) - int(note.value)
        grace = effect.grace
        if grace is not None:
            note_effects.append(
                "grace"
                if int(grace.fret) < 0
                else (
                    f"grace:f{int(grace.fret)}p{open_pitch + int(grace.fret)}:{int(grace.duration)}:"
                    f"{_enum_name(grace.transition)}:{int(grace.isOnBeat)}:{int(grace.isDead)}"
                )
            )
        trill = effect.trill
        if trill is not None:
            note_effects.append(
                "trill"
                if int(trill.fret) < 0
                else f"trill:f{int(trill.fret)}p{open_pitch + int(trill.fret)}:{int(trill.duration.value)}"
            )
        tremolo = effect.tremoloPicking
        if tremolo is not None:
            note_effects.append(f"trem:{int(tremolo.duration.value)}")
    fret: int | str = int(note.value)
    if note_type == "dead":
        fret = "x"
    velocity = int(note.velocity)
    # Some legacy GP3/GP4 files use -1 as an unspecified-dynamic sentinel.
    # It is not a playable MIDI velocity and is not a distinct printed symbol.
    if velocity == -1:
        velocity = 95
    elif not 0 <= velocity <= 127:
        raise ValueError(f"Invalid source note velocity: {velocity}")
    return {
        "string": int(note.string),
        "fret": fret,
        "pitch": int(note.realValue),
        "velocity": velocity,
        "swap_accidentals": note.swapAccidentals,
        "effects": list(dict.fromkeys(note_effects)),
    }


def _beat_effects(beat: Any) -> list[str]:
    effect = beat.effect
    if effect is None:
        return []
    values: list[str] = []
    octave = {
        "ottava": 12,
        "ottavaBassa": -12,
        "quindicesima": 24,
        "quindicesimaBassa": -24,
    }.get(beat.octave.name)
    if octave:
        values.append(f"ottava:{octave}")
    pick = effect.pickStroke.name
    if pick == "up":
        values.append("pick_up")
    elif pick == "down":
        values.append("pick_down")
    stroke = effect.stroke
    stroke_direction = stroke.direction.name
    if stroke_direction == "up":
        values.append("stroke_up")
    elif stroke_direction == "down":
        values.append("stroke_down")
    slap = effect.slapEffect.name
    if slap not in {"none", "0"}:
        values.append("slap:" + slap)
    if effect.fadeIn:
        values.append("fade")
    if effect.hasRasgueado:
        values.append("rasg")
    chord = effect.chord
    chord_name = chord.name.strip() if chord is not None else ""
    if chord is not None and chord_name:
        values.append("chord:" + quote(chord_name, safe=""))
    mix = effect.mixTableChange
    tempo = mix.tempo if mix is not None else None
    tempo_value = int(tempo.value) if tempo is not None else 0
    if tempo_value and not mix.hideTempo:
        values.append(f"tempo:{tempo_value}")
    text = beat.text
    text_value = (text or "").strip()
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
    key_signature = header.keySignature.name
    triplet_feel = header.tripletFeel.name
    measure_start = int(measure.start)
    voices = []
    for voice_index, voice in enumerate(measure.voices):
        events = []
        for beat in voice.beats:
            status = beat.status.name
            event = {
                "start": int(beat.start)
                - measure_start,
                "duration": _duration_dict(beat.duration),
                "status": status,
                "notes": [encode_note(note) for note in beat.notes],
                "effects": _beat_effects(beat),
            }
            events.append(event)
        # GP8 can render an empty primary voice as a measure rest. Secondary
        # voices containing only placeholder empty beats are not visible and
        # must not become impossible OCR targets.
        if events and (
            voice_index == 0 or any(event["status"] != "empty" for event in events)
        ):
            voices.append({"voice": voice_index, "events": events})
    bars = []
    if header.isRepeatOpen:
        bars.append("repeat_open")
    repeat_close = int(header.repeatClose)
    if repeat_close > 0:
        bars.append("repeat_close")
    if header.hasDoubleBar:
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
                            effect
                            for effect in event.get("effects", [])
                            if not str(effect).startswith("tempo:")
                        ]
    return {
        "index": int(measure.number) - 1,
        "number": int(measure.number),
        "time_signature": time_signature,
        "print_time_signature": previous_time_signature is None
        or time_signature != previous_time_signature,
        "key_signature": key_signature,
        "print_key_signature": previous_key_signature is None
        or key_signature != previous_key_signature,
        "tempo_quarter": visible_initial_tempo,
        "triplet_feel": triplet_feel if triplet_feel != "none" else None,
        "bars": bars,
        "repeat_count": repeat_close + 1 if repeat_close > 0 else None,
        "alternate_endings": int(header.repeatAlternative) or None,
        "section": header.marker.title.strip() if header.marker is not None else None,
        "direction": _enum_name(header.direction),
        "from_direction": _enum_name(header.fromDirection),
        "voices": voices,
    }


def _tied_value(self, note):
    current = note.beat.voice.measure
    voice_index = next(
        i for i, voice in enumerate(current.voices) if voice is note.beat.voice
    )
    for measure in reversed(current.track.measures):
        voice = measure.voices[voice_index]
        beats = voice.beats
        if measure is current:
            index = next(i for i, beat in enumerate(beats) if beat is note.beat)
            beats = beats[:index]
        for beat in reversed(beats):
            if beat.status.name != "empty":
                for previous in beat.notes:
                    if previous.string == note.string:
                        return previous.value
    return -1


def read_gp(stream, encoding="cp1252"):
    from guitarpro.io import _open

    reader, close = _open(None, stream, "rb", encoding=encoding)
    # PyGuitarPro uses list.index while the current beat is only partly read.
    # Equal notes can then match an earlier beat whose duration differs, making
    # a later tie incorrectly inherit an older pitch. Identity is unambiguous.
    reader.getTiedNoteValue = MethodType(_tied_value, reader)
    try:
        return reader.readSong()
    finally:
        if close:
            reader.close()


def write_targets_gp5(
    targets,
    output,
    *,
    mode="both",
    title="Guitar OCR",
    artist="",
    tuning=None,
    capo=0,
    instrument="guitar",
    midi_program=None,
):
    targets = list(targets)
    if not targets:
        raise ValueError("At least one measure is required")
    result = {
        "title": title,
        "artist": artist,
        "mode": mode,
        "instrument": instrument,
        "tuning_used": list(
            tuning if tuning is not None else DEFAULT_TUNINGS[instrument]
        ),
        "capo": capo,
        "midi_program": midi_program
        if midi_program is not None
        else DEFAULT_PROGRAMS[instrument],
        "records": [
            {"target": t, "measure_number": i, "bar_index": i - 1}
            for i, t in enumerate(targets, 1)
        ],
    }
    return write_score_gp5(result, output)


def targets_to_song(targets, **options):
    with TemporaryDirectory(prefix="scorelib-") as directory:
        output = write_targets_gp5(targets, Path(directory) / "score.gp5", **options)
        return read_gp(str(output), encoding="cp936")
