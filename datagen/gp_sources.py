from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote

from shared.m2 import format_measure_target


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
        if getattr(effect, "grace", None) is not None:
            note_effects.append("grace")
        if getattr(effect, "trill", None) is not None:
            note_effects.append("trill")
        if getattr(effect, "tremoloPicking", None) is not None:
            note_effects.append("trem")
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


def _track_rank(track: Any) -> tuple[int, int, int, int]:
    note_count = 0
    event_count = 0
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if _enum_name(getattr(beat, "status", "normal")) != "empty":
                    event_count += 1
                note_count += len(beat.notes)
    string_count = len(track.strings)
    guitar_bonus = 5000 if string_count == 6 else max(0, 3000 - abs(string_count - 6) * 700)
    playable_bonus = 0 if track.isPercussionTrack else 100000
    return playable_bonus + guitar_bonus + note_count * 3 + event_count, note_count, event_count, string_count


def select_target_track(song: Any) -> tuple[int, Any]:
    ranked = [(_track_rank(track), -index, index, track) for index, track in enumerate(song.tracks)]
    if not ranked:
        raise ValueError("Song has no tracks")
    _rank, _negative_index, index, track = max(ranked, key=lambda item: (item[0], item[1]))
    if track.isPercussionTrack:
        raise ValueError("Song has no non-percussion target track")
    return index, track


def song_sequence_payload(song: Any, track_index: int, source_path: Path, source_hash: str) -> dict[str, Any]:
    track = song.tracks[track_index]
    measures = []
    previous_signature = None
    previous_key_signature = None
    counters: Counter[str] = Counter()
    maximum_fret = 0
    note_count = 0
    event_count = 0
    multi_voice_measures = 0
    for measure in track.measures:
        encoded = encode_measure(
            measure,
            previous_signature,
            previous_key_signature,
            initial_tempo=int(getattr(song, "tempo", 120) or 120) if not measures else None,
        )
        previous_signature = encoded["time_signature"]
        previous_key_signature = encoded["key_signature"]
        if len(encoded["voices"]) > 1:
            multi_voice_measures += 1
        for voice in encoded["voices"]:
            event_count += len(voice["events"])
            for event in voice["events"]:
                notes = event.get("notes") or []
                if event.get("status") == "normal" and notes:
                    # MIDI velocity is not visible in GP8's printed output.
                    # Normalize it so source-only playback data cannot leak
                    # into the image-to-sequence target.
                    for note in notes:
                        note["velocity"] = 95
                duration = event["duration"]
                if duration["dotted"] or duration["double_dotted"]:
                    counters["dotted"] += 1
                if duration["tuplet_enters"] != 1 or duration["tuplet_times"] != 1:
                    counters["tuplet"] += 1
                if event["status"] == "rest":
                    counters["rest"] += 1
                if event["status"] == "empty":
                    counters["empty"] += 1
                for effect in event.get("effects", []):
                    counters[str(effect).split(":", 1)[0]] += 1
                for note in notes:
                    note_count += 1
                    if isinstance(note["fret"], int):
                        maximum_fret = max(maximum_fret, note["fret"])
                    for effect in note["effects"]:
                        counters[str(effect).split(":", 1)[0]] += 1
        encoded["targets"] = {
            mode: format_measure_target(encoded, mode)
            for mode in ("tab", "notation", "both")
        }
        measures.append(encoded)
    tuning = [int(string.value) for string in track.strings]
    diversity_tags = (
        "bend", "hammer", "slide", "sl", "ss", "sib", "sia", "sod", "sou",
        "pm", "tie", "dead", "dotted", "tuplet", "rest", "harm", "grace",
        "trill", "trem", "tap", "ghost", "accent", "heavy", "let", "stacc",
        "pick_up", "pick_down", "stroke_up", "stroke_down", "slap", "fade",
        "rasg", "chord", "tempo", "text", "dyn",
    )
    tags = sorted(key for key in diversity_tags if counters[key])
    if multi_voice_measures:
        tags.append("multi_voice")
    return {
        "schema_version": "2.0",
        "source_id": source_hash[:16],
        "sha256": source_hash,
        "source_path": str(source_path.resolve()),
        "source_format": source_path.suffix.lower().lstrip("."),
        "song": {
            "title": str(getattr(song, "title", "") or source_path.stem),
            "artist": str(getattr(song, "artist", "") or ""),
            "tempo_quarter": int(getattr(song, "tempo", 120) or 120),
        },
        "track": {
            "index": track_index,
            "number": int(getattr(track, "number", track_index + 1)),
            "name": str(getattr(track, "name", "") or ""),
            "string_count": len(track.strings),
            "tuning_midi_high_to_low": tuning,
            "capo": int(getattr(track, "offset", 0) or 0),
        },
        "statistics": {
            "measure_count": len(measures),
            "event_count": event_count,
            "note_count": note_count,
            "multi_voice_measure_count": multi_voice_measures,
            "maximum_fret": maximum_fret,
            "technique_counts": dict(sorted(counters.items())),
            "tags": tags,
        },
        "measures": measures,
    }


def parse_song(path: Path) -> tuple[Any, str]:
    import guitarpro

    last_error: Exception | None = None
    for encoding in ("utf-8", "gbk", "cp936", "cp1252", "latin-1"):
        try:
            song = guitarpro.parse(str(path), encoding=encoding)
            _repair_tied_note_values(song)
            return song, encoding
        except Exception as error:  # pragma: no cover - corpus dependent
            last_error = error
    raise RuntimeError(f"Could not parse {path}: {last_error!r}")


def _repair_tied_note_values(song: Any) -> None:
    """Resolve tie frets by identity-safe chronological traversal.

    PyGuitarPro's GP3/4/5 reader uses ``list.index(beat)`` while a beat is
    being parsed. Beats are attrs classes with structural equality, so two
    visually identical tie beats can compare equal and the reader can attach
    the later tie to an older fret. Guitar Pro itself uses the immediately
    preceding note on the same string. Recompute that state explicitly before
    any source label or round-trip metric reads ``note.realValue``.
    """

    for track in getattr(song, "tracks", []):
        previous_by_voice: dict[int, dict[int, int]] = {}
        for measure in getattr(track, "measures", []):
            for voice_index, voice in enumerate(getattr(measure, "voices", [])):
                previous = previous_by_voice.setdefault(voice_index, {})
                for beat in getattr(voice, "beats", []):
                    for note in getattr(beat, "notes", []):
                        note_type = _enum_name(getattr(note, "type", "normal"))
                        string = int(getattr(note, "string", 0) or 0)
                        if note_type == "tie" and string in previous:
                            note.value = previous[string]
                        value = int(getattr(note, "value", -1) or 0)
                        if note_type != "dead" and string > 0 and value >= 0:
                            previous[string] = value


def analyze_source(path: Path) -> dict[str, Any]:
    song, encoding = parse_song(path)
    track_index, _track = select_target_track(song)
    source_hash = sha256(path.read_bytes()).hexdigest()
    payload = song_sequence_payload(song, track_index, path, source_hash)
    payload["source_encoding"] = encoding
    return payload


def prepare_single_track_gp5(path: Path, output: Path, mode: str) -> dict[str, Any]:
    import guitarpro
    from guitarpro import models as gm

    song, encoding = parse_song(path)
    track_index, track = select_target_track(song)
    source_hash = sha256(path.read_bytes()).hexdigest()
    payload = song_sequence_payload(song, track_index, path, source_hash)
    song.tracks = [track]
    track.number = 1
    track.settings = gm.TrackSettings(
        tablature=mode in {"tab", "both"},
        notation=mode in {"notation", "both"},
        diagramsAreBelow=False,
        showRhythm=True,
        forceHorizontal=False,
        forceChannels=False,
        diagramList=True,
        diagramsInScore=True,
        autoLetRing=False,
        autoBrush=False,
        extendRhythmic=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(output), version=(5, 1, 0), encoding="utf-8")
    payload["source_encoding"] = encoding
    payload["prepared_gp5"] = str(output.resolve())
    return payload
