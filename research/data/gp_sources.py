from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scorelib.m2 import format_measure_target
from scorelib.instruments import track_instrument
from scorelib.percussion import visible_drum_key


from scorelib.guitarpro import encode_measure, _enum_name


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
    source_capo = int(getattr(track, "offset", 0) or 0)
    normalized_offset = track_instrument(track) == "pitched" and source_capo != 0
    if normalized_offset:
        # Some piano/voice imports use the GP capo field as a pitch offset.
        # Melodic labels store sounding pitches, including grace/trill notes;
        # normalize the backing strings before encoding without editing the source.
        track = deepcopy(track)
        for string in track.strings:
            string.value += source_capo
        track.offset = 0
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
        if track.isPercussionTrack:
            # A percussion clef has no pitched key signature or accidentals.
            encoded["print_key_signature"] = False
            encoded["key_signature"] = None
            for voice in encoded["voices"]:
                for event in voice["events"]:
                    for note in event.get("notes", []):
                        note["swap_accidentals"] = False
                        note["pitch"] = visible_drum_key(int(note["pitch"]))
                    pitches = [note["pitch"] for note in event.get("notes", [])]
                    if len(set(pitches)) != len(pitches):
                        raise ValueError("Overlaid drum symbols cannot establish the source's separate MIDI notes")
        previous_signature = encoded["time_signature"]
        previous_key_signature = encoded["key_signature"]
        if len(encoded["voices"]) > 1:
            multi_voice_measures += 1
        for voice in encoded["voices"]:
            event_count += len(voice["events"])
            for event in voice["events"]:
                notes = event.get("notes") or []
                if event.get("status") == "normal" and notes:
                    # GP8 can print dynamic marks, but source per-note velocity
                    # is not a reliable label for their location/visibility.
                    # This dataset does not yet supervise printed dynamics;
                    # do not require a velocity value on every note.
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
            "instrument": track_instrument(track),
            "midi_program": int(track.channel.instrument),
            "index": track_index,
            "number": int(getattr(track, "number", track_index + 1)),
            "name": str(getattr(track, "name", "") or ""),
            "string_count": len(track.strings),
            "tuning_midi_high_to_low": tuning,
            "capo": int(getattr(track, "offset", 0) or 0),
            **({"source_capo": source_capo} if normalized_offset else {}),
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


def analyze_source(path: Path, track_index: int | None = None) -> dict[str, Any]:
    song, encoding = parse_song(path)
    if track_index is None:
        track_index, _track = select_target_track(song)
    source_hash = sha256(path.read_bytes()).hexdigest()
    payload = song_sequence_payload(song, track_index, path, source_hash)
    payload["source_encoding"] = encoding
    return payload


def prepare_single_track_gp5(path: Path, output: Path, mode: str, track_index: int | None = None) -> dict[str, Any]:
    import guitarpro
    from guitarpro import models as gm

    song, encoding = parse_song(path)
    if track_index is None:
        track_index, track = select_target_track(song)
    else:
        track = song.tracks[track_index]
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
    write_text_encoding(output, 'utf-8')
    payload["source_encoding"] = encoding
    payload["prepared_gp5"] = str(output.resolve())
    return payload


def write_text_encoding(path: Path, encoding: str, metadata_source: Path | None = None) -> None:
    """Declare the bytes' encoding for GP's legacy importer, preserving overrides."""
    destination = path.with_name(path.name + '.metadata.json')
    source = metadata_source or destination
    metadata = json.loads(source.read_text(encoding='utf-8')) if source.is_file() else {}
    metadata['text_encoding'] = encoding
    destination.write_text(json.dumps(metadata, ensure_ascii=False) + '\n', encoding='utf-8')
