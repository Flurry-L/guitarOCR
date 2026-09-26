from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from shared.m2 import parse_measure_target, full_measure_rest_target
from shared.score_text import model_score_text, display_error
from shared.tuning import DEFAULT_TUNING
from gp5_export.fingering import _assign_positions, _tie_reservation_note_ids, _plan_notation_voice_positions
from gp5_export.effects import _duration, _enum_member, _note, _apply_beat_effects


def _display_settings(mode: str, gm: Any) -> Any:
    return gm.TrackSettings(
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


def targets_to_song(
    targets: Iterable[str],
    *,
    mode: str = "both",
    title: str = "Guitar OCR",
    artist: str = "",
    tuning: Iterable[int] = DEFAULT_TUNING,
    capo: int = 0,
) -> Any:
    from guitarpro import models as gm

    if mode not in {"tab", "notation", "both"}:
        raise ValueError(f"Unsupported display mode: {mode}")
    tuning_values = [int(value) for value in tuning]
    try:
        measures = [parse_measure_target(model_score_text(target)) for target in targets if target.strip()]
    except ValueError as error:
        raise ValueError(display_error(str(error))) from None
    if not measures:
        raise ValueError("At least one measure is required")
    if any(int(v["voice"]) not in {0, 1} for m in measures for v in m["voices"]):
        raise ValueError("GP5 supports only V0 and V1; refusing to discard additional voices")
    if any(len({v["voice"] for v in m["voices"]}) != len(m["voices"]) for m in measures):
        raise ValueError("Duplicate voices would lose notes in GP5")

    song = gm.Song(title=title, artist=artist)
    song.measureHeaders = []
    song.tracks = []
    first_tempo = next(
        (int(measure["tempo_quarter"]) for measure in measures if measure.get("tempo_quarter")),
        120,
    )
    song.tempo = first_tempo
    track = gm.Track(song, number=1, name="Guitar OCR", measures=[], strings=[])
    track.offset = int(capo)
    track.settings = _display_settings(mode, gm)
    track.channel.instrument = 25
    track.strings = [
        gm.GuitarString(number=index, value=value)
        for index, value in enumerate(tuning_values, start=1)
    ]
    song.tracks.append(track)

    current_time = "4/4"
    current_key = gm.KeySignature.CMajor
    start = gm.Duration.quarterTime
    previous_positions: dict[int, dict[int, tuple[int, int]]] = {0: {}, 1: {}}
    reserved_strings: dict[int, dict[int, int]] = {0: {}, 1: {}}
    current_velocity: dict[int, int] = {0: 95, 1: 95}
    reservation_note_ids = _tie_reservation_note_ids(measures)
    notation_position_plans = (
        {
            voice_index: _plan_notation_voice_positions(
                measures, voice_index, tuning_values
            )
            for voice_index in range(2)
        }
        if mode == "notation" else {}
    )
    for index, measure_data in enumerate(measures, start=1):
        if measure_data.get("time_signature"):
            current_time = str(measure_data["time_signature"])
        numerator_text, _slash, denominator_text = current_time.partition("/")
        numerator = int(numerator_text or 4)
        denominator = int(denominator_text or 4)
        if measure_data.get("key_signature"):
            current_key = _enum_member(
                gm.KeySignature, str(measure_data["key_signature"]), current_key
            )
        header = gm.MeasureHeader(number=index, start=start)
        header.timeSignature = gm.TimeSignature(
            numerator=numerator, denominator=gm.Duration(value=denominator)
        )
        header.keySignature = current_key
        header.hasDoubleBar = "double" in measure_data["bars"]
        header.isRepeatOpen = "repeat_open" in measure_data["bars"]
        if "repeat_close" in measure_data["bars"]:
            header.repeatClose = max(1, int(measure_data.get("repeat_count") or 2) - 1)
        header.repeatAlternative = int(measure_data.get("alternate_endings") or 0)
        if measure_data.get("section"):
            header.marker = gm.Marker(title=str(measure_data["section"]))
        header.tripletFeel = _enum_member(
            gm.TripletFeel, measure_data.get("triplet_feel"), gm.TripletFeel.none
        )
        header.direction = (
            gm.DirectionSign(str(measure_data["direction"]))
            if measure_data.get("direction") else None
        )
        header.fromDirection = (
            gm.DirectionSign(str(measure_data["from_direction"]))
            if measure_data.get("from_direction") else None
        )
        song.measureHeaders.append(header)

        measure = gm.Measure(track, header, voices=[])
        voices_by_index = {
            int(voice["voice"]): voice for voice in measure_data.get("voices", [])
        }
        for voice_index in range(2):
            voice = gm.Voice(measure, beats=[])
            voice_data = voices_by_index.get(voice_index)
            if voice_data is not None:
                cursor = 0
                for event in voice_data["events"]:
                    event_start = int(event.get("start", 0))
                    if event_start < cursor:
                        raise ValueError(f"Overlapping events in measure {index} V{voice_index}; GP5 cannot preserve them")
                    if event_start > cursor:
                        # GP5 serializes durations, not Beat.start. Materialize
                        # an explicit M2 gap so saving cannot shift later notes.
                        rests = parse_measure_target(full_measure_rest_target((event_start - cursor, 3840)))["voices"][0]["events"]
                        for rest in rests:
                            duration = _duration(rest["duration"], gm)
                            voice.beats.append(gm.Beat(voice=voice, start=start + cursor,
                                                      duration=duration, status=gm.BeatStatus.rest))
                            cursor += duration.time
                        if cursor != event_start:
                            raise ValueError(f"Unrepresentable event gap in measure {index} V{voice_index}")
                    for effect in event.get("effects") or []:
                        if str(effect).startswith("dyn:"):
                            current_velocity[voice_index] = int(
                                str(effect).partition(":")[2]
                            )
                    status = str(event.get("status", "normal"))
                    beat = gm.Beat(
                        voice=voice,
                        duration=_duration(event["duration"], gm),
                        start=start + int(event.get("start", 0)),
                        status={
                            "empty": gm.BeatStatus.empty,
                            "rest": gm.BeatStatus.rest,
                        }.get(status, gm.BeatStatus.normal),
                    )
                    note_values = list(event.get("notes", []))
                    planned_positions = [
                        notation_position_plans.get(voice_index, {}).get(id(note))
                        for note in note_values
                    ]
                    if all(position is not None for position in planned_positions):
                        positions = [
                            position for position in planned_positions
                            if position is not None
                        ]
                    else:
                        positions = _assign_positions(
                            note_values,
                            tuning_values,
                            previous_positions[voice_index],
                            reserved_strings[voice_index],
                        )
                    beat.notes = [
                        _note(
                            note_data,
                            beat,
                            tuning_values,
                            position,
                            gm,
                            current_velocity[voice_index],
                        )
                        for note_data, position in zip(note_values, positions)
                    ]
                    for note_data, position in zip(note_values, positions):
                        if "dead" in (note_data.get("effects") or []):
                            continue
                        string, fret = position
                        pitch = int(
                            note_data.get("pitch", tuning_values[string - 1] + fret)
                        )
                        previous_positions[voice_index][string] = (pitch, fret)
                        is_tie = "tie" in (note_data.get("effects") or [])
                        continues = id(note_data) in reservation_note_ids.get(
                            voice_index, set()
                        )
                        if is_tie and not continues:
                            reserved_strings[voice_index].pop(string, None)
                        if continues:
                            reserved_strings[voice_index][string] = pitch
                    _apply_beat_effects(
                        beat, list(event.get("effects") or []), len(tuning_values), gm
                    )
                    voice.beats.append(beat)
                    cursor = event_start + beat.duration.time
            measure.voices.append(voice)
        track.measures.append(measure)
        start += numerator * gm.Duration.quarterTime * 4 // denominator

    song.key = song.measureHeaders[0].keySignature
    return song


def write_targets_gp5(
    targets: Iterable[str],
    output: str | Path,
    *,
    mode: str = "both",
    title: str = "Guitar OCR",
    artist: str = "",
    tuning: Iterable[int] = DEFAULT_TUNING,
    capo: int = 0,
) -> Path:
    import guitarpro

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    song = targets_to_song(
        targets, mode=mode, title=title, artist=artist, tuning=tuning, capo=capo
    )
    replacements: list[dict[str, str]] = []

    def cp936_text(value: str, field: str) -> str:
        encoded = value.encode("cp936", errors="replace").decode("cp936")
        if encoded != value:
            replacements.append({"field": field, "original": value, "written": encoded})
        return encoded

    song.title = cp936_text(song.title, "title")
    song.artist = cp936_text(song.artist, "artist")
    for measure_index, measure in enumerate(song.tracks[0].measures, start=1):
        if measure.header.marker is not None:
            marker = measure.header.marker
            marker.title = cp936_text(marker.title, f"measure {measure_index} marker")
        for voice_index, voice in enumerate(measure.voices, start=1):
            for beat_index, beat in enumerate(voice.beats, start=1):
                field = f"measure {measure_index} voice {voice_index} beat {beat_index}"
                if beat.text:
                    beat.text = cp936_text(beat.text, f"{field} text")
                if beat.effect.chord is not None:
                    chord = beat.effect.chord
                    chord.name = cp936_text(chord.name, f"{field} chord")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        guitarpro.write(song, str(temporary), version=(5, 1, 0), encoding="cp936")
        guitarpro.parse(str(temporary), encoding="cp936")
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    report_path = output_path.with_name(f"{output_path.name}.encoding.json")
    report_path.write_text(
        json.dumps({"encoding": "cp936", "replacements": replacements}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert score text (one measure per line) to GP5.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("tab", "notation", "both"), default="both")
    parser.add_argument("--title", default="Guitar OCR")
    parser.add_argument("--artist", default="")
    parser.add_argument("--tuning", default=",".join(str(value) for value in DEFAULT_TUNING))
    parser.add_argument("--capo", type=int, default=0)
    args = parser.parse_args()
    targets = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines()]
    tuning = [int(value) for value in args.tuning.split(",") if value.strip()]
    write_targets_gp5(
        targets,
        args.output,
        mode=args.mode,
        title=args.title,
        artist=args.artist,
        tuning=tuning,
        capo=args.capo,
    )


if __name__ == "__main__":
    main()
