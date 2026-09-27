"""Assemble candidate total scores only from structurally aligned source parts."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import guitarpro

from datagen.gp_sources import parse_song
from shared.instruments import track_instrument


def timeline(song, track):
    tempo = int(song.tempo)
    rows = []
    for measure in track.measures:
        header = measure.header
        changes = []
        for voice in measure.voices:
            for beat in voice.beats:
                start = beat.start - header.start
                if start < 0 or start + beat.duration.time > header.length:
                    raise ValueError("Source has notes outside its measure duration")
                change = beat.effect.mixTableChange
                if change and change.tempo:
                    changes.append((start, change.tempo.value))
        changes = sorted(set(changes))
        rows.append(
            (
                header.timeSignature.numerator,
                header.timeSignature.denominator.value,
                header.keySignature.name,
                tempo,
                tuple(changes),
                header.isRepeatOpen,
                header.repeatClose,
                header.repeatAlternative,
                header.tripletFeel.name,
            )
        )
        if changes:
            tempo = int(changes[-1][1])
    return rows


def build(roots, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    families = defaultdict(list)
    for root in roots:
        for row in json.loads((root / "source_catalog.json").read_text())["sources"]:
            families[row["family"]].append((root, row))
    accepted, rejected = [], []
    for family, entries in sorted(families.items()):
        if len(entries) < 2:
            continue
        try:
            if len({row["split"] for _root, row in entries}) != 1:
                raise ValueError("Related parts cross dataset splits")
            selected = []
            for root, row in entries:
                label = json.loads(
                    (root / "labels" / f"{row['source_id']}.json").read_text()
                )
                song, _encoding = parse_song(Path(row["source_path"]))
                track = song.tracks[label["track"]["index"]]
                selected.append((song, track, label, row))
            reference = timeline(selected[0][0], selected[0][1])
            if any(
                timeline(song, track) != reference
                for song, track, _label, _row in selected[1:]
            ):
                raise ValueError(
                    "Measure count, meter, key, tempo or repeat structure differs"
                )
            if (
                len(
                    {track_instrument(track) for _song, track, _label, _row in selected}
                )
                < 2
            ):
                raise ValueError(
                    "Same-instrument variants do not establish distinct ensemble parts"
                )
            if (
                sum(track.isPercussionTrack for _song, track, _label, _row in selected)
                > 1
            ):
                raise ValueError(
                    "Multiple drum parts need explicit kit/channel selection"
                )
            result = deepcopy(selected[0][0])
            result.tracks = []
            channels = iter([i for i in range(16) if i != 9])
            parts = []
            for index, (_song, track, label, row) in enumerate(selected, 1):
                part = deepcopy(track)
                part.song = result
                part.number = index
                channel = 9 if part.isPercussionTrack else next(channels)
                part.channel.channel = part.channel.effectChannel = channel
                for measure, header in zip(
                    part.measures, result.measureHeaders, strict=True
                ):
                    measure.header = header
                    measure.track = part
                result.tracks.append(part)
                parts.append(
                    {
                        "id": f"part-{index}",
                        "name": track.name,
                        "instrument": track_instrument(track),
                        "source_id": row["source_id"],
                        "source_path": row["source_path"],
                        "midi_program": track.channel.instrument,
                        "measures": label["measures"],
                    }
                )
            destination = output / "scores" / f"{family}.gp5"
            destination.parent.mkdir(exist_ok=True)
            guitarpro.write(
                result, str(destination), version=(5, 1, 0), encoding="cp936"
            )
            restored = guitarpro.parse(str(destination), encoding="cp936")
            if len(restored.tracks) != len(parts) or any(
                timeline(restored, track) != reference for track in restored.tracks
            ):
                raise ValueError("Round-trip changed the aligned timeline")
            payload = {
                "family": family,
                "split": entries[0][1]["split"],
                "source": str(destination.resolve()),
                "title": result.title,
                "artist": result.artist,
                "parts": parts,
                "alignment": "same source family; matching measure count, meter, key, tempo and repeats",
                "review_required": True,
            }
            (output / "scores" / f"{family}.json").write_text(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            accepted.append(
                {k: v for k, v in payload.items() if k != "parts"}
                | {"parts": len(parts)}
            )
        except Exception as error:
            rejected.append({"family": family, "reason": str(error)})
    report = {
        "candidates": accepted,
        "rejected": rejected,
        "counts": dict(Counter(row["split"] for row in accepted)),
        "scope": "Candidate GP files; alignment checks are not manual musical validation and do not establish total-score OCR support",
    }
    (output / "catalog.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "counts": report["counts"],
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
