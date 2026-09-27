"""Inventory GP5 instrument coverage before selecting song-disjoint training data."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re

from datagen.gp_sources import parse_song


def examine(path):
    try:
        song, _ = parse_song(path)
        track = song.tracks[0]
        notes = [
            n
            for m in track.measures
            for v in m.voices
            for b in v.beats
            for n in b.notes
        ]
        family = re.search(r"_([0-9a-fA-F]{8})(?:_\d+)?\.gp5$", path.name)
        return dict(
            path=str(path),
            family=family[1].lower()
            if family
            else sha256(path.read_bytes()).hexdigest(),
            category=path.parent.name,
            program=track.channel.instrument,
            percussion=track.isPercussionTrack,
            strings=len(track.strings),
            tuning=[s.value for s in track.strings],
            measures=len(track.measures),
            notes=len(notes),
            max_chord=max(
                (
                    len(b.notes)
                    for m in track.measures
                    for v in m.voices
                    for b in v.beats
                ),
                default=0,
            ),
            voices=max(
                (sum(bool(v.beats) for v in m.voices) for m in track.measures),
                default=0,
            ),
            pitches=sorted({n.realValue for n in notes}),
            title=song.title,
            artist=song.artist,
        )
    except Exception as error:
        return dict(path=str(path), error=repr(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum files per instrument; otherwise inspect every file",
    )
    args = parser.parse_args()
    paths = []
    for directory in sorted(args.corpus.resolve().iterdir()):
        if directory.is_dir():
            group = sorted(
                directory.glob("*.gp5"), key=lambda p: sha256(p.name.encode()).digest()
            )
            paths.extend(group[: args.limit] if args.limit else group)
    counts, errors = Counter(), Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        ProcessPoolExecutor(args.workers) as pool,
        args.output.open("w", encoding="utf-8") as stream,
    ):
        for index, row in enumerate(pool.map(examine, paths, chunksize=16), 1):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if "error" in row:
                errors[row["error"].split(":")[0]] += 1
            else:
                counts[f"{row['category']}/{row['strings']}/{row['percussion']}"] += 1
            if index % 1000 == 0:
                stream.flush()
                print(f"Audited {index}/{len(paths)}", flush=True)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(dict(counts=counts, errors=errors), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
