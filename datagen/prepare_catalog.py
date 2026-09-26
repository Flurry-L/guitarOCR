"""Prepare all display modes from an explicitly grouped source catalog."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

from datagen.files import _write_json
from datagen.gp_sources import analyze_source, parse_song, prepare_single_track_gp5
from datagen.inventory import source_catalog


def _prepare(arguments):
    row, output, modes = arguments
    source = Path(row["source_path"])
    song, _ = parse_song(source)
    label = analyze_source(source)
    for mode in modes:
        temporary = output / "prepared" / mode / f"{row['family']}.gp5"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if len(song.tracks) == 1 and source.suffix.lower() == ".gp5":
            # The native renderer applies display_mode. Preserve the original
            # GP5 encoding and uncommon effects instead of round-tripping it.
            shutil.copy2(source, temporary)
        else:
            label = prepare_single_track_gp5(source, temporary, mode)
        destination = temporary.with_name(f"{label['source_id']}.gp5")
        temporary.replace(destination)
        metadata = source.with_name(source.name + ".metadata.json")
        if metadata.is_file():
            shutil.copy2(metadata, destination.with_name(destination.name + ".metadata.json"))
        label["prepared_gp5"] = str(destination.resolve())
    label["family"] = row["family"]
    _write_json(output / "labels" / f"{label['source_id']}.json", label)
    return {
        "source_id": label["source_id"],
        "family": row["family"],
        "split": row["split"],
        "source_path": row["source_path"],
    }


def prepare(catalog: Path, output: Path, modes: list[str], workers: int = 4):
    rows = json.loads(catalog.read_text(encoding="utf-8"))["sources"]
    if not rows or len({row["family"] for row in rows}) != len(rows):
        raise ValueError("Provide one source per family")
    if not modes or set(modes) - {"tab", "notation", "both"}:
        raise ValueError("Invalid display modes")
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        if Path(row["family"]).name != row["family"] or row["family"] in {".", ".."}:
            raise ValueError("Unsafe family name")
    output = output.resolve()
    if (output / "source_catalog.json").exists():
        raise FileExistsError(output / "source_catalog.json")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prepared = []
        for index, row in enumerate(
            pool.map(_prepare, ((row, output, modes) for row in rows)), 1
        ):
            prepared.append(row)
            if index % 25 == 0 or index == len(rows):
                print(f"Prepared {index}/{len(rows)} sources", flush=True)
    _write_json(
        output / "source_catalog.json", {"schema_version": "1.0", "sources": prepared}
    )
    source_catalog(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", action="append", choices=("tab", "notation", "both"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(
        args.catalog,
        args.output,
        args.mode or ["tab", "notation", "both"],
        args.workers,
    )


if __name__ == "__main__":
    main()
