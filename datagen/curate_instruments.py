"""Validate instrument targets and reuse the corresponding native renders."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from datagen.gp_sources import analyze_source
from datagen.native_alignment import note_differences
from shared.constraints import validate_measure_target


def relabel(job):
    root, row = job
    old = json.loads((root / "labels" / f"{row['source_id']}.json").read_text())
    try:
        label = analyze_source(Path(row["source_path"]), old["track"]["index"])
        for key in ("family", "modes", "prepared_gp5"):
            if key in old:
                label[key] = old[key]
        for measure in label["measures"]:
            for mode in row["modes"]:
                _parsed, errors = validate_measure_target(
                    measure["targets"][mode],
                    mode,
                    string_count=label["track"]["string_count"],
                )
                if errors:
                    raise ValueError(f"Measure {measure['index']}/{mode}: {errors}")
        paths = list(
            (
                root
                / "native-export"
                / "documents"
                / f"notation-{row['source_id']}"
                / "tracks"
            ).glob("*/official-score.json")
        )
        if len(paths) != 1:
            raise ValueError(
                "Expected one native notation export for label verification"
            )
        differences = note_differences(label, json.loads(paths[0].read_text()))
        if differences:
            raise ValueError(
                f"Native import changed {len(differences)} note events: {differences[:3]}"
            )
        return row, label, None
    except Exception as error:
        return row, None, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    root, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "labels").mkdir(parents=True)
    rows = json.loads((root / "source_catalog.json").read_text())["sources"]
    kept, excluded = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for index, (row, label, error) in enumerate(
            pool.map(relabel, ((root, r) for r in rows)), 1
        ):
            if error:
                excluded.append({**row, "reason": error})
            else:
                kept.append(row)
                (output / "labels" / f"{row['source_id']}.json").write_text(
                    json.dumps(label, ensure_ascii=False) + "\n"
                )
            if index % 100 == 0:
                print(f"Validated {index}/{len(rows)}", flush=True)
    for folder in ("pdf", "layout", "native-export", "prepared"):
        (output / folder).symlink_to(root / folder, target_is_directory=True)
    (output / "source_catalog.json").write_text(
        json.dumps({"sources": kept}, indent=2) + "\n"
    )
    (output / "excluded_sources.json").write_text(json.dumps(excluded, indent=2) + "\n")
    print({"kept": len(kept), "excluded": len(excluded)}, flush=True)


if __name__ == "__main__":
    main()
