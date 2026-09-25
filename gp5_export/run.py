"""Export an OCR stage result without loading recognition models."""

from __future__ import annotations

import argparse
from pathlib import Path

from gp5_export.writer import write_targets_gp5
from shared.artifacts import read_result, write_result


def run(recognition: Path, output: Path, *, allow_unreviewed: bool = False) -> Path:
    source = read_result(recognition, "measure_ocr")
    if source.get("review_measures") and not allow_unreviewed:
        raise ValueError(f"Review substituted measures before export: {source['review_measures']}")
    targets = Path(source["m2"]).read_text(encoding="utf-8").splitlines()
    gp5 = write_targets_gp5(
        targets, output.resolve() / "PRE.gp5", mode=source["mode"],
        title=source["title"], artist=source["artist"],
        tuning=source["tuning_used"], capo=source["capo"],
    )
    return write_result(
        output, "gp5_export", recognition=str(recognition.resolve()),
        gp5=str(gp5), encoding_report=str(gp5.with_name(gp5.name + ".encoding.json")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a measure OCR stage result to GP5.")
    parser.add_argument("--recognition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true", help="Explicitly export substituted rests")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
