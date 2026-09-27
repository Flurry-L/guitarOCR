"""Export an OCR stage result without loading recognition models."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from gp5_export.writer import write_targets_gp5
from shared.artifacts import read_result, write_result


def score_filename(title, fallback="乐谱"):
    name = str(title or "").strip()
    if not name or name.lower() in {"untitled", "未命名乐谱"}:
        name = Path(fallback).stem
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", name).strip(" .")
    if name.lower().endswith(".gp5"):
        name = name[:-4]
    # Leave room for the extension and encoding report on all supported systems.
    name = name.encode("utf-8")[:160].decode("utf-8", errors="ignore").rstrip(" .") or "乐谱"
    if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name):
        name = "_" + name
    return name + ".gp5"


def run(recognition: Path, output: Path, *, allow_unreviewed: bool = False, fallback_name="乐谱") -> Path:
    source = read_result(recognition, "measure_ocr")
    if source.get("review_measures") and not allow_unreviewed:
        raise ValueError(f"请先校对这些小节再导出：{source['review_measures']}")
    if len({(row.get("part_id", "part-1"), row.get("staff_id", "staff-1")) for row in source["records"]}) > 1:
        raise ValueError("Multi-part GP5 export is not supported yet; the separate parts remain in score.json")
    targets = [row["target"] for row in source["records"]]
    gp5 = write_targets_gp5(
        targets, output.resolve() / score_filename(source["title"], fallback_name), mode=source["mode"],
        title=source["title"], artist=source["artist"],
        tuning=source["tuning_used"], capo=source["capo"],
        instrument=source.get("instrument", "guitar"), midi_program=source.get("midi_program"),
    )
    return write_result(
        output, "gp5_export", recognition=str(recognition.resolve()),
        gp5=str(gp5), encoding_report=str(gp5.with_name(gp5.name + ".encoding.json")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a measure OCR stage result to GP5.")
    parser.add_argument("--recognition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true", help="Export measures still flagged for review")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
