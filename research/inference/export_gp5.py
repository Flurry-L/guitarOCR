"""Export an OCR stage result without loading recognition models."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

from scorelib.gp5.writer import GP5ReadbackError
from scorelib.gp5.writer import GP5TimingError
from scorelib.gp5.writer import write_targets_gp5
from scorelib.gp5.score import write_score_gp5
from research.common.artifacts import read_result
from research.common.artifacts import write_result


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
    source = read_result(recognition, 'measure_ocr')
    if source.get("review_measures") and not allow_unreviewed:
        raise ValueError(f"请先校对这些小节再导出：{source['review_measures']}")
    targets = [row["target"] for row in source["records"]]
    try:
        grouped = any('part_id' in row for row in source['records'])
        gp5 = write_score_gp5(source, output.resolve() / score_filename(source['title'], fallback_name)) if grouped else write_targets_gp5(
            targets, output.resolve() / score_filename(source["title"], fallback_name), mode=source["mode"],
            title=source["title"], artist=source["artist"],
            tuning=source["tuning_used"], capo=source["capo"],
            instrument=source.get("instrument", "guitar"), midi_program=source.get("midi_program"),
        )
    except (GP5ReadbackError, GP5TimingError) as error:
        from research.inference.measures.result import save_recognition

        for index in error.measures:
            source['records'][index - 1].update(
                needs_review=True,
                export_errors=['导出时无法保持本小节的音符，请核对延音、音高和时值'],
            )
        save_recognition(recognition.parent, source)
        return write_result(output, 'gp5_export', recognition=str(recognition.resolve()),
                            gp5=None, encoding_report=None, status='needs_review',
                            review_measures=[source['records'][i - 1]['measure_number'] for i in error.measures])
    from scorelib.musicxml import write_musicxml
    from scorelib.score_document import score_document

    projections = {}
    try:
        musicxml = gp5.with_suffix('.musicxml')
        if source.get('musicxml') and Path(source['musicxml']).is_file():
            if Path(source['musicxml']).resolve() != musicxml.resolve():
                shutil.copyfile(source['musicxml'], musicxml)
        else:
            write_musicxml(score_document(source), musicxml)
        projections['musicxml'] = str(musicxml)
    except ValueError as error:
        projections['musicxml_error'] = str(error)
    return write_result(
        output, 'gp5_export', recognition=str(recognition.resolve()),
        gp5=str(gp5), encoding_report=str(gp5.with_name(gp5.name + ".encoding.json")),
        **projections,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a measure OCR stage result to GP5.")
    parser.add_argument("--recognition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true", help="Export measures still flagged for review")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
