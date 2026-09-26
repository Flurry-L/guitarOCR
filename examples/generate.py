"""Regenerate the small, original example score and its expected transcription."""

from pathlib import Path

import pymupdf

from gp5_export.writer import write_targets_gp5
from shared.m2 import format_measure_target, parse_measure_target
from shared.score_text import display_score_text

TARGETS = [
    "M2 time=4/4 | V0{@0:q:s1f0 @960:q:s1f1 @1920:q:s1f3 @2880:q:s1f0}",
    "M2 time=4/4 | V0{@0:q:s2f0 @960:q:s2f1 @1920:q:s2f3 @2880:q:s2f0}",
    "M2 time=4/4 | V0{@0:q:s3f0 @960:q:s3f2 @1920:q:s3f0 @2880:q:s2f1}",
    "M2 time=4/4 | V0{@0:q:s1f3 @960:q:s1f1 @1920:q:s1f0 @2880:q:s2f3}",
]


def generate(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    targets = list(TARGETS)
    for index in range(1, len(targets)):
        targets[index] = targets[index].replace(" time=4/4", "")
    first = parse_measure_target(targets[0])
    first["tempo_quarter"] = 120
    targets[0] = format_measure_target(first, "tab", preserve_playback=True)
    (root / "expected.m2").write_text("\n".join(targets) + "\n", encoding="utf-8")
    (root / "expected.score.txt").write_text(display_score_text("\n".join(targets) + "\n"), encoding="utf-8")
    write_targets_gp5(
        targets,
        root / "expected.gp5",
        mode="tab",
        title="First Steps",
        artist="GuitarOCR example",
        tuning=[64, 59, 55, 50, 45, 40],
        capo=0,
    )
    with pymupdf.open() as document:
        page = document.new_page(width=595, height=420)
        page.insert_text((220, 40), "First Steps", fontsize=24)
        page.insert_text((215, 65), "GuitarOCR example", fontsize=12)
        page.insert_text((55, 95), "Standard tuning    Quarter = 120", fontsize=10)
        for system in range(2):
            top = 130 + system * 130
            for string in range(6):
                page.draw_line(
                    (55, top + string * 9), (545, top + string * 9), width=0.6
                )
            page.insert_text((32, top + 25), "TAB", fontsize=10)
            if system == 0:
                page.insert_text((64, top + 17), "4", fontsize=14)
                page.insert_text((64, top + 35), "4", fontsize=14)
            for x in (55, 300, 545):
                page.draw_line((x, top), (x, top + 45), width=0.8)
            for column in range(2):
                number = system * 2 + column
                measure = parse_measure_target(targets[number])
                page.insert_text(
                    (58 + column * 245, top - 10), str(number + 1), fontsize=8
                )
                for beat, event in enumerate(measure["voices"][0]["events"]):
                    note = event["notes"][0]
                    x = 92 + column * 245 + beat * 50
                    y = top + (note["string"] - 1) * 9
                    page.draw_rect(
                        pymupdf.Rect(x - 4, y - 6, x + 5, y + 5),
                        color=None,
                        fill=(1, 1, 1),
                        overlay=True,
                    )
                    page.insert_text((x - 3, y + 3), str(note["fret"]), fontsize=11)
                    page.draw_line((x, top + 55), (x, top + 80), width=0.8)
        document.save(root / "demo.pdf")
        page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5)).save(root / "demo.png")


if __name__ == "__main__":
    generate(Path(__file__).parent)
