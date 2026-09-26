"""Regenerate the small, original example score and its expected transcription."""

from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

from shared.pdf import open_pdf

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
    canvas = Canvas(str(root / "demo.pdf"), pagesize=(595, 420), bottomup=0, invariant=1)

    def text_at(point, text, size):
        canvas.setFont("Helvetica", size)
        canvas.drawString(*point, text)

    def line(start, end, width):
        canvas.setLineWidth(width)
        canvas.line(*start, *end)

    text_at((220, 40), "First Steps", 24)
    text_at((215, 65), "GuitarOCR example", 12)
    text_at((55, 95), "Standard tuning    Quarter = 120", 10)
    for system in range(2):
        top = 130 + system * 130
        for string in range(6):
            line((55, top + string * 9), (545, top + string * 9), 0.6)
        text_at((32, top + 25), "TAB", 10)
        if system == 0:
            text_at((64, top + 17), "4", 14)
            text_at((64, top + 35), "4", 14)
        for x in (55, 300, 545):
            line((x, top), (x, top + 45), 0.8)
        for column in range(2):
            number = system * 2 + column
            measure = parse_measure_target(targets[number])
            text_at((58 + column * 245, top - 10), str(number + 1), 8)
            for beat, event in enumerate(measure["voices"][0]["events"]):
                note = event["notes"][0]
                x = 92 + column * 245 + beat * 50
                y = top + (note["string"] - 1) * 9
                canvas.setFillColorRGB(1, 1, 1)
                canvas.rect(x - 4, y - 6, 9, 11, stroke=0, fill=1)
                canvas.setFillColorRGB(0, 0, 0)
                text_at((x - 3, y + 3), str(note["fret"]), 11)
                line((x, top + 55), (x, top + 80), 0.8)
    canvas.save()
    with open_pdf(root / "demo.pdf") as document:
        document[0].render(180, grayscale=False).save(root / "demo.png")



if __name__ == "__main__":
    generate(Path(__file__).parent)
