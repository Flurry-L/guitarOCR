"""Create an original, model-free two-track project for editor evaluation.

This is authored synthetic score data, never an OCR accuracy demonstration.
Run from the checkout: python -m scripts.create_demo --output output/demo
"""

import argparse
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from measure_ocr.result import save_recognition
from pipeline.archive import export_project
from pipeline.workspace import Workspace
from shared.artifacts import read_result, write_result


TITLE = "Harbor Light - synthetic editor demo"
TUNINGS = ([64, 59, 55, 50, 45, 40], [43, 38, 33, 28])
# Original four-quarter-note patterns. No third-party score or recording.
PATTERNS = (
    [(1, [0, 3, 5, 3]), (2, [1, 3, 5, 3]), (1, [5, 7, 8, 7]), (2, [5, 3, 1, 0]),
     (1, [0, 3, 7, 5]), (2, [1, 5, 3, 1]), (1, [3, 5, 7, 3]), (1, [0, 0, 0, 0])],
    [(4, [0, 0, 3, 3]), (3, [3, 3, 5, 5]), (4, [5, 5, 3, 3]), (3, [2, 2, 0, 0]),
     (4, [0, 0, 3, 3]), (3, [3, 3, 5, 5]), (4, [3, 3, 5, 5]), (4, [0, 0, 0, 0])],
)


def create_demo(output: Path):
    output = output.resolve()
    workspace = Workspace(output / "sessions", device="cpu")
    sid = uuid4().hex
    root = workspace.directory(sid)
    root.mkdir(parents=True)
    source = root / "Harbor-Light-synthetic.png"
    page = Image.new("RGB", (1600, 1240), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=18)
    draw.text((80, 42), "HARBOR LIGHT", fill="#172330", font=ImageFont.load_default(size=44))
    draw.text((82, 106), "Original synthetic score | Guitar + Bass | 4/4 | Quarter note = 96", fill="#334155", font=font)
    draw.text((82, 148), "PRESET EDITOR DEMO - NOT AN OCR RESULT. All notes below are quarter notes.", fill="#475569", font=small)
    boxes, profiles, targets = [], [], []
    for system in range(2):
        for part in range(2):
            top = 240 + system * 460 + part * 205
            name = "Guitar" if part == 0 else "Bass"
            draw.text((80, top - 34), name, fill="#172330", font=font)
            strings = len(TUNINGS[part])
            for string in range(strings):
                draw.line((150, top + string * 22, 1510, top + string * 22), fill="#64748b", width=2)
            for local in range(4):
                bar = system * 4 + local
                x = 150 + local * 340
                draw.line((x, top, x, top + (strings - 1) * 22), fill="#172330", width=2)
                draw.text((x + 12, top - 27), str(bar + 1), fill="#64748b", font=small)
                string, frets = PATTERNS[part][bar]
                for beat, fret in enumerate(frets):
                    nx, ny = x + 62 + beat * 72, top + (string - 1) * 22
                    draw.rectangle((nx - 8, ny - 13, nx + 21, ny + 17), fill="white")
                    draw.text((nx - 3, ny - 12), str(fret), fill="#172330", font=font)
                    # Printed rhythm must stay inside each crop; a page-header
                    # instruction alone would leave isolated TAB timing ambiguous.
                    bottom = top + (strings - 1) * 22
                    draw.line((nx + 5, bottom + 14, nx + 5, bottom + 46), fill="#172330", width=2)
                number = len(boxes) + 1
                boxes.append({"kind": "measure", "page": 1, "bbox": [x, top - 10, 340, strings * 22 + 54]})
                profiles.append({"measure_number": number, "part_id": f"part-{part + 1}",
                                 "part_name": name, "staff_id": "staff-1", "bar_index": bar,
                                 "instrument": "guitar" if part == 0 else "bass",
                                 "midi_program": 25 if part == 0 else 33,
                                 "tuning": list(TUNINGS[part]), "capo": 0, "tuning_explicit": True})
                targets.append("M2 time=4/4 tempo=96 | V0{" + " ".join(
                    f"@{beat * 960}:q:s{string}f{fret}" for beat, fret in enumerate(frets)) + "}")
            draw.line((1510, top, 1510, top + (strings - 1) * 22), fill="#172330", width=3)
    draw.text((80, 1150), "Created for GuitarOCR. Synthetic source and preset may be used, modified and redistributed (CC0).", fill="#475569", font=small)
    page.save(source)
    workspace.create(sid, [source])
    workspace.boxes(sid, boxes, "tab")
    workspace.metadata(sid, {"title": TITLE, "artist": "GuitarOCR original example (CC0)",
                             "tempo_quarter": 96, "capo": 0, "tuning_used": list(TUNINGS[0])})
    state = workspace.load(sid)
    layout = read_result(Path(state["layout"]), "layout")
    info = read_result(Path(state["info"]), "document_info")
    parts = [{"id": f"part-{i + 1}", "name": "Guitar" if i == 0 else "Bass",
              "instrument": "guitar" if i == 0 else "bass", "midi_program": 25 if i == 0 else 33,
              "tuning_used": list(TUNINGS[i]), "capo": 0, "transpose": None,
              "document_metadata": {"tempo_quarter": 96}} for i in range(2)]
    info.update(parts=parts, measure_profiles=profiles)
    write_result(Path(state["info"]).parent, "document_info", **{
        k: v for k, v in info.items() if k not in {"stage", "schema_version"}})
    records = [{**row, **profile, "target": target, "reviewed": False,
                "needs_review": True, "fallback_reason": [], "manually_edited": False}
               for row, profile, target in zip(layout["records"], profiles, targets)]
    result = {"mode": "tab", "title": TITLE, "artist": info["artist"], "instrument": "guitar",
              "midi_program": 25, "tuning_used": list(TUNINGS[0]), "capo": 0,
              "document_metadata": {"tempo_quarter": 96}, "parts": parts, "records": records,
              "layout": state["layout"], "info": state["info"],
              "provenance": {"kind": "synthetic-preset", "ocr_executed": False, "license": "CC0-1.0"}}
    state["recognition"] = str(save_recognition(workspace.output(sid, "synthetic"), result))
    state["revision"] += 1
    workspace.store(state)
    destination = output / "Harbor-Light-synthetic-project.zip"
    export_project(workspace, sid, destination)
    workspace.close()
    return sid, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    args = parser.parse_args()
    sid, archive = create_demo(args.output)
    print(f"Created synthetic preset (no OCR): {sid}\nProject ZIP: {archive}")
    print(f"Run: python -m webapp.app --edit-only --device cpu --output {args.output / 'sessions'}")


if __name__ == "__main__":
    main()
