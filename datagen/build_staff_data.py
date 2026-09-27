"""Teach the information adapter to read visible instrument labels and TAB line counts."""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datagen.native.annotations import printed_text
from datagen.training_samples import dataset_entry
from document_info.prompts import STAFF_PROMPT
from shared.pdf import open_pdf


def visible_instrument(name, mode, percussion):
    """Classify only printed names; GM programs are not visible in a score."""
    if percussion:
        return "drums"
    name = printed_text(name or "")
    if re.search(r"double\s+bass|contrabass|contrebass|kontrabass|低音提琴", name):
        return "pitched"
    if re.search(
        r"\bbass\w*|\bbs\b|\bbasse\b|\bbas\b|\bbaixo\b|\bbajo\b|\bbasso\b|贝斯|貝斯|低音吉他|ベース|\bбас\w*",
        name,
    ):
        return "bass"
    if re.search(
        r"\bguit\w*|\bgtr\b|\bgitarr\w*|\bgitara\b|\bkitara\b|\bkytara\b|吉他|ギター|\bгитар\w*",
        name,
    ):
        return "guitar"
    return "pitched" if mode == "notation" else None


def sample(job):
    root, output, row, mode = job
    paths = sorted(
        (
            root
            / "native-export"
            / "documents"
            / f"{mode}-{row['source_id']}"
            / "tracks"
        ).glob("*/layout.json")
    )
    if len(paths) != 1:
        raise ValueError(f"Expected one completed export: {row['source_id']}/{mode}")
    path = paths[0]
    layout = json.loads(path.read_text())
    score = json.loads(path.with_name("official-score.json").read_text())
    label = json.loads((root / "labels" / f"{row['source_id']}.json").read_text())
    system = layout["systems"][0]
    track = score["tracks"][0]
    percussion = label["track"].get("instrument") == "drums"
    page_number = int(system["page"]) - 1
    with open_pdf(path.with_name("score.pdf")) as pdf:
        page = pdf[page_number]
        image = page.render(180)
        # Match the runtime's first-measure vertical crop and include the label
        # to its left. A full row preserves the visible TAB line count.
        _x, y, _w, h = system["measure_boxes"][0]["bbox_mm"]
        scale = 180 / 25.4
        top, bottom = (
            max(0, int(y * scale) - 12),
            min(image.height, int((y + h) * scale) + 13),
        )
        printed = page.text((0, top * 72 / 180, page.width, bottom * 72 / 180))
        visible_name = track.get(system.get("track_label", ""))
        # Native PDFs may extract "el.bs." as "el.\nb\ns.". Ignore
        # extraction whitespace when checking that the name is printed.
        if not visible_name or "".join(
            printed_text(visible_name).split()
        ) not in "".join(printed_text(printed).split()):
            visible_name = None
        instrument = visible_instrument(visible_name, mode, percussion)
        destination = output / "images" / f"{row['source_id']}-{mode}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.crop((0, top, image.width, bottom)).convert("RGB").save(destination)
    target = {
        "instrument": instrument,
        "string_count": label["track"]["string_count"]
        if mode in {"tab", "both"}
        else None,
    }
    return row["split"], {
        "messages": [
            {"role": "user", "content": "<image>" + STAFF_PROMPT},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ],
        "images": [str(destination.resolve())],
        "provenance": {
            **row,
            "mode": mode,
            "kind": "staff",
            "visible_name": visible_name,
        },
    }


def build(roots, output, workers):
    jobs, families = [], {}
    for root in roots:
        for row in json.loads((root / "source_catalog.json").read_text())["sources"]:
            if families.setdefault(row["family"], row["split"]) != row["split"]:
                raise ValueError("Song family occurs in multiple splits")
            jobs.extend(
                (root, output, row, mode)
                for mode in row.get("modes", ["tab", "both", "notation"])
            )
    samples = {split: [] for split in ("train", "validation", "test")}
    with ProcessPoolExecutor(workers) as pool:
        for index, (split, value) in enumerate(pool.map(sample, jobs), 1):
            samples[split].append(value)
            if index % 100 == 0:
                print(f"Staff crops {index}/{len(jobs)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, values in samples.items():
        name = f"staff_{split}"
        (output / f"{name}.json").write_text(
            json.dumps(values, ensure_ascii=False) + "\n"
        )
        info[name] = dataset_entry(f"{name}.json")
    (output / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print({split: len(values) for split, values in samples.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    build(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
