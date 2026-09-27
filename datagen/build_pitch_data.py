"""Build pitch-context training data from native GP8 geometry and visible text."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from datagen.native.annotations import (
    header_target,
    measure_boxes,
    printed_text,
    tempo_target,
)
from datagen.native_alignment import note_differences
from datagen.training_samples import dataset_entry
from document_info.prompts import (
    CLEF_PROMPT,
    HEADER_PROMPT,
    TEMPO_PROMPT,
    TRANSPOSITION_PROMPT,
)
from measure_ocr.prompts import recognition_prompt
from shared.constraints import validate_measure_target
from shared.crops import crop_measure, crop_region
from shared.layout_labels import TYPED_CATEGORIES
from shared.m2 import format_history_context, format_measure_target
from shared.pdf import open_pdf
from shared.pitch_context import apply_pitch_regions

OCTAVE_GLYPHS = {"\ue511": 12, "\ue51c": -12, "\ue515": 24, "\ue51d": -24}
OCTAVE_NAMES = {12: "8va", -12: "8vb", 24: "15ma", -24: "15mb"}
NATIVE_OCTAVES = {None: 0, "8va": 12, "8vb": -12, "15ma": 24, "15mb": -24}
INFO_PROMPTS = {"clef": CLEF_PROMPT, "transposition": TRANSPOSITION_PROMPT,
                "header": HEADER_PROMPT, "tempo": TEMPO_PROMPT}


def visible_clef(page, region, bar):
    """BarView also exposes the TAB symbol as a clef element in mixed mode."""
    x, y, w, h = region["bbox_mm"]
    printed = page.text((x * 72 / 25.4, y * 72 / 25.4,
                         (x + w) * 72 / 25.4, (y + h) * 72 / 25.4))
    if any(glyph in printed for glyph in ("\ue06d", "\ue06e")):
        return {"clef": "tab", "clef_octave": 0}
    clef = "percussion" if bar["clef"] == "Neutral" else bar["clef"]
    if clef not in {"G2", "F4", "C3", "C4", "percussion"}:
        raise ValueError(f"Unmapped native clef: {clef}")
    return {"clef": clef, "clef_octave": NATIVE_OCTAVES[bar.get("ottavia_name")]}


def _chat(prompt, target, image, provenance):
    return {"messages": [{"role": "user", "content": "<image>" + prompt},
                         {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))
                          if isinstance(target, dict) else target}],
            "images": [str(image)], "provenance": provenance}


def _text_box(page, text):
    """Use native PDF character positions, never a guessed header rectangle."""
    words = page.words()
    target = "".join(text.lower().split())
    for start in range(len(words)):
        found = ""
        for end in range(start, min(len(words), start + 18)):
            found += "".join(words[end][4].lower().split())
            if not target.startswith(found):
                break
            if found == target:
                group = words[start:end + 1]
                return [min(w[0] for w in group), min(w[1] for w in group),
                        max(w[2] for w in group), max(w[3] for w in group)]
    return None


def _source(job):
    root, output, row, native_export = job
    label_path = root / "labels" / (row["source_id"] + ".json")
    label = json.loads(label_path.read_text())
    pages_out, infos, measures = [], [], []
    for mode in row["modes"]:
        paths = list((native_export / "documents" / (mode + "-" + row["source_id"]) / "tracks").glob("*/layout.json"))
        if len(paths) != 1:
            raise ValueError(f"Missing native export: {row['source_id']}/{mode}")
        path = paths[0]
        layout, score = json.loads(path.read_text()), json.loads(path.with_name("official-score.json").read_text())
        native_track = score["tracks"][0]
        native_bars = native_track["staves"][0]["measures"]
        if native_track["view_transposition_offset"] != row["expected_native_transpose"]:
            raise ValueError(f"Native transposition differs: {row['source_id']}/{mode}")
        if differences := note_differences(label, score):
            raise ValueError(f"Native note mismatch: {row['source_id']}: {differences[:2]}")
        boxes = measure_boxes(layout)
        if len(boxes) != len(label["measures"]):
            raise ValueError("Native measure count changed")
        provenance = {**row, "mode": mode}
        records, predictions = [], []
        system_of = {box["measure_index"]: system["system_index"]
                     for system in layout["systems"] for box in system["measure_boxes"]}
        with open_pdf(path.with_name("score.pdf")) as pdf:
            for page_index, page in enumerate(pdf, 1):
                image = page.render(180)
                name = f"{row['source_id']}-{mode}-{page_index}.png"
                image_path = output / "layout/images" / name
                image.save(image_path, compress_level=3)
                annotations = []

                def add_box(kind, box):
                    x, y, w, h = box
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(image.width, x + w), min(image.height, y + h)
                    if x1 <= x0 or y1 <= y0:
                        raise ValueError("Empty native box")
                    annotations.append({"label": kind, "bbox": [x0, y0, x1 - x0, y1 - y0]})

                def info(kind, box, target):
                    x, y, w, h = box
                    crop = output / "info/images" / f"{name[:-4]}-{len(infos)}-{kind}.png"
                    crop_region(image, [x, y, x + w, y + h], 6 if kind != "header" else 0).save(crop)
                    infos.append(_chat(INFO_PROMPTS[kind], target, crop, {**provenance, "kind": kind}))
                    predictions.append({"kind": kind, "page": page_index, "bbox": box,
                                        "image": str(crop), "parsed": target})

                for index, box in sorted(boxes.items()):
                    if box["page"] != page_index:
                        continue
                    pixel_box = [v * 180 / 25.4 for v in box["bbox_mm"]]
                    add_box("measure_" + mode, pixel_box)
                    crop = output / "measure/images" / f"{row['source_id']}-{mode}-{index}.png"
                    crop_measure(image, [v * 180 / 25.4 for v in box["bbox_mm"]]).save(crop, compress_level=3)
                    records.append({"measure_number": index + 1, "measure_index": index, "page": page_index,
                                    "system_index": system_of[index], "bbox": pixel_box, "image": str(crop)})
                for region in layout["pitch_regions"]:
                    if region["page"] != page_index:
                        continue
                    bbox = [v * 180 / 25.4 for v in region["bbox_mm"]]
                    if region["kind"] == "clef":
                        bar = native_bars[region["measure_index"]]
                        add_box("clef_region", bbox)
                        info("clef", bbox, visible_clef(page, region, bar))
                    elif region["type_code"] == 23:
                        x, y, w, h = region["bbox_mm"]
                        printed = page.text((x * 72 / 25.4, y * 72 / 25.4, (x + w) * 72 / 25.4, (y + h) * 72 / 25.4))
                        shifts = {shift for glyph, shift in OCTAVE_GLYPHS.items() if glyph in printed}
                        shift = next(iter(shifts)) if len(shifts) == 1 else None
                        add_box("transposition_region", bbox)
                        info("transposition", bbox, {"kind": "ottava", "semitones": shift,
                                                     "capo": None, "text": OCTAVE_NAMES.get(shift)})
                if page_index == 1:
                    instruction = row.get("instruction")
                    if instruction:
                        box = _text_box(page, instruction)
                        if box is None:
                            raise ValueError(f"Pitch instruction is not visible: {row['source_id']}/{mode}")
                        x0, y0, x1, y1 = [v * 180 / 72 for v in box]
                        bbox = [x0, y0, x1 - x0, y1 - y0]
                        add_box("transposition_region", bbox)
                        info("transposition", bbox, {"kind": "instrument", "semitones": row["expected_native_transpose"],
                                                     "capo": None, "text": instruction})
                    bottom = min(r["bbox"][1] for r in records if r["page"] == 1)
                    target = header_target(score)
                    visible = page.text((0, 0, page.width, bottom * 72 / 180))
                    if all(printed_text(v) in printed_text(visible) for v in target.values() if v):
                        info("header", [0, 0, image.width, bottom], target)
                for tempo in layout["tempo_indications"]:
                    if tempo["page"] != page_index:
                        continue
                    bbox = [v * 180 / 25.4 for v in tempo["bbox_mm"]]
                    add_box("tempo_region", bbox)
                    if target := tempo_target(score, tempo):
                        info("tempo", bbox, target)
                pages_out.append({"image": {"file_name": name, "width": image.width, "height": image.height,
                                             "source_id": row["source_id"], "family": row["family"],
                                             "mode": mode, "instrument": row["instrument"]}, "annotations": annotations})
        contexts = apply_pitch_regions(records, predictions, instrument=row["instrument"])
        # Unlabelled transposing instruments cannot supply their hidden offset.
        # Keep their detector/info data, but do not invent a measure input.
        if row["instrument"] == "pitched" and row["expected_native_transpose"] and not row.get("instruction"):
            continue
        history = []
        for record in contexts:
            index = record["measure_index"]
            measure = deepcopy(label["measures"][index])
            if mode == "tab":
                for voice in measure["voices"]:
                    for event in voice["events"]:
                        event["effects"] = [e for e in event["effects"] if not e.startswith("ottava:")]
            target = format_measure_target(measure, mode)
            _, errors = validate_measure_target(target, mode, tuning=label["track"]["tuning_midi_high_to_low"])
            if errors:
                raise ValueError(f"Invalid generated target: {errors}")
            previous = format_history_context(history, mode) if history else "START"
            history.append(target)
            measures.append({**record, "id": f"{row['source_id']}_{mode}_{index}", "source_id": row["source_id"],
                             "family": row["family"], "split": row["split"], "mode": mode,
                             "instrument": row["instrument"], "label": str(label_path),
                             "label_json": str(label_path), "target": target, "previous_context": previous,
                             "string_count": label["track"]["string_count"],
                             "tuning": label["track"]["tuning_midi_high_to_low"] if row["instrument"] in {"guitar", "bass"} else []})
    return row, pages_out, infos, measures


def _safe_source(job):
    try:
        return _source(job), None
    except Exception as error:
        return None, {"source_id": job[2]["source_id"], "error": str(error)}


def build(source, output, workers, native_export=None):
    output = output.resolve()
    native_export = (native_export or source / "native-export").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("layout/images", "layout/annotations", "info/images", "measure/images", "measure/manifests", "measure/llamafactory"):
        (output / folder).mkdir(parents=True)
    groups = {s: {"pages": [], "info": [], "measures": []} for s in ("train", "validation", "test")}
    catalog = json.loads((source / "source_catalog.json").read_text())["sources"]
    failures, families = [], {}
    with ProcessPoolExecutor(workers) as pool:
        for index, (result, error) in enumerate(pool.map(_safe_source, ((source.resolve(), output, r, native_export) for r in catalog)), 1):
            if error:
                failures.append(error)
            else:
                row, pages, infos, measures = result
                if families.setdefault(row["family"], row["split"]) != row["split"]:
                    raise ValueError("Source family crosses splits")
                group = groups[row["split"]]
                group["pages"].extend(pages)
                group["info"].extend(infos)
                group["measures"].extend(measures)
            if index % 50 == 0:
                print(f"Built {index}/{len(catalog)}; rejected {len(failures)}", flush=True)
    llama_info = {"info": {}, "measure": {}}
    report = {}
    for split, group in groups.items():
        images, annotations = [], []
        for page in group["pages"]:
            identifier = len(images) + 1
            images.append({**page["image"], "id": identifier})
            for order, box in enumerate(sorted(page["annotations"], key=lambda r: (r["bbox"][1], r["bbox"][0]))):
                x, y, w, h = box["bbox"]
                annotations.append({"id": len(annotations) + 1, "image_id": identifier,
                                    "category_id": TYPED_CATEGORIES.index(box["label"]) + 1,
                                    "bbox": box["bbox"], "area": w * h, "iscrowd": 0, "read_order": order,
                                    "segmentation": [[x, y, x+w, y, x+w, y+h, x, y+h]]})
        categories = [{"id": i, "name": label, "supercategory": "score"} for i, label in enumerate(TYPED_CATEGORIES, 1)]
        (output / "layout/annotations" / f"instance_{'val' if split == 'validation' else split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": categories}))
        (output / "measure/manifests" / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group["measures"]))
        chats = [_chat(recognition_prompt(r["mode"], r["previous_context"], r["instrument"], r["pitch_context"]),
                       r["target"], r["image"], {k: r[k] for k in ("source_id", "family", "split", "mode")}) for r in group["measures"]]
        for kind, rows, folder in (("info", group["info"], output / "info"), ("measure", chats, output / "measure/llamafactory")):
            name = f"pitch_{kind}_{split}"
            (folder / (name + ".json")).write_text(json.dumps(rows, ensure_ascii=False))
            llama_info[kind][name] = dataset_entry(name + ".json")
        report[split] = {"pages": len(images), "boxes": len(annotations), "info": len(group["info"]), "measures": len(chats)}
    for kind, folder in (("info", "info"), ("measure", "measure/llamafactory")):
        (output / folder / "dataset_info.json").write_text(json.dumps(llama_info[kind], indent=2))
    (output / "layout/images_mask").symlink_to("images", target_is_directory=True)
    (output / "rejected.json").write_text(json.dumps(failures, indent=2))
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--native-export", type=Path, help="Native export directory, defaults to SOURCE/native-export")
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
