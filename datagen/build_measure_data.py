from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from datagen.files import _write_json, _write_jsonl


from PIL import Image
from shared.m2 import format_previous_measure_context
from measure_ocr.prompts import recognition_prompt


MODES = ("tab",)
# These classes are visually or structurally important but are much rarer than
# ordinary fretted notes.  The main training file still contains every measure;
# a second, bounded file repeats representative training-only hard cases once.
# This avoids allowing common quarter/eighth-note measures to dominate LoRA.
HARDCASE_TAGS = {
    "accent",
    "bend",
    "chord",
    "dead",
    "fade",
    "ghost",
    "grace",
    "hammer",
    "harm",
    "heavy",
    "multi_voice",
    "pick_down",
    "pick_up",
    "pm",
    "rasg",
    "sia",
    "sib",
    "sl",
    "slap",
    "sod",
    "sou",
    "ss",
    "stacc",
    "stroke_down",
    "stroke_up",
    "tap",
    "trem",
    "trill",
}


def _measure_semantic_tags(measure: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    nonempty_voices = 0
    for voice in measure.get("voices", []):
        events = voice.get("events", [])
        if events:
            nonempty_voices += 1
        for event in events:
            status = str(event.get("status", "normal"))
            if status in {"rest", "empty"}:
                tags.add(status)
            duration = event.get("duration") or {}
            if duration.get("dotted") or duration.get("double_dotted"):
                tags.add("dotted")
            if (
                int(duration.get("tuplet_enters", 1) or 1) != 1
                or int(duration.get("tuplet_times", 1) or 1) != 1
            ):
                tags.add("tuplet")
            notes = event.get("notes") or []
            if len(notes) > 1:
                tags.add("chord")
            for effect in event.get("effects", []):
                tags.add(str(effect).split(":", 1)[0])
            for note in notes:
                for effect in note.get("effects", []):
                    tags.add(str(effect).split(":", 1)[0])
    if nonempty_voices > 1:
        tags.add("multi_voice")
    return sorted(tags)


def _balanced_hardcase_rows(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Return a deterministic, bounded union of rare-class examples.

    Each mode/tag bucket contributes at most 1.5% of that mode's ordinary
    training rows (and at least 256 when available).  Rare buckets smaller than
    the cap are therefore preserved in full, while common classes such as palm
    mute cannot overwhelm the rest of the dataset.
    """
    train_rows = [row for row in rows if row["split"] == "train"]
    per_mode = Counter(row["mode"] for row in train_rows)
    selected: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, int]] = defaultdict(dict)
    for mode in MODES:
        cap = max(256, (int(per_mode[mode]) * 15 + 999) // 1000)
        mode_rows = [row for row in train_rows if row["mode"] == mode]
        for tag in sorted(HARDCASE_TAGS):
            candidates = [
                row for row in mode_rows if tag in row.get("semantic_tags", [])
            ]
            candidates.sort(
                key=lambda row: sha256(
                    f"{seed}:hard:{tag}:{row['id']}".encode("utf-8")
                ).digest()
            )
            chosen = candidates[:cap]
            coverage[mode][tag] = len(chosen)
            for row in chosen:
                selected[row["id"]] = row
    hardcases = sorted(
        selected.values(),
        key=lambda row: (row["source_id"], row["mode"], row["measure_index"]),
    )
    return hardcases, {
        mode: dict(sorted(values.items())) for mode, values in sorted(coverage.items())
    }


def _split(source_id: str, seed: int) -> str:
    value = int.from_bytes(
        sha256(f"{seed}:{source_id}".encode("utf-8")).digest()[:8], "big"
    )
    ratio = value / float(2**64)
    if ratio < 0.80:
        return "train"
    if ratio < 0.90:
        return "validation"
    return "test"


def _stratified_source_splits(
    labels: list[dict[str, Any]], seed: int
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Create deterministic source-disjoint splits with long-tail coverage."""

    split_names = ("train", "validation", "test")
    ratios = {"train": 0.80, "validation": 0.10, "test": 0.10}
    total = len(labels)
    capacities = {
        "train": round(total * ratios["train"]),
        "validation": round(total * ratios["validation"]),
    }
    capacities["test"] = total - capacities["train"] - capacities["validation"]

    semantic_tags: dict[str, set[str]] = {}
    all_tags: dict[str, set[str]] = {}
    label_by_id = {label["source_id"]: label for label in labels}
    for label in labels:
        source_id = label["source_id"]
        semantics = set(label["statistics"].get("tags", []))
        if int(label["statistics"].get("multi_voice_measure_count", 0)):
            semantics.add("multi_voice")
        semantic_tags[source_id] = semantics
        all_tags[source_id] = {
            *semantics,
            f"format:{label['source_format']}",
            f"strings:{label['track']['string_count']}",
        }

    frequencies = Counter(tag for tags in all_tags.values() for tag in tags)
    semantic_universe = set().union(*semantic_tags.values())
    desired = {
        split: {
            tag: max(
                1
                if split != "train" and tag in semantic_universe and count >= 3
                else 0,
                round(count * ratios[split]),
            )
            for tag, count in frequencies.items()
        }
        for split in split_names
    }
    assignments: dict[str, str] = {}
    current = {split: Counter() for split in split_names}
    assigned_counts = Counter()

    def stable(source_id: str, split: str) -> float:
        digest = sha256(f"{seed}:{split}:{source_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def assign(source_id: str, split: str) -> None:
        assignments[source_id] = split
        assigned_counts[split] += 1
        current[split].update(all_tags[source_id])

    # Give both held-out splits at least one source for every semantic tag
    # represented by three or more independent songs.
    semantic_frequency = Counter(tag for tags in semantic_tags.values() for tag in tags)
    for split in ("validation", "test"):
        for tag, count in sorted(
            semantic_frequency.items(), key=lambda item: (item[1], item[0])
        ):
            if (
                count < 3
                or current[split][tag]
                or assigned_counts[split] >= capacities[split]
            ):
                continue
            candidates = [
                source_id
                for source_id, tags in semantic_tags.items()
                if source_id not in assignments and tag in tags
            ]
            if not candidates:
                continue
            source_id = max(
                candidates,
                key=lambda value: (
                    sum(
                        1.0 / semantic_frequency[candidate_tag]
                        for candidate_tag in semantic_tags[value]
                        if not current[split][candidate_tag]
                    ),
                    stable(value, split),
                ),
            )
            assign(source_id, split)

    remaining = sorted(
        (source_id for source_id in label_by_id if source_id not in assignments),
        key=lambda source_id: (
            -sum(1.0 / frequencies[tag] for tag in all_tags[source_id]),
            stable(source_id, "order"),
        ),
    )
    for source_id in remaining:
        choices = [
            split for split in split_names if assigned_counts[split] < capacities[split]
        ]
        split = max(
            choices,
            key=lambda value: (
                sum(
                    max(0, desired[value][tag] - current[value][tag])
                    / max(1, desired[value][tag])
                    for tag in all_tags[source_id]
                ),
                (capacities[value] - assigned_counts[value])
                / max(1, capacities[value]),
                stable(source_id, value),
            ),
        )
        assign(source_id, split)

    coverage = {
        split: dict(
            sorted(
                Counter(
                    tag
                    for source_id, assigned_split in assignments.items()
                    if assigned_split == split
                    for tag in semantic_tags[source_id]
                ).items()
            )
        )
        for split in split_names
    }
    return assignments, coverage


def _measure_boxes(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if layout.get("schema") == "gpomr.render-layout":
        boxes = {}
        for system in layout["systems"]:
            for measure in system["measure_boxes"]:
                index = int(measure["measure_index"])
                if index in boxes:
                    raise ValueError(f"Duplicate native measure index: {index}")
                boxes[index] = {
                    "page": int(system["page"]),
                    "bbox_mm": measure["bbox_mm"],
                    "staff_types": [],
                }
        return boxes
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in layout.get("records", []):
        if record.get("kind") != "bar":
            continue
        index = int(record.get("first_bar_index", -1))
        if index >= 0:
            grouped[index].append(record)
    boxes = {}
    for index, records in grouped.items():
        pages = {int(record["page"]) for record in records}
        if len(pages) != 1:
            continue
        coordinates = [
            record.get("page_bbox_mm") or record["bbox_mm"] for record in records
        ]
        left = min(float(value[0]) for value in coordinates)
        top = min(float(value[1]) for value in coordinates)
        right = max(float(value[0]) + float(value[2]) for value in coordinates)
        bottom = max(float(value[1]) + float(value[3]) for value in coordinates)
        boxes[index] = {
            "page": pages.pop(),
            "bbox_mm": [left, top, right - left, bottom - top],
            "staff_types": sorted(
                {int(record.get("staff_type", -1)) for record in records}
            ),
        }
    return boxes


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    import pymupdf

    scale = dpi / 72.0
    images = []
    with pymupdf.open(pdf_path) as document:
        for page in document:
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY
            )
            images.append(
                Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
            )
    return images


def _crop_measure(page: Image.Image, bbox_mm: list[float], dpi: int) -> Image.Image:
    pixels_per_mm = dpi / 25.4
    left, top, width, height = (float(value) for value in bbox_mm)
    pad_x = 1.5
    pad_y = 4.0
    x0 = max(0, round((left - pad_x) * pixels_per_mm))
    y0 = max(0, round((top - pad_y) * pixels_per_mm))
    x1 = min(page.width, round((left + width + pad_x) * pixels_per_mm))
    y1 = min(page.height, round((top + height + pad_y) * pixels_per_mm))
    return page.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))


def crop_and_manifest(
    output: Path, modes: list[str], dpi: int, seed: int
) -> dict[str, Any]:
    rows = []
    failures = []
    label_paths = sorted((output / "labels").glob("*.json"))
    label_values = [
        json.loads(path.read_text(encoding="utf-8")) for path in label_paths
    ]
    from datagen.inventory import source_catalog

    catalog = source_catalog(output, seed)
    source_splits = {sid: row["split"] for sid, row in catalog.items()}
    split_technique_sources = {
        split: dict(
            Counter(
                tag
                for label in label_values
                if source_splits[label["source_id"]] == split
                for tag in label["statistics"].get("tags", [])
            )
        )
        for split in ("train", "validation", "test")
    }
    _write_json(output / "source_splits.json", source_splits)
    source_formats: Counter[str] = Counter()
    technique_counts: Counter[str] = Counter()
    multi_voice_sources = 0
    for source_index, (label_path, label) in enumerate(
        zip(label_paths, label_values), start=1
    ):
        source_formats[label["source_format"]] += 1
        technique_counts.update(label["statistics"].get("technique_counts", {}))
        if int(label["statistics"].get("multi_voice_measure_count", 0)):
            multi_voice_sources += 1
        source_id = label["source_id"]
        split = source_splits[source_id]
        for mode in modes:
            pdf_path = output / "pdf" / mode / f"{source_id}.pdf"
            layout_path = output / "layout" / mode / f"{source_id}.layout.json"
            if not pdf_path.is_file() or not layout_path.is_file():
                failures.append(
                    {
                        "source_id": source_id,
                        "mode": mode,
                        "error": "missing_pdf_or_layout",
                    }
                )
                continue
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            boxes = _measure_boxes(layout)
            if len(boxes) != len(label["measures"]):
                failures.append(
                    {
                        "source_id": source_id,
                        "mode": mode,
                        "error": "measure_count_mismatch",
                        "labels": len(label["measures"]),
                        "boxes": len(boxes),
                    }
                )
                continue
            crop_paths = [
                output
                / "crops"
                / mode
                / source_id
                / f"m{int(measure['index']) + 1:04d}.png"
                for measure in label["measures"]
            ]
            import pymupdf

            with pymupdf.open(pdf_path) as document:
                page_count = document.page_count
            # Relabeling changes targets but not official PDF geometry. Avoid
            # rasterizing thousands of PDFs again when every crop already
            # exists; only the manifests/ShareGPT records need rewriting.
            pages = (
                _render_pdf_pages(pdf_path, dpi)
                if any(not path.is_file() for path in crop_paths)
                else None
            )
            for measure in label["measures"]:
                index = int(measure["index"])
                box = boxes[index]
                page_index = int(box["page"]) - 1
                if not 0 <= page_index < page_count:
                    failures.append(
                        {
                            "source_id": source_id,
                            "mode": mode,
                            "measure_index": index,
                            "error": "page_index_out_of_range",
                            "page": int(box["page"]),
                            "pdf_pages": page_count,
                        }
                    )
                    continue
                crop_path = (
                    output / "crops" / mode / source_id / f"m{index + 1:04d}.png"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not crop_path.is_file():
                    assert pages is not None
                    crop = _crop_measure(pages[page_index], box["bbox_mm"], dpi)
                    crop.save(crop_path, format="PNG", compress_level=3)
                target = measure["targets"][mode]
                previous_context = "START"
                if index > 0:
                    previous_context = format_previous_measure_context(
                        label["measures"][index - 1]["targets"][mode], mode
                    )
                rows.append(
                    {
                        "id": f"{source_id}_{mode}_m{index + 1:04d}",
                        "source_id": source_id,
                        "split": split,
                        "mode": mode,
                        "measure_index": index,
                        "image": str(crop_path.resolve()),
                        "target": target,
                        "previous_context": previous_context,
                        "target_utf8_bytes": len(target.encode("utf-8")),
                        "page": int(box["page"]),
                        "bbox_mm": box["bbox_mm"],
                        "staff_types": box["staff_types"],
                        "semantic_tags": _measure_semantic_tags(measure),
                        "label_json": str(label_path.resolve()),
                        "source_gp": label["source_path"],
                    }
                )
        print(f"[crop {source_index}/{len(label_paths)}] {source_id}", flush=True)
    expected_samples = sum(len(label["measures"]) for label in label_values) * len(
        modes
    )
    if failures or len(rows) != expected_samples:
        _write_json(output / "crop_failures.json", failures)
        raise RuntimeError(
            f"Crop gate produced {len(rows)}/{expected_samples} samples with "
            f"{len(failures)} failures; see {output / 'crop_failures.json'}"
        )
    rows.sort(
        key=lambda row: (
            row["split"],
            row["source_id"],
            row["mode"],
            row["measure_index"],
        )
    )
    manifests = {}
    for split_name in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        path = output / "manifests" / f"{split_name}.jsonl"
        _write_jsonl(path, split_rows)
        manifests[split_name] = str(path.resolve())
    _write_json(output / "crop_failures.json", failures)

    llama_root = output / "llamafactory"
    llama_root.mkdir(parents=True, exist_ok=True)
    hardcase_rows, hardcase_coverage = _balanced_hardcase_rows(rows, seed)
    for split_name in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        values = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>{recognition_prompt(row['mode'], row['previous_context'])}",
                    },
                    {"role": "assistant", "content": row["target"]},
                ],
                "images": [row["image"].replace("\\", "/")],
            }
            for row in split_rows
        ]
        _write_json(llama_root / f"gp8_measure_sequence_{split_name}.json", values)
    hardcase_values = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>{recognition_prompt(row['mode'], row['previous_context'])}",
                },
                {"role": "assistant", "content": row["target"]},
            ],
            "images": [row["image"].replace("\\", "/")],
        }
        for row in hardcase_rows
    ]
    _write_json(
        llama_root / "gp8_measure_sequence_train_hardcases.json", hardcase_values
    )
    dataset_info = {
        f"gp8_measure_sequence_{split_name}": {
            "file_name": f"gp8_measure_sequence_{split_name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
        for split_name in ("train", "validation", "test")
    }
    dataset_info["gp8_measure_sequence_train_hardcases"] = {
        "file_name": "gp8_measure_sequence_train_hardcases.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    _write_json(llama_root / "dataset_info.json", dataset_info)
    split_sources = {
        name: sorted({row["source_id"] for row in rows if row["split"] == name})
        for name in ("train", "validation", "test")
    }
    if any(
        set(split_sources[left]) & set(split_sources[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("Source leakage detected across dataset splits")
    summary = {
        "schema_version": "2.0",
        "sources": len({row["source_id"] for row in rows}),
        "samples": len(rows),
        "source_formats": dict(sorted(source_formats.items())),
        "multi_voice_sources": multi_voice_sources,
        "technique_occurrences": dict(sorted(technique_counts.items())),
        "modes": dict(Counter(row["mode"] for row in rows)),
        "splits": {
            name: {
                "sources": len(split_sources[name]),
                "samples": sum(row["split"] == name for row in rows),
            }
            for name in split_sources
        },
        "split_technique_sources": split_technique_sources,
        "hardcase_samples": len(hardcase_rows),
        "hardcase_coverage": hardcase_coverage,
        "maximum_target_utf8_bytes": max(
            (row["target_utf8_bytes"] for row in rows), default=0
        ),
        "manifests": manifests,
        "llamafactory": str(llama_root.resolve()),
        "failures": len(failures),
        "scope": (
            "Official Guitar Pro 8 TAB measure crops paired with exact source-GP multi-voice "
            "event sequences. Splits are disjoint by source SHA."
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary
