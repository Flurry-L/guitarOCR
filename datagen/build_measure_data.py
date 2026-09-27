from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PIL import Image

from datagen.native.annotations import measure_boxes
from datagen.sampling import balanced_hardcase_rows, measure_semantic_tags
from datagen.training_samples import dataset_entry, measure_sample
from shared.artifacts import write_json, write_jsonl
from shared.crops import crop_measure
from shared.m2 import format_previous_measure_context


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    from shared.pdf import open_pdf

    with open_pdf(pdf_path) as document:
        return [page.render(dpi) for page in document]


def _crop_source(job):
    label_path, label, output, modes, dpi, split = job
    rows, failures = [], []
    source_id = label["source_id"]
    for mode in modes:
        if mode not in label.get("modes", modes):
            continue
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
        boxes = measure_boxes(layout)
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
        from shared.pdf import open_pdf

        with open_pdf(pdf_path) as document:
            page_count = len(document)
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
            crop_path = output / "crops" / mode / source_id / f"m{index + 1:04d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            if not crop_path.is_file():
                assert pages is not None
                crop = crop_measure(
                    pages[page_index], [v * dpi / 25.4 for v in box["bbox_mm"]], dpi
                )
                crop.save(crop_path, format="PNG", compress_level=3)
            target = measure["targets"][mode]
            previous_context = "START"
            if index > 0:
                previous_context = format_previous_measure_context(
                    label["measures"][index - 1]["targets"][mode],
                    mode,
                    active_metadata=label["measures"][index - 1],
                )
            rows.append(
                {
                    "id": f"{source_id}_{mode}_m{index + 1:04d}",
                    "source_id": source_id,
                    "split": split,
                    "mode": mode,
                    "measure_index": index,
                    "instrument": label["track"].get("instrument", "guitar"),
                    "string_count": label["track"]["string_count"],
                    "family": label.get("family", source_id),
                    "image": str(crop_path.resolve()),
                    "target": target,
                    "previous_context": previous_context,
                    "target_utf8_bytes": len(target.encode("utf-8")),
                    "page": int(box["page"]),
                    "bbox_mm": box["bbox_mm"],
                    "staff_types": box["staff_types"],
                    "semantic_tags": measure_semantic_tags(measure),
                    "label_json": str(label_path.resolve()),
                    "source_gp": label["source_path"],
                }
            )
    return label["source_id"], rows, failures


def crop_and_manifest(
    output: Path, modes: list[str], dpi: int, seed: int, workers: int = 1
) -> dict[str, Any]:
    rows = []
    failures = []
    label_paths = sorted((output / "labels").glob("*.json"))
    label_values = [
        json.loads(path.read_text(encoding="utf-8")) for path in label_paths
    ]
    from datagen.catalog import source_catalog

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
    write_json(output / "source_splits.json", source_splits)
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
    jobs = (
        (path, label, output, modes, dpi, source_splits[label["source_id"]])
        for path, label in zip(label_paths, label_values)
    )
    with (
        ProcessPoolExecutor(max_workers=workers)
        if workers > 1
        else nullcontext() as pool
    ):
        results = pool.map(_crop_source, jobs) if pool else map(_crop_source, jobs)
        for index, (source_id, source_rows, source_failures) in enumerate(results, 1):
            rows.extend(source_rows)
            failures.extend(source_failures)
            print(f"[crop {index}/{len(label_paths)}] {source_id}", flush=True)
    expected_samples = sum(
        len(label["measures"]) * len(set(modes) & set(label.get("modes", modes)))
        for label in label_values
    )
    if failures or len(rows) != expected_samples:
        write_json(output / "crop_failures.json", failures)
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
        write_jsonl(path, split_rows)
        manifests[split_name] = str(path.resolve())
    write_json(output / "crop_failures.json", failures)

    llama_root = output / "llamafactory"
    llama_root.mkdir(parents=True, exist_ok=True)
    hardcase_rows, hardcase_coverage = balanced_hardcase_rows(rows, seed)
    for split_name in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        values = [measure_sample(row) for row in split_rows]
        write_json(llama_root / f"gp8_measure_sequence_{split_name}.json", values)
    hardcase_values = [measure_sample(row) for row in hardcase_rows]
    write_json(
        llama_root / "gp8_measure_sequence_train_hardcases.json", hardcase_values
    )
    dataset_info = {
        f"gp8_measure_sequence_{split_name}": dataset_entry(
            f"gp8_measure_sequence_{split_name}.json"
        )
        for split_name in ("train", "validation", "test")
    }
    dataset_info["gp8_measure_sequence_train_hardcases"] = dataset_entry(
        "gp8_measure_sequence_train_hardcases.json"
    )
    write_json(llama_root / "dataset_info.json", dataset_info)
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
            "Official Guitar Pro 8 TAB, notation and score+TAB measure crops paired with source-GP "
            "multi-voice event sequences. Splits follow the grouped source catalog."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary
