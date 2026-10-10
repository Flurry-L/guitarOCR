from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PIL import Image
from scorelib.pitch_context import transpose_key

from research.common.images import crop_measure
from research.data.catalog import write_json, write_jsonl
from research.data.native_alignment import native_text_target
from research.data.native_supervision import (
    measure_boxes,
    native_pitch_contexts,
    native_layout,
    prepare_native_measure,
    prepare_written_target,
    source_profile,
    visible_text_box,
)
from research.data.prompts import attach_neighbours, key_fifths
from research.data.technique_visibility import prepare_visual_techniques
from research.data.training_samples import (
    dataset_entry,
    iter_training_samples,
    visual_measure_sample,
)


def measure_semantic_tags(measure: dict[str, Any]) -> list[str]:
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


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    from research.common.pdf import open_pdf

    with open_pdf(pdf_path) as document:
        return [page.render(dpi) for page in document]


def _crop_source(job):
    label_path, label, output, modes, dpi, split = job
    profile = source_profile(str(label_path))
    rows, failures = [], []
    source_id = label["source_id"]
    for mode in modes:
        if mode not in label.get("modes", modes):
            continue
        layout, layout_path = native_layout(label_path, mode)
        pdf_path = Path(layout_path).with_name("score.pdf") if layout_path else None
        if layout is None or not pdf_path.is_file():
            failures.append(
                {
                    "source_id": source_id,
                    "mode": mode,
                    "error": "missing_pdf_or_layout",
                }
            )
            continue
        if layout['display_mode'] != mode:
            raise ValueError(f'Native display mode differs from {mode}: {layout_path}')
        contexts = {}
        if mode != 'tab' and profile['instrument'] != 'drums':
            score = json.loads(Path(layout_path).with_name('official-score.json').read_text())
            contexts = native_pitch_contexts(score, label['track']['capo'])
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
        from research.common.pdf import open_pdf

        with open_pdf(pdf_path) as document:
            page_count = len(document)
            instruction_visible = bool(label.get("instruction")
                and visible_text_box(document[0], label["instruction"]))
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
            visible = prepare_visual_techniques({'target': measure['targets'][mode],
                                                  'mode': mode, 'renderer': 'guitar_pro',
                                                  'label_json': str(label_path)})
            target = native_text_target(visible['target'], label_path, mode, index)
            prepared = prepare_written_target({
                'mode': mode, 'instrument': profile['instrument'], 'target': target,
                'pitch_reference': 'sounding', 'pitch_context': contexts.get(index),
            })
            key = measure.get('key_signature') or 'CMajor'
            if prepared.get('written_pitch'):
                key = transpose_key(key, -contexts[index]['instrument_transpose'])
            state = {'time': measure['time_signature'],
                     'key': 0 if mode == 'tab' else key_fifths(key)}
            if profile['tuning']:
                state['tuning'] = profile['tuning']
            rows.append(prepare_native_measure(
                {
                    "id": f"{source_id}_{mode}_m{index + 1:04d}",
                    "source_id": source_id,
                    "split": split,
                    "mode": mode,
                    "measure_index": index,
                    "instrument": profile["instrument"],
                    "midi_program": profile["midi_program"],
                    "string_count": profile["string_count"],
                    "family": label.get("family", source_id),
                    "image": str(crop_path.resolve()),
                    **prepared,
                    **({'image_supervision': {'status': 'review',
                        'reason': 'unverified_instrument_transposition'}}
                       if profile['instrument'] == 'pitched' and mode != 'tab'
                       and contexts[index]['instrument_transpose'] and not instruction_visible else {}),
                    "tuning": profile['tuning'],
                    "score_state": state,
                    "source_measures": len(label['measures']),
                    "source_target": measure["targets"][mode],
                    "renderer": "guitar_pro",
                    **({'technique_visibility': visible['technique_visibility']}
                       if visible.get('technique_visibility') else {}),
                    "target_utf8_bytes": len(prepared["target"].encode("utf-8")),
                    "page": int(box["page"]),
                    "bbox_mm": box["bbox_mm"],
                    "staff_types": box["staff_types"],
                    "semantic_tags": measure_semantic_tags(measure),
                    "label_json": str(label_path.resolve()),
                    "source_gp": label["source_path"],
                    "source_pdf": str(pdf_path.resolve()),
                    "part_id": "part-1",
                    "staff_id": "staff-1",
                    "bar_index": index,
                }
            ))
    for mode in modes:
        attach_neighbours([row for row in rows if row['mode'] == mode])
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
    from research.data.catalog import source_catalog

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
        (path, {**label, "family": catalog[label["source_id"]]["family"],
                "instruction": catalog[label["source_id"]].get("instruction")},
         output, modes, dpi, source_splits[label["source_id"]])
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
    visibility_review = {}
    for split_name in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        values = list(iter_training_samples(split_rows, serializer=visual_measure_sample,
                      review_path=output / 'manifests' / f'visibility_review_{split_name}.jsonl'))
        visibility_review[split_name] = len(split_rows) - len(values)
        write_json(llama_root / f"gp8_measure_sequence_{split_name}.json", values)
    dataset_info = {
        f"gp8_measure_sequence_{split_name}": dataset_entry(
            f"gp8_measure_sequence_{split_name}.json"
        )
        for split_name in ("train", "validation", "test")
    }
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
        "sources": len({row["source_id"] for row in rows}),
        "samples": len(rows),
        "visibility_review_rows": visibility_review,
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
