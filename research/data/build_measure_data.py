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
from research.data.prompts import key_fifths
from research.data.technique_visibility import prepare_visual_techniques


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
        "supervision": "source_manifest_only",
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
        "failures": len(failures),
        "scope": (
            "Official Guitar Pro 8 TAB, notation and score+TAB measure crops paired with source-GP "
            "source event sequences. These are assembly references, not local OCR answers. "
            "Training requires renderer-observed labels from research.data.observations. "
            "Splits follow the grouped source catalog."
        ),
    }
    if 'tab' in modes:
        documents = []
        for label_path, label in zip(label_paths, label_values):
            if 'tab' not in label.get('modes', modes):
                continue
            _, layout_path = native_layout(label_path, 'tab')
            documents.append({'folder': str(Path(layout_path).parent),
                'source_id': 'guitarpro-' + label['source_id'],
                'family': catalog[label['source_id']]['family'],
                'split': source_splits[label['source_id']]})
        summary['observations'] = build_native_observations(documents, output / 'observations', dpi)
    write_json(output / "summary.json", summary)
    return summary

def build_native_observations(documents, output, dpi=180, *, scopes_only=False):
    """Build GP observation data directly from native export triplets.

    documents declares folder/source_id/family/split. Unsupported measures stay
    in rejections.jsonl and never become empty-technique training negatives.
    """
    from research.common.pdf import open_pdf
    from research.data.native_supervision import native_observations
    from research.data.observations import write_dataset

    output = Path(output)
    rows, rejected, families = [], [], {}
    for source_number, source in enumerate(documents, 1):
        if source['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Native source requires an explicit dataset split')
        if families.setdefault(source['family'], source['split']) != source['split']:
            raise ValueError(f"Native source family crosses splits: {source['family']}")
        folder = Path(source['folder'])
        try:
            layout = json.loads((folder / 'layout.json').read_text())
            score = json.loads((folder / 'official-score.json').read_text())
            with open_pdf(folder / 'score.pdf') as document:
                mode = layout['display_mode']
                observations = native_observations(layout, score, document, dpi)
                pages = {}
                for system in layout['systems']:
                    for box in system['measure_boxes']:
                        index = box['measure_index']
                        observed = observations[index]
                        identity = {k: source[k] for k in ('source_id', 'family', 'split')}
                        identity.update(id=f"{source['source_id']}-{mode}-{index}", bar_index=index,
                            part_id=f"part-{score['source_track_index']}", staff_id='staff-0',
                            mode=mode, renderer='guitarpro')
                        if source.get('instrument'):
                            identity['instrument'] = source['instrument']
                        if all(m['clef'] == 'Neutral' for m in score['tracks'][0]['staves'][0]['measures']):
                            identity['instrument'] = 'drums'
                        if 'error' in observed:
                            rejected.append({**identity, 'reason': observed['error']})
                            continue
                        if scopes_only and not any(r.get('kind') in ('pm', 'let') or r.get('shape') == 'extension'
                                                   for r in observed['observation']['relations']):
                            continue
                        page = system['page']
                        if page not in pages:
                            pages[page] = document[page - 1].render(dpi, grayscale=False)
                        image = pages[page]
                        crop = observed['bbox']
                        path = output / 'images' / source['source_id'] / f'm{index + 1:04d}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        image.crop(crop).save(path)
                        rows.append({**identity, **observed, 'image': str(path.resolve()),
                            'page': page, 'system_index': system['system_index'], 'bbox': list(crop),
                            'source_pdf': str((folder / 'score.pdf').resolve()),
                            'native_layout': str((folder / 'layout.json').resolve()),
                            'native_official_score': str((folder / 'official-score.json').resolve()),
                            'supervision': 'native_drawn_glyphs_connections_rhythm_and_scopes'})
        except (ValueError, KeyError, OSError) as error:
            rejected.append({**source, 'folder': str(folder), 'reason': str(error)})
        if source_number % 10 == 0:
            print(f"[native {source_number}/{len(documents)}] accepted={len(rows)}", flush=True)
    counts = write_dataset(output, rows)
    kept_images = {Path(row['image']) for row in rows}
    for path in (output / 'images').glob('guitarpro-*/*.png'):
        if path.resolve() not in kept_images:
            path.unlink()
    write_jsonl(output / 'rejections.jsonl', rejected)
    coverage = {split: Counter() for split in counts}
    for row in rows:
        counted, observation = coverage[row['split']], row['observation']
        counted['measures'] += 1
        counted['multiple_voices'] += len(observation['voices']) > 1
        counted['with_diagram_marks'] += row.get('visible_diagram_marks', 0) > 0
        counted['clipped_chord_names'] += sum(not c['complete'] for c in row.get('visible_chords', []))
        for voice in observation['voices']:
            for event in voice['events']:
                duration = event.get('duration') or ''
                counted['tuplets'] += '[' in duration
                counted['dotted_rhythm'] += '.' in duration
                counted['unknown_rhythm'] += event.get('duration') is None
                counted['rests'] += event.get('status') == 'rest'
                counted['chord_names'] += sum(e.startswith('chord:') for e in event.get('effects', []))
                for note in event.get('notes') or []:
                    counted['hidden_frets'] += note.get('fret') is None
                    counted['visible_frets'] += note.get('fret') is not None
                    counted.update('effect_' + effect for effect in note.get('effects', []))
        ranges = []
        for relation in observation['relations']:
            kind = relation.get('kind') or 'unknown'
            counted['relation_' + kind] += 1
            counted['entering_' + kind] += relation['start'] == 'left'
            counted['leaving_' + kind] += relation['end'] == 'right'
            if kind in ('pm', 'let') or relation.get('shape') == 'extension':
                ranges.append(relation)
                counted['label_only_' + kind] += relation['end'] is None
                counted['ambiguous_scope_owner'] += relation.get('voice') is None and relation['start'] is None
        counted['parallel_scopes'] += len(ranges) > 1
    summary = {'renderer': 'guitarpro', 'sources': len(documents), 'accepted': counts,
               'rejected': dict(Counter(row['reason'] for row in rejected)),
               'visible_coverage': {split: dict(values) for split, values in coverage.items()},
               'scope': 'Single-staff TAB, notation and paired staves; labels use actual fret/notehead/rest drawing, connected curves and slides, '
                        'printed rhythmic evidence and PM/let scopes. Invisible values and ambiguous owners remain null.'}
    write_json(output / 'native_summary.json', summary)
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compile visible GP observations from native exports')
    parser.add_argument('--native-manifest', nargs='+', type=Path, required=True,
                        help='Native document manifests; each lives beside its documents directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-per-split', type=int, default=0)
    parser.add_argument('--scopes-only', action='store_true', help='Keep only measures with a traced PDF PM/let scope')
    args = parser.parse_args()
    documents, counts = [], Counter()
    for manifest in args.native_manifest:
        for line in manifest.read_text().splitlines():
            source = json.loads(line)
            split = 'validation' if source['split'] == 'dev' else source['split']
            if args.limit_per_split and counts[split] >= args.limit_per_split:
                continue
            folder = manifest.parent / 'documents' / source['document_id']
            tracks = sorted((folder / 'tracks').glob('*/official-score.json'))
            if not tracks:
                raise ValueError(f'Native export missing for {source["document_id"]}')
            for track in tracks:
                documents.append({'folder': str(track.parent),
                    'source_id': 'guitarpro-' + source['document_id'] + '-' + track.parent.name,
                    'family': source['source_family_id'], 'split': split,
                    'instrument': source['instrument_kind']})
            counts[split] += 1
    print(json.dumps(build_native_observations(documents, args.output, scopes_only=args.scopes_only), indent=2))
