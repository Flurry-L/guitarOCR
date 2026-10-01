"""Recognize ordered crops and write score text plus the saved model sequence."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from measure_ocr.result import save_recognition
from gp5_export.timing import gp5_timing_errors

from shared.defaults import MODEL, MEASURE_ADAPTER

from measure_ocr.recognizer import recognize_crops
from shared.artifacts import read_result, write_json
from shared.m2 import format_measure_target, parse_measure_target


def run(
    layout: Path,
    info: Path,
    output: Path,
    *,
    model: Path = MODEL,
    adapter: Path | None = MEASURE_ADAPTER,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    max_new_tokens_ceiling: int = 4096,
    maximum_attempts: int = 3,
    resume: bool = False,
    backend=None,
    progress=None,
    initial_records=None,
    retry_measures=None,
    cancelled=None,
) -> Path:
    if (
        max_new_tokens <= 0
        or max_new_tokens_ceiling < max_new_tokens
        or maximum_attempts < 1
    ):
        raise ValueError(
            "Token limits and attempt count must be positive; ceiling must be >= initial limit"
        )
    source = read_result(layout, "layout")
    information = read_result(info, "document_info")
    if Path(information["layout"]).resolve() != layout.resolve():
        raise ValueError("Document information belongs to a different layout result")
    records = information.get('resolved_records', source["records"])
    contexts = {row["measure_number"]: row for row in information.get("measure_pitch_contexts", [])}
    profiles = {row['measure_number']: row for row in information.get('measure_profiles', [])}
    capabilities_path = (adapter or model) / 'capabilities.json'
    capabilities = json.loads(capabilities_path.read_text()) if capabilities_path and capabilities_path.exists() else {}
    for row in records:
        row.update(profiles.get(row['measure_number'], {}))
        row.update(contexts.get(row["measure_number"], {}))
        row.setdefault('tuning_source', information.get('tuning_source', 'default'))
        row.setdefault('tuning_explicit', row['tuning_source'] in {'manual', 'printed'})
        row['fingering_tunings'] = row.get('fingering_tunings') or information.get('tuning_candidates') or [information['tuning_used']]
        if row.get("pitch_context"):
            row["pitch_context"] = {**row["pitch_context"], "capo": row.get('capo', information.get("capo", 0))}
        if (row.get('capo', information.get("capo")) and row.get('instrument', information.get("instrument", "guitar")) in {"guitar", "bass"}
                and (row.get("mode") or source["mode"]) != "tab" and not capabilities.get('capo_pitch')):
            row["pitch_needs_review"] = True
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "recognition.jsonl"
    # Reusing accepted measures requires the same crops, metadata, models and options.
    def identity(path, *, content=False):
        path = Path(path).resolve()
        stat = path.stat()
        if content:
            return {"path": str(path), "bytes": stat.st_size,
                    "sha256": sha256(path.read_bytes()).hexdigest()}
        return [str(path), stat.st_size, stat.st_mtime_ns]

    context = {
        # The CLI regenerates upstream artifacts on resume. Identical inputs
        # must remain reusable despite atomic replacements changing mtimes.
        'input_identity': 'sha256',
        'layout': identity(layout, content=True), 'info': identity(info, content=True),
        'images': [identity(row['image'], content=True) for row in records],
        'models': [], 'options': [device, max_new_tokens, max_new_tokens_ceiling, maximum_attempts],
        'initial': initial_records, 'retry': retry_measures,
    }
    for path in (model, adapter):
        if path is not None:
            artifacts = []
            for name in (
                "config.json",
                "adapter_config.json",
                "capabilities.json",
                "adapter_model.safetensors",
                "model.safetensors",
                "model.safetensors.index.json",
                "inference.json",
            ):
                artifact = path / name
                if artifact.is_file():
                    artifacts.append(identity(artifact))
            artifacts.extend(identity(p) for p in sorted(path.glob('model-*-of-*.safetensors')))
            inference_config = path / 'inference.json'
            if inference_config.is_file():
                settings = json.loads(inference_config.read_text())
                merged = path / settings['model']
                artifacts.extend(identity(p) for p in sorted(merged.glob('*.safetensors')))
            context['models'].append([str(path.resolve()), artifacts])
    if capabilities.get('state_reader'):
        from measure_ocr.state_reader import state_model_path
        context['state_reader'] = identity(state_model_path(capabilities_path.parent / capabilities['state_reader']))
    signature_path = output / "recognition_context.json"
    if resume and log.is_file():
        previous = (
            json.loads(signature_path.read_text(encoding="utf-8"))
            if signature_path.is_file()
            else {}
        )
        comparable = context
        if 'input_identity' not in previous:
            # Existing workspaces retain the original, strict stat-based check
            # once before upgrading their checkpoint to content identities.
            comparable = {key: value for key, value in context.items() if key != 'input_identity'}
            comparable.update(layout=identity(layout), info=identity(info),
                              images=[identity(row['image']) for row in records])
        if previous != comparable:
            raise ValueError(
                "Recognition inputs or options changed; use a new output or omit --resume"
            )
    if not resume:
        log.unlink(missing_ok=True)
    write_json(signature_path, context)
    targets = recognize_crops(
        records,
        source["mode"],
        model.resolve(),
        adapter.resolve() if adapter else None,
        device,
        max_new_tokens,
        max_new_tokens_ceiling,
        information["tuning_used"],
        maximum_attempts,
        log,
        resume,
        backend,
        progress,
        initial_records,
        retry_measures,
        cancelled,
        instrument=information.get("instrument", "guitar"),
    )
    metadata = information["document_metadata"]
    if targets and metadata.get("tempo_quarter"):
        first = parse_measure_target(targets[0])
        first["tempo_quarter"] = int(metadata["tempo_quarter"])
        targets[0] = format_measure_target(
            first, records[0].get("mode") or source["mode"], preserve_playback=True
        )
    for row, target in zip(records, targets):
        row["target"] = target
        row["timing_errors"] = gp5_timing_errors(target)
        if row["timing_errors"]:
            row["needs_review"] = True
    from shared.chords import attach_chord_annotations
    annotations = [{**p, 'part_id': part.get('id', 'part-1')}
                   for part in information.get('parts') or [information]
                   for p in part.get('document_metadata', {}).get('score_annotations', [])]
    attach_chord_annotations(records, annotations)
    from collections import defaultdict
    from gp5_export.fingering import notation_fingering_errors

    by_part = defaultdict(list)
    for row in records:
        by_part[row.get('part_id', 'part-1')].append(row)
    for part_id, rows in by_part.items():
        if any(row.get('mode', source['mode']) != 'notation' for row in rows):
            continue
        if rows[0].get('instrument', information.get('instrument')) not in {'guitar', 'bass'}:
            continue
        candidates = rows[0].get('fingering_tunings') or [rows[0]['tuning']]
        parsed = [parse_measure_target(row['target']) for row in rows]
        evaluations = [[notation_fingering_errors(measure, tuning) for measure in parsed] for tuning in candidates]
        selected = min(range(len(candidates)), key=lambda i: (sum(bool(e) for e in evaluations[i]), i))
        for part in information.get('parts', []):
            if part['id'] == part_id:
                part['tuning_used'] = candidates[selected]
                if selected:
                    part['document_metadata']['export_tuning_inferred_from_pitch_range'] = candidates[selected]
        if not information.get('parts') and selected:
            metadata['export_tuning_inferred_from_pitch_range'] = candidates[selected]
        for row, errors in zip(rows, evaluations[selected], strict=True):
            row['tuning'] = candidates[selected]
            row['fingering_errors'] = errors
            if errors and row.get('tuning_explicit'):
                row['needs_review'] = True
    for part in information.get('parts', []):
        rows = by_part.get(part['id'])
        if rows:
            part['tuning_used'] = rows[0]['tuning']
            part['tuning_source'] = rows[0].get('tuning_source', part.get('tuning_source'))
            if part['tuning_source'] == 'notation_tab_consensus':
                details = part['document_metadata']
                details['tuning_midi_high_to_low'] = rows[0]['tuning']
                details['warnings'] = [warning for warning in details.get('warnings', [])
                                       if not warning.startswith('无法确定这件乐器的')]
    if records and records[0].get('tuning') is not None:
        information['tuning_used'] = records[0]['tuning']
        if records[0].get('tuning_source') == 'notation_tab_consensus':
            metadata['tuning_midi_high_to_low'] = records[0]['tuning']
            metadata['warnings'] = [warning for warning in metadata.get('warnings', [])
                                    if not warning.startswith('无法确定这件乐器的')]
    return save_recognition(output, dict(
        layout=str(layout.resolve()),
        info=str(info.resolve()),
        mode=source["mode"],
        measures=len(records),
        recognition_log=str(log),
        records=records,
        instrument=information.get("instrument", "guitar"),
        midi_program=information.get("midi_program", 25),
        transpose=information.get("transpose"),
        measure_pitch_contexts=information.get("measure_pitch_contexts", []),
        parts=information.get('parts', []),
        **{
            key: information[key]
            for key in ("document_metadata", "title", "artist", "tuning_used", "capo")
        },
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize crops from layout and document-info manifests."
    )
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=MEASURE_ADAPTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens-ceiling", type=int, default=4096)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
