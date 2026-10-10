"""Evaluate raw measure OCR on reference crops with shared musical metrics."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from research.data.training_samples import visual_measure_inputs
from research.evaluation.measure_metrics import MeasureSequenceMetrics


def base_messages(row):
    sample = visual_measure_inputs(row)
    prompt = sample['messages'][0]['content'].replace('<image>', '')
    return [{'role': 'user', 'content': [*[{'type': 'image', 'url': p} for p in sample['images']],
                                        {'type': 'text', 'text': prompt}]}]


def select_rows(rows, limit):
    if not limit or len(rows) <= limit:
        return rows
    # Round-robin groups prevents source ordering from producing an all-TAB or
    # all-negative baseline. Selection is deterministic and persisted verbatim.
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get('corpus'), row.get('scenario'), row['mode'], row['instrument'])].append(row)
    selected = []
    while len(selected) < limit and groups:
        for key in list(groups):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == limit:
                break
    return selected


def summarize(rows, predictions):
    by_id = {row['id']: row for row in rows}
    groups = defaultdict(MeasureSequenceMetrics)
    failures = []
    for prediction in predictions:
        row = by_id[prediction['id']]
        text = prediction['raw']
        sample = MeasureSequenceMetrics()
        sample.update(row['target'], text, row['mode'], instrument=row['instrument'],
                      string_count=len(row.get('tuning', [])) or None)
        music_errors = [name for name, valid in (
            ('invalid_syntax', sample.syntax_valid), ('invalid_constraints', sample.constraint_valid),
            ('metadata', sample.metadata_exact), ('rhythm', sample.rhythm_exact),
            ('note_fields', sample.note_fields_exact), ('techniques', sample.technique_exact)) if not valid]
        categories = ['all', f"mode/{row['mode']}", f"instrument/{row['instrument']}"]
        categories.extend(f'{key}/{row[key]}' for key in ('corpus', 'scenario') if row.get(key))
        for category in categories:
            groups[category].merge(sample)
        if music_errors:
            failures.append({'id': row['id'], 'music_errors': music_errors,
                             'target': row['target'], 'raw': text, 'image': row.get('image')})
    return {'groups': {category: {'music': core.result()} for category, core in groups.items()},
            'failures': failures}


def run(manifest, output, model, device='cuda:0', limit=0, batch_size=8, max_tokens=2048):
    from research.inference.vllm import VllmBackend

    if batch_size < 1 or max_tokens < 1 or limit < 0:
        raise ValueError('Batch size and token budget must be positive; limit must be nonnegative')
    rows = select_rows([json.loads(line) for line in manifest.read_text().splitlines()], limit)
    if not rows:
        raise ValueError('No reference measures to evaluate')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'selection.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    metadata = {'model': str(model.resolve()), 'device': device, 'samples': len(rows),
                'batch_size': batch_size, 'max_tokens': max_tokens,
                'context': 'training inputs: reference crops and printed state; raw predictions'}
    predictions = []
    started = time.monotonic()
    backend = VllmBackend(model, device, options={'max_num_seqs': batch_size, 'max_model_len': 8192})
    try:
        with (output / 'predictions.jsonl').open('w') as handle:
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                values = backend.generate_batch([base_messages(row) for row in batch], max_tokens)
                for row, (raw, tokens) in zip(batch, values, strict=True):
                    prediction = {'id': row['id'], 'raw': raw, 'tokens': tokens}
                    predictions.append(prediction)
                    handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                handle.flush()
                print(min(offset + batch_size, len(rows)), '/', len(rows), flush=True)
    finally:
        backend.close()
    result = summarize(rows, predictions)
    result['run'] = {**metadata, 'seconds': round(time.monotonic() - started, 3)}
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=2048)
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
