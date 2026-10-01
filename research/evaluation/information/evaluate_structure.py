"""Evaluate complete held-out page structure and retained metadata tasks."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from research.inference.information import prompts
from research.inference.information.image_ocr import parse_info_response
from research.inference.layout.structure import STRUCTURE_PROMPT
from research.inference.layout.structure import marked_systems
from research.inference.layout.structure import parse_structure
from research.inference.layout.structure import structure_model_image
from research.inference.layout.structure import structure_schema
from research.inference.backends.vllm_backend import VllmBackend


def canonical_rows(rows):
    systems, parts, staves = {}, {}, defaultdict(dict)
    result = []
    for system, part, staff in rows:
        result.append([systems.setdefault(system, len(systems)), parts.setdefault(part, len(parts)),
                       staves[part].setdefault(staff, len(staves[part]))])
    return result


def compare(expected, predicted, kind):
    counts = Counter(samples=1)
    if kind != 'structure':
        for key, value in expected.items():
            counts[key + '_total'] += 1
            counts[key + '_correct'] += key in predicted and type(predicted[key]) is type(value) and predicted[key] == value
        counts['exact'] += expected == predicted
        return counts
    counts['valid'] += bool(predicted)
    counts['rows'] += len(expected['rows'])
    for key in ('instrument', 'strings', 'program', 'name'):
        counts[key + '_total'] += len(expected['rows'])
    if not predicted:
        return counts
    gold_rows, rows = canonical_rows(expected['rows']), canonical_rows(predicted['rows'])
    counts['part_count_correct'] += len(expected['parts']) == len(predicted['parts'])
    counts['grouping_exact'] += gold_rows == rows
    counts['rows_correct'] += sum(a == b for a, b in zip(gold_rows, rows, strict=True))
    profiles_correct = True
    for gold_row, row in zip(expected['rows'], predicted['rows'], strict=True):
        gold, value = expected['parts'][gold_row[1]], predicted['parts'][row[1]]
        for key in ('instrument', 'strings', 'program', 'name'):
            counts[key + '_correct'] += gold[key] == value[key]
        profiles_correct &= all(gold[key] == value[key] for key in ('instrument', 'strings', 'program'))
    counts['exact'] += gold_rows == rows and profiles_correct
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--speculative-tokens', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    capability_path = args.model / 'capabilities.json'
    if not capability_path.is_file():
        capability_path = args.model.parent / 'capabilities.json'
    capabilities = json.loads(capability_path.read_text()) if capability_path.exists() else {}
    tasks = []
    for path in args.dataset:
        rows = json.loads(path.read_text()) if path.suffix == '.json' else (json.loads(line) for line in path.open())
        for index, row in enumerate(rows):
            prompt = row['messages'][0]['content'].replace('<image>', '')
            if prompt.startswith(STRUCTURE_PROMPT):
                kind = 'structure'
            elif prompt.startswith('Read the first staff and its instrument label.'):
                kind = 'staff'
            else:
                kind = next(key for key in ('header', 'tempo', 'clef', 'transposition')
                            if prompt == getattr(prompts, key.upper() + '_PROMPT'))
            tasks.append({'id': str(path) + ':' + str(index), 'corpus': str(path.parent), 'kind': kind,
                          'image': row.get('geometry_image', row['images'][0]), 'prompt': prompt,
                          'row_modes': row.get('row_modes'),
                          'expected': json.loads(row['messages'][-1]['content'])})
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'predictions.jsonl'
    completed, metrics = set(), defaultdict(Counter)

    def record(row):
        sample = compare(row['expected'], row['predicted'], row['kind'])
        for group in (row['kind'], row['corpus'] + '/' + row['kind']):
            metrics[group].update(sample)
        if row['kind'] == 'structure' and 'model_predicted' in row:
            raw_sample = compare(row['expected'], row['model_predicted'], 'structure')
            for group in ('structure_model', row['corpus'] + '/structure_model'):
                metrics[group].update(raw_sample)
        completed.add(row['id'])

    if args.resume and destination.exists():
        for line in destination.open():
            record(json.loads(line))
    pending = [row for row in tasks if row['id'] not in completed]
    loading = time.perf_counter()
    engine = VllmBackend(args.model, args.device, options={'max_model_len': 8192, 'max_num_seqs': args.batch_size,
                        'kv_cache_memory_bytes': 4 * 1024 ** 3, 'speculative_tokens': args.speculative_tokens})
    started = time.perf_counter()
    try:
        with destination.open('a' if args.resume else 'w') as output:
            for kind in ('header', 'tempo', 'staff', 'clef', 'transposition', 'structure'):
                subset = [row for row in pending if row['kind'] == kind]
                for offset in range(0, len(subset), args.batch_size):
                    batch = subset[offset:offset + args.batch_size]
                    systems_by_row = ([marked_systems(row['image'], len(row['expected']['rows'])) for row in batch]
                                      if kind == 'structure' else [None] * len(batch))
                    messages = [[{'role': 'user', 'content': [{'type': 'image', 'url': structure_model_image(row['image'])
                                 if kind == 'structure' and capabilities.get('compact_structure') else row['image']},
                                 {'type': 'text', 'text': row['prompt']}]}] for row in batch]
                    generation = {'json_schema': [structure_schema(len(r['expected']['rows']), systems)
                                  for r, systems in zip(batch, systems_by_row, strict=True)]} if kind == 'structure' else {}
                    for batch_index, (row, (raw, tokens)) in enumerate(zip(batch, engine.generate_batch(messages, 2048 if kind == 'structure' else 512, **generation), strict=True)):
                        error = None
                        extra = {}
                        if kind == 'structure':
                            systems = systems_by_row[batch_index]
                            try:
                                model_predicted = parse_structure(raw, len(row['expected']['rows']), modes=row['row_modes'])
                            except (ValueError, KeyError, TypeError, IndexError):
                                model_predicted = {}
                            extra = {'systems_from_image': systems, 'model_predicted': model_predicted}
                        try:
                            predicted = parse_structure(raw, len(row['expected']['rows']), systems, row['row_modes']) if kind == 'structure' else parse_info_response(raw, kind)
                        except (ValueError, KeyError, TypeError, IndexError) as exc:
                            predicted, error = {}, str(exc)
                            if kind == 'structure':
                                # Match the production structure reader's single
                                # correction attempt and include it in latency.
                                extra['initial_raw'] = raw
                                correction = [*messages[batch_index],
                                              {'role': 'assistant', 'content': [{'type': 'text', 'text': raw}]},
                                              {'role': 'user', 'content': [{'type': 'text', 'text': error + '. Return the complete corrected JSON.'}]}]
                                raw, count = engine.generate(correction, 2048, json_schema=structure_schema(len(row['expected']['rows']), systems))
                                tokens += count
                                try:
                                    predicted = parse_structure(raw, len(row['expected']['rows']), systems, row['row_modes'])
                                    error = None
                                except (ValueError, KeyError, TypeError, IndexError) as exc:
                                    error = str(exc)
                        value = {**row, **extra, 'predicted': predicted, 'raw': raw, 'tokens': tokens, 'error': error}
                        record(value)
                        output.write(json.dumps(value, ensure_ascii=False) + '\n')
                    output.flush()
                    print(json.dumps({'completed': len(completed), 'total': len(tasks), 'kind': kind, 'seconds': time.perf_counter() - started}), flush=True)
    finally:
        runtime = {'engine_startup_seconds': started - loading,
                   'worker_seconds': time.perf_counter() - started,
                   'speculative_tokens': args.speculative_tokens,
                   'speculation': engine.metrics}
        (args.output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        engine.close()
    if len(completed) != len(tasks):
        raise ValueError('Missing evaluation samples')
    (args.output / 'metrics.json').write_text(json.dumps(dict(metrics), indent=2))
    print(json.dumps(dict(metrics)), flush=True)


if __name__ == '__main__':
    main()
