"""Evaluate complete scores with predicted context, including score latency."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from measure_ocr.evaluate import evaluate
from measure_ocr.parallel import recognize_independent
from measure_ocr.recognizer import recognize_crops
from shared.defaults import MODEL, MEASURE_ADAPTER
from shared.artifacts import write_json
from shared.glm_backend import GlmBackend
from shared.score_state import parse_signature


def selected_scores(manifest, limit, seed):
    groups = defaultdict(list)
    with manifest.open() as source:
        for line in source:
            row = json.loads(line)
            groups[(row['source_id'], row['mode'])].append(row)
    buckets = defaultdict(list)
    for key, rows in groups.items():
        rows.sort(key=lambda r: r['measure_index'])
        if [r['measure_index'] for r in rows] != list(range(len(rows))):
            raise ValueError(f'Incomplete measure sequence: {key}')
        if rows[0].get('source_measures', len(rows)) != len(rows):
            raise ValueError(f'Missing end of score: {key}')
        buckets[(rows[0].get('instrument', 'guitar'), key[1])].append(key)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    keys = []
    while any(buckets.values()) and (not limit or len(keys) < limit):
        for name in sorted(buckets):
            if buckets[name] and (not limit or len(keys) < limit):
                keys.append(buckets[name].pop())
    return [(key, groups[key]) for key in keys]


def shard_scores(scores, index, shards):
    if shards == 1:
        return scores
    # Instrument buckets repeat in a fixed cycle; striding can put almost all
    # long guitar scores on one GPU. Assign whole sequences by bar count and
    # retain their original order within each worker.
    assignments = [[] for _ in range(shards)]
    loads = [0] * shards
    for position, score in sorted(enumerate(scores), key=lambda item: len(item[1][1]), reverse=True):
        rank = min(range(shards), key=lambda i: (loads[i], len(assignments[i]), i))
        assignments[rank].append((position, score))
        loads[rank] += len(score[1])
    return [score for _position, score in sorted(assignments[index])]


def worker(args):
    scores = shard_scores(selected_scores(args.manifest, args.max_scores, args.seed), args.shard_index, args.shards)
    capability_path = Path(args.adapter or args.model) / 'capabilities.json'
    capabilities = json.loads(capability_path.read_text()) if capability_path.is_file() else {}
    if capabilities.get('visual_pitch') and any(
        row['mode'] == 'both' and row.get('instrument', 'guitar') != 'drums' and not row.get('pitch_context')
        for _key, rows in scores for row in rows
    ):
        raise ValueError('This model reads written pitches from notation+TAB. Rebuild missing pitch contexts with datagen.score_support_data before evaluation.')
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = args.output / f'predictions-{args.shard_index}.jsonl'
    timings = args.output / f'timings-{args.shard_index}.jsonl'
    completed = set()
    if args.resume and predictions.exists():
        for line in predictions.open():
            completed.add(json.loads(line)['id'])
        assigned = {row['id'] for _key, rows in scores for row in rows}
        if not completed.issubset(assigned):
            raise ValueError('Saved predictions use a different GPU partition; use a fresh output directory')
    loading = time.perf_counter()
    if args.engine == 'vllm':
        from shared.vllm_backend import VllmBackend

        backend = VllmBackend(args.model, 'cuda:0', options={
            'speculative_tokens': args.speculative_tokens,
            'gpu_memory_utilization': args.gpu_memory,
            'kv_cache_memory_bytes': args.kv_cache_mb * 1024 ** 2,
            'max_num_seqs': max(32, args.batch_size),
            **args.engine_options,
        })
    else:
        backend = GlmBackend(args.model, args.adapter, 'cuda:0')
    state_reader = None
    if args.state_model:
        from measure_ocr.state_reader import StateReader

        state_reader = StateReader(args.state_model, 'cuda:0')
    started = time.perf_counter()
    with predictions.open('a' if args.resume else 'w') as output, timings.open('a' if args.resume else 'w') as times:
        for index, ((source_id, mode), truth) in enumerate(scores):
            if all(r['id'] in completed for r in truth):
                continue
            rows = copy.deepcopy(truth)
            label = truth[0].get('label_json')
            track = json.loads(Path(label).read_text()).get('track', {}) if label else {}
            corpus = truth[0].get('corpus') or ('engraved' if source_id.startswith('engraved-') else
                     'techniques' if 'parallel_techniques' in str(label) else
                     'synthetic' if 'parallel_synthetic' in str(label) else 'original')
            for row in rows:
                row['measure_number'] = row['measure_index'] + 1
                row.pop('target')
                row.pop('score_state', None)
            directory = args.output / 'scores' / f'{source_id}-{mode}'
            infer = recognize_crops if args.legacy else recognize_independent
            kwargs = {} if args.legacy else {'batch_size': args.batch_size, 'state_reader': state_reader,
                                             'batch_order': args.batch_order, 'constrained_decoding': args.constrained_decoding}
            before = time.perf_counter()
            infer(rows, mode, args.model, args.adapter, 'cuda:0', args.max_new_tokens,
                  4096, truth[0].get('tuning', []), args.maximum_attempts,
                  directory / 'recognition.jsonl', args.resume, backend=backend,
                  instrument=truth[0].get('instrument', 'guitar'), **kwargs)
            elapsed = time.perf_counter() - before
            raw_by_number = {}
            for line in (directory / 'recognition.jsonl').open():
                r = json.loads(line)
                if 'raw' in r:
                    raw_by_number[r['measure_number']] = r['target']
            for expected, row in zip(truth, rows, strict=True):
                value = {k: expected.get(k) for k in ('id', 'source_id', 'mode', 'image', 'instrument', 'pitch_context')}
                value.update(expected=expected.get('sounding_target', expected['target']), predicted=row['target'],
                             corpus=corpus, midi_program=expected.get('midi_program', track.get('midi_program')),
                             raw_prediction=raw_by_number.get(row['measure_number'], row['target']),
                             needs_review=row.get('needs_review', False), tuning=row.get('tuning'),
                             string_count=expected.get('string_count', len(expected.get('tuning') or [])),
                             context_source='predicted', measure_index=row['measure_index'],
                             score_state=row.get('score_state'), expected_state=expected['score_state'],
                             printed_state=row.get('printed_state'), expected_signature=expected['signature_target'],
                             boundary_resolved=row.get('boundary_resolved', False))
                if expected['id'] not in completed:
                    output.write(json.dumps(value, ensure_ascii=False) + '\n')
            output.flush()
            timing = {'source_id': source_id, 'mode': mode, 'measures': len(rows), 'seconds': elapsed,
                      'speculation': getattr(backend, 'metrics', [])}
            times.write(json.dumps(timing) + '\n')
            times.flush()
            print(json.dumps({'score': index + 1, 'total': len(scores), **timing}), flush=True)
    runtime = {'engine_startup_seconds': started - loading,
               'worker_seconds': time.perf_counter() - started,
               'speculation': getattr(backend, 'metrics', [])}
    (args.output / f'runtime-{args.shard_index}.json').write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime), flush=True)
    if hasattr(backend, 'close'):
        backend.close()


def aggregate(args, shards):
    predictions = args.output / 'predictions.jsonl'
    with predictions.open('w') as output:
        for i in range(shards):
            with (args.output / f'predictions-{i}.jsonl').open() as source:
                for line in source:
                    output.write(line)
    result = evaluate(predictions, None)
    counts = defaultdict(int)
    for line in predictions.open():
        row = json.loads(line)
        if row.get('printed_state'):
            state = row['printed_state']
            gold = parse_signature(row['expected_signature'])
            for field in ('time', 'key'):
                counts[f'{field}_printed_correct'] += state.get(field) == gold[field] and 'error' not in state
                if field != 'key' or row['mode'] != 'tab':
                    counts[f'{field}_effective_samples'] += 1
                    counts[f'{field}_effective_correct'] += row['score_state'][field] == row['expected_state'][field]
                if gold[field] is not None:
                    counts[f'{field}_positive'] += 1
                    counts[f'{field}_positive_correct'] += state.get(field) == gold[field]
            counts['state_samples'] += 1
    result['state_counts'] = dict(counts)
    durations = [json.loads(line) for i in range(shards) for line in (args.output / f'timings-{i}.jsonl').open()]
    import numpy as np

    elapsed = [r['seconds'] for r in durations]
    result['latency'] = {'scores': len(durations), 'measures': sum(r['measures'] for r in durations),
                         'sum_score_seconds': sum(elapsed), 'score_median_seconds': float(np.median(elapsed)),
                         'score_p95_seconds': float(np.percentile(elapsed, 95))}
    result['engine'] = args.engine
    result['engine_options'] = args.engine_options
    result['kv_cache_memory_bytes'] = args.kv_cache_mb * 1024 ** 2
    result['speculative_tokens'] = args.speculative_tokens
    state_model = args.state_model
    capability_path = Path(args.adapter or args.model) / 'capabilities.json'
    if not state_model and not args.legacy and capability_path.exists():
        configured = json.loads(capability_path.read_text()).get('state_reader')
        if configured:
            state_model = (capability_path.parent / configured).resolve()
    result['state_model'] = str(state_model) if state_model else None
    result['model'] = str(args.model)
    result['batch_size'] = args.batch_size
    result['batch_order'] = args.batch_order or ('score' if args.engine == 'vllm' else 'area')
    result['constrained_decoding'] = (bool(json.loads(capability_path.read_text()).get('m2_constraints'))
                                     if args.constrained_decoding is None and capability_path.is_file()
                                     else bool(args.constrained_decoding)) and not args.legacy
    capabilities = json.loads(capability_path.read_text()) if capability_path.is_file() else {}
    result['retry_constrained_decoding'] = not args.legacy and args.engine == 'vllm' and (
        result['constrained_decoding'] or (args.constrained_decoding is not False
                                         and bool(capabilities.get('m2_retry_constraints'))))
    result['postprocessing'] = [] if args.legacy else ['printed_state', 'fretted_pitches', 'cross_bar_ties']
    result['workers'] = [json.loads(path.read_text()) for i in range(shards)
                         if (path := args.output / f'runtime-{i}.json').exists()]
    result['scope'] = 'complete_score_crops_with_predicted_context'
    result['input_conditions'] = {
        'measure_crops': 'reference',
        'track_instrument_tuning_and_pitch_context': 'reference',
        'previous_generated_measures': 'predicted' if args.legacy else 'unused',
        'time_and_key_signatures': 'decoder' if args.legacy else 'predicted_visual_state',
        'gpu_work_assignment': 'complete_scores_balanced_by_bar_count' if shards > 1 else 'single_gpu',
    }
    write_json(args.output / 'metrics.json', result)
    print(json.dumps({'overall': result['overall'], 'latency': result['latency'], 'state': result['state_counts']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Merged model for vLLM; defaults to the adapter inference configuration')
    parser.add_argument('--adapter', type=Path, default=MEASURE_ADAPTER)
    parser.add_argument('--engine', choices=['transformers', 'vllm'], default='vllm')
    parser.add_argument('--engine-options', type=json.loads, default={})
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--speculative-tokens', type=int, default=0)
    parser.add_argument('--constrained-decoding', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--state-model', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-order', choices=['score', 'area'])
    parser.add_argument('--max-scores', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--maximum-attempts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--gpu-memory', type=float, default=.4)
    parser.add_argument('--kv-cache-mb', type=int, default=2048)
    parser.add_argument('--shard-index', type=int, default=-1)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.model is None:
        if args.engine == 'vllm':
            inference = args.adapter / 'inference.json'
            if not inference.is_file():
                parser.error('vLLM requires a merged model: pass --model or publish the adapter inference.json')
            args.model = (inference.parent / json.loads(inference.read_text())['model']).resolve()
        else:
            args.model = MODEL
    devices = args.gpus.split(',')
    if args.aggregate:
        aggregate(args, len(devices))
    elif args.shard_index >= 0:
        worker(args)
    else:
        args.output.mkdir(parents=True, exist_ok=True)

        def launch(item):
            i, device = item
            command = [sys.executable, '-m', 'measure_ocr.evaluate_scores', *sys.argv[1:],
                       '--shard-index', str(i), '--shards', str(len(devices))]
            with (args.output / f'worker-{i}.log').open('a' if args.resume else 'w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': device, 'OMP_NUM_THREADS': '1'})
        with ThreadPoolExecutor(len(devices)) as pool:
            list(pool.map(launch, enumerate(devices)))
        aggregate(args, len(devices))


if __name__ == '__main__':
    main()
