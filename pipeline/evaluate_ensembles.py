"""Evaluate whole ensemble PDFs with predicted layout, parts and music content."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from document_info.evaluate_structure import canonical_rows
from layout.persistent import LayoutBackend
from measure_ocr.metrics import MeasureSequenceMetrics
from pipeline.config import parse_args
from pipeline.evaluate_scores import overlap
from pipeline.run import _run
from shared.artifacts import read_result
from shared.defaults import INFO_ADAPTER, LAYOUT_MODEL, MEASURE_ADAPTER, MODEL
from shared.glm_backend import BackendPool


def sources(roots, split):
    return sorted(path for root in roots for path in (root / split).glob('score-*/score.json'))


def worker(args):
    scores = sources(args.data, args.split)[args.shard_index::args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    pool = BackendPool(args.model, args.device)
    detector = LayoutBackend(args.layout_model, device=args.device)
    with ExitStack() as resources, (args.output / f'scores-{args.shard_index}.jsonl').open('w') as reports, \
            (args.output / f'predictions-{args.shard_index}.jsonl').open('w') as predictions:
        resources.callback(pool.close)
        resources.callback(detector.close)
        loader = resources.enter_context(ThreadPoolExecutor(max_workers=1))
        preparing = loader.submit(pool.prepare, [args.info_adapter, args.adapter])
        for path in scores:
            truth = json.loads(path.read_text())
            directory = args.output / 'scores' / truth['id']
            manifest = directory / 'manifest.json'
            if args.resume and manifest.exists() and json.loads(manifest.read_text()).get('status') in {'complete', 'needs_review'}:
                result = json.loads(manifest.read_text())
            else:
                options = parse_args([truth['pdf'], '--output', str(directory), '--model', str(args.model),
                                      '--adapter', str(args.adapter), '--info-adapter', str(args.info_adapter),
                                      '--layout-model-dir', str(args.layout_model), '--device', args.device,
                                      '--layout-source', 'image', '--max-new-tokens', '2048', '--max-new-tokens-ceiling', '4096'])
                before = time.perf_counter()
                try:
                    result = _run(options, pool, detector, preparing=preparing)
                except Exception as exc:
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'error.log').write_text(traceback.format_exc())
                    result = json.loads(manifest.read_text()) if manifest.exists() else {}
                    result.update(status='failed', error=str(exc), seconds=time.perf_counter() - before)
            recognition = result.get('stages', {}).get('measure_ocr', {}).get('manifest')
            data = read_result(Path(recognition), 'measure_ocr') if recognition and Path(recognition).exists() else {}
            rows, pairs = data.get('records', []), []
            for i, gold in enumerate(truth['records']):
                for j, row in enumerate(rows):
                    if gold['page'] == row['page'] and (iou := overlap(gold['bbox'], row['bbox'])) >= .5:
                        pairs.append((iou, i, j))
            matched, used = {}, set()
            for _iou, i, j in sorted(pairs, reverse=True):
                if i not in matched and j not in used:
                    matched[i] = j
                    used.add(j)
            gold_groups, predicted_groups = [], []
            counts = Counter(scores=1, expected_bars=len(truth['records']), detected_bars=len(rows), matched_bars=len(matched),
                             extra_bars=len(rows) - len(used), exported=bool(result.get('gp5')), failed=result.get('status') == 'failed')
            for i, gold in enumerate(truth['records']):
                row = rows[matched[i]] if i in matched else {}
                if row:
                    gold_groups.append([gold['bar_index'], gold['part_id'], gold['staff_id']])
                    predicted_groups.append([row.get('bar_index'), row.get('part_id'), row.get('staff_id')])
                counts['aligned_bars'] += bool(row) and row.get('bar_index') == gold['bar_index']
                counts['mode_correct'] += row.get('mode') == gold['mode']
                counts['instrument_correct'] += row.get('instrument') == gold['instrument']
                counts['program_total'] += gold.get('midi_program') is not None
                counts['program_correct'] += gold.get('midi_program') is not None and row.get('midi_program') == gold['midi_program']
                value = {k: gold.get(k) for k in ('id', 'mode', 'instrument', 'tuning', 'midi_program', 'part_name')}
                value.update(score=truth['id'], corpus=path.parents[2].name, expected=gold.get('sounding_target', gold['target']),
                             predicted=row.get('target', ''), matched=i in matched, needs_review=row.get('needs_review', True))
                predictions.write(json.dumps(value, ensure_ascii=False) + '\n')
            counts['grouping_exact'] += (len(matched) == len(truth['records']) == len(rows)
                                         and canonical_rows(gold_groups) == canonical_rows(predicted_groups)
                                         and counts['aligned_bars'] == len(rows))
            report = {'score': truth['id'], 'corpus': path.parents[2].name, 'counts': dict(counts),
                      'seconds': result.get('seconds'), 'error': result.get('error'), 'gp5': result.get('gp5')}
            reports.write(json.dumps(report, ensure_ascii=False) + '\n')
            reports.flush()
            predictions.flush()
            print(json.dumps(report, ensure_ascii=False), flush=True)


def aggregate(args):
    counts, metrics, timings = defaultdict(Counter), defaultdict(MeasureSequenceMetrics), []
    seen = set()
    with (args.output / 'predictions.jsonl').open('w') as output:
        for shard in range(args.shards):
            for line in (args.output / f'predictions-{shard}.jsonl').open():
                row = json.loads(line)
                output.write(line)
                sample = MeasureSequenceMetrics()
                sample.update(row['expected'], row['predicted'], mode=row['mode'], tuning=row.get('tuning'),
                              string_count=len(row.get('tuning') or []))
                for group in ('overall', row['corpus'], 'mode/' + row['mode'], 'instrument/' + row['instrument'], 'program/' + str(row['midi_program'])):
                    metrics[group].merge(sample)
            for line in (args.output / f'scores-{shard}.jsonl').open():
                row = json.loads(line)
                if row['score'] in seen:
                    raise ValueError('Duplicate score evaluation')
                seen.add(row['score'])
                counts['overall'].update(row['counts'])
                counts[row['corpus']].update(row['counts'])
                if row['seconds'] is not None:
                    timings.append(row['seconds'])
    if len(seen) != len(sources(args.data, args.split)):
        raise ValueError('Incomplete whole-score evaluation')
    import numpy as np

    result = {'scope': 'whole_pdf_predicted_layout_structure_metadata_and_notes', 'counts': dict(counts),
              'metrics': {k: v.result() for k, v in metrics.items()}, 'latency': {'sum_seconds': sum(timings),
              'median_seconds': float(np.median(timings)) if timings else None}}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'counts': dict(counts), 'overall': result['metrics'].get('overall')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, action='append', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--adapter', type=Path, default=MEASURE_ADAPTER)
    parser.add_argument('--info-adapter', type=Path, default=INFO_ADAPTER)
    parser.add_argument('--layout-model', type=Path, default=LAYOUT_MODEL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--shard-index', type=int, default=-1)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    devices = args.gpus.split(',')
    if args.shard_index >= 0:
        worker(args)
    else:
        args.shards = len(devices)
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.aggregate:
            def launch(item):
                index, device = item
                with (args.output / f'worker-{index}.log').open('a' if args.resume else 'w') as log:
                    subprocess.run([sys.executable, '-m', 'pipeline.evaluate_ensembles', *sys.argv[1:], '--shard-index', str(index),
                                    '--shards', str(args.shards)], check=True, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': device, 'OMP_NUM_THREADS': '1'})
            with ThreadPoolExecutor(len(devices)) as pool:
                list(pool.map(launch, enumerate(devices)))
        aggregate(args)


if __name__ == '__main__':
    main()
