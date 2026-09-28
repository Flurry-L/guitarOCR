"""Evaluate complete PDFs through detection, metadata, recognition and GP5 export."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from layout.persistent import LayoutBackend
from measure_ocr.evaluate_scores import selected_scores
from measure_ocr.metrics import MeasureSequenceMetrics
from pipeline.config import parse_args
from pipeline.run import _run
from shared.artifacts import read_result
from shared.defaults import INFO_ADAPTER, LAYOUT_MODEL, MEASURE_ADAPTER, MODEL
from shared.glm_backend import BackendPool


def score_pdf(row):
    root = Path(row['label_json']).parent.parent
    document = f"{row['mode']}-{row['source_id']}"
    folders = [root / 'native-export', root.with_name(root.name + '_native')]
    for folder in folders:
        paths = list((folder / 'documents' / document / 'tracks').glob('*/score.pdf'))
        if len(paths) == 1:
            return paths[0]
    path = root / 'pdf' / row['mode'] / (row['source_id'] + '.pdf')
    if path.is_file():
        return path
    raise FileNotFoundError(f'No complete PDF for {document}')


def overlap(a, b):
    x, y, w, h = a
    u, v, s, t = b
    area = max(0., min(x + w, u + s) - max(x, u)) * max(0., min(y + h, v + t) - max(y, v))
    return area / max(1., w * h + s * t - area)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--adapter', type=Path, default=MEASURE_ADAPTER)
    parser.add_argument('--info-adapter', type=Path, default=INFO_ADAPTER)
    parser.add_argument('--layout-model', type=Path, default=LAYOUT_MODEL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-scores', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    scores = selected_scores(args.manifest, args.max_scores, args.seed)
    # Resolve inputs before allocating GPU resources; no missing source is skipped.
    pdfs = [score_pdf(rows[0]) for _, rows in scores]
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, counts = MeasureSequenceMetrics(), Counter()
    by_mode = defaultdict(MeasureSequenceMetrics)
    timings = []
    pool = BackendPool(args.model, args.device)
    detector = LayoutBackend(args.layout_model, device=args.device)
    with ExitStack() as resources, (args.output/'predictions.jsonl').open('w') as predictions, (args.output/'scores.jsonl').open('w') as reports:
        resources.callback(pool.close)
        resources.callback(detector.close)
        loader = resources.enter_context(ThreadPoolExecutor(max_workers=1))
        preparing = loader.submit(pool.prepare, [args.info_adapter, args.adapter])
        for ((source, mode), truth), pdf in zip(scores, pdfs, strict=True):
            directory = args.output / 'scores' / f'{source}-{mode}'
            result_path = directory / 'manifest.json'
            if args.resume and result_path.exists() and json.loads(result_path.read_text()).get('status') in {'complete','needs_review'}:
                result = json.loads(result_path.read_text())
            else:
                options = parse_args([str(pdf), '--output', str(directory), '--model', str(args.model),
                    '--adapter', str(args.adapter), '--info-adapter', str(args.info_adapter),
                    '--layout-model-dir', str(args.layout_model), '--device', args.device,
                    '--layout-source', 'image', '--max-new-tokens', '2048', '--max-new-tokens-ceiling', '4096'])
                before = time.perf_counter()
                try:
                    result = _run(options, pool, detector, preparing=preparing)
                except Exception as error:
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory/'error.log').write_text(traceback.format_exc())
                    partial = json.loads(result_path.read_text()) if result_path.exists() else {}
                    result = {**partial, 'status':'failed','error':str(error), 'mode':partial.get('mode'),
                              'gp5':None, 'review_measures':[], 'seconds':time.perf_counter()-before}
                timings.append(time.perf_counter()-before)
            recognized = result.get('stages', {}).get('measure_ocr', {}).get('manifest')
            recognition = read_result(Path(recognized), 'measure_ocr') if recognized and Path(recognized).is_file() else {}
            rows = recognition.get('records', [])
            if recognition:
                result['mode'] = recognition['mode']
                result['review_measures'] = recognition.get('review_measures', [])
            pairs = []
            for i, expected in enumerate(truth):
                box = ([v * 180 / 25.4 for v in expected['bbox_mm']] if 'bbox_mm' in expected else expected['bbox'])
                for j, row in enumerate(rows):
                    if row['page'] == expected['page']:
                        iou = overlap(box, row['bbox'])
                        if iou >= .5:
                            pairs.append((iou, i, j))
            matched, used = {}, set()
            for _, i, j in sorted(pairs, reverse=True):
                if i not in matched and j not in used:
                    matched[i] = j
                    used.add(j)
            for i, expected in enumerate(truth):
                row = rows[matched[i]] if i in matched else {}
                target = row.get('target', '')
                gold = expected.get('sounding_target', expected['target'])
                sample = MeasureSequenceMetrics()
                sample.update(gold, target, mode=mode, tuning=expected.get('tuning'), string_count=expected.get('string_count'))
                for metric in (metrics, by_mode[mode]):
                    metric.merge(sample)
                predictions.write(json.dumps({'id':expected['id'], 'source_id':source,
                    'expected':gold, 'predicted':target, 'mode':mode,
                    'instrument':expected.get('instrument', 'guitar'),
                    'tuning':expected.get('tuning'), 'string_count':expected.get('string_count'),
                    'pitch_context':expected.get('pitch_context'), 'context_source':'predicted',
                    'matched':i in matched, 'needs_review':row.get('needs_review', True)}, ensure_ascii=False)+'\n')
            counts.update(scores=1, expected_bars=len(truth), detected_bars=len(rows), matched_bars=len(matched),
                          extra_bars=len(rows)-len(used), exact_bar_count=int(len(rows)==len(truth)),
                          correct_mode=int(result['mode']==mode), exported=int(bool(result.get('gp5'))),
                          failed=int(result['status']=='failed'))
            report = {'source_id':source, 'mode':mode, 'expected_bars':len(truth), 'detected_bars':len(rows),
                      'matched_bars':len(matched), 'seconds':result['seconds'], 'gp5':result.get('gp5'),
                      'review_measures':result['review_measures'], 'error':result.get('error')}
            reports.write(json.dumps(report,ensure_ascii=False)+'\n')
            reports.flush()
            predictions.flush()
            print(json.dumps(report,ensure_ascii=False),flush=True)
    report = {'scope':'complete_pdfs_with_predicted_layout_and_metadata', 'counts':dict(counts),
              'overall':metrics.result(), 'by_mode':{k:v.result() for k,v in by_mode.items()},
              'fresh_run_seconds':sum(timings)}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'counts':dict(counts),'core_exact':metrics.result()['core_exact_rate']}),flush=True)


if __name__ == '__main__':
    main()
