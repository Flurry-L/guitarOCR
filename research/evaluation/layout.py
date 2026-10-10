"""Evaluate layout through the native application, including missed regions."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

from research.inference.run import NativeClient

MATCH_IOU = .5


def summarize(rows):
    totals = {key: sum(row[key] for row in rows)
              for key in ("expected", "predicted", "matched", "correct_mode")}
    return {"pages": len(rows), **totals,
            "recall": totals["matched"] / max(1, totals["expected"]),
            "precision": totals["matched"] / max(1, totals["predicted"]),
            "matched_mode_accuracy": totals["correct_mode"] / max(1, totals["matched"]),
            "perfect_pages": sum(row["expected"] == row["predicted"] == row["matched"] == row["correct_mode"]
                                 for row in rows)}


def iou(first, second):
    a, b, c, d = first['coordinate']
    x, y, r, s = second['coordinate']
    hit = max(0, min(c, r) - max(a, x)) * max(0, min(d, s) - max(b, y))
    return hit / max(1e-9, (c - a) * (d - b) + (r - x) * (s - y) - hit)


def measure_metrics(truth, predictions):
    available = set(range(len(truth)))
    used, matches = set(), []
    pairs = [(iou(box, gold), index, candidate) for index, box in enumerate(predictions)
             for candidate, gold in enumerate(truth)]
    for overlap, index, candidate in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if overlap >= MATCH_IOU and index not in used and candidate in available:
            matches.append({'prediction': index, 'truth': candidate, 'iou': overlap,
                            'correct_mode': predictions[index]['label'] == truth[candidate]['label']})
            available.remove(candidate)
            used.add(index)
    return {'expected': len(truth), 'predicted': len(predictions), 'matched': len(matches),
            'correct_mode': sum(m['correct_mode'] for m in matches),
            'missed_truth': sorted(available), 'extra_predictions': sorted(set(range(len(predictions))) - used),
            'matches': matches}


def coco_pages(annotations, images):
    data = json.loads(Path(annotations).read_text())
    categories = {row['id']: row['name'] for row in data['categories']}
    boxes = defaultdict(list)
    for row in data['annotations']:
        if categories[row['category_id']].startswith('measure_'):
            x, y, w, h = row['bbox']
            boxes[row['image_id']].append({'label': categories[row['category_id']],
                'coordinate': [x, y, x + w, y + h], 'annotation_id': row['id'],
                'musical_case': row.get('musical_case')})
    return [{'id': str(row['id']), 'dataset': row.get('selection_reason', 'coco'),
             'image': str(Path(images) / row['file_name']),
             'source_families': row.get('source_families', [row.get('family', row.get('source_id'))]),
             'truth': boxes[row['id']]} for row in data['images']]


def evaluate(pages, endpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    with NativeClient(endpoint) as client:
        for index, source in enumerate(pages):
            page_started = time.monotonic()
            try:
                project = client.recognize([source['image']], output / 'pages' / str(index), action='detect')
            except (RuntimeError, OSError, httpx.HTTPError) as error:
                project = {'error': str(error)}
            boxes = []
            for box in project.get('boxes', []):
                if box['kind'] != 'measure':
                    continue
                x, y, width, height = box['bbox']
                boxes.append({'label': 'measure_' + box['mode'],
                              'coordinate': [x, y, x + width, y + height]})
            row = {'source': source, 'prediction': boxes, 'project': project.get('id'),
                   'seconds': time.monotonic() - page_started, 'error': project.get('error'),
                   'metrics': measure_metrics(source['truth'], boxes)}
            rows.append(row)
            (output / f'{index}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
            print(source['id'], row['metrics'], flush=True)
    result = {'scope': 'native_application_layout', 'seconds': time.monotonic() - started,
              'failed_pages': sum(bool(row['error']) for row in rows),
              'overall': summarize([row['metrics'] for row in rows])}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:7860')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    pages = coco_pages(args.annotations, args.images)
    print(json.dumps(evaluate(pages, args.endpoint, args.output), ensure_ascii=False, indent=2))
