"""Train page localization on explicit ensemble boxes and retained pitch regions."""

import argparse
from collections import Counter
import json
from pathlib import Path

from research.inference.layout.persistent import LayoutBackend


def add_engraved(source, engraved, output):
    """Add renderer geometry directly, without using predicted boxes as labels."""
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(exist_ok=True)
    link = output / 'images/base'
    if not link.exists():
        link.symlink_to((source / 'images').resolve(), target_is_directory=True)
    for split, tag in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        data = json.loads((source / 'annotations' / f'instance_{tag}.json').read_text())
        for image in data['images']:
            image['file_name'] = 'base/' + image['file_name']
        categories = {c['name']: c['id'] for c in data['categories']}
        image_id = max(i['id'] for i in data['images'])
        ann_id = max(a['id'] for a in data['annotations'])
        for path in sorted((engraved / split).glob('score-*/score.json')):
            score = json.loads(path.read_text())
            for page_number, page in enumerate(score['pages'], 1):
                image_id += 1
                name = f"{score['id']}-{page_number}.png"
                link = output / 'images' / name
                if not link.exists():
                    link.symlink_to(Path(page['image']).resolve())
                data['images'].append({'id': image_id, 'file_name': name, 'width': page['width'], 'height': page['height'],
                                       'source_id': score['id'], 'mode': 'notation', 'source_families': [score['id']]})
                boxes = [(categories['measure_notation'], r['bbox']) for r in score['records'] if r['page'] == page_number]
                boxes.extend((categories[r['kind'] + '_region'], r['bbox']) for r in page.get('regions', []))
                for order, (category, bbox) in enumerate(boxes):
                    ann_id += 1
                    x, y, w, h = bbox
                    data['annotations'].append({'id': ann_id, 'image_id': image_id, 'category_id': category,
                        'bbox': bbox, 'area': w * h, 'iscrowd': 0, 'read_order': order,
                        'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]]})
        (output / 'annotations' / f'instance_{tag}.json').write_text(json.dumps(data))
        print(split, {'pages': len(data['images']), 'regions': len(data['annotations'])}, flush=True)


def build(output, device='cuda:7'):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    (output / 'annotations').mkdir(exist_ok=True)
    original = Path('database/parallel_score_final/layout')
    link = output / 'images/original'
    if not link.exists():
        link.symlink_to((original / 'images').resolve(), target_is_directory=True)
    detector = LayoutBackend(device=device)
    reports = {}
    try:
        for split, tag in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            data = json.loads((original / 'annotations' / f'instance_{tag}.json').read_text())
            for image in data['images']:
                image['file_name'] = 'original/' + image['file_name']
            category_ids = {c['name']: c['id'] for c in data['categories']}
            image_id = max(i['id'] for i in data['images'])
            ann_id = max(a['id'] for a in data['annotations'])
            scores = [json.loads(line) for line in Path(f'database/ensemble_pages/scores_{split}.jsonl').open()]
            count = Counter()
            for index, score in enumerate(scores):
                path = Path(score['pdf']).with_name('score.json')
                score = json.loads(path.read_text())
                predictions = detector([p['image'] for p in score['pages']])
                for page_number, (page, prediction) in enumerate(zip(score['pages'], predictions, strict=True), 1):
                    image_id += 1
                    name = f'ensemble-{split}-{index}-{page_number}.png'
                    destination = output / 'images' / name
                    if not destination.exists():
                        destination.symlink_to(Path(page['image']).resolve())
                    data['images'].append({'id': image_id, 'file_name': name, 'width': page['width'], 'height': page['height'],
                                           'source_id': score['id'], 'mode': 'mixed', 'source_families': score['source_families']})
                    rows = [r for r in score['records'] if r['page'] == page_number]
                    boxes = [(category_ids['measure_' + r['mode']], r['bbox']) for r in rows]
                    for region in [*prediction.get('tempo_regions', []), *prediction.get('pitch_regions', [])]:
                        if region.get('score', 0) < .6:
                            continue
                        x0, y0, x1, y1 = region['coordinate']
                        boxes.append((category_ids[region['label']], [x0, y0, x1 - x0, y1 - y0]))
                    for order, (category, bbox) in enumerate(boxes):
                        ann_id += 1
                        x, y, w, h = bbox
                        data['annotations'].append({'id': ann_id, 'image_id': image_id, 'category_id': category,
                            'bbox': bbox, 'area': w * h, 'iscrowd': 0, 'read_order': order,
                            'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]]})
                    count.update(pages=1, expected_bars=len(rows), detected_bars=len(prediction['measures']),
                                 exact_count=int(len(rows) == len(prediction['measures'])))
                if (index + 1) % 100 == 0:
                    print(split, index + 1, dict(count), flush=True)
            (output / 'annotations' / f'instance_{tag}.json').write_text(json.dumps(data))
            reports[split] = dict(count)
            (output / 'baseline.json').write_text(json.dumps(reports, indent=2))
    finally:
        detector.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/score_support/layout'))
    parser.add_argument('--device', default='cuda:7')
    parser.add_argument('--extend-native', type=Path)
    parser.add_argument('--source', type=Path, default=Path('database/score_support/layout'))
    args = parser.parse_args()
    if args.extend_native:
        add_engraved(args.source, args.extend_native, args.output)
    else:
        build(args.output, args.device)
