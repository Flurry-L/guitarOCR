"""Add short pages without cutting any labelled score or text region.

Unlike arbitrary detector crops, a retained notation+TAB label always retains
both staves. Every intersecting annotation expands the crop until stable.
Original train/validation membership is preserved; predictions are never labels.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random

from PIL import Image


def complete_band(image, annotations, anchor, *, include_header=False, margin=20):
    """Close a horizontal band over all intersecting annotation intervals."""
    top = 0 if include_header else max(0, math.floor(anchor['bbox'][1] - margin))
    bottom = min(image['height'], math.ceil(anchor['bbox'][1] + anchor['bbox'][3] + margin))
    while True:
        previous = top, bottom
        for item in annotations:
            _, y, _, height = item['bbox']
            if y < bottom and y + height > top:
                top = max(0, min(top, math.floor(y)))
                bottom = min(image['height'], max(bottom, math.ceil(y + height)))
        if previous == (top, bottom):
            return top, bottom


def family_keys(image):
    return set(image.get('source_families') or [image.get('family', image.get('source_id'))]) - {None}


def build(dataset, output, split, per_group, *, include_original=False, exclude=None, minimum_modes=1):
    dataset, output = Path(dataset), Path(output)
    annotation_path = dataset / 'annotations' / f'instance_{split}.json'
    data = json.loads(annotation_path.read_text())
    by_image = defaultdict(list)
    for annotation in data['annotations']:
        by_image[annotation['image_id']].append(annotation)
    measures = {row['id'] for row in data['categories'] if row['name'].startswith('measure_')}
    excluded_families = set()
    exclusions = [exclude] if isinstance(exclude, (str, Path)) else (exclude or [])
    for excluded_manifest in exclusions:
        previous = json.loads(Path(excluded_manifest).read_text())
        for sample in previous['samples']:
            excluded_families.update(sample['source_families'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images' / 'short').mkdir(parents=True)
    (output / 'annotations').mkdir()
    (output / 'images' / 'original').symlink_to((dataset / 'images').resolve(), target_is_directory=True)
    images = [{**row, 'file_name': 'original/' + row['file_name']} for row in data['images']] if include_original else []
    annotations = list(data['annotations']) if include_original else []
    image_id = max(row['id'] for row in data['images'])
    annotation_id = max(row['id'] for row in data['annotations'])
    counts, variants, samples = Counter(), Counter(), []
    ordered = list(data['images'])
    random.Random(split).shuffle(ordered)
    for image in ordered:
        if family_keys(image) & excluded_families:
            continue
        group = (image.get('instrument', 'ensemble'), image.get('mode', 'mixed'))
        if counts[group] >= per_group:
            continue
        boxes = by_image[image['id']]
        music = sorted((a for a in boxes if a['category_id'] in measures), key=lambda a: (a['bbox'][1], a['bbox'][0]))
        if not music or any(not isinstance(a.get('segmentation'), list) for a in boxes):
            continue
        rng = random.Random(f'{split}:{image["id"]}')
        with_header = counts[group] % 2 == 0
        anchor = music[0] if with_header else rng.choice(music)
        chosen = {anchor['category_id']}
        if len({row['category_id'] for row in music}) < minimum_modes:
            continue
        while len(chosen) < minimum_modes:
            other = min((row for row in music if row['category_id'] not in chosen),
                        key=lambda row: abs(row['bbox'][1] - anchor['bbox'][1]))
            top = min(anchor['bbox'][1], other['bbox'][1])
            bottom = max(anchor['bbox'][1] + anchor['bbox'][3], other['bbox'][1] + other['bbox'][3])
            anchor = {'bbox': [0, top, image['width'], bottom - top]}
            chosen.add(other['category_id'])
        top, bottom = complete_band(image, boxes, anchor, include_header=with_header)
        if bottom - top >= image['width'] * .9:
            continue
        selected = [a for a in boxes if a['bbox'][1] < bottom and a['bbox'][1] + a['bbox'][3] > top]
        if any(a['bbox'][1] < top or a['bbox'][1] + a['bbox'][3] > bottom for a in selected):
            raise ValueError('A labelled region crosses the complete crop boundary')
        actual_modes = sorted({a['category_id'] for a in selected if a['category_id'] in measures})
        if len(actual_modes) < minimum_modes:
            raise ValueError('The crop does not retain the requested notation modes')
        image_id += 1
        filename = f'short/{split}-{image["id"]}.png'
        with Image.open(dataset / 'images' / image['file_name']) as source:
            if source.size != (image['width'], image['height']):
                raise ValueError(f"Source image dimensions differ from annotations: {image['file_name']}")
            source.crop((0, top, image['width'], bottom)).save(output / 'images' / filename, compress_level=1)
        images.append({**image, 'id': image_id, 'file_name': filename, 'height': bottom - top,
                       'source_file_name': image['file_name'], 'source_crop': [0, top, image['width'], bottom]})
        for order, source in enumerate(sorted(selected, key=lambda a: a.get('read_order', 0))):
            row = deepcopy(source)
            annotation_id += 1
            row.update(id=annotation_id, image_id=image_id, read_order=order)
            row['bbox'][1] -= top
            row['segmentation'] = [[value - top if index % 2 else value for index, value in enumerate(polygon)]
                                   for polygon in row['segmentation']]
            annotations.append(row)
        counts[group] += 1
        variants['with_header' if with_header else 'complete_band'] += 1
        samples.append({'image_id': image_id, 'source_image_id': image['id'], 'source_families': sorted(family_keys(image)),
                        'group': list(group), 'crop': [0, top, image['width'], bottom], 'with_header': with_header,
                        'annotations': len(selected), 'retained_measure_category_ids': actual_modes})
        if len(samples) % 100 == 0:
            print(split, 'short pages', len(samples), flush=True)
    result = {**data, 'images': images, 'annotations': annotations}
    (output / 'annotations' / f'instance_{split}.json').write_text(json.dumps(result, ensure_ascii=False))
    report = {'source': str(dataset.resolve()),
              'split': split, 'training_allowed': split == 'train', 'original_pages': len(data['images']) if include_original else 0,
              'short_pages': len(samples), 'total_pages': len(images), 'annotations': len(annotations),
              'groups': {'/'.join(key): count for key, count in counts.items()}, 'variants': dict(variants),
              'excluded_families': sorted(excluded_families), 'samples': samples,
              'minimum_retained_measure_modes': minimum_modes,
              'contract': 'No stretching, partial annotations, detector labels, or cross-split source mixing.'}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--per-group', type=int, default=300)
    parser.add_argument('--include-original', action='store_true')
    parser.add_argument('--exclude', type=Path, nargs='*')
    parser.add_argument('--minimum-modes', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    report = build(args.dataset, args.output, args.split, args.per_group,
                   include_original=args.include_original, exclude=args.exclude, minimum_modes=args.minimum_modes)
    print(json.dumps({key: value for key, value in report.items() if key not in {'samples', 'excluded_families'}}, indent=2))
