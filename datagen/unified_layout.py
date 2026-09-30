"""Localize score regions using renderer geometry, including composed pages."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random

from PIL import Image

from shared.layout_labels import TYPED_CATEGORIES


@lru_cache(maxsize=2048)
def native_regions(label, mode):
    path = Path(label)
    root, source = path.parent.parent, path.stem
    native = {'pitch_training': 'pitch_native', 'pitch_named': 'pitch_named_native',
              'parallel_synthetic': 'parallel_synthetic_native', 'parallel_techniques': 'parallel_techniques_native'}
    export = Path('database') / native[root.name] if root.name in native else root / 'native-export'
    paths = list((export / 'documents' / f'{mode}-{source}').glob('tracks/*/layout.json'))
    if not paths:
        return {}
    layout = json.loads(paths[0].read_text())
    regions = defaultdict(list)
    for region in layout.get('pitch_regions', []):
        label = 'clef_region' if region['kind'] == 'clef' else 'annotation_region'
        regions[region['page']].append((label, [v * 180 / 25.4 for v in region['bbox_mm']]))
    for region in layout.get('tempo_indications', []):
        regions[region['page']].append(('tempo_region', [v * 180 / 25.4 for v in region['bbox_mm']]))
    return regions


def transfer_regions(row, reference):
    label = reference.get('label_json') or reference.get('label')
    if not label or not reference.get('bbox'):
        return []
    x, y, w, h = reference['bbox']
    origin = [max(0, round(x - 1.5 * 180 / 25.4)), max(0, round(y - 4 * 180 / 25.4))]
    with Image.open(reference['image']) as image:
        width, height = image.size
    dx, dy, out_width, out_height = row['bbox']
    sx, sy = out_width / width, out_height / height
    result = []
    for kind, box in native_regions(label, reference['mode']).get(reference['page'], []):
        a, b, c, d = box
        left, top = max(origin[0], a), max(origin[1], b)
        right, bottom = min(origin[0] + width, a + c), min(origin[1] + height, b + d)
        if right - left < 5 or bottom - top < 5:
            continue
        if kind != 'annotation_region' and (right - left) * (bottom - top) < .85 * c * d:
            continue
        result.append((kind, [dx + (left - origin[0]) * sx, dy + (top - origin[1]) * sy,
                              (right - left) * sx, (bottom - top) * sy]))
    return result


def build(output):
    output = output.resolve()
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(exist_ok=True)
    categories = [{'id': i + 1, 'name': name, 'supercategory': 'score'} for i, name in enumerate(TYPED_CATEGORIES)]
    ids = {c['name']: c['id'] for c in categories}
    report = {}
    for split, tag in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        source = Path('database/parallel_score_final/layout')
        data = json.loads((source / 'annotations' / f'instance_{tag}.json').read_text())
        images, annotations = [], []

        def add(image, boxes, metadata=None, overlay=None):
            with Image.open(image) as im:
                width, height = im.size
            name = f'{tag}-{len(images) + 1}.png'
            link = output / 'images' / name
            if overlay is None:
                if not link.exists():
                    link.symlink_to(Path(image).resolve())
            else:
                if link.is_symlink():
                    link.unlink()
                overlay.save(link, compress_level=1)
            image_id = len(images) + 1
            images.append({**(metadata or {}), 'id': image_id, 'file_name': name, 'width': width, 'height': height})
            for order, (kind, bbox) in enumerate(sorted(boxes, key=lambda b: (b[1][1], b[1][0]))):
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                annotations.append({'id': len(annotations) + 1, 'image_id': image_id,
                    'category_id': ids[kind], 'bbox': bbox, 'area': w * h, 'iscrowd': 0,
                    'read_order': order, 'segmentation': [[x, y, x+w, y, x+w, y+h, x, y+h]]})

        by_image = defaultdict(list)
        names = {c['id']: c['name'] for c in data['categories']}
        for a in data['annotations']:
            kind = names[a['category_id']]
            by_image[a['image_id']].append(('annotation_region' if kind == 'transposition_region' else kind, a['bbox']))
        for im in data['images']:
            add(source / 'images' / im['file_name'], by_image[im['id']], im)
        references = {}
        with Path(f'database/score_support/manifest_{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                label = row.get('label_json') or row.get('label')
                if label:
                    references.setdefault((label, row['mode'], row['measure_index']), row)
        count = Counter()
        for corpus in ['ensemble_quality', 'engraved_scores']:
            for path in sorted((Path('database') / corpus / split).glob('score-*/score.json')):
                score = json.loads(path.read_text())
                for page_number, page in enumerate(score['pages'], 1):
                    boxes = []
                    for row in score['records']:
                        if row['page'] != page_number:
                            continue
                        boxes.append(('measure_' + row['mode'], row['bbox']))
                        reference = references.get((row.get('label_json'), row['mode'], row.get('bar_index', row['measure_index'])))
                        if reference:
                            regions = transfer_regions(row, reference)
                            boxes.extend(regions)
                            count.update(k for k, _ in regions)
                    boxes.extend((r['kind'] + '_region', r['bbox']) for r in page.get('regions', []))
                    # Coincident source crop padding can repeat the same region.
                    unique = {}
                    for kind, box in boxes:
                        unique[(kind, tuple(round(x) for x in box))] = (kind, box)
                    add(page['image'], list(unique.values()), {'source_id': score['id'], 'mode': 'mixed',
                        'family': score['id'], 'source_families': score.get('source_families', [score['id']])})
        # Train candidate localization on known printed markers. The answer is
        # their rendered rectangle, not the previous detector's guess.
        rng = random.Random(3845 + len(split))
        base_images = images.copy()
        old_boxes = defaultdict(list)
        for a in annotations:
            old_boxes[a['image_id']].append((TYPED_CATEGORIES[a['category_id'] - 1], a['bbox']))
        marker_images = sorted((Path('database/unified_score/annotation_images') / split).glob('*.png'))
        for i in range(3000 if split == 'train' else 240):
            im = rng.choice(base_images)
            source_path = output / 'images' / im['file_name']
            with Image.open(source_path) as page:
                page = page.convert('RGB')
            boxes = list(old_boxes[im['id']])
            # Put a diverse annotation strip in the blank top margin; occupied
            # pixels are never erased and reference music geometry is unchanged.
            occupied_top = min((b[1] for _, b in boxes), default=page.height)
            top = max(4, int(occupied_top) - 110)
            for j in range(rng.randrange(1, 4)):
                with Image.open(rng.choice(marker_images)) as marker:
                    marker = marker.convert('RGB')
                    marker.thumbnail((max(80, page.width // 4), 75))
                    x, y = 30 + j * (page.width // 3), top
                    if y + marker.height >= occupied_top - 4 or x + marker.width >= page.width:
                        continue
                    # Reject occupied header areas rather than hiding a title.
                    area = page.crop((x, y, x + marker.width, y + marker.height)).convert('L')
                    if sum(v * n for v, n in enumerate(area.histogram())) / (area.width * area.height) < 252:
                        continue
                    page.paste(marker, (x, y))
                    boxes.append(('annotation_region', [x, y, marker.width, marker.height]))
            add(source_path, boxes, {**im, 'augmentation': 'visible_annotations'}, page)
        payload = {'images': images, 'annotations': annotations, 'categories': categories}
        (output / 'annotations' / f'instance_{tag}.json').write_text(json.dumps(payload))
        report[split] = {'pages': len(images), 'boxes': len(annotations), 'native_transferred': dict(count), 'detector_pseudo_labels': 0}
        (output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(split, report[split], flush=True)


def add_chord_pages(base, output, chords):
    output = output.resolve()
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(exist_ok=True)
    for split in ('train', 'val', 'test'):
        data = json.loads((base / 'annotations' / f'instance_{split}.json').read_text())
        images, annotations = [], []
        for prefix, root in [('score', base), ('chord', chords)]:
            payload = data if root == base else json.loads((root / 'annotations' / f'instance_{split}.json').read_text())
            mapping = {}
            for im in payload['images']:
                identifier = len(images) + 1
                mapping[im['id']] = identifier
                name = prefix + '-' + im['file_name']
                link = output / 'images' / name
                if not link.exists():
                    link.symlink_to((root / 'images' / im['file_name']).resolve())
                images.append({**im, 'id': identifier, 'file_name': name})
            for a in payload['annotations']:
                annotations.append({**a, 'id': len(annotations) + 1, 'image_id': mapping[a['image_id']]})
        (output / 'annotations' / f'instance_{split}.json').write_text(json.dumps({
            'images': images, 'annotations': annotations, 'categories': data['categories']}))
        print(split, len(images), 'pages', len(annotations), 'boxes', flush=True)
    mask = output / 'images_mask'
    if not mask.exists():
        mask.symlink_to('images', target_is_directory=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/unified_layout'))
    parser.add_argument('--base', type=Path)
    parser.add_argument('--chords', type=Path)
    args = parser.parse_args()
    if args.base and args.chords:
        add_chord_pages(args.base, args.output, args.chords)
    else:
        build(args.output)
