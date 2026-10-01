"""Build visible chord tasks, timed measure targets and detector rectangles."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path

from datagen.native_chords import visible_chords
from datagen.training_samples import annotation_sample, dataset_entry, visual_measure_sample
from shared.pdf import open_pdf
from shared.pitch_context import convert_pitch_target
from shared.score_state import attach_neighbours


def document(job):
    row, mode = job
    source = row['source_id']
    pdf_path = next((Path('database/chord_native/documents') / f'{mode}-{source}').glob('tracks/*/score.pdf'))
    metadata, pages = [], {}
    output = Path('database/chord_training/info/chords')
    with open_pdf(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, 1):
            regions = visible_chords(page)
            pages[f'{source}-{mode}-{page_number}.png'] = [
                [b[0] * 2.5, b[1] * 2.5, (b[2] - b[0]) * 2.5, (b[3] - b[1]) * 2.5]
                for b, _ in regions]
            # Each geometry still trains the detector. OCR doesn't need three
            # identical copies of the header diagram on different staff modes.
            if mode != 'both':
                continue
            image = page.render(180)
            for i, (box, value) in enumerate(regions):
                path = output / f'{source}-{page_number}-{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                bounds = [max(0, round(v * 2.5)) for v in box]
                bounds[2], bounds[3] = min(image.width, bounds[2]), min(image.height, bounds[3])
                image.crop(tuple(bounds)).save(path)
                metadata.append(annotation_sample(path, value))
    return row['split'], metadata, pages


def build():
    root = Path('database/unified_score')
    source = Path('database/chord_training')
    counts = defaultdict(Counter)
    catalog = json.loads(Path('database/chord_scores/source_catalog.json').read_text())['sources']
    pages = {}
    handles = {split: (root / f'extra_{split}.jsonl').open('w') for split in ('train', 'validation', 'test')}
    try:
        def write(split, row, task):
            line = json.dumps(row, ensure_ascii=False) + '\n'
            handles[split].write(line)
            counts[split][task] += 1

        with ProcessPoolExecutor(16) as pool:
            jobs = [(row, mode) for row in catalog for mode in row['modes']]
            for i, (split, samples, boxes) in enumerate(pool.map(document, jobs, chunksize=1), 1):
                pages.update(boxes)
                for row in samples:
                    write(split, row, 'chord_annotation')
                if i % 150 == 0:
                    print('documents', i, '/', len(jobs), flush=True)
        for split, tag in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            grouped = defaultdict(list)
            with (source / 'measure/manifests' / f'{split}.jsonl').open() as handle:
                for line in handle:
                    row = json.loads(line)
                    grouped[row['source_id'], row['mode']].append(row)
            with (root / f'chord_manifest_{split}.jsonl').open('w') as manifest:
                for rows in grouped.values():
                    rows.sort(key=lambda r: r['measure_index'])
                    attach_neighbours(rows)
                    for row in rows:
                        row.update(visual_pitch=True, source_measures=len(rows),
                                   score_state={'time': '4/4', 'key': 0, 'tuning': row['tuning']},
                                   bar_index=row['measure_index'], part_id='part-1', staff_id='staff-1',
                                   midi_program=24, part_name='Guitar',
                                   source_page=str((source / 'layout/images' /
                                       f"{row['source_id']}-{row['mode']}-{row['page']}.png").resolve()))
                        row['sounding_target'] = row['target']
                        if row['mode'] != 'tab':
                            row['target'] = convert_pitch_target(row['target'], row['pitch_context'], to_written=True, mode=row['mode'])
                        write(split, visual_measure_sample(row), 'chord_measure')
                        manifest.write(json.dumps(row) + '\n')
            with (root / f'paired_{split}.jsonl').open() as handle:
                for line in handle:
                    write(split, json.loads(line), 'paired_structure')
            path = source / 'layout/annotations' / f'instance_{tag}.json'
            data = json.loads(path.read_text())
            category = next(c['id'] for c in data['categories'] if c['name'] == 'annotation_region')
            existing = {(a['image_id'], tuple(a['bbox'])) for a in data['annotations']
                        if a['category_id'] == category}
            next_id = max((a['id'] for a in data['annotations']), default=0) + 1
            for im in data['images']:
                for box in pages[im['file_name']]:
                    if (im['id'], tuple(box)) in existing:
                        continue
                    x, y, w, h = box
                    data['annotations'].append({'id': next_id, 'image_id': im['id'],
                        'category_id': category, 'bbox': box, 'area': w * h, 'iscrowd': 0,
                        'segmentation': [[x, y, x+w, y, x+w, y+h, x, y+h]]})
                    next_id += 1
                    existing.add((im['id'], tuple(box)))
            by_image = defaultdict(list)
            for a in data['annotations']:
                by_image[a['image_id']].append(a)
            for annotations in by_image.values():
                for order, annotation in enumerate(sorted(annotations, key=lambda a: (a['bbox'][1], a['bbox'][0]))):
                    annotation['read_order'] = order
            path.write_text(json.dumps(data))
    finally:
        for handle in handles.values():
            handle.close()
    info_path = root / 'dataset_info.json'
    info = json.loads(info_path.read_text())
    for split in handles:
        info[f'extra_{split}'] = dataset_entry(f'extra_{split}.jsonl')
    info_path.write_text(json.dumps(info, indent=2))
    (root / 'extra_summary.json').write_text(json.dumps(dict(counts), indent=2))
    print(dict(counts), flush=True)
    write_score_truth()


def write_score_truth():
    """Keep held-out full PDFs alongside their beat-aligned reference records."""
    for split in ('validation', 'test'):
        grouped = defaultdict(list)
        with Path(f'database/unified_score/chord_manifest_{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                row.update(bar_index=row['measure_index'], part_id='part-1', staff_id='staff-1',
                           midi_program=24, part_name='Guitar',
                           midi_program_visible=False,
                           source_page=str(Path('database/chord_training/layout/images',
                               f"{row['source_id']}-{row['mode']}-{row['page']}.png").resolve()))
                grouped[row['source_id'], row['mode']].append(row)
        for (source, mode), records in grouped.items():
            pdf = next(Path(f'database/chord_native/documents/{mode}-{source}').glob('tracks/*/score.pdf'))
            annotations = []
            with open_pdf(pdf) as document:
                for page_number, page in enumerate(document, 1):
                    for box, value in visible_chords(page):
                        x, y, right, bottom = box
                        annotations.append({'page': page_number, 'part_id': 'part-1',
                                            'bbox': [x * 2.5, y * 2.5, (right - x) * 2.5, (bottom - y) * 2.5],
                                            'parsed': value})
            directory = Path(f'database/chord_score_pages/{split}/score-{source}-{mode}')
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'score.json').write_text(json.dumps({
                'id': f'{source}-{mode}', 'split': split, 'pdf': str(pdf.resolve()),
                'parts': [{'name': 'Guitar', 'instrument': 'guitar', 'strings': 6, 'program': 24}],
                'annotations': annotations,
                'records': sorted(records, key=lambda row: row['bar_index']),
            }, ensure_ascii=False))
        print(f'{split}: {len(grouped)} chord PDFs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-only', action='store_true')
    args = parser.parse_args()
    write_score_truth() if args.scores_only else build()
