"""Mix original single-part pages with ensemble structure and metadata tasks."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from PIL import Image

from research.data.training_samples import dataset_entry
from research.data.source_profile import apply_source_profile
from research.inference.layout.postprocess import order_measure_boxes
from research.inference.layout.structure import STRUCTURE_PROMPT
from research.inference.layout.structure import structure_image
from research.inference.layout.structure import structure_model_image
from scorelib.instruments import DEFAULT_PROGRAMS


def single_page(job):
    image, annotations, categories, reference, source, output = job
    boxes = [{'coordinate': [a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]],
              'score': 1., 'label': categories[a['category_id']]} for a in annotations]
    measures = order_measure_boxes(boxes)
    groups = defaultdict(list)
    for box in measures:
        groups[box['system_index']].append(box)
    rows = list(groups.values())
    if not rows:
        return None
    reference = apply_source_profile(reference)
    instrument = reference.get('instrument') or image.get('instrument', 'guitar')
    part = {'name': {'guitar': 'Guitar', 'bass': 'Bass', 'pitched': 'Piano', 'drums': 'Drums'}[instrument],
            'instrument': instrument, 'strings': len(reference.get('tuning') or []) if image['mode'] in {'tab', 'both'} else None,
            'program': DEFAULT_PROGRAMS[instrument]}
    path = structure_image(Path(source) / image['file_name'], rows, Path(output) / f"single-{image['id']}.png")
    target = {'parts': [part], 'rows': [[i, 0, 0] for i in range(len(rows))]}
    return {'messages': [{'role': 'user', 'content': '<image>' + STRUCTURE_PROMPT + f'There are {len(rows)} marked rows.'},
                         {'role': 'assistant', 'content': json.dumps(target, separators=(',', ':'))}], 'images': [path]}


def resize_marked(path):
    with Image.open(path) as image:
        if image.width <= 1200 and image.height <= 1680:
            return
        image.thumbnail((1200, 1680), Image.Resampling.LANCZOS)
        image.save(path)


def build(output, workers=16, engraved=None, native_focus=False, compact=False):
    output.mkdir(parents=True, exist_ok=True)
    info, report = {}, {}
    old = Path('database/parallel_score_final/info')
    layout = Path('database/parallel_score_final/layout')
    for split in ('train', 'validation', 'test'):
        reference = {}
        with Path(f'database/score_support/manifest_{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                reference.setdefault(row['source_id'], row)
        data = json.loads((layout / 'annotations' / f"instance_{'val' if split == 'validation' else split}.json").read_text())
        annotations = defaultdict(list)
        for row in data['annotations']:
            annotations[row['image_id']].append(row)
        categories = {c['id']: c['name'] for c in data['categories']}
        with ProcessPoolExecutor(workers) as pool:
            jobs = ((im, annotations[im['id']], categories, reference.get(im.get('source_id'), {}),
                     str(layout / 'images'), str(output / 'pages' / split)) for im in data['images'])
            singles = [s for s in pool.map(single_page, jobs, chunksize=4) if s]
            ensemble = [json.loads(line) for line in Path(f'database/ensemble_pages/structure_{split}.jsonl').open()]
            list(pool.map(resize_marked, (s['images'][0] for s in ensemble), chunksize=4))
        base = json.loads((old / f'document_info_{split}.json').read_text())
        native_structure, native_info = [], []
        if engraved:
            native_structure = [json.loads(line) for line in (engraved / f'structure_{split}.jsonl').open()]
            native_info = [json.loads(line) for line in (engraved / f'info_{split}.jsonl').open()]
        if split != 'train':
            samples = base + singles + ensemble + native_structure + native_info
        elif native_focus:
            if not native_structure:
                raise ValueError('--native-focus requires native engraved training pages')
            samples = base[::8] + singles[::3] + ensemble * 4 + native_structure * 32 + native_info[::4]
        else:
            samples = base[::2] + singles * 2 + ensemble * 20 + native_structure * 12 + native_info * 2
        if compact:
            paths = {s['images'][0] for s in samples if s['messages'][0]['content'].startswith('<image>' + STRUCTURE_PROMPT)}
            with ProcessPoolExecutor(workers) as pool:
                converted = dict(zip(paths, pool.map(structure_model_image, paths), strict=True))
            samples = [{**s, 'geometry_image': s['images'][0], 'images': [converted[s['images'][0]]]}
                       if s['images'][0] in converted else s for s in samples]
        with (output / f'info_{split}.jsonl').open('w') as handle:
            handle.writelines(json.dumps({k: s[k] for k in ('messages', 'images', 'geometry_image') if k in s}, ensure_ascii=False) + '\n' for s in samples)
        info[f'info_{split}'] = dataset_entry(f'info_{split}.jsonl')
        report[split] = {'metadata': len(base), 'single_pages': len(singles), 'ensemble_pages': len(ensemble),
                         'native_pages': len(native_structure), 'native_metadata': len(native_info), 'samples': len(samples)}
        (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))
        (output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(split, report[split], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/score_support/info'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--engraved', type=Path, help='Add native renderer page structure and metadata')
    parser.add_argument('--native-focus', action='store_true', help='Emphasize instrument grouping and grand staves')
    parser.add_argument('--compact', action='store_true', help='Use only staff identities and page margins for structure')
    build(**vars(parser.parse_args()))
