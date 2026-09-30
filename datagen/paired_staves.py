"""Teach separately marked notation/TAB rows to refer to the same part."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from layout.structure import structure_prompt, structure_image, marked_systems
from datagen.training_samples import dataset_entry


def staff_boundary(page, row):
    candidates = []
    strings = row[0].get('string_count') or len(row[0].get('tuning') or []) or 6
    for box in row[:4]:
        x, y, w, h = map(round, box['bbox'])
        crop = np.asarray(page.crop((x, y, x + w, y + h)).convert('L'))
        if min(crop.shape) < 30:
            continue
        strength = (crop < 210).mean(1)
        hits = np.flatnonzero(strength > max(.28, float(strength.max()) * .55))
        runs = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)
        centers = [float(r.mean()) for r in runs if len(r)]
        groups = []
        for start in range(len(centers)):
            for count in sorted({5, strings}):
                group = centers[start:start + count]
                if len(group) != count:
                    continue
                gaps = np.diff(group)
                gap = float(np.median(gaps))
                if 4 <= gap <= 40 and max(abs(gaps - gap)) <= max(1.2, gap * .14):
                    groups.append((count, group))
        for count, notation in groups:
            if count != 5:
                continue
            tabs = [g for n, g in groups if n == strings and g[0] > notation[-1] + 12]
            if tabs:
                tab = max(tabs, key=len)
                candidates.append(y + (notation[-1] + tab[0]) / 2)
    return float(np.median(candidates)) if candidates else None


def render(path, write_images=True, image_tag='paired-structure'):
    score = json.loads(path.read_text())
    rows_by_page = defaultdict(lambda: defaultdict(list))
    for r in score['records']:
        rows_by_page[r['page']][(r['system_index'], r['part_id'], r['staff_id'])].append(r)
    parts = score['parts']
    profiles = [{'name': p['name'], 'instrument': p['instrument'], 'strings': p.get('strings') or len(p.get('tuning', [])) or None,
                 'program': p.get('midi_program', p.get('program', 0))} for p in parts]
    indices = {p.get('id', f'part-{i + 1}'): i for i, p in enumerate(parts)}
    result = []
    for page_number, keyed in rows_by_page.items():
        rows, labels, changed = [], [], False
        first = next(iter(keyed.values()))[0]
        with Image.open(first['source_page']) as page:
            for (system, part, staff), records in keyed.items():
                records.sort(key=lambda r: r['bbox'][0])
                label = [system, indices[part], int(staff.rsplit('-', 1)[-1]) - 1]
                boundary = staff_boundary(page, records) if all(r['mode'] == 'both' for r in records) else None
                if boundary is None:
                    rows.append(records)
                    labels.append(label)
                    continue
                changed = True
                for mode in ('notation', 'tab'):
                    split = []
                    for r in records:
                        x, y, w, h = r['bbox']
                        cut = min(y + h - 10, max(y + 10, boundary))
                        bbox = [x, y, w, cut - y] if mode == 'notation' else [x, cut, w, y + h - cut]
                        split.append({**r, 'mode': mode, 'bbox': bbox})
                    rows.append(split)
                    labels.append(label)
        if changed:
            image_path = path.parent / f'{image_tag}-{page_number}.png'
            image = structure_image(first['source_page'], rows, image_path) if write_images else str(image_path.resolve())
            row_modes = [Counter(r['mode'] for r in row).most_common(1)[0][0] for row in rows]
            systems = {}
            labels = [[systems.setdefault(system, len(systems)), part, staff] for system, part, staff in labels]
            result.append({'messages': [{'role': 'user', 'content': '<image>' + structure_prompt(
                                           len(rows), row_modes, marked_systems(image, len(rows)))},
                                        {'role': 'assistant', 'content': json.dumps({'parts': profiles, 'rows': labels}, separators=(',', ':'))}],
                           'images': [image],
                           'row_modes': row_modes})
    return result


def build(name='paired', image_tag='paired-structure', repeats=3):
    counts = Counter()
    output = Path('database/unified_score')
    with ProcessPoolExecutor(12) as pool:
        for split in ('train', 'validation', 'test'):
            paths = sorted((Path('database/ensemble_quality') / split).glob('score-*/score.json'))
            with (output / f'{name}_{split}.jsonl').open('w') as handle:
                for samples in pool.map(partial(render, image_tag=image_tag), paths, chunksize=4):
                    for sample in samples:
                        line = json.dumps(sample) + '\n'
                        handle.write(line * (repeats if split == 'train' else 1))
                        counts[split] += 1
    path = output / 'dataset_info.json'
    info = json.loads(path.read_text())
    info.update({f'{name}_{split}': dataset_entry(f'{name}_{split}.jsonl') for split in counts})
    path.write_text(json.dumps(info, indent=2))
    print(dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='paired')
    parser.add_argument('--image-tag', default='paired-structure')
    parser.add_argument('--repeats', type=int, default=3)
    build(**vars(parser.parse_args()))
