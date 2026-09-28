"""Rehearse metadata on page crops with detector-sized shifts and padding."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import random
import shutil

from PIL import Image
from datagen.training_samples import dataset_entry


def augment_score(task):
    index, directory, samples, output, variants, seed = task
    score = json.loads((directory / 'score.json').read_text())
    samples = iter(samples)
    rng = random.Random(seed + index)
    result = []
    for page_index, page in enumerate(score['pages']):
        with Image.open(page['image']) as image:
            for region_index, region in enumerate(page.get('regions', [])):
                if region['kind'] != 'clef':
                    continue
                source = next(samples)
                x, y, width, height = region['bbox']
                for variant in range(variants):
                    dx, dy, pad = rng.randint(-5, 5), rng.randint(-5, 5), rng.randint(0, 14)
                    crop = image.crop((max(0, int(x + dx - pad)), max(0, int(y + dy - pad)),
                                       min(image.width, int(x + width + dx + pad + 1)),
                                       min(image.height, int(y + height + dy + pad + 1)))).convert('RGB')
                    scale = rng.uniform(.75, 1.3)
                    crop = crop.resize((max(20, round(crop.width * scale)), max(20, round(crop.height * scale))),
                                       Image.Resampling.LANCZOS)
                    if variant:
                        buffer = io.BytesIO()
                        crop.save(buffer, format='JPEG', quality=rng.randint(72, 97))
                        buffer.seek(0)
                        crop = Image.open(buffer).convert('RGB')
                    destination = output / 'images' / f'{index}-{page_index}-{region_index}-{variant}.png'
                    crop.save(destination)
                    result.append({**source, 'images': [str(destination.resolve())]})
    if next(samples, None) is not None:
        raise ValueError('Clef crops and page labels disagree')
    return result


def build(source, rehearsal, output, variants=3, workers=16, seed=92401):
    (output / 'images').mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    with (source / 'info_train.jsonl').open() as labels:
        for line in labels:
            row = json.loads(line)
            path = Path(row['images'][0])
            if path.name.startswith('clef-'):
                groups[path.parent.parent].append(row)
    total = 0
    with ProcessPoolExecutor(workers) as pool, (output / 'info_train.jsonl').open('w') as handle:
        with (rehearsal / 'info_train.jsonl').open() as retained:
            shutil.copyfileobj(retained, handle)
        jobs = [(i, path, rows, output, variants, seed) for i, (path, rows) in enumerate(groups.items())]
        for rows in pool.map(augment_score, jobs):
            handle.writelines(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            total += len(rows)
    splits = ['train']
    for split in ('validation', 'test'):
        source_path = rehearsal / f'info_{split}.jsonl'
        if source_path.is_file():
            shutil.copyfile(source_path, output / source_path.name)
            splits.append(split)
    (output / 'dataset_info.json').write_text(json.dumps({f'info_{s}': dataset_entry(f'info_{s}.jsonl') for s in splits}))
    print(json.dumps({'train_scores': len(groups), 'augmented_clefs': total}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('database/engraved_scores'))
    parser.add_argument('--rehearsal', type=Path, default=Path('database/score_support/info_structure_compact'))
    parser.add_argument('--output', type=Path, default=Path('database/score_support/info_crop_rehearsal'))
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=92401)
    build(**vars(parser.parse_args()))
