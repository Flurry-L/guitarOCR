"""Recover short printed instructions above the first staff of each part."""

from pathlib import Path

import numpy as np
from PIL import Image


def opening_annotations(source, output):
    first = {}
    for row in source['records']:
        first.setdefault(row.get('part_id', 'part-1'), row)
    regions = []
    for part, row in first.items():
        if not row.get('source_page'):
            continue
        x, y, width, height = row['bbox']
        # A detected clef bounds the text area without guessing which high
        # notes or ledger lines are annotations.
        clefs = [r for r in source['regions'] if r['kind'] == 'clef' and r['page'] == row['page']
                 and x - 20 <= r['bbox'][0] <= x + min(120, width / 3)
                 and y <= r['bbox'][1] < y + height]
        if not clefs:
            continue
        bottom = int(min(r['bbox'][1] for r in clefs)) - 4
        top = max(0, int(y) - 80)
        previous = [r for r in source['records'] if r['page'] == row['page']
                    and r['bbox'][1] + r['bbox'][3] < y]
        if previous:
            top = max(top, int(max(r['bbox'][1] + r['bbox'][3] for r in previous)) + 8)
        if bottom - top < 8:
            continue
        with Image.open(row['source_page']) as image:
            left, right = max(0, int(x)), min(image.width, int(x + max(width, image.width * .4)))
            ink = np.asarray(image.crop((left, top, right, bottom)).convert('L')) < 200
            lines = np.flatnonzero(ink.sum(1) >= 3)
            bands = np.split(lines, np.flatnonzero(np.diff(lines) > 3) + 1)
            candidates = []
            for band in bands:
                if len(band) < 6 or band[-1] - band[0] > 64:
                    continue
                strip = ink[band[0]:band[-1] + 1]
                h = int(band[-1] - band[0] + 1)
                columns = np.flatnonzero(strip.any(0))
                words = np.split(columns, np.flatnonzero(np.diff(columns) > max(24, h * 1.5)) + 1)
                for word in words:
                    if len(word) < 3:
                        continue
                    w = int(word[-1] - word[0] + 1)
                    # Separated high noteheads and stems above the clef are
                    # not text instructions. Keep connected phrases together.
                    if w < h * 1.5 or strip[:, word].sum() < 20:
                        continue
                    candidates.append((left + int(word[0]), top + int(band[0]), w, h))
            for a, b, w, h in candidates:
                covered = False
                for region in source['regions']:
                    if region['page'] != row['page'] or region['kind'] not in {'annotation', 'transposition', 'tempo'}:
                        continue
                    u, v, s, t = region['bbox']
                    overlap = max(0, min(a+w, u+s)-max(a,u)) * max(0, min(b+h,v+t)-max(b,v))
                    if overlap / (w*h) > .7:
                        covered = True
                        break
                if covered:
                    continue
                path = Path(output) / part / f'opening-{len(regions):03d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((max(0,a-6), max(0,b-6), min(image.width,a+w+6), min(image.height,b+h+6))).convert('RGB').save(path)
                regions.append({'kind': 'annotation', 'page': row['page'], 'bbox': [a,b,w,h],
                                'part_id': part, 'image': str(path.resolve()), 'source': 'opening_text_line'})
    return regions
