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
        # Keep recovery above the measure: high notes and slurs can extend
        # above the clef while still belonging to the music.
        clefs = [r for r in source['regions'] if r['kind'] == 'clef' and r['page'] == row['page']
                 and x - 20 <= r['bbox'][0] <= x + min(120, width / 3)
                 and y <= r['bbox'][1] < y + height]
        if not clefs:
            continue
        bottom = int(min(y, min(r['bbox'][1] for r in clefs) - 4))
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
                    if region['page'] != row['page'] or region['kind'] not in {'annotation', 'transposition', 'tempo', 'title', 'subtitle', 'credit', 'tuning', 'header_text'}:
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


def prepare_chord_regions(records, predictions, output):
    """Keep header libraries separate and retain local chord evidence in crops."""
    from research.common.crops import crop_measure

    for p in predictions:
        if p.get('parsed', {}).get('kind') not in {'chord', 'chord_diagram'}:
            continue
        a, b, c, d = p['bbox']
        candidates = []
        for r in records:
            x, y, w, h = r['bbox']
            if (r['page'] == p['page'] and x - 8 <= a + c / 2 < x + w + 8
                    and y - max(48, h * .2) <= b + d <= y + h):
                candidates.append((max(0, y - b - d), abs(a + c / 2 - x - w / 2), r['measure_number'], r))
        if candidates:
            row = min(candidates, key=lambda v: v[:3])[-1]
            p.update(scope='measure', measure_number=row['measure_number'], part_id=row['part_id'])
        else:
            p['scope'] = 'library'
    page = None
    source = None
    try:
        for row in records:
            local = [p for p in predictions if p.get('scope') == 'measure' and p['measure_number'] == row['measure_number']]
            if not local:
                continue
            row['chord_annotations'] = local
            x, y, w, h = row['bbox']
            left, top, right, bottom = x, y, x + w, y + h
            for p in local:
                a, b, c, d = p['bbox']
                left, top, right, bottom = min(left, a), min(top, b), max(right, a + c), max(bottom, b + d)
            if (left, top, right, bottom) == (x, y, x + w, y + h):
                continue
            if source != row['source_page']:
                if page is not None:
                    page.close()
                source = row['source_page']
                page = Image.open(source).convert('RGB')
            bounds = [left, top, right - left, bottom - top]
            output.mkdir(parents=True, exist_ok=True)
            path = output / f"measure-{row['measure_number']}-chords.png"
            crop_measure(page, bounds).save(path)
            row.update(image=str(path.resolve()), content_bbox=bounds)
    finally:
        if page is not None:
            page.close()
