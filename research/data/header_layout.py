"""Compose labelled score headers over existing, split-preserving music pages.

Text boxes come from font bounds or native PDF text, never detector predictions.
Original unlabelled headers are removed before composition.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from research.common.layout_labels import TYPED_CATEGORIES
from research.common.pdf import open_pdf
from research.data.annotations import printed_text
from research.data.header_rehearsal import FONTS, WORDS, KINDS, SUBTITLES, TUNINGS, CHARACTERS


def normalized(text):
    return ''.join(printed_text(text).split())


def native_headers(root, destination):
    """Extract visible first-page fields using the GP style and PDF glyph boxes."""
    pool = defaultdict(list)
    catalog = json.loads((root / 'source_catalog.json').read_text())['sources']
    for source in catalog:
        paths = list((root / 'native-export' / 'documents' / ('tab-' + source['source_id'])).glob('tracks/*/official-score.json'))
        if len(paths) != 1:
            continue
        path = paths[0]
        score = json.loads(path.read_text())
        layout = json.loads(path.with_name('layout.json').read_text())
        body_top = min((b['bbox_mm'][1] for s in layout['systems'] if s['page'] == 1 for b in s['measure_boxes']), default=0)
        body_top = min([body_top] + [b['bbox_mm'][1] for b in layout.get('tempo_indications', []) if b['page'] == 1])
        top = body_top * 72 / 25.4 - 3
        if top < 30:
            continue
        metadata = score['document']['metadata']
        fields = []
        for key, style in metadata['first_page_header'].items():
            if not style['visible'] or not metadata['properties'].get(key):
                continue
            text = style['formatted_text']
            for prop, value in metadata['properties'].items():
                text = text.replace('%' + prop.upper() + '%', value)
            kind = {'title': 'title_region', 'subtitle': 'subtitle_region', 'album': 'header_text_region'}.get(key, 'credit_region')
            fields.append((kind, text))
        staff = score['tracks'][0]['staves'][0]
        if staff['tuning_label_visible'] and staff['tuning_displayed_label']:
            label = staff['tuning_displayed_label']
            fields.append(('tuning_region', 'Standard tuning' if label == 'Standard' else label))
        with open_pdf(path.with_name('score.pdf')) as pdf:
            page = pdf[0]
            words = [w for w in page.words() if w[3] <= top]
            boxes, used = [], set()
            for kind, text in fields:
                target = normalized(text)
                matches = []
                for i in range(len(words)):
                    joined = ''
                    for j in range(i, len(words)):
                        joined += normalized(words[j][4])
                        if joined == target:
                            matches.append((i, j + 1))
                            break
                        if not target.startswith(joined):
                            break
                if len(matches) != 1:
                    break
                start, end = matches[0]
                part = words[start:end]
                x0, y0 = min(w[0] for w in part), min(w[1] for w in part)
                x1, y1 = max(w[2] for w in part), max(w[3] for w in part)
                boxes.append((kind, [x0 * 2.5, y0 * 2.5, (x1-x0)*2.5, (y1-y0)*2.5]))
                used.update(range(start, end))
            else:
                # Reject templates with unaccounted visible text instead of
                # silently teaching the detector to ignore it.
                if not boxes or len(used) != len(words):
                    continue
                image = page.render(180).crop((0, 0, round(page.width * 2.5), round(top * 2.5)))
                target = destination / ('native-' + source['source_id'] + '.png')
                image.save(target, compress_level=1)
                pool[source['split']].append((str(target), boxes))
    return pool


@lru_cache(maxsize=256)
def font(path, size):
    return ImageFont.truetype(path, size)


def rendered_header(width, rng, fonts):
    scale = width / 1488
    height = round(rng.randrange(240, 470) * scale)
    image = Image.new('RGB', (width, max(height, round(720 * scale))), 'white')
    draw = ImageDraw.Draw(image)
    chinese = rng.random() < .65
    title = (rng.choice(WORDS) + rng.choice(KINDS) if chinese else rng.choice(['Moonlight Sonata', 'Prelude in E minor', 'Autumn Waltz', 'Guitar Studies', 'River Song']))
    author = (rng.choice(WORDS) + rng.choice(['工作室', '音乐教室', '吉他社', '']) if chinese else rng.choice(['J. S. Bach', 'River Studio', 'Maria Garcia', 'John Williams']))
    if rng.random() < .2:
        title = ''.join(rng.choices(CHARACTERS, k=rng.randrange(3, 9)))
        author = ''.join(rng.choices(CHARACTERS, k=rng.randrange(2, 6)))
    if rng.random() < .4:
        title += ' ' + str(rng.randrange(1, 300))
    alignment = rng.choices(['center', 'left'], [9, 1])[0]
    path = rng.choice(fonts)
    y, boxes = round(rng.randrange(30, 100) * scale), []

    def text(value, kind, size, align, top):
        f = font(path, max(9, round(size * scale)))
        bb = draw.textbbox((0, 0), value, font=f)
        while bb[2] - bb[0] > width * .88:
            f = font(path, max(8, f.size - 1))
            bb = draw.textbbox((0, 0), value, font=f)
        x = (width - bb[2] + bb[0]) / 2 if align == 'center' else (width * .91 - bb[2] if align == 'right' else width * .07)
        draw.text((x, top - bb[1]), value, font=f, fill=rng.choice(['black', '#202020', '#444444']))
        boxes.append((kind, [x + bb[0], top, bb[2]-bb[0], bb[3]-bb[1]]))
        return top + bb[3] - bb[1] + round(rng.randrange(12, 24) * scale)

    if rng.random() < .96:
        y = text(title, 'title_region', rng.randrange(40, 72), alignment, y)
    if rng.random() < .7:
        y = text(rng.choice(SUBTITLES) if chinese else rng.choice(['Intermediate Course', 'Solo arrangement', 'Op. 12 No. 3']), 'subtitle_region', rng.randrange(22, 36), alignment, y)
    if rng.random() < .88:
        y = text(author, 'credit_region', rng.randrange(22, 36), rng.choice([alignment, 'right']), y)
    if rng.random() < .25:
        prefix = rng.choice(['作曲：', '编曲：', '制谱：']) if chinese else rng.choice(['Music by ', 'Arranged by ', 'Tabbed by '])
        y = text(prefix + author, 'credit_region', rng.randrange(19, 29), rng.choice(['left', 'right']), y)
    if rng.random() < .25:
        y = text(rng.choice(['Album: Home', 'For classroom use', '吉他教程 第二册', '© River Studio']), 'header_text_region', rng.randrange(19, 27), alignment, y)
    tuning = rng.choice(TUNINGS)
    if tuning and rng.random() < .65:
        y = text('Standard tuning' if tuning == 'Standard' else tuning, 'tuning_region', rng.randrange(19, 28), 'left', max(y, height - round(45 * scale)))
    return image.crop((0, 0, width, max(height, y + 10))), boxes


def compose(job):
    im, boxes, split, source, output, fonts, templates = job
    rng = random.Random(48219 + im['id'] + {'train': 0, 'val': 1000000, 'test': 2000000}[split])
    with Image.open(Path(source) / 'images' / im['file_name']) as original:
        page = original.convert('RGB')
    music_top = max(0, int(min(b[1] for k, b in boxes if k.startswith('measure_'))) - 12)
    cleaned = page.copy()
    ImageDraw.Draw(cleaned).rectangle((0, 0, page.width, music_top - 1), fill='white')
    for kind, (x, y, w, h) in boxes:
        if not kind.startswith('measure_') and y < music_top:
            rect = (max(0, int(x) - 2), max(0, int(y) - 2),
                    min(page.width, round(x + w) + 2), min(music_top, round(y + h) + 2))
            cleaned.paste(page.crop(rect), rect[:2])
    page = cleaned
    # Include every labelled symbol, including tempo/chords above the first bar.
    top = max(0, int(min(b[1] for _, b in boxes)) - 12)
    if rng.random() < .22:
        header, added = Image.new('RGB', (page.width, rng.randrange(25, 100)), 'white'), []
    elif templates and rng.random() < .35:
        path, added = rng.choice(templates)
        with Image.open(path) as original:
            factor = page.width / original.width
            header = original.convert('RGB').resize((page.width, round(original.height * factor)))
        added = [(k, [v * factor for v in b]) for k, b in added]
    else:
        header, added = rendered_header(page.width, rng, fonts)
    offset = header.height + rng.randrange(10, 32)
    result = Image.new('RGB', (page.width, page.height - top + offset), 'white')
    result.paste(header, (0, 0))
    result.paste(page.crop((0, top, page.width, page.height)), (0, offset))
    name = split + '-' + str(im['id']) + '.png'
    result.save(Path(output) / 'images' / name, compress_level=1)
    shifted = [(k, [x, y - top + offset, w, h]) for k, (x,y,w,h) in boxes]
    return {**im, 'file_name': name, 'height': result.height}, added + shifted


def build(source, output, workers):
    output = output.resolve()
    for directory in ['images', 'annotations', 'headers']:
        (output / directory).mkdir(parents=True, exist_ok=True)
    templates = native_headers(Path('database/gp8_headers_v3'), output / 'headers')
    print('Native header templates', {k: len(v) for k,v in templates.items()}, flush=True)
    fonts = [str(p) for p in FONTS if p.is_file()]
    categories = [{'id': i+1, 'name': k, 'supercategory': 'score'} for i,k in enumerate(TYPED_CATEGORIES)]
    ids = {c['name']: c['id'] for c in categories}
    report = {}
    with ProcessPoolExecutor(workers) as pool:
        for split in ['train', 'val', 'test']:
            data = json.loads((source / 'annotations' / f'instance_{split}.json').read_text())
            names = {c['id']:c['name'] for c in data['categories']}
            by_image = defaultdict(list)
            for a in data['annotations']:
                by_image[a['image_id']].append((names[a['category_id']], a['bbox']))
            rows, annotations, counts = [], [], Counter()
            jobs = ((im, by_image[im['id']], split, str(source), str(output), fonts, templates.get('validation' if split=='val' else split, [])) for im in data['images'] if by_image[im['id']])
            for im, boxes in pool.map(compose, jobs, chunksize=8):
                rows.append(im)
                for order, (kind, (x,y,w,h)) in enumerate(sorted(boxes, key=lambda b:(b[1][1],b[1][0]))):
                    annotations.append({'id':len(annotations)+1,'image_id':im['id'],'category_id':ids[kind], 'bbox':[x,y,w,h], 'area':w*h,'iscrowd':0,'read_order':order,'segmentation':[[x,y,x+w,y,x+w,y+h,x,y+h]]})
                    counts[kind] += 1
            (output/'annotations'/f'instance_{split}.json').write_text(json.dumps({'images':rows,'annotations':annotations,'categories':categories}))
            report[split] = {'pages':len(rows), 'boxes':dict(counts)}
            print(split, report[split], flush=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2))
    mask = output/'images_mask'
    if not mask.exists():
        mask.symlink_to('images', target_is_directory=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('database/unified_layout_complete'))
    parser.add_argument('--output', type=Path, default=Path('database/unified_layout_headers'))
    parser.add_argument('--workers', type=int, default=24)
    build(**vars(parser.parse_args()))
