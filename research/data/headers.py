"""Generate visible header text and page-layout training samples."""

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from research.common.images import TYPED_CATEGORIES
from research.common.pdf import open_pdf
from research.data.native_supervision import printed_text
from research.data.prompts import HEADER_PROMPT
from research.data.training_samples import dataset_entry

FONTS = [Path('tools/native-build/fonts/NotoSerifCJKsc-Regular.otf'),
         Path.home() / '.local/share/fonts/NotoSansCJKsc-Regular.otf']


WORDS = ['春风', '夏夜', '秋雨', '冬雪', '星河', '远方', '故乡', '山川', '江南', '晨曦',
         '回忆', '旅行', '田园', '月光', '海浪', '花园', '梦想', '时光', '河流', '天边',
         '清晨', '晚霞', '归来', '相遇', '告别', '童年', '少年', '大地', '天空', '落叶']


KINDS = ['小夜曲', '圆舞曲', '奏鸣曲', '练习曲', '变奏曲', '进行曲', '幻想曲', '组曲', '随想曲']


SUBTITLES = ['基础课程', '进阶课程', '演奏示范', '独奏改编', '节奏与旋律', '技巧训练', '第二册', '课堂示例']


TUNINGS = ['Standard', 'Dropped D', 'DADGAD', 'Open D', 'Open G', 'Half step down', None]


def han_characters():
    # GB2312 level-one characters provide a broad, public character inventory.
    result = []
    for first in range(0xb0, 0xd8):
        for second in range(0xa1, 0xff):
            try:
                text = bytes((first, second)).decode('gb2312')
            except UnicodeDecodeError:
                continue
            if '\u4e00' <= text <= '\u9fff':
                result.append(text)
    return result


CHARACTERS = han_characters()


def render_text(job):
    index, split, root, font_paths = job
    rng = random.Random(20261006 + index + ['train', 'validation', 'test'].index(split) * 1000000)
    if index % 5 == 0:
        title = ''.join(rng.choices(CHARACTERS, k=rng.randrange(3, 10)))
        artist = ''.join(rng.choices(CHARACTERS, k=rng.randrange(2, 6)))
    elif index % 5 == 1:
        title = rng.choice(['Autumn', 'Moonlight', 'Starlight', 'Journey', 'Memories', 'Seaside']) + ' ' + rng.choice(['Waltz', 'Studies', 'Suite', 'Sonata', 'Variations'])
        artist = rng.choice(['River', 'Willow', 'Cedar', 'Maple', 'Meadow']) + ' ' + rng.choice(['Studio', 'Music School', 'Workshop'])
    else:
        title = rng.choice(WORDS) + rng.choice(KINDS)
        artist = rng.choice(WORDS) + rng.choice(['音乐工作室', '音乐教室', '艺术中心', '吉他社', ''])
    if index % 3 == 0:
        title += rng.choice([' ', ' 第']) + str(rng.randrange(1, 501))
    if index % 11 == 0:
        title = None
    if index % 4 == 0:
        artist = None
    subtitle = rng.choice(SUBTITLES) if index % 3 else None
    tuning = rng.choice(TUNINGS)
    width, height = rng.randrange(1000, 1801), rng.randrange(320, 521)
    image = Image.new('RGB', (width, height), rng.choice(['white', '#fffffa', '#f8f8f5']))
    draw = ImageDraw.Draw(image)
    font_path = rng.choice(font_paths)
    y = rng.randrange(30, 90)
    for text, size, align in [(title, rng.randrange(38, 66), 'center'),
                              (subtitle, rng.randrange(22, 33), 'center'),
                              (artist, rng.randrange(21, 34), rng.choice(['center', 'right']))]:
        if not text:
            continue
        font = ImageFont.truetype(font_path, size)
        bounds = font.getbbox(text)
        text_width = bounds[2] - bounds[0]
        x = (width - text_width) / 2 if align == 'center' else width - text_width - rng.randrange(60, 130)
        draw.text((x, y), text, font=font, fill=rng.choice(['black', '#222222']))
        y += size + rng.randrange(4, 18)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', rng.randrange(18, 25))
    if tuning:
        draw.text((70, height - 82), 'Standard tuning' if tuning == 'Standard' else tuning, font=font, fill='black')
    if index % 7:
        draw.text((90, height - 48), '♩ = ' + str(rng.randrange(40, 201)), font=font, fill='black')
    if index % 5 == 0:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .5)))
    path = Path(root) / 'header_images' / split / f'{index:06d}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=1)
    return {'messages': [{'role': 'user', 'content': '<image>' + HEADER_PROMPT},
                         {'role': 'assistant', 'content': json.dumps({'title': title, 'artist': artist, 'tuning_name': tuning}, ensure_ascii=False, separators=(',', ':'))}],
            'images': [str(path.resolve())]}




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


def build_layout(source, output, workers, headers):
    output = output.resolve()
    for directory in ['images', 'annotations', 'headers']:
        (output / directory).mkdir(parents=True, exist_ok=True)
    templates = native_headers(headers, output / 'headers')
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


def build_text(output, train, validation, test, workers):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fonts = [str(path.resolve()) for path in FONTS if path.is_file()]
    if not fonts:
        raise FileNotFoundError('Install Noto Sans CJK or Noto Serif CJK before rendering headers')
    registry = {}
    with ProcessPoolExecutor(workers) as pool:
        for split, count in [('train', train), ('validation', validation), ('test', test)]:
            filename = f'headers_{split}.jsonl'
            with (output / filename).open('w') as stream:
                for row in pool.map(render_text, ((i, split, output, fonts) for i in range(count))):
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            registry[f'headers_{split}'] = dataset_entry(filename)
    (output / 'dataset_info.json').write_text(json.dumps(registry, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    layout = commands.add_parser('layout')
    layout.add_argument('--source', type=Path, required=True)
    layout.add_argument('--headers', type=Path, required=True)
    text = commands.add_parser('text')
    text.add_argument('--train', type=int, default=12000)
    text.add_argument('--validation', type=int, default=2000)
    text.add_argument('--test', type=int, default=2000)
    for command in (layout, text):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--workers', type=int, default=12)
    args = vars(parser.parse_args())
    command = args.pop('command')
    (build_layout if command == 'layout' else build_text)(**args)
