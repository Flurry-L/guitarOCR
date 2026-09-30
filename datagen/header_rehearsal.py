"""Preserve general title/artist reading while learning structured music tasks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from datagen.training_samples import dataset_entry
from document_info.prompts import HEADER_PROMPT


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


def render(job):
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


def build(root, train=12000, validation=2000, test=2000, workers=16):
    root.mkdir(parents=True, exist_ok=True)
    fonts = [str(path.resolve()) for path in FONTS if path.is_file()]
    if not fonts:
        raise FileNotFoundError('Install Noto Sans CJK or Noto Serif CJK before rendering headers')
    definitions = root / 'dataset_info.json'
    datasets = json.loads(definitions.read_text()) if definitions.is_file() else {}
    with ProcessPoolExecutor(workers) as pool:
        for split, size in [('train', train), ('validation', validation), ('test', test)]:
            destination = root / f'headers_{split}.jsonl'
            count = 0
            with destination.open('w') as output:
                source = Path('database/gp8_headers_v3/datasets/info_mixed') / f'document_info_{split}.json'
                seen = set()
                for row in json.loads(source.read_text()):
                    if 'title' not in json.loads(row['messages'][-1]['content']) or row['images'][0] in seen:
                        continue
                    seen.add(row['images'][0])
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
                for row in pool.map(render, ((i, split, str(root), fonts) for i in range(size)), chunksize=32):
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
            datasets[f'headers_{split}'] = dataset_entry(destination.name)
            print(split, count, flush=True)
    definitions.write_text(json.dumps(datasets, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('database/unified_score'))
    parser.add_argument('--train', type=int, default=12000)
    parser.add_argument('--validation', type=int, default=2000)
    parser.add_argument('--test', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=16)
    build(**vars(parser.parse_args()))
