"""Compose aligned ensemble pages from native score glyphs, with source-held-out splits."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from datagen.training_samples import dataset_entry
from layout.structure import STRUCTURE_PROMPT, structure_image
from shared.instruments import DEFAULT_PROGRAMS
from shared.pitch_context import convert_pitch_target
from shared.layout_labels import MEASURE_LABELS


def compose(job):
    index, split, selected, output = job
    rng = random.Random(20261001 + index + 100000 * ['train', 'validation', 'test'].index(split))
    root = Path(output) / split / f'score-{index:05d}'
    root.mkdir(parents=True, exist_ok=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 16)
    title_font = ImageFont.truetype(font_path, 36)
    parts, staff_sources = [], []
    piano = index % 4 == 0
    for rows in selected:
        row = rows[0]
        instrument = row['instrument']
        part_index = len(parts)
        program = DEFAULT_PROGRAMS[instrument]
        name = {'guitar': 'Guitar', 'bass': 'Bass', 'drums': 'Drums', 'pitched': 'Piano'}[instrument]
        if instrument == 'pitched' and not piano:
            name, program = rng.choice([('Piano', 0), ('Violin', 40), ('Viola', 41), ('Cello', 42), ('Flute', 73), ('Oboe', 68)])
        name = name + (' II' if any(p['name'] == name for p in parts) else '')
        parts.append({'name': name, 'instrument': instrument,
                      'strings': len(row.get('tuning') or []) if row['mode'] in {'tab', 'both'} else None,
                      'program': program})
        staff_sources.append((part_index, 0, rows))
        if piano and instrument == 'pitched':
            # Independent left-hand notation is grouped under the same piano.
            # Written notes define its actual pitch; the source timbre is immaterial.
            lower = next((r for r in selected if r[0]['instrument'] == 'bass' and r[0]['mode'] == 'notation'), None)
            if lower:
                staff_sources.append((part_index, 1, lower))
    page, page_number, y, page_rows, row_labels, records, pages, chats = None, 0, 0, [], [], [], [], []
    labels = {v: k for k, v in MEASURE_LABELS.items()} if isinstance(MEASURE_LABELS, dict) else None
    del labels
    row_heights = [230 if rows[0]['mode'] == 'both' else 150 for _, _, rows in staff_sources]
    system_height = sum(row_heights) + 26 * (len(staff_sources) - 1)

    def finish():
        if page is None:
            return
        image_path = root / f'page-{page_number}.png'
        page.save(image_path)
        marked = structure_image(image_path, page_rows, root / f'structure-{page_number}.png')
        target = {'parts': parts, 'rows': row_labels.copy()}
        chats.append({'messages': [{'role': 'user', 'content': '<image>' + STRUCTURE_PROMPT + f'There are {len(page_rows)} marked rows.'},
                                  {'role': 'assistant', 'content': json.dumps(target, separators=(',', ':'))}], 'images': [marked]})
        pages.append({'image': str(image_path.resolve()), 'width': page.width, 'height': page.height})

    for system in range(4):
        if page is None or y + system_height + 70 > 2800:
            finish()
            page_number += 1
            page = Image.new('RGB', (2000, 2800), 'white')
            draw = ImageDraw.Draw(page)
            draw.text((650, 42), f'Ensemble Study {index + 1}', font=title_font, fill='black')
            y, page_rows, row_labels = 150, [], []
            page_system = 0
        first_y = y
        for (part_index, staff, source), height in zip(staff_sources, row_heights, strict=True):
            boxes = []
            for column in range(4):
                original = source[(system * 4 + column) % len(source)]
                x = 235 + column * 415
                with Image.open(original['image']) as crop:
                    im = crop.convert('RGB').resize((415, height), Image.Resampling.LANCZOS)
                page.paste(im, (x, y))
                number = len(records) + 1
                crop_path = root / 'crops' / f'{number}.png'
                crop_path.parent.mkdir(exist_ok=True)
                im.save(crop_path)
                row = {**deepcopy(original), 'id': f'{split}-{index}-{number}', 'source_id': f'ensemble-{split}-{index}',
                       'source_family': original['family'], 'family': f'ensemble-{split}-{index}',
                       'measure_number': number, 'measure_index': system * 4 + column,
                       'bar_index': system * 4 + column, 'page': page_number,
                       'part_id': f'part-{part_index + 1}', 'staff_id': f'staff-{staff + 1}',
                       'part_name': parts[part_index]['name'], 'instrument': parts[part_index]['instrument'],
                       'midi_program': parts[part_index]['program'], 'system_index': page_system,
                       'row_index': len(page_rows), 'system_measure_index': column,
                       'bbox': [x, y, 415, height], 'image': str(crop_path.resolve()),
                       'source_page': str((root / f'page-{page_number}.png').resolve())}
                if row['instrument'] == 'pitched':
                    row['tuning'] = []
                    context = dict(row.get('pitch_context') or {})
                    context.update(instrument_transpose=0, capo=0)
                    row['pitch_context'] = context
                    row['sounding_target'] = convert_pitch_target(row['target'], context)
                boxes.append(row)
                records.append(row)
            if staff == 0:
                draw.text((28, y + height // 2 - 14), parts[part_index]['name'], font=font, fill='black')
            else:
                draw.text((95, y + height // 2 - 10), 'L.H.', font=small, fill='black')
            if staff == 1:
                top = y - row_heights[max(0, staff_sources.index((part_index, staff, source)) - 1)] - 26
                draw.arc((205, top + 20, 231, y + height - 20), 90, 270, fill='black', width=3)
            page_rows.append(boxes)
            row_labels.append([page_system, part_index, staff])
            y += height + 26
        # The continuous system bracket is visual evidence of simultaneity.
        draw.line((227, first_y + 14, 227, y - 40), fill='black', width=3)
        draw.line((222, first_y + 14, 235, first_y + 14), fill='black', width=3)
        draw.line((222, y - 40, 235, y - 40), fill='black', width=3)
        y += 75
        page_system += 1
    finish()
    images = [Image.open(p['image']).convert('RGB') for p in pages]
    pdf = root / 'score.pdf'
    images[0].save(pdf, save_all=True, append_images=images[1:], resolution=180)
    for im in images:
        im.close()
    payload = {'id': f'ensemble-{split}-{index}', 'split': split, 'pdf': str(pdf.resolve()),
               'parts': parts, 'records': records, 'pages': pages,
               'source_families': sorted({r['source_family'] for r in records})}
    (root / 'score.json').write_text(json.dumps(payload, ensure_ascii=False))
    return payload, chats


def build(output, train=2400, validation=240, test=240, workers=12):
    output.mkdir(parents=True, exist_ok=True)
    info, summary = {}, {}
    for split, count in [('train', train), ('validation', validation), ('test', test)]:
        groups = defaultdict(list)
        with Path(f'database/score_support/manifest_{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['score_state']['time'] == '4/4':
                    groups[(row['source_id'], row['mode'])].append(row)
        pools = defaultdict(list)
        for rows in groups.values():
            rows.sort(key=lambda r: r['measure_index'])
            # Whole contiguous chunks keep ties and accidentals meaningful.
            for start in range(0, len(rows) - 15, 16):
                chunk = rows[start:start + 16]
                if chunk[-1]['measure_index'] - chunk[0]['measure_index'] == 15:
                    pools[(chunk[0]['instrument'], chunk[0]['mode'])].append(chunk)
        jobs = []
        for index in range(count):
            rng = random.Random(29491 + index)
            plan = [('pitched', 'notation'), ('bass', 'notation')] if index % 4 == 0 else [
                ('guitar', rng.choice(['tab', 'both', 'notation'])),
                ('bass', rng.choice(['tab', 'both', 'notation'])), ('drums', 'notation')]
            if index % 3 == 0:
                plan.append(('pitched', 'notation'))
            selected = [rng.choice(pools[key]) for key in plan if pools[key]]
            jobs.append((index, split, selected, str(output)))
        bars, page_count = 0, 0
        with ProcessPoolExecutor(workers) as pool, (output / f'scores_{split}.jsonl').open('w') as catalog, \
                (output / f'structure_{split}.jsonl').open('w') as chats:
            for i, (score, samples) in enumerate(pool.map(compose, jobs, chunksize=1), 1):
                catalog.write(json.dumps({k: v for k, v in score.items() if k not in {'records'}}, ensure_ascii=False) + '\n')
                chats.writelines(json.dumps(s, ensure_ascii=False) + '\n' for s in samples)
                bars += len(score['records'])
                page_count += len(samples)
                if i % 50 == 0:
                    print(split, i, '/', count, 'pages', page_count, 'bars', bars, flush=True)
        info[f'structure_{split}'] = dataset_entry(f'structure_{split}.jsonl')
        summary[split] = {'scores': count, 'pages': page_count, 'bars': bars}
        (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/ensemble_pages'))
    parser.add_argument('--train', type=int, default=2400)
    parser.add_argument('--validation', type=int, default=240)
    parser.add_argument('--test', type=int, default=240)
    parser.add_argument('--workers', type=int, default=12)
    build(**vars(parser.parse_args()))
