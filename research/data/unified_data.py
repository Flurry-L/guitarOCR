"""Build shared OCR tasks from rendered evidence and source-disjoint scores."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from research.data.training_samples import annotation_sample
from research.data.training_samples import dataset_entry
from research.inference.information.prompts import ANNOTATION_PROMPT
from scorelib.pitch_context import explicit_transposition
from research.data.chord_annotations import sample_diagram
from research.data.chord_annotations import draw_diagram


TECHNIQUES = ['let ring', 'Let Ring', 'P.M.', 'palm mute', 'vibrato', 'rit.',
              'pizz.', 'arco', 'simile', 'legato', 'sul pont.', 'let ring throughout']
CHORDS = [root + accidental + suffix for root in 'ABCDEFG'
          for accidental in ['', '#', 'b', '♯', '♭']
          for suffix in ['', 'm', '7', 'maj7', 'm7', 'mMaj7', '6', 'm6', '9', 'm9',
                         'maj9', '11', 'm11', '13', 'maj13', 'sus2', 'sus4', '7sus4',
                         'dim', 'dim7', 'aug', 'add9', 'add11', 'm7b5', '7b9', '7#9',
                         '6/9', '/F#', '/Bb']]
INSTRUMENTS = ['Trumpet in Bb', 'Bb Clarinet', 'Clarinet in A', 'Alto Saxophone in Eb',
               'Tenor Sax in Bb', 'Horn in F', 'English Horn in F', 'Guitar', 'Bass Guitar',
               'Piccolo', 'Double Bass', 'Concert pitch']
FONTS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf',
         '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
         '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']


def annotation(kind, text, semitones=None, capo=None):
    return {'kind': kind, 'semitones': semitones, 'capo': capo, 'text': text}


def render_annotation(job):
    index, split, output = job
    rng = random.Random(81293 + index + 1000000 * ['train', 'validation', 'test'].index(split))
    kind = ['technique', 'chord', 'chord_diagram', 'ottava', 'capo', 'instrument', 'other', 'tempo'][index % 8]
    shift, capo, diagram = None, None, None
    if kind == 'technique':
        text = rng.choice(TECHNIQUES)
    elif kind == 'chord_diagram':
        text, diagram = sample_diagram(rng, string_counts=(4, 5, 6, 6, 6, 7, 8))
    elif kind == 'chord':
        text = rng.choice(CHORDS)
    elif kind == 'ottava':
        text, shift = rng.choice([('8va', 12), ('8vb', -12), ('15ma', 24), ('15mb', -24)])
    elif kind == 'capo':
        capo = rng.randrange(13)
        roman = ['0', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'][capo]
        text = rng.choice([f'Capo {capo}', f'Capo {roman}', f'Capo fret {capo}', f'Capo: {capo}'])
    elif kind == 'instrument':
        text = rng.choice(INSTRUMENTS)
        shift = explicit_transposition(text)
    elif kind == 'tempo':
        text = rng.choice(['Moderate', 'Allegro', 'Andante', '']) + f' ♩ = {rng.randrange(40, 221)}'
        text = text.strip()
    else:
        text = rng.choice(['Verse', 'Chorus', 'Bridge', 'Solo', 'Drums', 'Standard tuning',
                           'DADGAD', 'D.S. al Coda', 'Fine', 'Copyright 2026', 'III', 'V', 'VII', '1.', '2.'])
    font_pool = FONTS[:4] if split == 'train' else FONTS[4:]
    font_path = rng.choice(font_pool)
    # Liberation omits several music glyphs: never label an undrawn accidental.
    if any(c in text for c in '♭♯♩'):
        font_path = FONTS[index % 2]
    font = ImageFont.truetype(font_path, rng.randrange(16, 39))
    left, top, right, bottom = font.getbbox(text)
    pad = rng.randrange(5, 25)
    width = max(90, right - left + pad * 2)
    height = bottom - top + pad * 2
    if kind in {'ottava', 'technique'} and index % 3:
        width += rng.randrange(40, 600)
    if kind == 'chord_diagram':
        height += 170
        width = max(width, 60 + 23 * (len(diagram['frets']) - 1))
    im = Image.new('RGB', (width, height), rng.choice(['white', '#fffff5', '#f4f4ef']))
    draw = ImageDraw.Draw(im)
    draw.text((pad - left, pad - top), text, font=font, fill=rng.choice(['black', '#222222', '#353535']))
    if kind in {'ottava', 'technique'} and index % 3:
        y = pad + (bottom - top) // 2
        for x in range(right - left + pad + 8, width - pad, 13):
            draw.line((x, y, min(x + 6, width - pad), y), fill='black', width=rng.choice([1, 2]))
        if kind == 'ottava':
            draw.line((width - pad, y - 7, width - pad, y), fill='black', width=1)
    if kind == 'chord_diagram':
        draw_diagram(im, diagram, bottom - top + pad + 8, FONTS[0], rng)
    if index % 5 == 0:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .65)))
    if index % 7 == 0:
        im = im.rotate(rng.uniform(-2, 2), resample=Image.Resampling.BICUBIC, expand=True, fillcolor='white')
    path = Path(output) / split / f'{index:06d}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, compress_level=1)
    value = annotation(kind, text, shift, capo)
    if diagram is not None:
        value['diagram'] = diagram
    return annotation_sample(path, value)


def native_annotations(job):
    """Text comes from visible PDF glyphs, never a detector prediction."""
    row, output = job
    from research.common.pdf import open_pdf

    source = row['source_id']
    paths = list((Path('database/pitch_native/documents') / ('both-' + source)).glob('tracks/*/score.pdf'))
    if not paths:
        paths = list((Path('database/pitch_native/documents') / ('notation-' + source)).glob('tracks/*/score.pdf'))
    if not paths:
        return []
    result = []
    with open_pdf(paths[0]) as pdf:
        for page_number, page in enumerate(pdf):
            if page_number >= 4:
                break
            words = page.words()
            targets, consumed = [], set()
            for i, word in enumerate(words):
                if i in consumed:
                    continue
                text = word[4].strip()
                group = [word]
                if text.casefold() == 'let' and i + 1 < len(words) and words[i + 1][4].casefold() == 'ring':
                    group.append(words[i + 1])
                    consumed.add(i + 1)
                    text = ' '.join(w[4] for w in group)
                normalized = re.sub(r'[.\s]', '', text.casefold())
                if normalized in {'letring', 'pm', 'vibrato', 'pizz', 'arco', 'legato', 'simile', 'rit', 'staccato'}:
                    kind = 'technique'
                elif re.fullmatch(r'[A-G](?:#|b|♯|♭)?(?:m|maj|sus|dim|aug|add)?(?:[2679]|11|13)?(?:/[A-G](?:#|b|♯|♭)?)?', text):
                    # Chords must occur over music, not in the title/header.
                    if len(text) < 2 or word[1] < page.height * .18:
                        continue
                    kind = 'chord'
                else:
                    continue
                box = [min(w[0] for w in group), min(w[1] for w in group),
                       max(w[2] for w in group), max(w[3] for w in group)]
                targets.append((box, kind, text))
            if not targets:
                continue
            image = page.render(180)
            for k, (box, kind, text) in enumerate(targets):
                x0, y0, x1, y1 = [v * 2.5 for v in box]
                crop = image.crop((max(0, int(x0) - 6), max(0, int(y0) - 6),
                                   min(image.width, int(x1) + 7), min(image.height, int(y1) + 7)))
                path = Path(output) / row['split'] / f'{source}-{page_number}-{k}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(path)
                result.append(annotation_sample(path, annotation(kind, text)))
    return result


def rows(path):
    if path.suffix == '.jsonl':
        with path.open() as handle:
            yield from (json.loads(line) for line in handle if line.strip())
    else:
        yield from json.loads(path.read_text())


def build(output, workers=16, annotations=32000):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    definitions = output / 'dataset_info.json'
    report, datasets = {}, json.loads(definitions.read_text()) if definitions.is_file() else {}
    catalog = json.loads(Path('database/pitch_training/source_catalog.json').read_text())['sources']
    # One plain export per source family: variants of a work stay in its original split.
    native = {}
    for row in catalog:
        if row.get('variant') == 'plain':
            native.setdefault(row['family'], row)
    with ProcessPoolExecutor(workers) as pool:
        for split in ('train', 'validation', 'test'):
            counts, seen = Counter(), set()
            destination = output / f'ocr_{split}.jsonl'
            with destination.open('w') as handle:
                def write(sample, task, repeat=1):
                    sample = {'messages': sample['messages'], 'images':
                              [sample['geometry_image']] if task == 'structure' and sample.get('geometry_image') else sample['images']}
                    identity = (tuple(sample['images']), sample['messages'][0]['content'], sample['messages'][-1]['content'])
                    if identity in seen:
                        return
                    seen.add(identity)
                    line = json.dumps(sample, ensure_ascii=False) + '\n'
                    for _ in range(repeat):
                        handle.write(line)
                    counts[task] += repeat

                for sample in rows(Path('database/score_quality_profiles') / f'measure_{split}.jsonl'):
                    write(sample, 'measure')
                info_path = (Path('database/score_support/info_crop_rehearsal') / f'info_{split}.jsonl'
                             if split != 'test' else Path('database/score_support/info_structure_compact/info_test.jsonl'))
                for sample in rows(info_path):
                    prompt = sample['messages'][0]['content']
                    task = 'structure' if 'score structure' in prompt else 'metadata'
                    if 'pitch instruction' in prompt or 'Classify and read the visible score annotation' in prompt:
                        sample['messages'][0]['content'] = '<image>' + ANNOTATION_PROMPT
                        task = 'annotation'
                    write(sample, task, 3 if split == 'train' and task in {'structure', 'annotation'} else 1)
                for sample in rows(Path('database/ensemble_quality') / f'structure_{split}.jsonl'):
                    write(sample, 'structure', 3 if split == 'train' else 1)
                amount = annotations if split == 'train' else annotations // 8
                for sample in pool.map(render_annotation, ((i, split, str(output / 'annotation_images')) for i in range(amount)), chunksize=32):
                    write(sample, 'annotation', 2 if split == 'train' else 1)
                jobs = [(row, str(output / 'native_annotations')) for row in native.values() if row['split'] == split]
                for i, samples in enumerate(pool.map(native_annotations, jobs, chunksize=1)):
                    for sample in samples:
                        write(sample, 'native_annotation', 3 if split == 'train' else 1)
                    if i % 100 == 0:
                        print(split, 'native sources', i, '/', len(jobs), dict(counts), flush=True)
            report[split] = dict(counts)
            datasets[f'ocr_{split}'] = dataset_entry(destination.name)
            (output / 'dataset_info.json').write_text(json.dumps(datasets, indent=2))
            (output / 'summary.json').write_text(json.dumps(report, indent=2))
            print(split, dict(counts), flush=True)


def assemble_training(output, inputs=None, text_sources=None, staff_data=None):
    """Write the shared task mixture once; tokenization needs no cache overlays."""
    from research.data.native_alignment import native_text_target

    inputs = inputs or output
    output.mkdir(parents=True, exist_ok=True)
    definitions = output / 'dataset_info.json'
    datasets = json.loads(definitions.read_text()) if definitions.is_file() else {}
    if text_sources is None and (output / 'text_sources.json').is_file():
        text_sources = output / 'text_sources.json'
    references = json.loads(text_sources.read_text()) if text_sources else {}
    staff_data = staff_data or Path('database/instrument_training/datasets/staff_visible')
    for split in ('train', 'validation', 'test'):
        correction_path = inputs / f'corrected_annotations_{split}.jsonl'
        corrections = {row['images'][0]: row for row in rows(correction_path)} if correction_path.is_file() else {}
        counts = Counter()
        staff = list(rows(staff_data / f'staff_{split}.json'))
        staff_counts = Counter(row.get('provenance', {}).get('visible_name') for row in staff)
        destination = output / f'unified_{split}.jsonl'
        with destination.open('w') as handle:
            def write(row, task, repeat=1):
                row = corrections.get(row['images'][0], row)
                if task != 'staff' and 'Read the first staff and its instrument label.' in row['messages'][0]['content']:
                    # Replace legacy two-field supervision with the visible
                    # name task below; absent provenance is not an absent label.
                    return
                reference = references.get(row['images'][0])
                if reference and row['messages'][-1]['content'].startswith('M2 '):
                    row = {**row, 'messages': [dict(message) for message in row['messages']]}
                    answer = row['messages'][-1]
                    answer['content'] = native_text_target(answer['content'], *reference)
                line = json.dumps(row, ensure_ascii=False) + '\n'
                for _ in range(repeat):
                    handle.write(line)
                counts[task] += repeat

            for source in ('ocr', 'extra', 'refinement'):
                for row in rows(inputs / f'{source}_{split}.jsonl'):
                    answer = row['messages'][-1]['content']
                    diagram = answer.startswith('{') and json.loads(answer).get('kind') == 'chord_diagram'
                    write(row, source, 3 if split == 'train' and diagram else 1)
            for row in rows(inputs / f'paired_complete_{split}.jsonl'):
                write(row, 'paired', 16 if split == 'train' else 1)
            for row in rows(inputs / f'headers_{split}.jsonl'):
                write(row, 'headers', 4 if split == 'train' else 1)
            from research.inference.information.prompts import STAFF_PROMPT
            from research.inference.information.staff_image import focus_staff

            for index, row in enumerate(staff):
                provenance = row['provenance']
                target = json.loads(row['messages'][-1]['content'])
                target['name'] = provenance['visible_name']
                image = row['images'][0]
                if not provenance.get('focused_staff'):
                    crop_path = output / 'staff_images' / split / f'{index}.png'
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(image) as source:
                        focus_staff(source).save(crop_path)
                    image = str(crop_path.resolve())
                sample = {'messages': [{'role': 'user', 'content': '<image>' + STAFF_PROMPT},
                                       {'role': 'assistant', 'content': json.dumps(target, separators=(',', ':'))}],
                          'images': [image]}
                repeat = max(1, min(80, 300 // staff_counts[target['name']])) if split == 'train' else 1
                write(sample, 'staff', repeat)
        datasets[f'unified_{split}'] = dataset_entry(destination.name)
        print(split, dict(counts), 'total', counts.total(), flush=True)
    definitions.write_text(json.dumps(datasets, indent=2))


def build_refinement(output, workers=16, annotations=32000):
    """Add broader chord names and four-to-eight-string diagrams."""
    definitions = output / 'dataset_info.json'
    datasets = json.loads(definitions.read_text()) if definitions.is_file() else {}
    with ProcessPoolExecutor(workers) as pool:
        for split in ('train', 'validation', 'test'):
            count = annotations if split == 'train' else annotations // 8
            destination = output / f'refinement_{split}.jsonl'
            jobs = ((80000 + i, split, str(output / 'refinement_images')) for i in range(count))
            with destination.open('w') as handle:
                for row in pool.map(render_annotation, jobs, chunksize=32):
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            datasets[f'refinement_{split}'] = dataset_entry(destination.name)
            print(split, count, flush=True)
    definitions.write_text(json.dumps(datasets, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/unified_score'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--annotations', type=int, default=32000)
    parser.add_argument('--assemble-training', action='store_true')
    parser.add_argument('--inputs', type=Path, help='Existing task datasets; defaults to --output')
    parser.add_argument('--text-sources', type=Path,
                        help='Legacy text corrections: image path to [source label, mode, measure index]')
    parser.add_argument('--refinement', action='store_true')
    parser.add_argument('--staff-data', type=Path, help='Directory containing staff_train/validation/test.json with visible-name provenance')
    args = parser.parse_args()
    if args.assemble_training:
        assemble_training(args.output, args.inputs, args.text_sources, args.staff_data)
    elif args.refinement:
        build_refinement(args.output, args.workers, args.annotations)
    else:
        build(args.output, args.workers, args.annotations)
