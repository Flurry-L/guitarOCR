"""Generate musical notation and multi-staff supervision with native SVG geometry."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image

from research.data.training_samples import dataset_entry
from research.inference.information.prompts import CLEF_PROMPT
from research.inference.information.prompts import HEADER_PROMPT
from research.inference.information.prompts import STAFF_PROMPT
from research.inference.layout.structure import STRUCTURE_PROMPT
from research.inference.layout.structure import structure_image
from scorelib.m2 import format_measure_target
from scorelib.m2 import parse_measure_target
from scorelib.musicxml import duration_ticks
from scorelib.musicxml import write_musicxml
from scorelib.pitch_context import convert_pitch_target
from scorelib.pitch_context import transpose_key
from scorelib.score_state import SIGNATURE_PROMPT
from scorelib.score_state import attach_neighbours
from scorelib.score_state import key_fifths
from scorelib.score_state import signature_target
from scorelib.score_state import state_prompt
from scorelib.techniques import canonical_chord_marks


PROFILES = [
    ('Piano', 'pitched', 0, 'G2', 0, 64), ('Violin', 'pitched', 40, 'G2', 0, 72),
    ('Viola', 'pitched', 41, 'C3', 0, 60), ('Cello', 'pitched', 42, 'F4', 0, 48),
    ('Flute', 'pitched', 73, 'G2', 0, 76), ('Oboe', 'pitched', 68, 'G2', 0, 72),
    ('Bb Clarinet', 'pitched', 71, 'G2', -2, 70), ('Bb Trumpet', 'pitched', 56, 'G2', -2, 70),
    ('Alto Saxophone in Eb', 'pitched', 65, 'G2', -9, 72), ('Horn in F', 'pitched', 60, 'G2', -7, 67),
    ('Guitar', 'guitar', 25, 'G2', -12, 67), ('Bass', 'bass', 33, 'F4', -12, 52),
]


def generate(index, split, bars=16):
    from guitarpro.models import KeySignature
    from scorelib.instruments import standard_tuning

    rng = random.Random(20261003 + index + 100000 * ['train', 'validation', 'test'].index(split))
    profiles = rng.sample(PROFILES, rng.choice([1, 1, 2, 3, 4]))
    if index % 4 == 0:
        profiles[0] = PROFILES[0]
    fifths = rng.randint(-5, 5)
    key = next(k.name for k in KeySignature if k.value == (fifths, 0))
    time = rng.choice(['4/4', '4/4', '3/4', '6/8'])
    quarters = 4 if time == '4/4' else 3
    tonic = (fifths * 7) % 12
    pitch_classes = {(tonic + n) % 12 for n in (0, 2, 4, 5, 7, 9, 11)}
    parts = []
    for pi, (name, instrument, program, clef, shift, middle) in enumerate(profiles):
        staff_count = 2 if program == 0 else 1
        tuning = standard_tuning(instrument) if instrument in {'guitar', 'bass'} else []
        part = {'id': f'part-{pi + 1}', 'name': name, 'instrument': instrument, 'midi_program': program,
                'tuning': tuning, 'capo': 0, 'staves': []}
        for si in range(staff_count):
            staff = {'id': f'staff-{si + 1}', 'measures': []}
            context = {'clef': 'F4' if si else clef, 'clef_octave': 0, 'instrument_transpose': shift, 'octave_spans': [], 'capo': 0}
            centre = 45 if si else middle
            pitch_pool = [p for p in range(centre - 10, centre + 14) if p % 12 in pitch_classes]
            last = {}
            for bi in range(bars):
                data = parse_measure_target('M2 | V0{@0:w:r}')
                data.update(time_signature=time if bi == 0 else None, key_signature=key if bi == 0 else None,
                            print_time_signature=bi == 0, print_key_signature=bi == 0,
                            tempo_quarter=None,
                            voices=[])
                for vi in range(2 if index % 3 == 0 and program in {0, 25} else 1):
                    events, start = [], 0
                    for _quarter in range(quarters):
                        pattern = rng.choice([(4,), (8, 8), (8, 8), (16, 16, 16, 16), (8, 16), (3, 3, 3)])
                        for k, denominator in enumerate(pattern):
                            duration = {'value': 8 if denominator == 3 else denominator,
                                        'dotted': pattern == (8, 16) and k == 0, 'double_dotted': False,
                                        'tuplet_enters': 3 if denominator == 3 else 1,
                                        'tuplet_times': 2 if denominator == 3 else 1}
                            event = {'start': start, 'duration': duration, 'status': 'normal', 'effects': [], 'notes': []}
                            if rng.random() < .1:
                                event['status'] = 'rest'
                                last[vi] = []
                            else:
                                base = rng.randrange(max(1, len(pitch_pool) - 6))
                                size = rng.choice([1, 1, 2, 3, 4]) if program in {0, 25} else 1
                                pitches = [pitch_pool[min(base + j * 2, len(pitch_pool) - 1)] - 12 * vi for j in range(size)]
                                if tuning:
                                    pitches = [max(p, min(tuning) - shift) for p in pitches]
                                tied = rng.random() < .12 and bool(last.get(vi))
                                # MusicXML readers can merge unison ties between piano
                                # staves; keep generated ties in disjoint registers.
                                if program == 0 and vi:
                                    tied = False
                                if tied:
                                    pitches = last[vi]
                                for pitch in sorted(set(pitches)):
                                    effects = ['tie'] if tied else []
                                    if not tied and rng.random() < .08:
                                        effects.append(rng.choice(['stacc', 'accent', 'trill']))
                                    event['notes'].append({'pitch': pitch, 'effects': effects})
                                last[vi] = pitches
                            events.append(event)
                            start += duration_ticks(duration)
                    data['voices'].append({'voice': vi, 'events': events})
                written = format_measure_target(canonical_chord_marks(data), 'notation')
                sounding = convert_pitch_target(written, context)
                staff['measures'].append({**parse_measure_target(sounding), 'index': bi, 'mode': 'notation',
                                          'pitch_context': context, 'written_target': written, 'sounding_target': sounding})
            part['staves'].append(staff)
        parts.append(part)
    return {'schema': 'guitarocr.score/2', 'ticks_per_quarter': 960, 'title': f'Notation Study {index + 1}',
            'artist': '', 'timeline': [{'index': i, 'time_signature': time} for i in range(bars)], 'parts': parts}


def render(job):
    index, split, output = job
    import cairosvg
    import verovio

    root = Path(output) / split / f'score-{index:05d}'
    root.mkdir(parents=True, exist_ok=True)
    score = generate(index, split)
    xml = write_musicxml(score, root / 'score.musicxml')
    renderer = verovio.toolkit()
    renderer.setOptions({'scale': 70, 'pageWidth': 2100, 'pageHeight': 2970, 'svgBoundingBoxes': True,
                         'header': 'auto', 'footer': 'none', 'font': ['Leipzig', 'Bravura', 'Gootville'][index % 3]})
    if not renderer.loadFile(str(xml)):
        raise ValueError(f'Native engraving failed: {xml}')
    flat = [(pi, si, p, s) for pi, p in enumerate(score['parts']) for si, s in enumerate(p['staves'])]
    records, pages, structures, metadata = [], [], [], []
    for page in range(1, renderer.getPageCount() + 1):
        svg = renderer.renderToSVG(page)
        tree = ET.fromstring(svg)
        ns = {'s': 'http://www.w3.org/2000/svg'}
        width = float(tree.get('width').removesuffix('px'))
        scale = width / 21000
        image_path = root / f'page-{page}.png'
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(image_path), background_color='white')
        page_rows = defaultdict(list)
        page_labels, regions = {}, []
        systems = [n for n in tree.iter() if n.get('class') == 'system']
        with Image.open(image_path) as image:
            for system_index, system in enumerate(systems):
                for measure in (n for n in system if n.get('class') == 'measure'):
                    match = re.fullmatch(r'p1m(\d+)', measure.get('id', ''))
                    if not match:
                        continue
                    bi = int(match[1]) - 1
                    staves = [n for n in measure if n.get('class') == 'staff']
                    if len(staves) != len(flat):
                        raise ValueError('Engraving changed staff count')
                    for staff_node, (pi, si, part, staff) in zip(staves, flat, strict=True):
                        boxes = [n for n in staff_node.findall('.//s:rect', ns) if float(n.get('width', 0)) > 0 and float(n.get('height', 0)) > 0]
                        left = min(float(n.get('x')) for n in boxes)
                        top = min(float(n.get('y')) for n in boxes)
                        right = max(float(n.get('x')) + float(n.get('width')) for n in boxes)
                        bottom = max(float(n.get('y')) + float(n.get('height')) for n in boxes)
                        x0, y0 = max(0, (left + 500) * scale - 3), max(0, (top + 500) * scale - 22)
                        x1, y1 = min(image.width, (right + 500) * scale + 3), min(image.height, (bottom + 500) * scale + 22)
                        box = [x0, y0, x1 - x0, y1 - y0]
                        source_id = f'engraved-{split}-{index}-p{pi}-s{si}'
                        crop = root / 'crops' / f'{pi}-{si}-{bi}.png'
                        crop.parent.mkdir(exist_ok=True)
                        image.crop((int(x0), int(y0), int(x1 + 1), int(y1 + 1))).save(crop)
                        data = staff['measures'][bi]
                        written_key = transpose_key(next(m['key_signature'] for m in staff['measures'] if m.get('key_signature')), -data['pitch_context']['instrument_transpose'])
                        state = {'time': score['timeline'][bi]['time_signature'], 'key': key_fifths(written_key)}
                        visible = {n.get('class') for n in staff_node.iter() if n.get('class') in {'meterSig', 'keySig'}
                                   and (n.findall('.//s:use', ns) or n.findall('.//s:text', ns))}
                        row = {'id': f'{source_id}-{bi}', 'source_id': source_id, 'family': f'engraved-{split}-{index}',
                               'split': split, 'measure_index': bi, 'bar_index': bi, 'source_measures': 16,
                               'measure_number': len(records) + 1, 'mode': 'notation', 'visual_pitch': True,
                               'page': page, 'system_index': system_index, 'bbox': box, 'source_page': str(image_path.resolve()),
                               'image': str(crop.resolve()), 'part_id': part['id'], 'staff_id': staff['id'],
                               'part_name': part['name'], 'instrument': part['instrument'], 'midi_program': part['midi_program'],
                               'tuning': part['tuning'], 'capo': 0, 'pitch_context': data['pitch_context'],
                               'target': data['written_target'], 'sounding_target': data['sounding_target'],
                               'score_state': state, 'signature_target': signature_target(state['time'] if 'meterSig' in visible else None,
                                                                                        state['key'] if 'keySig' in visible else None)}
                        records.append(row)
                        for clef_index, clef_node in enumerate(n for n in staff_node if n.get('class') == 'clef'):
                            clef_boxes = [n for n in clef_node.findall('.//s:rect', ns) if float(n.get('width', 0)) > 0 and float(n.get('height', 0)) > 0]
                            if not clef_boxes:
                                continue
                            cx0 = max(0, (min(float(n.get('x')) for n in clef_boxes) + 500) * scale - 8)
                            cy0 = max(0, (min(float(n.get('y')) for n in clef_boxes) + 500) * scale - 8)
                            cx1 = min(image.width, (max(float(n.get('x')) + float(n.get('width')) for n in clef_boxes) + 500) * scale + 8)
                            cy1 = min(image.height, (max(float(n.get('y')) + float(n.get('height')) for n in clef_boxes) + 500) * scale + 8)
                            clef_path = root / 'crops' / f'clef-{pi}-{si}-{bi}-{clef_index}.png'
                            image.crop((int(cx0), int(cy0), int(cx1 + 1), int(cy1 + 1))).save(clef_path)
                            regions.append({'kind': 'clef', 'bbox': [cx0, cy0, cx1 - cx0, cy1 - cy0]})
                            metadata.append(chat(clef_path, CLEF_PROMPT, {'clef': data['pitch_context']['clef'], 'clef_octave': 0}))
                        page_rows[(system_index, pi, si)].append(row)
                        page_labels[(system_index, pi, si)] = [system_index, pi, si]
            marked = structure_image(image_path, list(page_rows.values()), root / f'structure-{page}.png')
            parts = [{'name': p['name'], 'instrument': p['instrument'], 'strings': None, 'program': p['midi_program']} for p in score['parts']]
            structures.append({'messages': [{'role': 'user', 'content': '<image>' + STRUCTURE_PROMPT + f'There are {len(page_rows)} marked rows.'},
                                            {'role': 'assistant', 'content': json.dumps({'parts': parts, 'rows': list(page_labels.values())}, separators=(',', ':'))}], 'images': [marked]})
            for (_system, pi, si), rows in page_rows.items():
                if rows[0]['bar_index'] != 0:
                    continue
                top = max(0, min(r['bbox'][1] for r in rows) - 12)
                bottom = min(image.height, max(r['bbox'][1] + r['bbox'][3] for r in rows) + 13)
                staff_path = root / f'staff-{pi}-{si}.png'
                image.crop((0, int(top), image.width, int(bottom))).save(staff_path)
                metadata.append(chat(staff_path, STAFF_PROMPT, {'instrument': score['parts'][pi]['instrument'], 'string_count': None}))
            if page == 1:
                top = min(r['bbox'][1] for r in records if r['page'] == 1)
                header_path = root / 'header.png'
                image.crop((0, 0, image.width, max(1, int(top)))).save(header_path)
                metadata.append(chat(header_path, HEADER_PROMPT, {'title': score['title'], 'artist': None, 'tuning_name': None}))
            pages.append({'image': str(image_path.resolve()), 'width': image.width, 'height': image.height, 'regions': regions})
    attach_neighbours(records)
    images = [Image.open(p['image']).convert('RGB') for p in pages]
    pdf = root / 'score.pdf'
    images[0].save(pdf, save_all=True, append_images=images[1:], resolution=180)
    for im in images:
        im.close()
    payload = {'id': f'engraved-{split}-{index}', 'split': split, 'pdf': str(pdf.resolve()), 'records': records, 'pages': pages, 'parts': score['parts']}
    (root / 'score.json').write_text(json.dumps(payload))
    chats = [{'messages': [{'role': 'user', 'content': '<image><image><image>' + state_prompt('notation', r['instrument'], r['score_state'], r['pitch_context'], first=r['bar_index'] == 0, visual_pitch=True)},
                           {'role': 'assistant', 'content': r['target']}], 'images': [r['image'], r['previous_image'], r['next_image']]} for r in records]
    return records, chats, structures, metadata


def chat(image, prompt, value):
    return {'messages': [{'role': 'user', 'content': '<image>' + prompt},
                         {'role': 'assistant', 'content': json.dumps(value, separators=(',', ':'))}],
            'images': [str(Path(image).resolve())]}


def build(output, train=1600, validation=160, test=160, workers=12):
    output.mkdir(parents=True, exist_ok=True)
    summary, info = {}, {}
    for split, count in [('train', train), ('validation', validation), ('test', test)]:
        counts = {'scores': count, 'bars': 0, 'pages': 0}
        with ProcessPoolExecutor(workers) as pool, (output / f'manifest_{split}.jsonl').open('w') as manifest, \
                (output / f'measure_{split}.jsonl').open('w') as measures, (output / f'structure_{split}.jsonl').open('w') as structures, \
                (output / f'info_{split}.jsonl').open('w') as infos, (output / f'state_{split}.jsonl').open('w') as states:
            for i, (rows, chats, pages, metadata) in enumerate(pool.map(render, ((i, split, str(output)) for i in range(count))), 1):
                manifest.writelines(json.dumps(r) + '\n' for r in rows)
                measures.writelines(json.dumps(r) + '\n' for r in chats)
                structures.writelines(json.dumps(r) + '\n' for r in pages)
                infos.writelines(json.dumps(r) + '\n' for r in metadata)
                states.writelines(json.dumps({'messages': [{'role': 'user', 'content': '<image>' + SIGNATURE_PROMPT}, {'role': 'assistant', 'content': r['signature_target']}],
                                             'images': [r['image']]}) + '\n' for r in rows)
                counts['bars'] += len(rows)
                counts['pages'] += len(pages)
                if i % 50 == 0:
                    print(split, i, '/', count, counts, flush=True)
        summary[split] = counts
        for kind in ('measure', 'structure', 'info'):
            info[f'{kind}_{split}'] = dataset_entry(f'{kind}_{split}.jsonl')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/engraved_scores'))
    parser.add_argument('--train', type=int, default=1600)
    parser.add_argument('--validation', type=int, default=160)
    parser.add_argument('--test', type=int, default=160)
    parser.add_argument('--workers', type=int, default=12)
    build(**vars(parser.parse_args()))
