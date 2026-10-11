"""Generate musical notation and multi-staff supervision with native SVG geometry."""

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from scorelib.score import format_measure_target, parse_measure_target
from scorelib.score import duration_ticks, write_musicxml
from scorelib.pitch_context import convert_pitch_target, transpose_key
from research.data.prompts import (
    SIGNATURE_PROMPT,
    key_fifths,
    signature_target,
)
from scorelib.techniques import canonical_chord_marks

from research.data.prompts import CLEF_PROMPT, HEADER_PROMPT
from research.data.training_samples import dataset_entry, staff_sample, visual_measure_sample
from research.data.build_staff_data import visible_instrument

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
                data = parse_measure_target('M | V0{@0:w:r}')
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
                            start += int(duration_ticks(duration))
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
    root = Path(output) / split / f'score-{index:05d}'
    return render_score(generate(index, split), root, f'engraved-{split}-{index}', split,
                        font=['Leipzig', 'Bravura', 'Gootville'][index % 3],
                        spacing=[(10, 4), (16, 6), (12, 5)][index % 3])


def render_score(score, root, family, split, *, font='Leipzig', spacing=None,
                 join_octaves=False, omit_octave_labels=False, bars_per_system=None):
    """Render an authored score using the same geometry and task serializers."""
    import cairosvg
    import verovio
    from research.data.observations import glyph_position, note_identity, verovio_links, verovio_observation

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    xml = write_musicxml(score, root / 'score.musicxml')
    # A playback time-modification does not print a tuplet number. Supply the
    # visual MusicXML brackets for each complete authored group before rendering.
    xml_tree = ET.parse(xml)
    for measure_node in xml_tree.iter('measure'):
        for notation in measure_node.iter('notations'):
            for existing in list(notation.findall('tuplet')):
                notation.remove(existing)
        groups = {}
        for note_node in measure_node.findall('note'):
            if note_node.find('chord') is not None:
                continue
            modification = note_node.find('time-modification')
            if modification is None:
                continue
            actual = int(modification.findtext('actual-notes'))
            normal = int(modification.findtext('normal-notes'))
            key = (note_node.findtext('staff'), note_node.findtext('voice'), actual, normal)
            group = groups.setdefault(key, [])
            group.append(note_node)
            if len(group) == actual:
                for endpoint, kind in [(group[0], 'start'), (group[-1], 'stop')]:
                    notation = endpoint.find('notations')
                    if notation is None:
                        notation = ET.SubElement(endpoint, 'notations')
                    ET.SubElement(notation, 'tuplet', {'type': kind, 'number': '1',
                        **({'bracket': 'yes', 'show-number': 'actual'} if kind == 'start' else {})})
                groups[key] = []
        if any(groups.values()):
            raise ValueError('Incomplete visual tuplet group')
    if join_octaves:
        for part_node in xml_tree.iter('part'):
            measures = part_node.findall('measure')
            for previous, current in zip(measures, measures[1:]):
                stop = next((n for n in reversed(list(previous)) if n.tag == 'direction'
                             and n.find("direction-type/octave-shift[@type='stop']") is not None), None)
                start = next((n for n in current if n.tag == 'direction'
                              and n.find('direction-type/octave-shift') is not None), None)
                if stop is None or start is None:
                    continue
                a, b = stop.find('direction-type/octave-shift'), start.find('direction-type/octave-shift')
                after = list(previous)[list(previous).index(stop) + 1:]
                before = list(current)[:list(current).index(start)]
                if (b.get('type') != 'stop' and a.get('size') == b.get('size')
                        and stop.get('placement') == start.get('placement')
                        and stop.findtext('voice') == start.findtext('voice')
                        and stop.findtext('staff') == start.findtext('staff')
                        and not any(n.tag == 'note' for n in before + after)):
                    previous.remove(stop)
                    current.remove(start)
    if bars_per_system:
        for part_node in xml_tree.iter('part'):
            for index, measure_node in enumerate(part_node.findall('measure')):
                if index and index % bars_per_system == 0:
                    measure_node.insert(0, ET.Element('print', {'new-system': 'yes'}))
    xml_tree.write(xml, encoding='utf-8', xml_declaration=True)
    renderer = verovio.toolkit()
    renderer.setOptions({'scale': 70, 'pageWidth': 2100, 'pageHeight': 2970, 'svgBoundingBoxes': True,
                         'header': 'auto', 'footer': 'none', 'font': font})
    if spacing is not None:
        renderer.setOptions({'spacingStaff': spacing[0], 'spacingSystem': spacing[1]})
    if bars_per_system:
        renderer.setOptions({'breaks': 'encoded'})
    if not renderer.loadFile(str(xml)):
        raise ValueError(f'Native engraving failed: {xml}')
    flat = [(pi, si, p, s) for pi, p in enumerate(score['parts']) for si, s in enumerate(p['staves'])]
    records, pages, structures, metadata = [], [], [], []
    for page in range(1, renderer.getPageCount() + 1):
        svg = renderer.renderToSVG(page)
        tree = ET.fromstring(svg)
        links = verovio_links(renderer.getMEI())
        if join_octaves or omit_octave_labels:
            for node in tree.iter():
                classes = node.get('class', '').split()
                # Omit only the native repeated number on a continuation;
                # selected studies also omit the origin's now-unreadable label.
                if classes[:1] == ['octave'] and (omit_octave_labels or 'spanning' in classes):
                    for glyph in list(node.findall('s:use', {'s': 'http://www.w3.org/2000/svg'})):
                        node.remove(glyph)
            svg = ET.tostring(tree, encoding='unicode')
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
                        for span in system.iter():
                            classes = span.get('class', '').split()
                            if not classes or classes[0] not in ('octave', 'trill') or 'bounding-box' in classes:
                                continue
                            link_id = span.get('id') or next((c[3:] for c in classes if c.startswith('id-')), None)
                            link = links.get(link_id, {})
                            begin, end = note_identity(link.get('startid', '')), note_identity(link.get('endid', ''))
                            if classes[0] == 'trill':
                                end = begin
                            if begin is None or end is None or begin[:2] != (pi, si) or not begin[2] <= bi <= end[2]:
                                continue
                            if classes[0] == 'trill':
                                for rect in span.findall('.//s:rect', ns):
                                    rx, ry, rw, rh = (float(rect.get(key, 0)) for key in ('x', 'y', 'width', 'height'))
                                    if rw > 0 and rh > 0:
                                        top, bottom = min(top, ry), max(bottom, ry + rh)
                                continue
                            for line in span.findall('s:path', ns):
                                points = list(map(float, re.findall(r'[-+]?(?:\d*\.\d+|\d+)', line.get('d', ''))))
                                if points and max(points[::2]) >= left and min(points[::2]) <= right:
                                    top = min(top, min(points[1::2]) - 12)
                                    bottom = max(bottom, max(points[1::2]) + 12)
                            for glyph in span.findall('s:use', ns):
                                gx, gy = glyph_position(glyph)
                                if left <= gx <= right:
                                    # Include the number above its baseline.
                                    # Padding also works on cross-system spans,
                                    # whose aggregate bounding box is displaced.
                                    top, bottom = min(top, gy - 400), max(bottom, gy + 80)
                        x0, y0 = max(0, (left + 500) * scale - 3), max(0, (top + 500) * scale - 22)
                        x1, y1 = min(image.width, (right + 500) * scale + 3), min(image.height, (bottom + 500) * scale + 22)
                        neighbour_bounds = []
                        for neighbour in staves:
                            if neighbour is staff_node:
                                continue
                            rects = [n for n in neighbour.findall('.//s:rect', ns)
                                     if float(n.get('height', 0)) > 0 and float(n.get('width', 0)) > 0]
                            neighbour_bounds.append((min(float(n.get('y')) for n in rects),
                                max(float(n.get('y')) + float(n.get('height')) for n in rects)))
                        for neighbour_top, neighbour_bottom in neighbour_bounds:
                            if neighbour_bottom <= top:
                                y0 = max(y0, ((neighbour_bottom + top) / 2 + 500) * scale)
                            if neighbour_top >= bottom:
                                y1 = min(y1, ((bottom + neighbour_top) / 2 + 500) * scale)
                        box = [x0, y0, x1 - x0, y1 - y0]
                        source_id = f'{family}-p{pi}-s{si}'
                        crop = root / 'crops' / f'{pi}-{si}-{bi}.png'
                        crop.parent.mkdir(exist_ok=True)
                        image.crop((int(x0), int(y0), int(x1 + 1), int(y1 + 1))).save(crop)
                        data = staff['measures'][bi]
                        written_key = transpose_key(next(m['key_signature'] for m in staff['measures'] if m.get('key_signature')), -data['pitch_context']['instrument_transpose'])
                        state = {'time': score['timeline'][bi]['time_signature'], 'key': key_fifths(written_key)}
                        visible = {n.get('class') for n in staff_node.iter() if n.get('class') in {'meterSig', 'keySig'}
                                   and (n.findall('.//s:use', ns) or n.findall('.//s:text', ns))}
                        row = {'id': f'{source_id}-{bi}', 'source_id': source_id, 'family': family,
                               'split': split, 'measure_index': bi, 'bar_index': bi, 'source_measures': len(staff['measures']),
                               'measure_number': len(records) + 1, 'mode': 'notation', 'visual_pitch': True,
                               'page': page, 'system_index': system_index, 'bbox': box, 'source_page': str(image_path.resolve()),
                               'image': str(crop.resolve()), 'part_id': part['id'], 'staff_id': staff['id'],
                               'part_name': part['name'], 'instrument': part['instrument'], 'midi_program': part['midi_program'],
                               'tuning': part['tuning'], 'capo': 0, 'pitch_context': data['pitch_context'],
                               'target': data['written_target'], 'sounding_target': data['sounding_target'],
                               'score_state': state, 'signature_target': signature_target(state['time'] if 'meterSig' in visible else None,
                                                                                        state['key'] if 'keySig' in visible else None)}
                        records.append(row)
                        row['observation'] = verovio_observation(staff_node, tree, data, (pi, si, bi), links,
                            (x0 / scale - 500, y0 / scale - 500, x1 / scale - 500, y1 / scale - 500))
                        row['reference_target'] = data['written_target']
                        row['oracle_auxiliary'] = {'state': {key: {'status': 'observed', 'value': value}
                            for key, value in {**state, **data['pitch_context']}.items()},
                            'voice_map': {str(v['voice']): v['voice'] for v in data['voices']}}
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
            parts = [{'name': p['name'], 'instrument': p['instrument'], 'strings': None, 'program': p['midi_program']} for p in score['parts']]
            structures.append({'image': str(image_path.resolve()),
                               'centers': [(min(r['bbox'][1] for r in rows) +
                                            max(r['bbox'][1] + r['bbox'][3] for r in rows)) / 2
                                           for rows in page_rows.values()],
                               'expected': {'parts': parts, 'rows': list(page_labels.values())},
                               'row_modes': ['notation'] * len(page_rows),
                               'family': family, 'split': split})
            for (_system, pi, si), rows in page_rows.items():
                if rows[0]['bar_index'] != 0:
                    continue
                top = max(0, min(r['bbox'][1] for r in rows) - 12)
                bottom = min(image.height, max(r['bbox'][1] + r['bbox'][3] for r in rows) + 13)
                visible_names, clipped_label = [], False
                for label in systems[_system]:
                    if label.get('class') not in ('label', 'labelAbbr'):
                        continue
                    rects = [n for n in label.findall('.//s:rect', ns)
                             if float(n.get('width', 0)) > 0 and float(n.get('height', 0)) > 0]
                    if not rects:
                        continue
                    y0 = (min(float(n.get('y')) for n in rects) + 500) * scale
                    y1 = (max(float(n.get('y')) + float(n.get('height')) for n in rects) + 500) * scale
                    if y1 <= top or y0 >= bottom:
                        continue
                    if y0 < top or y1 > bottom:
                        clipped_label = True
                        continue
                    name = ' '.join(''.join(label.itertext()).split())
                    if name:
                        visible_names.append(name)
                if clipped_label or len(visible_names) > 1:
                    continue
                name = visible_names[0] if visible_names else None
                staff_path = root / f'staff-{pi}-{si}.png'
                image.crop((0, int(top), image.width, int(bottom))).save(staff_path)
                metadata.append(staff_sample(staff_path, {
                    'instrument': visible_instrument(name, 'notation', False),
                    'string_count': None, 'name': name}, family=family, split=split, renderer='verovio', task='staff'))
            if page == 1:
                top = min(r['bbox'][1] for r in records if r['page'] == 1)
                header_path = root / 'header.png'
                image.crop((0, 0, image.width, max(1, int(top)))).save(header_path)
                metadata.append(chat(header_path, HEADER_PROMPT, {'title': score['title'], 'artist': None, 'tuning_name': None}))
            pages.append({'image': str(image_path.resolve()), 'width': image.width, 'height': image.height, 'regions': regions})
    images = [Image.open(p['image']).convert('RGB') for p in pages]
    pdf = root / 'score.pdf'
    images[0].save(pdf, save_all=True, append_images=images[1:], resolution=180)
    for im in images:
        im.close()
    payload = {'id': family, 'split': split, 'pdf': str(pdf.resolve()), 'records': records, 'pages': pages, 'parts': score['parts']}
    (root / 'score.json').write_text(json.dumps(payload))
    chats = [visual_measure_sample(r) for r in records if 'observation' in r]
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
                                             'images': [r['image']],
                                             'provenance': {key: r[key] for key in ('family', 'split')}}) + '\n' for r in rows)
                counts['bars'] += len(rows)
                counts['pages'] += len(pages)
                if i % 50 == 0:
                    print(split, i, '/', count, counts, flush=True)
        summary[split] = counts
        for kind in ('measure', 'info'):
            info[f'{kind}_{split}'] = dataset_entry(f'{kind}_{split}.jsonl')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=int, default=1600)
    parser.add_argument('--validation', type=int, default=160)
    parser.add_argument('--test', type=int, default=160)
    parser.add_argument('--workers', type=int, default=12)
    build(**vars(parser.parse_args()))
