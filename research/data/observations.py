"""Single-crop labels from painted glyphs; source scores only join event identity."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from copy import deepcopy

from scorelib.score import _duration_token

SVG = {'s': 'http://www.w3.org/2000/svg'}
XLINK = '{http://www.w3.org/1999/xlink}href'
XML_ID = '{http://www.w3.org/XML/1998/namespace}id'
ACCIDENTALS = {'E260': -1, 'E261': 0, 'E262': 1, 'E263': 2, 'E264': -2}
TAB_TECHNIQUES = {'hammer', 'sl', 'ss', 'sib', 'sia', 'sou', 'sod'}
DYNAMIC_SYMBOLS = {58666: 15, 58667: 31, 58656: 47, 58668: 63,
                   58669: 79, 58658: 95, 58671: 111, 58672: 127}
PICK_SYMBOLS = {0xE610: 'pick_down', 0xE612: 'pick_up'}


def glyph_code(node):
    return node.get(XLINK, '').lstrip('#').split('-')[0]


def glyph_position(node):
    match = re.search(r'translate\(([-.\d]+)[, ]+([-.\d]+)\)', node.get('transform', ''))
    if not match:
        raise ValueError('Rendered glyph has no position')
    return tuple(map(float, match.groups()))


def note_identity(value):
    match = re.fullmatch(r'#?n(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', value)
    if not match:
        return None
    part, staff, bar, voice, event, note = map(int, match.groups())
    return part - 1, staff - 1, bar - 1, voice - (staff - 1) * 16 - 1, event, note


def visible_state(value=None):
    return {'status': 'absent'} if value is None else {'status': 'observed', 'value': value}


def verovio_links(mei):
    """MEI supplies endpoint ownership; the SVG supplies whether a curve exists."""
    root = ET.fromstring(mei)
    chords = {}
    for node in root.iter():
        if node.tag.rsplit('}', 1)[-1] == 'chord' and node.get(XML_ID):
            head = next((n.get(XML_ID) for n in node if note_identity(n.get(XML_ID, ''))), None)
            if head:
                chords['#' + node.get(XML_ID)] = '#' + head
    return {n.get(XML_ID): {key: chords.get(value, value) if key in ('startid', 'endid') else value
                           for key, value in n.attrib.items()}
            for n in root.iter() if n.tag.rsplit('}', 1)[-1] in {'tie', 'trill', 'octave'}}


def verovio_observation(staff_node, tree, data, identity, links, crop_svg):
    pi, si, bi = identity
    x0, y0, x1, y1 = crop_svg
    lines = []
    for node in staff_node:
        if node.tag.endswith('path'):
            values = re.fullmatch(r'M([-.\d]+) ([-.\d]+) L([-.\d]+) ([-.\d]+)', node.get('d', ''))
            if values and values[2] == values[4]:
                lines.append(float(values[2]))
    if len(lines) != 5:
        raise ValueError('Notation observation requires five painted staff lines')
    top, bottom = min(lines), max(lines)
    step = (bottom - top) / 8
    nodes = {n.get('id'): n for n in staff_node.iter() if n.get('id')}
    parents = {child: parent for parent in staff_node.iter() for child in parent}
    state = {key: visible_state() for key in ('time', 'key', 'clef')}
    for node in staff_node:
        uses = node.findall('.//s:use', SVG)
        if node.get('class') == 'keySig' and uses:
            values = [ACCIDENTALS.get(glyph_code(n)) for n in uses]
            if any(v is None for v in values):
                state['key'] = {'status': 'unknown'}
            else:
                state['key'] = visible_state(sum(values))
        elif node.get('class') == 'meterSig' and uses:
            rows = {}
            for glyph in uses:
                code = glyph_code(glyph)
                x, y = glyph_position(glyph)
                if re.fullmatch('E08[0-9]', code):
                    rows.setdefault(y, []).append((x, code[-1]))
            if len(rows) == 2:
                numbers = [''.join(digit for _, digit in sorted(row)) for _, row in sorted(rows.items())]
                state['time'] = visible_state('/'.join(numbers))
            else:
                state['time'] = {'status': 'unknown'}
        elif node.get('class') == 'clef' and uses:
            letter = {'E050': 'G', 'E062': 'F', 'E05C': 'C'}.get(glyph_code(uses[0]))
            if letter:
                line = round((bottom - glyph_position(uses[0])[1]) / (2 * step)) + 1
                state['clef'] = visible_state(f'{letter}{line}')
                state['clef_octave'] = visible_state(0)
            else:
                state['clef'] = {'status': 'unknown'}
    voices = []
    for voice in data['voices']:
        vi = voice['voice']
        events = []
        for ei, event in enumerate(voice['events']):
            prefix = f'n{pi + 1}_{si + 1}_{bi + 1}_{si * 16 + vi + 1}_{ei}_'
            if event['duration'].get('tuplet_enters', 1) != 1:
                current = nodes.get(prefix + '0')
                while current is not None and current.get('class') != 'tuplet':
                    current = parents.get(current)
                if current is None or not current.findall("s:g[@class='tupletNum']//s:use", SVG):
                    raise ValueError(f'Tuplet ratio has no printed number: {prefix}')
            notes, articulation_groups = [], set()
            for ni, _source_note in enumerate(event['notes']):
                node = nodes.get(prefix + str(ni))
                heads = [] if node is None else node.findall("s:g[@class='notehead']/s:use", SVG)
                if len(heads) != 1:
                    raise ValueError(f'No unique painted notehead: {prefix}{ni}')
                y = glyph_position(heads[0])[1]
                accidentals = node.findall("s:g[@class='accid']//s:use", SVG)
                accidental = None if not accidentals else ACCIDENTALS.get(glyph_code(accidentals[0]), 'unknown')
                note = {'staff_position': round((bottom - y) / step), 'accidental': accidental}
                notes.append(note)
                owner = parents.get(node)
                articulation_groups.add(owner if owner is not None and owner.get('class') == 'chord' else node)
            # Engravers attach articulations to the chord, independently of
            # source note ownership. Use the existing top-note convention.
            marks = []
            for owner in articulation_groups:
                for glyph in owner.findall(".//s:g[@class='artic']/s:use", SVG):
                    effect = {'E4A0': 'accent', 'E4A1': 'accent', 'E4A2': 'stacc', 'E4A3': 'stacc'}.get(glyph_code(glyph))
                    gx, gy = glyph_position(glyph)
                    if effect and x0 <= gx <= x1 and y0 <= gy <= y1:
                        marks.append(effect)
            ordered_notes = sorted(notes, key=lambda note: -note['staff_position'])
            for effect in sorted(set(marks)):
                for i in range(marks.count(effect)):
                    ordered_notes[i % len(ordered_notes)].setdefault('effects', []).append(effect)
            if not notes and prefix + '0' not in nodes:
                raise ValueError(f'No painted rest: {prefix}')
            events.append({'start': event['start'], 'duration': _duration_token(event['duration']),
                           'status': event['status'], 'notes': notes})
        voices.append({'voice': vi, 'events': events})
    relations = []
    for node in tree.iter():
        classes = node.get('class', '').split()
        if not classes or classes[0] not in {'tie', 'trill', 'octave'} or 'bounding-box' in classes:
            continue
        link_id = node.get('id') or next((c[3:] for c in classes if c.startswith('id-')), None)
        if link_id not in links:
            continue
        link = links[link_id]
        start = note_identity(link.get('startid', ''))
        end = note_identity(link.get('endid', ''))
        if start is None or start[:2] != (pi, si):
            continue
        if classes[0] == 'trill':
            # A local trill is a printed glyph, often with no extension path.
            printed = [glyph for glyph in node.findall('s:use', SVG)
                       if glyph_code(glyph) == 'E566' and x0 <= glyph_position(glyph)[0] <= x1
                       and y0 <= glyph_position(glyph)[1] <= y1]
            if printed and start[2] == bi:
                voice = next(v for v in voices if v['voice'] == start[3])
                voice['events'][start[4]]['notes'][start[5]].setdefault('effects', []).append('trill')
            continue
        # Verovio repeats the destination bounding box on the origin of a
        # broken tie. The actual path is the only reliable local geometry.
        coordinates = []
        for path in node.findall('s:path', SVG):
            values = list(map(float, re.findall(r'[-+]?(?:\d*\.\d+|\d+)', path.get('d', ''))))
            coordinates.extend(zip(values[::2], values[1::2]))
        if not coordinates:
            continue
        bx, by = min(x for x, _ in coordinates), min(y for _, y in coordinates)
        bw, bh = max(x for x, _ in coordinates) - bx, max(y for _, y in coordinates) - by
        if bx + bw <= x0 or bx >= x1 or by + bh <= y0 or by >= y1:
            continue
        if end is None or not start[2] <= bi <= end[2]:
            continue
        if classes[0] == 'octave':
            printed = [glyph for glyph in node.findall('s:use', SVG)
                       if x0 <= glyph_position(glyph)[0] <= x1 and y0 <= glyph_position(glyph)[1] <= y1]
            value = None
            if printed:
                size = {'E510': 12, 'E511': 12, 'E512': 12, 'E513': 12,
                        'E514': 24, 'E515': 24, 'E516': 24, 'E517': 24}.get(glyph_code(printed[0]))
                if size:
                    value = size if glyph_position(printed[0])[1] < top else -size
            relation = {'kind': 'ottava' if value is not None else None,
                        'start': list(start[3:5]) if start[2] == bi else 'left',
                        'end': list(end[3:5]) if end[2] == bi else 'right'}
            if value is not None:
                relation['value'] = value
            if start[2] != bi or end[2] != bi:
                relation.update(voice=start[3], lane=round((coordinates[0][1] - top) / (bottom - top), 3))
            relations.append(relation)
            continue
        if start[3] != end[3]:
            raise ValueError(f'Painted Verovio tie crosses unsupported voices: {start} -> {end}')
        relation = {'kind': 'tie', 'start': list(start[3:]) if start[2] == bi else 'left',
                    'end': list(end[3:]) if end[2] == bi else 'right'}
        if start[2] != bi or end[2] != bi:
            relation.update(voice=start[3], lane=round((coordinates[0][1] - top) / (bottom - top), 3))
        relations.append(relation)
    return {'state': state, 'voices': voices, 'relations': relations}


def write_dataset(output, rows):
    from research.data.training_samples import dataset_entry, visual_measure_sample

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    families, identities = {}, set()
    for row in rows:
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError(f"Unknown dataset split: {row['split']}")
        if families.setdefault(row['family'], row['split']) != row['split']:
            raise ValueError(f"Family crosses splits: {row['family']}")
        if row['id'] in identities:
            raise ValueError(f"Duplicate observation: {row['id']}")
        identities.add(row['id'])
        row['observation']['relations'] = sorted(row['observation']['relations'], key=relation_order)
    counts = {}
    for split in ('train', 'validation', 'test'):
        selected = [row for row in rows if row['split'] == split]
        counts[split] = len(selected)
        with (output / f'manifest_{split}.jsonl').open('w') as manifest, (output / f'{split}.jsonl').open('w') as chats:
            for row in selected:
                manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                chats.write(json.dumps(visual_measure_sample(row), ensure_ascii=False) + '\n')
    (output / 'dataset_info.json').write_text(json.dumps({split: dataset_entry(f'{split}.jsonl') for split in counts}, indent=2))
    (output / 'dataset.json').write_text(json.dumps({'complete': True, 'supervision': 'visible_observation',
        'measure_images': 1, 'image_geometry': 'preserve_aspect_ratio', 'splits': counts,
        'renderers': sorted({row.get('renderer', 'unknown') for row in rows}),
        'coverage': {split: {mode: sum(row['split'] == split and ('drums' if row.get('instrument') == 'drums' else row['mode']) == mode
                         for row in rows) for mode in ('tab', 'notation', 'both', 'drums')} for split in counts}}, indent=2))
    return counts


def relation_order(relation):
    """Read entering arcs first, then local endpoints, then exiting arcs."""
    def endpoint(value):
        if value == 'left':
            return (-1, -1, -1)
        if value is None or value == 'right':
            return (10**9, 10**9, 10**9)
        return tuple(value) + (-1,) * (3 - len(value))

    start, end = relation['start'], relation['end']
    group = 0 if start == 'left' else 2 if end == 'right' else 1
    return (group, endpoint(start), endpoint(end),
            json.dumps(relation, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def notation_job(job):
    from research.data.engraved_scores import generate, render_score
    from scorelib.score import format_measure_target, parse_measure_target

    output, split, index = job
    score = generate(index, split, bars=12)
    # Chromatic studies retain their printed key signature, so explicit
    # accidentals and within-bar accidental memory both appear in the images.
    shift = [0, 1, 0, -1][index % 4]
    for part in score['parts']:
        for staff in part['staves']:
            for measure in staff['measures']:
                for voice in measure['voices']:
                    for event in voice['events']:
                        for note in event['notes']:
                            note['pitch'] += shift
                for field in ('written_target', 'sounding_target'):
                    parsed = parse_measure_target(measure[field])
                    for voice in parsed['voices']:
                        for event in voice['events']:
                            for note in event['notes']:
                                note['pitch'] += shift
                    measure[field] = format_measure_target(parsed, 'notation')
    family = f'notation-{split}-{index:05d}'
    values, *_ = render_score(score, Path(output) / 'images' / family, family, split,
                              font=['Leipzig', 'Bravura', 'Gootville'][index % 3], spacing=(32, 12))
    for row in values:
        row['renderer'] = 'verovio'
    return values


def build_notation(output, train, validation, test, workers=8, index_offset=0):
    rows = []
    jobs = [(str(output), split, index) for split, count in
            [('train', train), ('validation', validation), ('test', test)]
            for index in range(index_offset, index_offset + count)]
    with ProcessPoolExecutor(workers) as pool:
        for index, values in enumerate(pool.map(notation_job, jobs), 1):
            rows.extend(values)
            print('notation', index, len(jobs), len(rows), flush=True)
    return rows


def tab_score(index, split):
    from scorelib.score import duration_ticks, format_measure_target, parse_measure_target

    # Eight views share one phrase: range marks on/off, independently combined
    # with bar numbers and ordinary tuning/instrument headers on/off.
    phrase = index - index % 8
    appearance = index % 8 // 2
    rng = random.Random(600000 * ['train', 'validation', 'test'].index(split) + phrase)
    identity, family = f'tab-{split}-{index:05d}', f'tab-{split}-{phrase:05d}'
    meter = rng.choice(['3/4', '4/4', '6/8'])
    tuning = rng.choice([[64, 59, 55, 50, 45, 40], [64, 59, 55, 50, 45, 38],
                         [62, 57, 55, 50, 45, 38]])
    part_name = rng.choice(['Acoustic Guitar', 'Electric Guitar', 'Classical Guitar'])
    quarters = 4 if meter == '4/4' else 3
    measures = []
    for mi in range(8):
        row = dict(part_id='part-1', staff_id='staff-1', instrument='guitar',
                   mode='tab', capo=0, visual_pitch=True, written_pitch=False,
                   pitch_reference='fingering', corpus='composed', renderer='alphatab',
                   pitch_context=dict(clef='G2', clef_octave=0, instrument_transpose=-12,
                                      capo=0, octave_spans=[]),
                   parsed=parse_measure_target('M | V0{@0:w:r}'))
        row.update(id=f'{identity}-{mi}', source_id=identity, family=family, split=split,
                   source_measures=8, measure_index=mi, bar_index=mi, measure_number=mi + 1,
                   tuning=tuning, part_name=part_name,
                   midi_program={'Classical Guitar': 24, 'Acoustic Guitar': 25, 'Electric Guitar': 27}[part_name])
        row['score_state'] = {'time': meter, 'key': 0, 'tuning': row['tuning']}
        parsed = row['parsed']
        parsed.update(time_signature=meter if mi == 0 else None, print_time_signature=mi == 0)
        events, start = [], 0
        patterns = [[(8, False)] * (2 * quarters)] * 3 + [[(4, False)] * quarters,
            [(8, True), (16, False)] + [(4, False)] * (quarters - 1),
            [(8, False), (16, False), (16, False)] + [(8, False)] * (2 * quarters - 2)]
        patterns += [[(2, True)], [(4, True), (4, True)]] if quarters == 3 else [
            [(2, False), (4, False), (4, False)], [(4, True), (8, False), (2, False)]]
        whole_rest = rng.random() < .12
        pattern = [(1, False)] if whole_rest and quarters == 4 else [(2, True)] if whole_rest else rng.choice(patterns)
        for ei, (value, dotted) in enumerate(pattern):
            duration = {'value': value, 'dotted': dotted, 'double_dotted': False,
                        'tuplet_enters': 1, 'tuplet_times': 1}
            count = rng.choice([1, 1, 2, 3, 4])
            notes = [{'string': string, 'fret': rng.randrange(0, 18), 'effects': []}
                     for string in sorted(rng.sample(range(1, 7), count))]
            for note in notes:
                note['pitch'] = row['tuning'][note['string'] - 1] + note['fret'] + 12
            status = 'normal'
            if whole_rest or rng.random() < .06:
                notes, status = [], 'rest'
            events.append({'start': start, 'duration': duration, 'status': status, 'notes': notes, 'effects': []})
            start += int(duration_ticks(duration))
        if start != 960 * quarters:
            raise ValueError(f'Authored TAB rhythm does not fill {meter}')
        # Reserve connected pairs before adding ties. Range marks are authored
        # independently below, so curves, hidden tied frets and lines coexist.
        reserved = set()
        pairs = list(range(len(events) - 1))
        rng.shuffle(pairs)
        for ei in pairs:
            if (ei in reserved or ei + 1 in reserved or not events[ei]['notes']
                    or not events[ei + 1]['notes'] or rng.random() < .4):
                continue
            string, fret = rng.randint(1, 6), rng.randint(7, 17)
            change = rng.choice([-7, -5, -3, -2, -1, 1, 2, 3, 5, 7])
            technique = rng.choice(['hammer', 'hammer', 'sl', 'ss'])
            for offset in (0, 1):
                value = fret + (change if offset else 0)
                events[ei + offset]['notes'] = [{'string': string, 'fret': value,
                    'pitch': row['tuning'][string - 1] + value + 12,
                    'effects': [technique] if offset == 0 else []}]
                reserved.add(ei + offset)
        for ei, event in enumerate(events):
            if ei not in reserved and event['notes'] and rng.random() < .32:
                rng.choice(event['notes'])['effects'].append(rng.choice(['sib', 'sia', 'sou', 'sod']))
        # Match H/P and slide phrases with independent picking signs. Views
        # with no sign retain exactly the same notes and local techniques.
        if appearance:
            for ei, event in enumerate(events):
                if event['notes'] and (ei == 0 or any('hammer' in n['effects'] for n in event['notes'])
                                       or ei % 3 == 0):
                    event['effects'].append('pick_down' if appearance == 1 else 'pick_up'
                                            if appearance == 2 else ['pick_down', 'pick_up'][ei % 2])
        if mi == 0 or rng.random() < .4:
            sounded = [event for event in events if event['notes']]
            if sounded:
                rng.choice(sounded)['effects'].append(f'dyn:{rng.choice([31, 47, 63, 79, 95, 111])}')
        parsed['voices'] = [{'voice': 0, 'events': events}]
        measures.append(row)
    flat = [event for row in measures for event in row['parsed']['voices'][0]['events']]
    for previous, event in zip(flat, flat[1:]):
        if (not previous['notes'] or not event['notes'] or rng.random() >= .32
                or any(n['effects'] for n in previous['notes'] + event['notes'])):
            continue
        tied = deepcopy(rng.sample(previous['notes'], rng.randint(1, len(previous['notes']))))
        for note in tied:
            note['effects'] = ['tie']
        strings = {note['string'] for note in tied}
        event['notes'] = sorted(tied + [n for n in event['notes'] if n['string'] not in strings],
                                key=lambda note: note['string'])
    scope_shape = phrase // 8 % 4
    ranges = []
    if scope_shape < 2:
        first = rng.randint(0, 2)
        last = min(7, first + rng.randint(3, 6))
        kind = ['pm', 'let'][scope_shape]
        for mi in range(first, last + 1):
            events = measures[mi]['parsed']['voices'][0]['events']
            begin = rng.randrange(len(events)) if mi == first else 0
            end = rng.randrange(begin, len(events)) + 1 if mi == last else len(events)
            ranges.append((kind, events[begin:end]))
    else:
        cursor = rng.randrange(min(5, len(flat)))
        while cursor < len(flat):
            length = rng.randint(1, 5) if scope_shape == 2 else rng.randint(4, 12)
            kind = rng.choice(['pm', 'let'])
            ranges.append((kind, flat[cursor:cursor + length]))
            cursor += length + rng.randint(1, 8)
    if index % 2:
        for kind, events in ranges:
            for event in events:
                for note in event['notes']:
                    note['effects'].append(kind)
    for row in measures:
        row['target'] = format_measure_target(row['parsed'], 'tab')
    scale = rng.choice([.85, 1.05, 1.25, 1.55, 1.85, 2.1])
    bars_per_row = rng.choice([2, 3, 4])
    widest = max(len(row['parsed']['voices'][0]['events']) for row in measures)
    # Larger notation needs a proportionally wider page. Forcing a fixed page
    # width can overlap adjacent frets until native slides disappear entirely.
    width = math.ceil(scale * max(rng.choice([700, 900, 1150, 1450]),
                                  bars_per_row * (90 + 24 * widest)))
    return {'id': identity, 'title': '', 'mode': 'tab', 'renderer': 'alphatab',
            'render_variant': appearance, 'measures': measures,
            'render_options': {'width': width, 'scale': scale, 'bars_per_row': bars_per_row,
                'effect_line_scale': rng.choice([.45, .65, 1.0]),
                'effect_padding': rng.choice([8, 20, 36, 50]),
                'let_label': rng.choice(['let ring', 'let ring', 'LetRing']),
                'extend_line_effects': scope_shape < 2 or rng.random() < .5,
                'crop_top_margin': rng.choice([1, 3, 8, 16]),
                'continuous_scopes': scope_shape < 2 or rng.random() < .65,
                'hide_headers': appearance < 2, 'hide_bar_numbers': appearance % 2 == 0,
                'capture_techniques': True, 'capture_observations': True, 'notation_mode': 0}}


def tab_chord_score(index, split):
    """The same music with no annotation, a name, a grid, and a clipped grid.

    The existing eight views vary techniques, numbers and headers. Each gets
    four chord views, all kept in one family, so diagram digits cannot explain
    changes in the note/rhythm labels.
    """
    from urllib.parse import quote
    from scorelib.chords import diagram_effect
    from scorelib.score import format_measure_target
    from research.data.synthetic_scores import sample_diagram

    appearance = index % 4
    state = tab_score(index // 4, split)
    identity = f'tab-chords-{split}-{index:05d}'
    family = f'tab-chords-{split}-{index - index % 32:05d}'
    rng = random.Random(f'chords:{split}:{index // 32}')
    shared_name, shared_diagram = sample_diagram(rng)
    while max(f for f in shared_diagram['frets'] if type(f) is int) > 12:
        shared_name, shared_diagram = sample_diagram(rng)
    # Octave transposition preserves the chord name but changes its voicing.
    # Muted strings stay muted; open strings become fretted an octave higher.
    alternative = deepcopy(shared_diagram)
    alternative['frets'] = [f + 12 if type(f) is int else f for f in alternative['frets']]
    alternative['barres'] = [[f + 12, a, b] for f, a, b in alternative['barres']]
    alternative['base_fret'] = min(f for f in alternative['frets'] if type(f) is int and f > 0)
    for mi, row in enumerate(state['measures']):
        row.update(id=f'{identity}-{mi}', source_id=identity, family=family)
        events = row['parsed']['voices'][0]['events']
        # One and several annotations, including two voicings of one name.
        anchors = [0] if mi % 2 == 0 else sorted({0, len(events) // 2})
        for ai, ei in enumerate(anchors):
            name, diagram = ((shared_name, shared_diagram if ai == 0 else alternative)
                             if mi % 3 == 0 else sample_diagram(rng))
            if appearance:
                events[ei]['effects'].append('chord:' + quote(name, safe=''))
                if appearance >= 2:
                    events[ei]['effects'].append(diagram_effect(diagram))
        row['target'] = format_measure_target(row['parsed'], 'tab')
    state.update(id=identity, render_variant=appearance,
                 chord_view=['none', 'name', 'diagram', 'clipped_diagram'][appearance])
    state['render_options'].update(chord_diagrams_in_score=appearance >= 2,
                                   chord_diagrams_on_top=False, justify_last_system=True)
    # Grids can be wider than a short beat; give the native engraver enough
    # page width to place several grids without colliding neighbouring notes.
    state['render_options']['width'] = max(state['render_options']['width'],
        math.ceil(state['render_options']['scale'] * state['render_options']['bars_per_row'] * 360))
    return state


def octave_job(job):
    from research.data.engraved_scores import generate, render_score
    from scorelib.score import format_measure_target, parse_measure_target
    from scorelib.pitch_context import convert_pitch_target

    output, split, index = job
    score = generate(index, split, bars=8)
    score['parts'] = score['parts'][:1]
    part = score['parts'][0]
    part.update(name='Flute' if index % 2 == 0 else 'Cello', instrument='pitched',
                midi_program=73 if index % 2 == 0 else 42, tuning=[])
    part['staves'] = part['staves'][:1]
    staff = part['staves'][0]
    shift = [12, -12, 24, -24][index % 4]
    context = {'clef': 'G2' if index % 2 == 0 else 'F4', 'clef_octave': 0,
               'instrument_transpose': 0, 'capo': 0, 'octave_spans': []}
    rng = random.Random(800000 * ['train', 'validation', 'test'].index(split) + index)
    for bi, measure in enumerate(staff['measures']):
        data = parse_measure_target('M | V0{@0:w:r}')
        data.update(time_signature='4/4' if bi == 0 else None, key_signature='GMajor' if bi == 0 else None,
                    print_time_signature=bi == 0, print_key_signature=bi == 0)
        events = []
        for ei in range(8):
            pitch = rng.choice([60, 62, 64, 65, 67, 69, 71, 72]) - (12 if index % 2 else 0)
            events.append({'start': ei * 480, 'duration': {'value': 8, 'dotted': False, 'double_dotted': False,
                'tuplet_enters': 1, 'tuplet_times': 1}, 'status': 'normal',
                'notes': [{'pitch': pitch, 'effects': []}],
                'effects': [f'ottava:{shift}'] if bi < 3 else [f'ottava:{-shift}'] if 4 <= bi < 7 else []})
        data['voices'] = [{'voice': 0, 'events': events}]
        written = format_measure_target(data, 'notation')
        sounding = convert_pitch_target(written, context)
        measure.update(**parse_measure_target(sounding), index=bi, mode='notation', pitch_context=context,
                       written_target=written, sounding_target=sounding)
    score['timeline'] = [{'index': i, 'time_signature': '4/4'} for i in range(8)]
    family = f'octave-{split}-{index:05d}'
    values, *_ = render_score(score, Path(output) / 'images' / family, family, split,
        font=['Leipzig', 'Bravura', 'Gootville'][index % 3], spacing=(20, 12),
        join_octaves=True, omit_octave_labels=index % 5 == 0, bars_per_system=2)
    for row in values:
        row['renderer'] = 'verovio'
    return values


def build_octave(output, train, validation, test, workers=8, index_offset=0):
    jobs = [(str(output), split, index) for split, count in
            [('train', train), ('validation', validation), ('test', test)]
            for index in range(index_offset, index_offset + count)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for index, values in enumerate(pool.map(octave_job, jobs), 1):
            rows.extend(values)
            print('octave', index, len(jobs), len(rows), flush=True)
    return rows


def intersects(box, crop):
    x, y, w, h = box
    return w > 0 and h > 0 and x + w > crop[0] and x < crop[2] and y + h > crop[1] and y < crop[3]


def continuation_image(image, geometry, index, state):
    """Change only traced text/hooks; draw each continuation with its own dashes."""
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(image)
    spans = geometry['techniqueSpans']
    boxes = geometry['boxes']
    for span in spans:
        span['label_visible'] = True
    for previous, current in zip(spans, spans[1:]):
        a, b = previous['events'][-1], current['events'][0]
        last = len(state['measures'][a['mi']]['parsed']['voices'][a['vi']]['events']) - 1
        linked = (previous['kind'] == current['kind'] and a['ei'] == last
                  and a['mi'] + 1 == b['mi'] and b['ei'] == 0)
        if not linked or not state['render_options']['continuous_scopes']:
            continue
        if any(sum(c['type'] == 'rect' and c['bbox'][2] > c['bbox'][3]
                   for c in span['commands']) < 2 for span in (previous, current)):
            continue
        for span, side in [(previous, 'right'), (current, 'left')]:
            rects = [c['bbox'] for c in span['commands'] if c['type'] == 'rect']
            horizontal = [b for b in rects if b[2] > b[3]]
            x, y, w, h = horizontal[0]
            spacing = horizontal[1][0] - x
            event = span['events'][-1 if side == 'right' else 0]
            box = boxes[event['mi']]
            if side == 'right':
                for rx, ry, rw, rh in rects:
                    if rh > rw:
                        draw.rectangle((rx - 1, ry - 1, rx + rw + 1, ry + rh + 1), fill='white')
                start, end = horizontal[-1][0] + spacing, box[0] + box[2] + 2
                span['open_right'] = True
            else:
                for command in span['commands']:
                    if command['type'] == 'text':
                        rx, ry, rw, rh = command['bbox']
                        draw.rectangle((rx - 1, ry - 1, rx + rw + 1, ry + rh + 1), fill='white')
                end = horizontal[0][0]
                start = end - math.ceil((end - box[0] + 2) / spacing) * spacing
                span.update(open_left=True, label_visible=False)
            left, upper = math.floor(start) - 1, math.floor(y) - 1
            size = (max(1, math.ceil(end) - left + 1), math.ceil(y + h) - upper + 1)
            overlay = Image.new('RGBA', (size[0] * 4, size[1] * 4))
            painter = ImageDraw.Draw(overlay)
            for step in range(max(0, math.ceil((end - start) / spacing))):
                dx = start + step * spacing
                dash = [dx, y, min(w, end - dx), h]
                x0, y0 = round((dx - left) * 4), round((y - upper) * 4)
                x1, y1 = round((dx + dash[2] - left) * 4), round((y + h - upper) * 4)
                if x1 > x0 and y1 > y0:
                    painter.rectangle((x0, y0, x1 - 1, y1 - 1), fill='black')
                    span['commands'].append({'type': 'rect',
                        'bbox': [left + x0 / 4, upper + y0 / 4, (x1 - x0) / 4, (y1 - y0) / 4]})
            overlay = overlay.resize(size, Image.Resampling.LANCZOS)
            image.paste(overlay, (left, upper), overlay)
    # Some pages genuinely start after an unreadable or lost effect label.
    if index % 11 == 0 and spans:
        span = spans[0]
        for command in span['commands']:
            if command['type'] == 'text':
                x, y, w, h = command['bbox']
                draw.rectangle((x - 1, y - 1, x + w + 1, y + h + 1), fill='white')
        span['label_visible'] = False


def tab_crops(geometry, size):
    painted, boxes = geometry['painted'], geometry['boxes']
    staff_type = 'tab' if any(p.get('staff_type') == 'tab' for p in painted) else 'score'
    centers = {p['mi']: (min(p['lines']) + max(p['lines'])) / 2
               for p in painted if p['kind'] == 'staff' and p.get('staff_type', 'tab') == staff_type}
    visible_tops, visible_bottoms = [], []
    for mi, box in enumerate(geometry['boxes']):
        staff_ink = [p['bbox'] for p in painted if p['kind'] == 'staff' and p['mi'] == mi]
        tops = [b[1] for b in staff_ink]
        bottoms = [b[1] + b[3] for b in staff_ink]
        tops.extend(p['bbox'][1] for p in painted
                    if p['kind'] in ('fret', 'signature', 'notehead', 'accidental') and p['mi'] == mi
                    and (p['texts'] or p['symbols']))
        marks = [p['bbox'] for p in painted if p['kind'] in ('chord', 'pick', 'dynamic') and p['mi'] == mi]
        tops.extend(b[1] for b in marks)
        bottoms.extend(b[1] + b[3] for b in marks)
        context = [p['bbox'] for p in geometry.get('context', []) if p['mi'] == mi]
        tops.extend(box[1] for box in context)
        bottoms.extend(box[1] + box[3] for box in context)
        tops.extend(c['bbox'][1] for span in geometry['techniqueSpans']
                    if any(e['mi'] == mi for e in span['events']) for c in span['commands'])
        for curve in (p for p in painted if p['kind'] in ('tie', 'technique') and p['paths']
                      and p['start']['mi'] <= mi <= p['end']['mi']):
            _, y, _, h = curve['bbox']
            distance = abs(y + h / 2 - centers[mi])
            nearest = min(abs(y + h / 2 - centers[curve[end]['mi']]) for end in ('start', 'end'))
            if distance <= nearest + .01:
                tops.append(y)
                bottoms.append(y + h)
        visible_tops.append(min(tops))
        visible_bottoms.append(max(bottoms))
    crops = []
    for mi, (x, y, w, h) in enumerate(boxes):
        crop = [max(0, math.floor(x) - 1),
                max(0, math.floor(visible_tops[mi]) - geometry.get('crop_top_margin', 2)),
                min(size[0], math.ceil(x + w) + 1), min(size[1], math.ceil(visible_bottoms[mi]) + 2)]
        for item in geometry.get('context', []):
            if item['mi'] == mi:
                bx, _, bw, _ = item['bbox']
                crop[0] = max(0, min(crop[0], math.floor(bx) - 1))
                crop[2] = min(size[0], max(crop[2], math.ceil(bx + bw) + 1))
        for technique in (p for p in painted if p['kind'] == 'technique' and p['paths']
                          and p['start']['mi'] == mi and p['end']['mi'] == mi):
            tx, _, tw, _ = technique['bbox']
            crop[0] = max(0, min(crop[0], math.floor(tx) - 1))
            crop[2] = min(size[0], max(crop[2], math.ceil(tx + tw) + 1))
        for span in geometry['techniqueSpans']:
            if span['events'][0]['mi'] == mi and span['label_visible']:
                for command in span['commands']:
                    if command['type'] == 'text':
                        crop[0] = max(0, min(crop[0], math.floor(command['bbox'][0]) - 1))
        # Local annotations give bars in the same system different top edges.
        # Only a lower staff can limit this crop's bottom padding.
        following = [visible_tops[i] for i in range(len(boxes)) if centers[i] > centers[mi] + 1]
        if following:
            crop[3] = min(crop[3], math.floor(min(following)) - 1)
        crops.append(crop)
    return crops


def clipped_chord_crop(crop, geometry, mi):
    """Cut an upper grid only if every other rendered symbol stays inside.

    This is a training crop perturbation, not a production masking rule. It
    exercises the real situation where a neighbouring grid enters the crop
    without its name; visible continuation marks and notes are never erased.
    """
    chords = [p for p in geometry['painted'] if p['kind'] == 'chord' and p['mi'] == mi
              and p.get('diagram_bbox') and p.get('name_bbox')]
    if not chords:
        return crop
    protected = [p['bbox'][1] for p in geometry['painted'] if p['kind'] != 'chord'
                 and (p['bbox'][2] > 0 or p['bbox'][3] > 0)
                 and (p.get('mi') == mi or p.get('start', {}).get('mi') == mi
                      or p.get('end', {}).get('mi') == mi)]
    protected.extend(p['bbox'][1] for p in geometry.get('context', []) if p['mi'] == mi)
    protected.extend(c['bbox'][1] for span in geometry['techniqueSpans']
                     if any(e['mi'] == mi for e in span['events']) for c in span['commands'])
    highest_music = min(protected, default=crop[3])
    cutoff = max(p['name_bbox'][1] + p['name_bbox'][3] for p in chords)
    cutoff = math.ceil(cutoff) + 1
    if crop[1] < cutoff < min(highest_music - 1, crop[3]):
        return [crop[0], cutoff, crop[2], crop[3]]
    return crop


def painted_state(painted, mi, staff):
    state = {key: visible_state() for key in ('time', 'key', 'clef', 'clef_octave')}
    points = [point for p in painted if p['kind'] == 'signature' and p['mi'] == mi
              and p.get('staff_type', 'tab') == staff.get('staff_type', 'tab')
              for point in p['symbol_points']]
    rows = {}
    for point in points:
        symbol = point['symbol']
        if 0xE080 <= symbol <= 0xE089:
            rows.setdefault(round(point['y'], 3), []).append((point['x'], str(symbol - 0xE080)))
        if symbol in (0xE06D, 0xE06E):
            state['clef'] = visible_state('tab')
        if symbol in (0xE069, 0xE06A):
            state['clef'] = visible_state('percussion')
        letter = {0xE050: 'G', 0xE062: 'F', 0xE05C: 'C'}.get(symbol)
        if letter:
            step = (max(staff['lines']) - min(staff['lines'])) / 8
            line = round((max(staff['lines']) - point['y']) / (2 * step)) + 1
            state['clef'] = visible_state(f'{letter}{line}')
            state['clef_octave'] = visible_state(0)
    if len(rows) == 2:
        state['time'] = visible_state('/'.join(''.join(d for _, d in sorted(values))
                                             for _, values in sorted(rows.items())))
    accidentals = [ACCIDENTALS[f'{p["symbol"]:04X}'] for p in points
                   if f'{p["symbol"]:04X}' in ACCIDENTALS]
    if accidentals:
        state['key'] = visible_state(sum(accidentals))
    return state


def painted_accidental(head, painted):
    owner = head['mi'], head['vi'], head['ei']
    y = head['symbol_points'][0]['y']
    marks = [point for item in painted if item['kind'] == 'accidental'
        and (item['mi'], item['vi'], item['ei']) == owner
        for point in item['symbol_points'] if abs(point['y'] - y) < .01]
    # A printed accidental belongs to the written staff position. Matching
    # coincident notation heads to TAB strings is an assembly operation.
    if not marks:
        return None
    heads = {(round(point['x'], 3), round(point['y'], 3), point['symbol'])
             for item in painted if item['kind'] == 'notehead'
             and (item['mi'], item['vi'], item['ei']) == owner
             for point in item['symbol_points'] if abs(point['y'] - y) < .01}
    # Separate heads at the same height can have different alterations. A
    # shared sign does not reveal its attachment; keep that evidence unknown.
    if len(heads) > 1:
        return 'unknown'
    codes = {mark['symbol'] for mark in marks}
    return ACCIDENTALS.get(f'{codes.pop():04X}', 'unknown') if len(codes) == 1 else 'unknown'


def painted_notehead(symbol):
    shapes = {0xE0A2: 'normal', 0xE0A3: 'normal', 0xE0A4: 'normal',
              0xE0A9: 'x', 0xE0B3: 'circle-x', 0xE0F7: 'circle-slash',
              0xE0D9: 'diamond', 0xE0DB: 'diamond', 0xE0DD: 'diamond',
              0xE0BA: 'triangle-up', 0xE0BB: 'triangle-up', 0xE0BC: 'triangle-up', 0xE0BE: 'triangle-up',
              0xE0F8: 'heavy-x', 0xE0F9: 'heavy-x-hat'}
    if symbol not in shapes:
        raise ValueError(f'Unsupported drawn percussion notehead U+{symbol:04X}')
    return shapes[symbol]


def tab_observations(state, geometry, page_image, output, index):
    from PIL import ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    painted = geometry['painted']
    systems = {s['mi']: s for s in geometry['systems']}
    geometry['crop_top_margin'] = state['render_options']['crop_top_margin']
    frets = {(p['mi'], p['vi'], p['ei'], p['ni']): p for p in painted if p['kind'] == 'fret'}
    staff_type = 'tab' if state['mode'] in ('tab', 'both') else 'score'
    staffs = {p['mi']: p for p in painted if p['kind'] == 'staff'
              and p.get('staff_type', 'tab') == staff_type}
    notation_staffs = {p['mi']: p for p in painted if p['kind'] == 'staff' and p.get('staff_type') == 'score'}
    heads = {(p['mi'], p['vi'], p['ei'], p['ni']): p for p in painted if p['kind'] == 'notehead'}
    continuation_image(page_image, geometry, index, state)
    if index % 7 == 0:
        candidates = [p for p in painted if p['kind'] == 'fret' and p['texts']
            and 'tie' in state['measures'][p['mi']]['parsed']['voices'][p['vi']]['events'][p['ei']]['notes'][p['ni']]['effects']]
        if candidates:
            chosen = candidates[0]
            x, y, w, h = chosen['bbox']
            ImageDraw.Draw(page_image).rectangle((x + w * .15, y - 1, x + w * .85, y + h + 1), fill='white')
            chosen['texts'] = []
            chosen['obscured'] = True
    page_path = output / 'page.png'
    page_image.save(page_path)
    rows = []
    for mi, (source, crop) in enumerate(zip(state['measures'], tab_crops(geometry, page_image.size))):
        if state.get('chord_view') == 'clipped_diagram':
            crop = clipped_chord_crop(crop, geometry, mi)
        path = output / f'measure-{mi}.png'
        page_image.crop(crop).save(path)
        lines = staffs[mi]['lines']
        top, bottom = min(lines), max(lines)
        local_state = painted_state(painted, mi, notation_staffs.get(mi) or staffs[mi])
        tab_endpoints = {(p[end]['vi'], p[end]['ei'], p[end]['ni'])
                         for p in painted if p['kind'] in ('tie', 'technique') and p['paths']
                         and p.get('staff_type') == 'tab' and intersects(p['bbox'], crop)
                         for end in ('start', 'end') if p[end]['mi'] == mi}
        voices, note_indexes = [], {}
        for vi, voice in enumerate(source['parsed']['voices']):
            events = []
            for ei, event in enumerate(voice['events']):
                tab_notes, notation_notes = [], []
                for ni, _ in enumerate(event['notes']):
                    if state['mode'] in ('tab', 'both'):
                        glyph = frets[mi, vi, ei, ni]
                        texts = glyph['texts']
                        text = texts[-1].strip('()<>[]') if texts else ''
                        fret = int(text) if text.isdigit() else 'x' if text.lower() == 'x' else None
                        if state['mode'] != 'both' or texts or (vi, ei, ni) in tab_endpoints:
                            tab_notes.append((ni, {'string': glyph['string'], 'fret': fret}))
                    if state['mode'] in ('notation', 'both'):
                        head = heads[mi, vi, ei, ni]
                        points = head['symbol_points']
                        if not points:
                            raise ValueError('Source note has no painted notation head')
                        y = points[0]['y']
                        staff_lines = notation_staffs[mi]['lines']
                        step = (max(staff_lines) - min(staff_lines)) / 8
                        position = (max(staff_lines) - y) / step
                        if abs(position - round(position)) > .01:
                            raise ValueError('Painted notehead is not on a staff line or space')
                        note = dict(staff_position=round(position), accidental=painted_accidental(head, painted))
                        if source['instrument'] == 'drums':
                            note['notehead'] = painted_notehead(points[0]['symbol'])
                        notation_notes.append((ni, note, points[0]))
                if state['mode'] == 'both':
                    tab_notes.sort(key=lambda item: item[1]['string'])
                    notation_notes.sort(key=lambda item: (item[2]['y'], item[2]['x']))
                groups = {'tab': [], 'notation': []}
                for ni, note in tab_notes:
                    note_indexes['tab', vi, ei, ni] = len(groups['tab'])
                    groups['tab'].append(note)
                visible_heads = {}
                for ni, note, point in notation_notes:
                    shape = (round(point['x'], 3), round(point['y'], 3), point['symbol'])
                    if state['mode'] == 'both' and shape in visible_heads:
                        note_indexes['notation', vi, ei, ni] = visible_heads[shape]
                    else:
                        note_indexes['notation', vi, ei, ni] = len(groups['notation'])
                        visible_heads[shape] = len(groups['notation'])
                        groups['notation'].append(note)
                observed_event = {'start': event['start'], 'duration': _duration_token(event['duration']),
                                  'status': event['status'],
                                  'notes': groups['notation' if state['mode'] == 'notation' else 'tab']}
                if state['mode'] == 'both':
                    observed_event['notation_notes'] = groups['notation']
                events.append(observed_event)
                if event['status'] == 'rest' and not any(p['kind'] == 'rest' and p['mi'] == mi
                    and p['vi'] == vi and p['ei'] == ei and p['symbols'] for p in painted):
                    raise ValueError(f'Source rest has no painted rest glyph: {source["id"]}/{vi}/{ei}')
            voices.append({'voice': voice['voice'], 'events': events})
        visible_chords = []
        for chord in (p for p in painted if p['kind'] == 'chord' and p['mi'] == mi):
            box = chord.get('name_bbox')
            if not chord.get('name') or not box:
                continue
            x, y, w, h = box
            if not (crop[0] <= x and x + w <= crop[2] and crop[1] <= y and y + h <= crop[3]):
                continue
            from urllib.parse import quote
            voices[chord['vi']]['events'][chord['ei']].setdefault('effects', []).append(
                'chord:' + quote(chord['name'], safe=''))
            visible_chords.append({'voice': chord['vi'], 'event': chord['ei'],
                                   'name': chord['name'], 'bbox': box})
        for dynamic in (p for p in painted if p['kind'] == 'dynamic' and p['mi'] == mi):
            values = {DYNAMIC_SYMBOLS[symbol] for symbol in dynamic['symbols'] if symbol in DYNAMIC_SYMBOLS}
            if len(values) != 1 or not intersects(dynamic['bbox'], crop):
                raise ValueError('Printed dynamic has no unique visible symbol in its measure crop')
            event = voices[dynamic['vi']]['events'][dynamic['ei']]
            event.setdefault('effects', []).append(f'dyn:{values.pop()}')
        for pick in (p for p in painted if p['kind'] == 'pick' and p['mi'] == mi):
            values = {PICK_SYMBOLS[symbol] for symbol in pick['symbols'] if symbol in PICK_SYMBOLS}
            if len(values) != 1 or not intersects(pick['bbox'], crop):
                raise ValueError('Printed picking sign has no unique visible symbol in its measure crop')
            event = voices[pick['vi']]['events'][pick['ei']]
            effect = values.pop()
            if effect not in event.setdefault('effects', []):
                event['effects'].append(effect)
        for technique in (p for p in painted if p['kind'] == 'technique' and p['paths']
                          and p['start']['mi'] == mi):
            a = technique['start']
            if not intersects(technique['bbox'], crop):
                continue
            effect = technique['effect']
            if effect not in TAB_TECHNIQUES:
                raise ValueError(f'Unsupported painted TAB technique: {effect}')
            representation = 'tab' if technique['staff_type'] == 'tab' else 'notation'
            field = 'notation_notes' if state['mode'] == 'both' and representation == 'notation' else 'notes'
            note = voices[a['vi']]['events'][a['ei']][field][note_indexes[representation, a['vi'], a['ei'], a['ni']]]
            if effect not in note.setdefault('effects', []):
                note['effects'].append(effect)
        for vi, voice in enumerate(source['parsed']['voices']):
            for ei, event in enumerate(voice['events']):
                for ni, note in enumerate(event['notes']):
                    expected = set(note['effects']) & TAB_TECHNIQUES
                    representation = 'notation' if state['mode'] == 'notation' else 'tab'
                    mapped = note_indexes.get((representation, vi, ei, ni))
                    observed = (set(voices[vi]['events'][ei]['notes'][mapped].get('effects', []))
                                if mapped is not None else set())
                    if expected != observed:
                        raise ValueError(f'Unpainted or wrongly attributed technique: {source["id"]}/{ei}/{ni}')
        relations = []
        for tie in (p for p in painted if p['kind'] == 'tie' and p['paths'] and intersects(p['bbox'], crop)):
            a, b = tie['start'], tie['end']
            if not a['mi'] <= mi <= b['mi']:
                continue
            representation = 'tab' if tie['staff_type'] == 'tab' else 'notation'
            relation = {'kind': 'tie',
                        'start': [a['vi'], a['ei'], note_indexes[representation, a['vi'], a['ei'], a['ni']]] if a['mi'] == mi else 'left',
                        'end': [b['vi'], b['ei'], note_indexes[representation, b['vi'], b['ei'], b['ni']]] if b['mi'] == mi else 'right'}
            if state['mode'] == 'both':
                relation['representation'] = representation
            if a['mi'] != mi or b['mi'] != mi:
                relation_lines = (notation_staffs if representation == 'notation' else staffs)[mi]['lines']
                relation_top, relation_bottom = min(relation_lines), max(relation_lines)
                relation.update(voice=a['vi'], lane=round((tie['paths'][0][1] - relation_top) / (relation_bottom - relation_top), 3))
            if relation not in relations:
                relations.append(relation)
        for span in geometry['techniqueSpans']:
            local = [event for event in span['events'] if event['mi'] == mi]
            commands = [c for c in span['commands'] if intersects(c['bbox'], crop)]
            horizontal = [c['bbox'] for c in commands if c['type'] == 'rect' and c['bbox'][2] > c['bbox'][3]]
            labelled = span['label_visible'] and any(c['type'] == 'text' for c in commands)
            if not local:
                continue
            if not horizontal:
                if labelled:
                    event = span['events'][0]
                    relations.append({'kind': span['kind'], 'start': [event['vi'], event['ei']],
                                      'end': None})
                continue
            a, b = span['events'][0], span['events'][-1]
            relation = {'kind': span['kind'] if labelled else None,
                'start': [a['vi'], a['ei']] if a['mi'] == mi and not span.get('open_left') else 'left',
                'end': [b['vi'], b['ei']] if b['mi'] == mi and not span.get('open_right') else 'right'}
            if relation['start'] == 'left' or relation['end'] == 'right':
                relation.update(voice=a['vi'], lane=round((horizontal[0][1] - top) / (bottom - top), 3))
            relations.append(relation)
        row = {key: source[key] for key in ('id', 'family', 'source_id', 'split', 'part_id', 'staff_id', 'bar_index', 'mode')}
        row.update(image=str(path.resolve()), source_page=str(page_path.resolve()), bbox=crop,
                   instrument=source['instrument'],
                   renderer='alphatab', render_variant=state['render_variant'],
                   page=systems[mi]['page_index'] + 1, system_index=systems[mi]['system_index'],
                   observation={'state': local_state, 'voices': voices, 'relations': relations},
                   reference_target=source['target'],
                   oracle_auxiliary={'staff_start': source['bar_index'] == 0, 'state': {k: visible_state(v) for k, v in source['score_state'].items()},
                                     'voice_map': {str(v['voice']): v['voice'] for v in voices}})
        if 'chord_view' in state:
            row.update(chord_view=state['chord_view'], visible_chords=visible_chords,
                       visible_diagrams=sum(p['kind'] == 'chord' and p['mi'] == mi
                           and bool(p.get('diagram_bbox')) and intersects(p['diagram_bbox'], crop)
                           for p in painted))
        rows.append(row)
    (output / 'geometry.json').write_text(json.dumps(geometry))
    return rows


def standard_score(index, split):
    from research.data.engraved_scores import generate, key_fifths
    from scorelib.pitch_context import transpose_key
    from scorelib.score import format_measure_target, parse_measure_target

    score = generate(index, split, bars=8)
    state = tab_score(index, split)
    state.update(id=f'standard-{split}-{index:05d}', mode='notation', measures=[])
    for pi, part in enumerate(score['parts']):
        for si, staff in enumerate(part['staves']):
            context = dict(staff['measures'][0]['pitch_context'])
            if index % 5 == 4 and context['clef'] == 'C3':
                context['clef'] = 'C4'
            first_key = next(m['key_signature'] for m in staff['measures'] if m.get('key_signature'))
            written_key = key_fifths(transpose_key(first_key, -context['instrument_transpose']))
            for mi, measure in enumerate(staff['measures']):
                parsed = parse_measure_target(measure['written_target'])
                parsed['key_signature'] = measure['key_signature']
                for voice in parsed['voices']:
                    for event in voice['events']:
                        if voice['voice'] > 0 and event['status'] == 'rest':
                            event.update(status='normal', notes=[{'pitch': next(n['pitch'] for e in voice['events']
                                for n in e['notes']), 'effects': []}])
                        for note in event['notes']:
                            note['effects'] = [e for e in note['effects'] if e == 'tie']
                row = dict(id=f'{state["id"]}-p{pi}-s{si}-{mi}', source_id=state['id'],
                    family=state['id'], split=split, mode='notation', instrument=part['instrument'],
                    part_id=part['id'], staff_id=staff['id'], bar_index=mi, measure_index=mi,
                    part_name=part['name'], midi_program=part['midi_program'],
                    tuning=part['tuning'], capo=0, pitch_context=context, parsed=parsed,
                    target=format_measure_target(parsed, 'notation'),
                    score_state=dict(time=score['timeline'][mi]['time_signature'], key=written_key,
                        tuning=part['tuning'], clef=context['clef'], clef_octave=0,
                        instrument_transpose=context['instrument_transpose'],
                        instrument_transpose_explicit=True, capo=0))
                state['measures'].append(row)
    state['render_options']['width'] = max(state['render_options']['width'], 2200)
    state['render_options']['bars_per_row'] = 2
    state['render_options']['staff_padding'] = 35
    return state


def score_mode(index, split, mode):
    """Independent dual/standard/drum families with one visible measure target."""
    from scorelib.score import format_measure_target
    from guitarpro.models import KeySignature

    if mode == 'standard':
        return standard_score(index, split)
    state = tab_score(index, split)
    state['id'] = f'{mode}-{split}-{index:05d}'
    state['mode'] = 'both' if mode == 'both' else 'notation'
    family = f'{mode}-{split}-{index - index % 8:05d}'
    rng = random.Random(f'{mode}:{split}:{index - index % 8}')
    drum_keys = [35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
    for mi, row in enumerate(state['measures']):
        row.update(id=f'{state["id"]}-{mi}', source_id=state['id'], family=family, mode=state['mode'])
        row['score_state'].update(clef='G2', clef_octave=0,
            instrument_transpose=-12, instrument_transpose_explicit=False, capo=0)
        if mode == 'both':
            initial_key = index // 8 % 15 - 7
            key = initial_key if mi < 4 else (0 if initial_key else 3)
            row['score_state']['key'] = key
            row['parsed'].update(key_signature=KeySignature((key, 0)).name,
                                 print_key_signature=mi in (0, 4))
        # Local techniques also occur on dual staves. Their labels come from
        # each staff's actual paint, just as the two note groups do.
        for voice in row['parsed']['voices']:
            for event in voice['events']:
                for note in event['notes']:
                    note['effects'] = [effect for effect in note['effects']
                                       if effect == 'tie' or mode == 'both' and effect in TAB_TECHNIQUES]
        if mode == 'drums':
            row.update(instrument='drums', part_name='Drums', midi_program=0, tuning=[])
            row['pitch_context'].update(clef='percussion', instrument_transpose=0)
            row['score_state'].update(clef='percussion', instrument_transpose=0, tuning=[])
            # One merged rhythm or two stem directions, with common kit
            # combinations and isolated hits to cover every visible symbol.
            count = 1 if index % 3 == 0 else 2
            length = 2880 if row['score_state']['time'] in ('3/4', '6/8') else 3840
            voices = []
            for vi in range(count):
                events = []
                step = rng.choice([480, 960])
                for onset in range(0, length, step):
                    pool = drum_keys if count == 1 else ([42, 46, 49, 51, 52, 53, 55, 56, 57] if vi == 0
                                                        else [35, 36, 37, 38, 43, 44, 45, 47, 48, 50, 54])
                    pitches = rng.sample(pool, rng.choice([1, 1, 1, 2, 3]))
                    rest = vi == 0 and rng.random() < .12
                    events.append(dict(start=onset, duration=dict(value=8 if step == 480 else 4,
                        dotted=False, double_dotted=False, tuplet_enters=1, tuplet_times=1),
                        status='rest' if rest else 'normal', effects=[],
                        notes=[] if rest else [dict(pitch=pitch, effects=[]) for pitch in pitches]))
                voices.append(dict(voice=vi, events=events))
            row['parsed']['voices'] = voices
        row['target'] = format_measure_target(row['parsed'], state['mode'])
    return state


def tab_batch(job):
    from PIL import Image
    from research.data.score_rendering import browser_page

    output, jobs, variant = job
    rows = []
    with browser_page() as (page, origin):
        for split, index in jobs:
            state = (tab_chord_score(index, split) if variant == 'tab-chords' else
                     tab_score(index, split) if variant == 'tab' else score_mode(index, split, variant))
            directory = Path(output) / 'images' / state['id']
            directory.mkdir(parents=True, exist_ok=True)
            page.goto(origin + '/research/data/score_renderer.js')
            geometry = page.evaluate('async s => (await import("/research/data/score_renderer.js")).render(s)', state)
            (directory / 'geometry.json').write_text(json.dumps(geometry))
            image = directory / 'page.png'
            page.locator('#score').screenshot(path=str(image))
            with Image.open(image) as loaded:
                rows.extend(tab_observations(state, geometry, loaded.convert('RGB'), directory, index))
    return rows


def build_tab(output, train, validation, test, workers=4, index_offset=0, *, with_chords=False, mode='tab'):
    jobs = [(split, index) for split, count in [('train', train), ('validation', validation), ('test', test)]
            for index in range(index_offset, index_offset + count)]
    batches = [(str(output), jobs[i:i + 20], 'tab-chords' if with_chords else mode) for i in range(0, len(jobs), 20)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for index, values in enumerate(pool.map(tab_batch, batches), 1):
            rows.extend(values)
            print('tab', index, len(batches), len(rows), flush=True)
    return rows


def build_tab_chords(output, train, validation, test, workers=4, index_offset=0):
    return build_tab(output, train, validation, test, workers, index_offset, with_chords=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/observations'))
    parser.add_argument('--train', type=int, default=240)
    parser.add_argument('--validation', type=int, default=24)
    parser.add_argument('--test', type=int, default=24)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--index-offset', type=int, default=0,
                        help='Start score IDs and random seeds here when adding new families')
    parser.add_argument('--mode', choices=['notation', 'tab', 'tab-chords', 'both', 'standard', 'drums', 'octave'], default='notation')
    parser.add_argument('--merge', nargs='+', type=Path, help='Merge existing observation manifests and refresh their prompts')
    args = parser.parse_args()
    if args.merge:
        rows = [json.loads(line) for directory in args.merge for split in ('train', 'validation', 'test')
                for line in (directory / f'manifest_{split}.jsonl').read_text().splitlines() if line.strip()]
    else:
        builder = {'notation': build_notation, 'tab': build_tab, 'tab-chords': build_tab_chords,
                   'octave': build_octave,
                   **{mode: (lambda *values, mode=mode: build_tab(*values, mode=mode))
                      for mode in ('both', 'standard', 'drums')}}[args.mode]
        rows = builder(args.output, args.train, args.validation, args.test, args.workers, args.index_offset)
    print(write_dataset(args.output, rows), flush=True)


if __name__ == '__main__':
    main()
