"""Rehearsal, real page crops and native note-to-technique grounding supervision."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw

from datagen.training_samples import dataset_entry, visual_measure_sample
from datagen.source_profile import apply_composed_profile
from shared.m2 import parse_measure_target, format_measure_target
from shared.score_state import attach_neighbours


def ensemble(path):
    data = json.loads(path.read_text())
    rows = [apply_composed_profile(row) for row in data['records']]
    attach_neighbours(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['part_id'], row['staff_id']].append(row)
    for (part, staff), bars in groups.items():
        for row in bars:
            row.update(source_id=f"{data['id']}-{part}-{staff}", source_measures=len(bars),
                       corpus=path.parents[2].name, measure_index=row['bar_index'], visual_pitch=True)
    return rows


def note_glyphs(layout, official):
    """Join drawn frets to musical onsets, excluding separate grace beats."""
    events = {}
    for track in official['tracks']:
        if track['track_index'] != official['source_track_index']:
            continue
        for staff in track['staves']:
            for measure in staff['measures']:
                for voice in measure['voices']:
                    for event in voice['events']:
                        if not event['grace'] and not event['placeholder']:
                            key = (staff['staff_index'], measure['measure_index'],
                                   voice['voice_index'], event['event_index'])
                            events[key] = round(960 * Fraction(*event['offset']))
    glyphs = defaultdict(list)
    for glyph in layout['note_geometry']['glyphs']:
        key = tuple(glyph[name] for name in ('staff_index', 'measure_index', 'voice_index', 'event_index'))
        if glyph['kind'] == 'fret' and key in events:
            glyphs[glyph['measure_index']].append({**glyph, 'onset': events[key]})
    return glyphs


def glyph_matches_note(glyph, note):
    # Trills can draw an extra parenthesized fret using the same owning Note.
    # The anchor must enclose the main note's fret, including ghost/harmonic
    # decorations, rather than that second trill glyph.
    text = glyph['text'].strip().strip('()<>[]').lower().replace('×', 'x')
    fret = 'x' if 'dead' in note.get('effects', []) else str(note.get('fret')).lower()
    return text == fret


def grounding(job):
    source, rows, output = job
    row = rows[0]
    paths = list((Path(row['label_json']).parent.parent / 'native-export/documents' /
                  f"{row['mode']}-{source}").glob('tracks/*/layout.json'))
    if not paths:
        return []
    layout = json.loads(paths[0].read_text())
    geometry = layout.get('note_geometry') or {}
    if geometry.get('status') != 'complete':
        return []
    official = json.loads(paths[0].with_name('official-score.json').read_text())
    glyphs = note_glyphs(layout, official)
    result = []
    for row in rows:
        if not row.get('bbox'):
            continue
        target = parse_measure_target(row['target'])
        events = {(v['voice'], e['start']): e for v in target['voices'] for e in v['events']}
        candidates = []
        for g in glyphs[row['measure_index']]:
            e = events.get((g['voice_index'], g['onset']))
            if not e:
                continue
            string = len(row['tuning']) - g['native_string_index']
            n = next((n for n in e['notes'] if n.get('string') == string), None)
            if not n or not glyph_matches_note(g, n) or not (n.get('effects') or e.get('effects')):
                continue
            candidates.append((g, e, n))
        if not candidates:
            continue
        # Use all measures containing anchored techniques, without repeating
        # numerous equivalent notes from the same measure.
        g, event, note = candidates[row['measure_index'] % len(candidates)]
        with Image.open(row['image']) as source_image:
            image = source_image.convert('RGB')
        x, y, w, h = g['bbox_mm']
        left = max(0, round(row['bbox'][0] - 1.5 * 180 / 25.4))
        top = max(0, round(row['bbox'][1] - 4 * 180 / 25.4))
        box = [x * 180 / 25.4 - left, y * 180 / 25.4 - top,
               (x + w) * 180 / 25.4 - left, (y + h) * 180 / 25.4 - top]
        if min(box[:2]) < 0 or box[2] > image.width or box[3] > image.height:
            continue
        ImageDraw.Draw(image).rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], outline=(30, 90, 220), width=1)
        path = Path(output) / 'anchors' / row['mode'] / source / f"{row['measure_index']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, compress_level=1)
        answer = deepcopy(target)
        answer['voices'] = [{'voice': g['voice_index'], 'events': [{**event, 'notes': [note]}]}]
        for key in ('time_signature', 'key_signature', 'tempo_quarter', 'section'):
            answer[key] = None
        answer['print_time_signature'] = answer['print_key_signature'] = False
        result.append({'messages': [
            {'role': 'user', 'content': '<image>Read ONLY the note outlined in blue. Return an M2 fragment with '
             'its voice, original onset, duration, string, fret, written pitch if present, and all techniques '
             'that apply to this note. Include technique parameters and span effects; do not copy nearby notes.'},
            {'role': 'assistant', 'content': format_measure_target(answer, row['mode'])}], 'images': [str(path.resolve())]})
    return result


def build_warmup(output, seed=20261005):
    """Teach the expanded vocabulary serialization before image finetuning."""
    rng, seen, count = random.Random(seed), set(), 0

    def write(target, destination):
        nonlocal count
        score = parse_measure_target(target)
        value = {'messages': [
            {'role': 'user', 'content': 'Serialize these musical events into one complete M2 fragment. '
             'Preserve every note, onset, duration and technique. ' + json.dumps(score, separators=(',', ':'))},
            {'role': 'assistant', 'content': target}], 'images': []}
        destination.write(json.dumps(value) + '\n')
        count += 1

    with (output / 'vocabulary_warmup.jsonl').open('w') as destination:
        for index, line in enumerate((output / 'measure_train.jsonl').open()):
            if index % 8:
                continue
            row = json.loads(line)
            # Single-note anchor fragments deliberately omit bar metadata.
            if any('/anchors/' in path for path in row.get('images', [])):
                continue
            target = row['messages'][-1]['content']
            if target not in seen:
                seen.add(target)
                write(target, destination)
        for index in range(18000):
            mode = ['tab', 'both', 'notation'][index % 3]
            events = []
            for start in sorted(rng.sample(range(0, 15360, 40), rng.randint(2, 8))):
                notes = []
                for string in rng.sample(range(1, 13), rng.randint(1, 4)):
                    note = {'string': string, 'fret': rng.randint(0, 36), 'pitch': rng.randint(0, 127), 'effects': []}
                    if rng.random() < .18:
                        note['effects'] = [rng.choice(['tie', 'hammer', 'pm', 'let', 'stacc', 'vib', 'dead'])]
                    notes.append(note)
                events.append({'start': start, 'duration': {'value': 128, 'dotted': False, 'double_dotted': False,
                               'tuplet_enters': 1, 'tuplet_times': 1}, 'status': 'normal', 'effects': [], 'notes': notes})
            score = parse_measure_target('M2 time=16/4 | V0{@0:w:r}')
            score['voices'] = [{'voice': 0, 'events': events}]
            write(format_measure_target(score, mode), destination)
    return count


def build(output, workers=24, anchor_repeats=4):
    output.mkdir(parents=True, exist_ok=True)
    info, report = {}, {}
    replay = Path('database/score_support_rehearsal')
    for split in ('train', 'validation', 'test'):
        counts = Counter()
        with (output / f'measure_{split}.jsonl').open('w') as dest:
            for line in (replay / f'measure_{split}.jsonl').open():
                dest.write(line)
                counts['rehearsal'] += 1
            with (output / f'ensemble_manifest_{split}.jsonl').open('w') as manifest:
                for corpus in ('ensemble_pages', 'ensemble_quality'):
                    for path in sorted((Path('database') / corpus / split).glob('*/score.json')):
                        rows = ensemble(path)
                        for row in rows:
                            row['id'] = corpus + '-' + row['id']
                            row['source_id'] = corpus + '-' + row['source_id']
                            dest.write(json.dumps(visual_measure_sample(row), ensure_ascii=False) + '\n')
                            manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                            counts[corpus] += 1
            if split == 'train':
                groups = defaultdict(list)
                for line in (replay / 'manifest_train.jsonl').open():
                    row = json.loads(line)
                    if row['mode'] in {'tab', 'both'} and 'parallel_techniques/' in row['image']:
                        groups[row['source_id'], row['mode']].append(row)
                jobs = [(source, rows, str(output)) for (source, mode), rows in groups.items()]
                with ProcessPoolExecutor(workers) as pool, (output / 'anchors_train.jsonl').open('w') as anchors:
                    for samples in pool.map(grounding, jobs):
                        for row in samples:
                            line = json.dumps(row, ensure_ascii=False) + '\n'
                            anchors.write(line)
                            dest.write(line * anchor_repeats)
                            counts['technique_anchors'] += anchor_repeats
        report[split] = dict(counts)
        info[f'measure_{split}'] = dataset_entry(f'measure_{split}.jsonl')
        print(split, counts, flush=True)
    report['vocabulary_warmup'] = build_warmup(output)
    info.update(anchors_train=dataset_entry('anchors_train.jsonl'),
                vocabulary_warmup=dataset_entry('vocabulary_warmup.jsonl'))
    (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))
    (output / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/score_quality'))
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--anchor-repeats', type=int, default=4)
    build(**vars(parser.parse_args()))
