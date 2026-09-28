"""Build full-source visual-context OCR and printed-state supervision."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import random

from PIL import Image

from datagen.scan_augment import degrade
from datagen.training_samples import dataset_entry
from datagen.written_pitch_data import prepare
from shared.pitch_context import transpose_key
from shared.score_state import SIGNATURE_PROMPT, attach_neighbours, key_fifths, signature_target, state_prompt


def native_key_target(target):
    for source, visible in {'FMajorFlat': 'EMajor', 'GMajorSharp': 'AMajorFlat',
                            'DMinorFlat': 'CMinorSharp', 'EMinorSharp': 'FMinor'}.items():
        target = target.replace(' key=' + source, ' key=' + visible)
    return target


def native_layout(row):
    root = Path(row['label_json']).parent.parent
    path = root / 'layout' / row['mode'] / (row['source_id'] + '.layout.json')
    if path.exists():
        return json.loads(path.read_text())
    matches = list((root / 'native-export/documents' / f"{row['mode']}-{row['source_id']}" / 'tracks').glob('*/layout.json'))
    return json.loads(matches[0].read_text()) if matches else {}


def native_pitch_context(rows):
    row = rows[0]
    if row.get('instrument') != 'pitched' or row['mode'] != 'notation' or row.get('pitch_context'):
        return None
    root = Path(row['label_json']).parent.parent
    paths = list((root / 'native-export/documents' / f"notation-{row['source_id']}" / 'tracks').glob('*/official-score.json'))
    if len(paths) != 1:
        raise ValueError(f"Missing native pitch reference: {row['source_id']}")
    track = json.loads(paths[0].read_text())['tracks'][0]
    if 'view_transposition_offset' not in track:
        raise ValueError(f"Re-export legacy native pitch metadata: {row['source_id']}")
    shifts = {None: 0, '8va': 12, '8vb': -12, '15ma': 24, '15mb': -24}
    return {bar['measure_index']: {
        'instrument_transpose': track['view_transposition_offset'],
        'clef': bar['clef'], 'clef_octave': shifts[bar.get('ottavia_name')], 'octave_spans': [],
    } for bar in track['staves'][0]['measures']}


def process_source(job):
    rows, output, augment = job
    rows.sort(key=lambda r: r['measure_index'])
    label = json.loads(Path(rows[0]['label_json']).read_text())
    layout = native_layout(rows[0])
    native_pitch = native_pitch_context(rows)
    printed, systems = defaultdict(dict), {}
    for system in layout.get('systems', []):
        for box in system.get('measure_boxes', []):
            systems[box['measure_index']] = system['system_index']
        for s in system.get('time_signatures', []):
            printed[s['measure_index']]['time'] = f"{s['numerator']}/{s['denominator']}"
        for s in system.get('key_signatures', []):
            printed[s['measure_index']]['key'] = s['accidental_count']
    attach_neighbours(rows)
    chats, states, manifests, counts = [], [], [], Counter()
    source_number = int(rows[0]['source_id'][:8], 16)
    rng = random.Random(source_number)
    for row in rows:
        index = row['measure_index']
        measure = label['measures'][index]
        row['instrument'] = row.get('instrument', label['track'].get('instrument', 'guitar'))
        row['tuning'] = label['track'].get('tuning_midi_high_to_low', []) if row['instrument'] in {'guitar', 'bass'} else []
        sounding_target = row['target'] = native_key_target(row['target'])
        if native_pitch is not None:
            row['pitch_context'] = native_pitch[index]
        row = prepare(row)
        key = measure.get('key_signature') or 'CMajor'
        if row.get('written_pitch'):
            key = transpose_key(key, -(row.get('pitch_context') or {}).get('instrument_transpose', 0))
        row['score_state'] = {'time': measure.get('time_signature') or '4/4', 'key': key_fifths(key)}
        if row['mode'] == 'tab':
            row['score_state']['key'] = 0
        if row['tuning']:
            row['score_state']['tuning'] = row['tuning']
        row['system_index'] = systems.get(index, row.get('system_index', 0))
        row['source_measures'] = len(label['measures'])
        row['sounding_target'] = sounding_target
        signatures = printed.get(index, {})
        if not layout.get('systems'):
            signatures = {
                'time': measure.get('time_signature') if measure.get('print_time_signature') else None,
                'key': row['score_state']['key'] if measure.get('print_key_signature') and row['mode'] != 'tab' else None,
            }
        row['signature_target'] = signature_target(signatures.get('time'), signatures.get('key'))
        prompt = state_prompt(row['mode'], row['instrument'], row['score_state'], row.get('pitch_context'), first=index == 0)
        images = [row['image'], row['previous_image'], row['next_image']]
        chat = {'messages': [{'role': 'user', 'content': '<image><image><image>' + prompt},
                             {'role': 'assistant', 'content': row['target']}], 'images': images}
        chats.append(chat)
        # Keep every positive state example, plus a source-diverse set of negatives.
        if signatures or index == 0 or rng.random() < .12 or row['split'] != 'train':
            states.append({'messages': [{'role': 'user', 'content': '<image>' + SIGNATURE_PROMPT},
                                        {'role': 'assistant', 'content': row['signature_target']}],
                           'images': [row['image']]})
        counts[f"{row['instrument']}/{row['mode']}"] += 1
        if row['split'] == 'train':
            target = row['target']
            hard = any(t in target for t in ('trill', 'grace', 'bend:', 'tie', 'V1{', 'ottava:'))
            if hard:
                chats.append(chat)
            if augment and (hard or rng.random() < .15):
                path = Path(output) / 'scan' / row['mode'] / row['source_id'] / f'{index}.png'
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(row['image']) as im:
                        degrade(im, row['id'], crop=True).save(path, compress_level=1)
                changed = deepcopy(chat)
                changed['images'][0] = str(path.resolve())
                chats.append(changed)
        manifests.append(row)
    return chats, states, manifests, dict(counts)


def build(output, workers=24, augment=True, sources=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = sources or [
        Path('database/gp8_joint_v3/datasets/measure_ocr'),
        Path('database/instrument_training/datasets/measure_verified'),
        Path('database/pitch_training/datasets/native_pitch/measure'),
        Path('database/pitch_named/datasets/native_pitch/measure'),
    ]
    excluded = set()
    for name in ('instrument_training-alignment.json', 'gp8_joint_v3-alignment.json'):
        excluded.update(r['source_id'] for r in json.loads((Path('output/instrument-training') / name).read_text()) if r['different_events'])
    families, info, totals = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        merged = {}
        for root in roots:
            path = root / 'manifests' / f'{split}.jsonl'
            for line in path.open():
                row = json.loads(line)
                if row['source_id'] in excluded:
                    continue
                if families.setdefault(row['family'], split) != split:
                    raise ValueError(f"Family occurs in different splits: {row['family']}")
                row['split'] = split
                row.setdefault('label_json', row.get('label'))
                merged[(row['source_id'], row['mode'], row['measure_index'])] = row
        groups = defaultdict(list)
        for row in merged.values():
            groups[(row['source_id'], row['mode'])].append(row)
        del merged
        paths = {kind: output / f'{kind}_{split}.jsonl' for kind in ('measure', 'state', 'manifest')}
        counts, total, state_total = Counter(), 0, 0
        handles = {k: p.open('w') for k, p in paths.items()}
        try:
            with ProcessPoolExecutor(workers) as pool:
                jobs = ((rows, str(output), augment) for _, rows in sorted(groups.items()))
                for i, (chats, states, manifests, c) in enumerate(pool.map(process_source, jobs, chunksize=1), 1):
                    for kind, values in (('measure', chats), ('state', states), ('manifest', manifests)):
                        handles[kind].writelines(json.dumps(v, ensure_ascii=False) + '\n' for v in values)
                    counts.update(c)
                    total += len(chats)
                    state_total += len(states)
                    if i % 100 == 0:
                        print(split, i, '/', len(groups), 'measure', total, 'state', state_total, flush=True)
        finally:
            for h in handles.values():
                h.close()
        for kind in ('measure', 'state'):
            info[f'{kind}_{split}'] = dataset_entry(paths[kind].name)
        totals[split] = {'sequences': len(groups), 'measure_samples': total, 'state_samples': state_total, 'groups': dict(counts)}
        (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))
        (output / 'summary.json').write_text(json.dumps(totals, indent=2))
        print(split, totals[split], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/parallel_score'))
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--source', type=Path, action='append', help='Measure dataset root containing manifests/')
    args = parser.parse_args()
    build(args.output, args.workers, not args.no_augment, args.source)
