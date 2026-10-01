"""Prepare tuning-independent visual pitches and balanced full-source supervision."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from research.data.training_samples import dataset_entry
from research.data.training_samples import visual_measure_sample
from research.data.source_profile import apply_source_profile
from scorelib.pitch_context import convert_pitch_target
from scorelib.m2 import format_measure_target
from scorelib.m2 import parse_measure_target
from scorelib.techniques import canonical_chord_marks


def prepare_source(job):
    rows, corpus, native_focus, rehearsal_focus = job
    if corpus == 'original':
        rows = [apply_source_profile(row) for row in rows]
    contexts = {r['measure_index']: r.get('pitch_context') for r in rows if r['mode'] == 'notation'}
    chats, result, counts = [], [], Counter()
    for row in rows:
        row = dict(row)
        row['corpus'] = corpus
        row['visual_pitch'] = True
        if corpus == 'engraved':
            for key in ('target', 'sounding_target'):
                if row.get(key):
                    row[key] = format_measure_target(canonical_chord_marks(parse_measure_target(row[key])), 'notation')
        instrument = row.get('instrument', 'guitar')
        mode = row['mode']
        if mode == 'both' and instrument != 'drums':
            context = row.get('pitch_context') or contexts.get(row['measure_index']) or {
                'clef': 'F4' if instrument == 'bass' else 'G2',
                'clef_octave': 0, 'instrument_transpose': -12, 'octave_spans': [],
            }
            row['pitch_context'] = context
            row['target'] = convert_pitch_target(row.get('sounding_target', row['target']),
                                                 context, to_written=True, mode='both')
        sample = visual_measure_sample(row, first=row['measure_index'] == 0)
        repeat = 1
        if row['split'] == 'train':
            repeat = 3 if corpus != 'original' else 2 if mode == 'both' else 1
            if mode == 'notation' and (instrument not in {'guitar', 'bass'} or
                                      any(t in row['target'] for t in ('bend:', 'grace:', 'trill', 'V1{', 'ottava:'))):
                repeat += 2
            if corpus == 'engraved':
                repeat = (15 if ' || ' in row['target'] else 3) if rehearsal_focus else 30 if native_focus and ' || ' in row['target'] else 6
        chats.extend([sample] * repeat)
        result.append(row)
        counts[f'{corpus}/{instrument}/{mode}'] += 1
    return chats, result, dict(counts), corpus


def build(output, workers=16, engraved=None, native_focus=False, rehearsal_focus=False):
    output.mkdir(parents=True, exist_ok=True)
    roots = [('original', Path('database/parallel_score_corrected')),
             ('synthetic', Path('database/parallel_synthetic/datasets/independent')),
             ('techniques', Path('database/parallel_techniques/datasets/independent'))]
    if engraved:
        roots.append(('engraved', engraved))
    families, summary, info = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        groups = defaultdict(list)
        for corpus, root in roots:
            with (root / f'manifest_{split}.jsonl').open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if families.setdefault(row['family'], split) != split:
                        raise ValueError('Source family crosses splits')
                    groups[(corpus, row['source_id'])].append(row)
        counts, examples, original_examples = Counter(), 0, 0
        with (output / f'measure_{split}.jsonl').open('w') as chats, \
                (output / f'manifest_{split}.jsonl').open('w') as manifest, \
                ProcessPoolExecutor(workers) as pool:
            jobs = ((rows, corpus, native_focus, rehearsal_focus) for (corpus, _source), rows in groups.items())
            for index, (samples, rows, count, corpus) in enumerate(pool.map(prepare_source, jobs, chunksize=1), 1):
                if (native_focus or rehearsal_focus) and split == 'train' and corpus != 'engraved':
                    available = len(samples)
                    samples = samples[(-original_examples) % 3::3]
                    original_examples += available
                    retained = samples
                    samples = retained + [s for s in retained if ' || ' in s['messages'][-1]['content']] * 3
                    if rehearsal_focus:
                        if corpus == 'synthetic' and rows[0]['instrument'] == 'drums':
                            samples += retained * 10
                        elif corpus == 'techniques':
                            notation_images = {r['image'] for r in rows if r['mode'] != 'tab'}
                            samples += [s for s in retained if s['images'][0] in notation_images] * 2
                chats.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in samples)
                manifest.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
                counts.update(count)
                examples += len(samples)
                if index % 200 == 0:
                    print(split, index, '/', len(groups), examples, flush=True)
        info[f'measure_{split}'] = dataset_entry(f'measure_{split}.jsonl')
        summary[split] = {'examples': examples, 'groups': dict(counts), 'sources': len(groups)}
        (output / 'dataset_info.json').write_text(json.dumps(info, indent=2))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(split, summary[split], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/score_support'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--engraved', type=Path, help='Add rendered notation and ensemble supervision')
    parser.add_argument('--native-focus', action='store_true', help='Emphasize original and native polyphony while retaining notation and TAB rehearsal')
    parser.add_argument('--rehearsal-focus', action='store_true', help='Balance native polyphony with percussion and fretted techniques')
    build(**vars(parser.parse_args()))
