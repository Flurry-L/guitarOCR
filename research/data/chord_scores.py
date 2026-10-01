"""Generate complete Guitar Pro scores with beat-aligned chord symbols."""

import argparse
import json
from pathlib import Path
import random
from urllib.parse import quote

import guitarpro

from research.data.chord_annotations import sample_diagram
from research.data.gp_sources import analyze_source
from scorelib.gp5.writer import targets_to_song
from scorelib.chords import diagram_effect
from scorelib.m2 import format_measure_target
from scorelib.m2 import parse_measure_target


def build(output, sources=600, bars=24):
    output = output.resolve()
    (output / 'sources').mkdir(parents=True, exist_ok=True)
    (output / 'labels').mkdir(exist_ok=True)
    catalog, manifest = [], []
    tuning = [64, 59, 55, 50, 45, 40]
    for index in range(sources):
        rng = random.Random(430927 + index)
        targets = []
        for bar in range(bars):
            measure = parse_measure_target('M2 time=4/4 key=CMajor | V0{@0:w:r}')
            events = []
            name, diagram = sample_diagram(rng)
            name = name.replace('♯', '#').replace('♭', 'b')
            for beat in range(8):
                changed = beat == 0 or (beat in {2, 4, 6} and rng.random() < .5)
                if changed and beat:
                    name, diagram = sample_diagram(rng)
                    name = name.replace('♯', '#').replace('♭', 'b')
                available = [(6 - i, fret) for i, fret in enumerate(diagram['frets']) if fret != 'x']
                selected = available if index % 3 == 0 else rng.sample(available, rng.choice([1, 1, min(3, len(available))]))
                notes = [{'string': s, 'fret': f, 'pitch': tuning[s - 1] + f,
                          'effects': ['let'] if index % 7 == 0 else ['pm'] if index % 11 == 0 else []}
                         for s, f in selected]
                effects = ['chord:' + quote(name, safe=''), diagram_effect(diagram)] if changed else []
                events.append({'start': beat * 480, 'duration': {'value': 8}, 'status': 'normal',
                               'notes': notes, 'effects': effects})
            measure['voices'][0]['events'] = events
            targets.append(format_measure_target(measure, 'both'))
        song = targets_to_song(targets, mode='both', title=f'Chord Study {index + 1}',
                               artist='Score Studies', tuning=tuning, midi_program=24)
        # Train both in-score names and actual diagrams; the header library is
        # deliberately separate from the timed event sequence.
        song.tracks[0].settings.diagramList = index % 3 == 0
        song.tracks[0].settings.diagramsInScore = index % 2 == 0
        source_id = f'chords-{index:05d}'
        path = output / 'sources' / f'{source_id}.gp5'
        guitarpro.write(song, str(path))
        split = 'validation' if index % 10 == 8 else 'test' if index % 10 == 9 else 'train'
        modes = ['tab', 'both', 'notation']
        label = analyze_source(path, 0)
        label.update(source_id=source_id, family=source_id, modes=modes)
        (output / 'labels' / f'{source_id}.json').write_text(json.dumps(label, ensure_ascii=False))
        catalog.append({'source_id': source_id, 'source_path': str(path), 'family': source_id,
                        'split': split, 'instrument': 'guitar', 'midi_program': 24, 'modes': modes,
                        'expected_native_transpose': -12, 'instruction': None})
        manifest.extend({'document_id': f'{mode}-{source_id}', 'source_family_id': source_id,
                         'split': 'dev' if split == 'validation' else split, 'instrument_kind': 'guitar',
                         'source': f'sources/{source_id}.gp5', 'display_mode': mode} for mode in modes)
    (output / 'source_catalog.json').write_text(json.dumps({'sources': catalog}, indent=2))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    print({'scores': sources, 'documents': len(manifest), 'bars': bars * len(manifest)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/chord_scores'))
    parser.add_argument('--sources', type=int, default=600)
    parser.add_argument('--bars', type=int, default=24)
    build(**vars(parser.parse_args()))
