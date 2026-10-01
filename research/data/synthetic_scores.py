"""Generate complete native scores with varied meters, keys and techniques."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import guitarpro as gp
from guitarpro import models as m

from research.data.gp_sources import analyze_source
from research.data.prepare_pitch_data import INSTRUMENT_LABELS
from research.data.prepare_pitch_data import PROGRAM_OFFSETS
from scorelib.percussion import VISIBLE_DRUM_KEYS
from scorelib.percussion import DRUM_KEY_ALIASES


METERS = [(4, 4), (3, 4), (2, 4), (6, 8), (5, 4), (7, 8), (9, 8), (12, 8), (5, 8), (3, 8), (2, 2)]
MAJOR_KEYS = [key for key in m.KeySignature if key.value[1] == 0 and abs(key.value[0]) <= 7]
DRUMS = sorted(VISIBLE_DRUM_KEYS - set(DRUM_KEY_ALIASES))


def create_score(index, seed, bars):
    rng = random.Random(seed + index)
    instrument = ('guitar', 'bass', 'pitched', 'drums')[index % 4]
    program = 24 if instrument == 'guitar' else 33 if instrument == 'bass' else 0
    if instrument == 'pitched':
        program = (0, 56, 60, 64, 65, 66, 67, 69, 71)[index // 4 % 9]
    song = m.Song(title=f'Study {index + 1}', artist='Score Studies', tempo=rng.randrange(55, 201))
    song.measureHeaders = []
    track = m.Track(song, name=instrument.title(), measures=[], isPercussionTrack=instrument == 'drums')
    song.tracks = [track]
    track.channel.instrument = program
    track.channel.channel = 9 if instrument == 'drums' else 0
    track.channel.effectChannel = 9 if instrument == 'drums' else 1
    if instrument == 'guitar':
        tunings = [[64, 59, 55, 50, 45, 40], [64, 59, 55, 50, 45, 38], [64, 59, 55, 50, 45, 40, 35]]
    elif instrument == 'bass':
        tunings = [[43, 38, 33, 28], [43, 38, 33, 28, 23], [48, 43, 38, 33, 28, 23]]
    else:
        tunings = [[0] * 7] if instrument == 'drums' else [[48] * 7]
    tuning = rng.choice(tunings)
    track.strings = [m.GuitarString(i + 1, pitch) for i, pitch in enumerate(tuning)]
    shift = 0 if instrument == 'drums' else PROGRAM_OFFSETS[program]
    instruction = INSTRUMENT_LABELS[program][index % 2] if program in INSTRUMENT_LABELS else 'Concert pitch' if instrument == 'pitched' else None
    # Keep instrumental rendering and pitch conversion independently observable.
    start, previous = 960, {}
    voices = 2 if index % 3 == 0 else 1
    for bar in range(bars):
        if bar % 4 == 0:
            meter = METERS[(index // 4 + bar // 4) % len(METERS)]
            key = MAJOR_KEYS[(index // 4 + bar // 4 * 7) % len(MAJOR_KEYS)]
        if instrument == 'drums':
            key = m.KeySignature.CMajor
        numerator, denominator = meter
        header = m.MeasureHeader(number=bar + 1, start=start, keySignature=key,
                                 timeSignature=m.TimeSignature(numerator, m.Duration(denominator)))
        if bar == bars - 1:
            header.hasDoubleBar = True
        song.measureHeaders.append(header)
        measure = m.Measure(track, header)
        track.measures.append(measure)
        scale = {(key.value[0] * 7 + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}
        for voice_index in range(voices):
            voice = measure.voices[voice_index]
            cursor = start
            for unit in range(numerator):
                pattern = rng.randrange(4) if denominator < 8 else rng.randrange(3)
                if voice_index == 1:
                    pattern = min(pattern, 1)
                durations = ([m.Duration(denominator)] if pattern == 0 else
                             [m.Duration(denominator * 2)] * 2 if pattern == 1 else
                             [m.Duration(denominator * 2, True), m.Duration(denominator * 4)] if pattern == 2 else
                             [m.Duration(denominator * 2, tuplet=m.Tuplet(3, 2))] * 3)
                for offset, duration in enumerate(durations):
                    beat = m.Beat(voice, duration=deepcopy(duration), start=cursor, status=m.BeatStatus.normal)
                    voice.beats.append(beat)
                    cursor += duration.time
                    if bar == 0 and unit == 0 and offset == 0 and voice_index == 0 and instruction:
                        beat.text = instruction
                    prev = previous.get(voice_index, [])
                    tied = bool(prev) and instrument != 'drums' and unit == 0 and offset == 0 and rng.random() < .3
                    if not tied and rng.random() < .10:
                        beat.status = m.BeatStatus.rest
                        previous[voice_index] = []
                        continue
                    if tied:
                        positions = [(note.string, note.value) for note in prev]
                    elif instrument == 'drums':
                        pitches = rng.sample(DRUMS if index % 5 == 0 else [36, 38, 42, 44, 46, 45, 47, 48, 49, 51, 53, 55, 57, 59], rng.randrange(1, 4))
                        positions = list(enumerate(pitches, 1 if voice_index == 0 else 5))
                    elif instrument == 'pitched':
                        pitches = rng.sample([p for p in range(48, 73) if p % 12 in scale], rng.randrange(1, 4) if voice_index == 0 else 1)
                        positions = [(i + (1 if voice_index == 0 else 7), p - 48) for i, p in enumerate(pitches)]
                    else:
                        strings = rng.sample(range(1, len(tuning) if voices == 2 else len(tuning) + 1), rng.randrange(1, min(4, len(tuning)))) if voice_index == 0 else [len(tuning)]
                        positions = [(s, rng.choice([f for f in range(1, 16) if (tuning[s - 1] + f) % 12 in scale])) for s in strings]
                    for string, fret in positions:
                        note = m.Note(beat, value=fret, string=string, type=m.NoteType.tie if tied else m.NoteType.normal)
                        beat.notes.append(note)
                        if not tied:
                            effect = rng.randrange(18)
                            if effect == 0:
                                note.effect.ghostNote = True
                            elif effect == 1:
                                note.effect.accentuatedNote = True
                            elif effect == 2:
                                note.effect.staccato = True
                            elif instrument != 'drums':
                                if effect == 3:
                                    note.effect.grace = m.GraceEffect(fret=max(0, fret - rng.choice([1, 2, 3])), isOnBeat=rng.choice([True, False]))
                                elif effect == 4:
                                    note.effect.trill = m.TrillEffect(fret=min(27, fret + 2), duration=m.Duration(16))
                                elif effect == 5:
                                    note.effect.tremoloPicking = m.TremoloPickingEffect(m.Duration(rng.choice([8, 16, 32])))
                                elif instrument in {'guitar', 'bass'}:
                                    if effect == 6:
                                        note.effect.bend = m.BendEffect(m.BendType.bend, 50, [m.BendPoint(0, 0), m.BendPoint(6, 2), m.BendPoint(12, 2)])
                                    elif effect == 7:
                                        note.effect.vibrato = True
                                    elif effect == 8:
                                        note.effect.palmMute = True
                                    elif effect == 9:
                                        note.effect.letRing = True
                    previous[voice_index] = beat.notes
        start += header.length
    return song, instrument, program, shift, instruction


def create_technique_score(index, seed, bars):
    song, instrument, program, shift, _ = create_score(index // 2 * 4 + index % 2, seed, bars)
    rng = random.Random(seed + index)
    track = song.tracks[0]
    capo = rng.randint(1, 9) if instrument == 'guitar' and index % 4 == 0 else 0
    track.offset = capo
    song.title = f'Technique Study {index + 1}'
    previous = None
    for bar, measure in enumerate(track.measures):
        for voice in measure.voices:
            voice.beats.clear()
        voice = measure.voices[0]
        denominator = max(8, measure.header.timeSignature.denominator.value)
        count = measure.header.length // m.Duration(denominator).time
        technique = (index // 2 + bar // 4) % 7
        phrase_size = rng.choice([2, 3, 4])
        phrase = []
        for event in range(count):
            beat = m.Beat(voice, duration=m.Duration(denominator),
                          start=measure.header.start + event * m.Duration(denominator).time,
                          status=m.BeatStatus.normal)
            voice.beats.append(beat)
            if event % phrase_size == 0:
                string = rng.randint(1, len(track.strings))
                start = rng.randint(2, 12)
                phrase = [start + 2 * j for j in range(phrase_size)]
                if rng.random() < .5:
                    phrase.reverse()
                if rng.random() < .8:
                    beat.effect.pickStroke = rng.choice([m.BeatStrokeDirection.up, m.BeatStrokeDirection.down])
            fret = phrase[event % phrase_size]
            tied = event == 0 and previous is not None and bar % 5 == 0
            if tied:
                string, fret = previous
            note = m.Note(beat, value=fret, string=string, type=m.NoteType.tie if tied else m.NoteType.normal)
            beat.notes.append(note)
            if not tied:
                continues = event % phrase_size + 1 < phrase_size and event + 1 < count
                if technique == 0 and continues:
                    note.effect.hammer = True
                elif technique == 1 and continues:
                    note.effect.slides = [rng.choice([m.SlideType.shiftSlideTo, m.SlideType.legatoSlideTo])]
                elif technique == 2:
                    note.value = rng.choice([5, 7, 12])
                    note.effect.harmonic = m.NaturalHarmonic()
                elif technique == 3:
                    if event % 3 == 0:
                        note.effect.bend = m.BendEffect(m.BendType.bend, 50, [m.BendPoint(0, 0), m.BendPoint(6, 2), m.BendPoint(12, 2)])
                    elif event % 3 == 1:
                        note.effect.ghostNote = True
                    else:
                        note.effect.staccato = True
                elif technique == 4:
                    if event % 2 == 0:
                        note.effect.grace = m.GraceEffect(fret=max(0, fret - 2), isOnBeat=bar % 2 == 0)
                    else:
                        note.effect.trill = m.TrillEffect(fret=fret + 2, duration=m.Duration(16))
                elif technique == 5:
                    note.effect.palmMute = event % 4 < 2
                    note.effect.letRing = event % 4 >= 2
                elif technique == 6:
                    note.effect.tremoloPicking = m.TremoloPickingEffect(m.Duration(32))
                    note.effect.vibrato = event % 3 == 0
            if bar % 6 in {2, 3} and index % 3 == 0:
                beat.octave = m.Octave.ottava
            previous = (note.string, note.value)
        if bar == 0 and capo:
            voice.beats[0].text = f'Capo {capo}'
    return song, instrument, program, shift, f'Capo {capo}' if capo else None


def build(output, sources=1200, bars=24, seed=20260928, profile='general'):
    output = output.resolve()
    (output / 'sources').mkdir(parents=True, exist_ok=True)
    (output / 'labels').mkdir(exist_ok=True)
    catalog, manifest = [], []
    for index in range(sources):
        creator = create_technique_score if profile == 'techniques' else create_score
        song, instrument, program, shift, instruction = creator(index, seed, bars)
        source_id = f'{seed:08x}{index:08x}'
        # Rotate holdouts so every instrument has independently generated scores.
        split = ('train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'validation', 'test')[(index // 4) % 10]
        path = output / 'sources' / f'{source_id}.gp5'
        gp.write(song, str(path))
        label = analyze_source(path, 0)
        modes = ['notation', 'both', 'tab'] if instrument in {'guitar', 'bass'} else ['notation']
        family = f'synthetic-{seed}-{index}'
        label.update(source_id=source_id, family=family, modes=modes)
        (output / 'labels' / f'{source_id}.json').write_text(json.dumps(label, ensure_ascii=False))
        row = {'source_id': source_id, 'source_path': str(path), 'family': family, 'split': split,
               'instrument': instrument, 'midi_program': program, 'modes': modes,
               'expected_native_transpose': shift, 'instruction': instruction}
        if profile == 'techniques' and instruction:
            row.update(instruction_kind='capo', capo=song.tracks[0].offset)
        catalog.append(row)
        manifest.extend({'document_id': f'{mode}-{source_id}', 'source_family_id': family,
                         'split': 'dev' if split == 'validation' else split, 'instrument_kind': instrument,
                         'source': f'sources/{source_id}.gp5', 'display_mode': mode} for mode in modes)
        if (index + 1) % 100 == 0:
            print('Generated', index + 1, 'scores', flush=True)
    (output / 'source_catalog.json').write_text(json.dumps({'sources': catalog}, indent=2))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    print({'sources': len(catalog), 'documents': len(manifest), 'measures': len(manifest) * bars}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('database/parallel_synthetic'))
    parser.add_argument('--sources', type=int, default=1200)
    parser.add_argument('--bars', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--profile', choices=['general', 'techniques'], default='general')
    build(**vars(parser.parse_args()))
