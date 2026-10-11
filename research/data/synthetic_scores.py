"""Generate complete native scores with varied meters, keys and techniques."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import subprocess
from urllib.parse import quote

import guitarpro as gp
from guitarpro import models as m

from research.data.gp_sources import analyze_source
from scorelib.instruments import VISIBLE_DRUM_KEYS
from scorelib.instruments import DRUM_KEY_ALIASES
from scorelib.chords import diagram_effect
from scorelib.guitarpro import targets_to_song
from scorelib.score import format_measure_target, parse_measure_target


# Verified with the bundled native GP8 importer. These are data-generation
# settings, never an inference lookup from a hidden MIDI program.
PROGRAM_OFFSETS = {0: 0, 24: -12, 33: -12, 56: -2, 60: -7,
                   64: -2, 65: -9, 66: -14, 67: -21, 69: -7, 71: 0}
INSTRUMENT_LABELS = {
    56: ["Trumpet in Bb", "B-flat Trumpet"],
    60: ["Horn in F", "F Horn"],
    64: ["Soprano Sax in Bb", "Bb Soprano Saxophone"],
    65: ["Alto Sax in Eb", "E-flat Alto Saxophone"],
    66: ["Tenor Sax in Bb", "Bb Tenor Saxophone"],
    67: ["Baritone Sax in Eb", "E-flat Baritone Saxophone"],
    69: ["English Horn in F", "Cor anglais in F"],
}


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


def create_native_coverage_score(index, seed, bars):
    """Controlled GP sources for checking visible relations, rhythm and scopes.

    Every source contains every case. Different sources vary pitches, strings,
    tempo and spacing; their families are split before any native rendering.
    These cases describe source intent, never substitute for visible labels.
    """
    if bars < 32:
        raise ValueError('native-coverage requires at least 32 bars')
    rng = random.Random(seed + index)
    song = m.Song(title=f'Visible notation study {index + 1}',
                  artist='Reading studies', tempo=rng.randrange(70, 151))
    song.measureHeaders = []
    track = m.Track(song, name='Guitar', measures=[])
    song.tracks = [track]
    track.channel.instrument = 24
    track.strings = [m.GuitarString(i + 1, pitch)
                     for i, pitch in enumerate([64, 59, 55, 50, 45, 40])]
    track.settings.showRhythm = True
    track.settings.diagramList = False
    cases = []
    primary_string = rng.choice([2, 3, 4])
    fret = rng.randrange(3, 10)

    def fill(bar, durations=None, voice_index=0, string=None):
        measure = track.measures[bar]
        voice = measure.voices[voice_index]
        voice.beats.clear()
        cursor = measure.header.start
        for duration in durations or [m.Duration(8) for _ in range(8)]:
            beat = m.Beat(voice, start=cursor, duration=deepcopy(duration),
                          status=m.BeatStatus.normal)
            voice.beats.append(beat)
            beat.notes.append(m.Note(beat, string=string or primary_string,
                                     value=rng.randrange(2, 14)))
            cursor += duration.time
        if cursor != measure.header.start + measure.header.length:
            raise ValueError(f'Incomplete coverage measure {bar}')
        return voice.beats

    def beat(bar, event, voice=0):
        return track.measures[bar].voices[voice].beats[event]

    def pair(kind, start, end, *, interval=0, voice=0, chord=False):
        origin, destination = beat(*start, voice), beat(*end, voice)
        a, b = origin.notes[0], destination.notes[0]
        a.string = b.string = primary_string
        a.value, b.value = fret, fret + interval
        if kind == 'tie':
            b.type = m.NoteType.tie
        elif kind in {'hammer', 'pull'}:
            a.effect.hammer = True
        else:
            a.effect.slides = [m.SlideType.shiftSlideTo if kind == 'ss'
                               else m.SlideType.legatoSlideTo]
        if chord:
            for item, value in [(origin, fret + 2), (destination, fret + 3)]:
                item.notes.append(m.Note(item, string=primary_string + 1,
                                         value=value))
        cases.append({'kind': kind, 'start': list(start), 'end': list(end),
                      'voice': voice, 'interval': interval,
                      'partial_chord': chord})

    def scope(kind, start, end, voice=0):
        for bar in range(start[0], end[0] + 1):
            events = track.measures[bar].voices[voice].beats
            left = start[1] if bar == start[0] else 0
            right = end[1] if bar == end[0] else len(events) - 1
            for event in events[left:right + 1]:
                for note in event.notes:
                    setattr(note.effect, 'palmMute' if kind == 'pm' else 'letRing', True)
        cases.append({'kind': kind, 'start': list(start), 'end': list(end),
                      'voice': voice})

    for bar in range(bars):
        header = m.MeasureHeader(number=bar + 1, start=960 + bar * 3840)
        header.hasDoubleBar = bar == bars - 1
        song.measureHeaders.append(header)
        track.measures.append(m.Measure(track, header))
        fill(bar)
    # GP5 stores a break after the measure. Native layout is checked after
    # importing; an import that ignores a requested break is not called coverage.
    for bar in [6, 7, 15, 24, 26, 27]:
        track.measures[bar].lineBreak = m.LineBreak.break_
    cases.append({'kind': 'negative', 'bars': [0, 29]})
    pair('tie', (1, 1), (1, 2))
    pair('tie', (1, 5), (1, 6))
    pair('tie', (2, 2), (2, 3), chord=True)
    pair('tie', (2, 7), (3, 0))
    pair('tie', (3, 7), (4, 0))
    pair('tie', (5, 7), (6, 0), chord=True)
    pair('pull', (6, 7), (7, 0), interval=-1)
    pair('tie', (7, 7), (8, 0))
    pair('hammer', (9, 0), (9, 1), interval=1)
    pair('pull', (9, 2), (9, 3), interval=-1)
    pair('hammer', (9, 5), (9, 6), interval=rng.choice([2, 3, 5]))
    pair('hammer' if index % 2 == 0 else 'pull', (10, 7), (11, 0),
         interval=1 if index % 2 == 0 else -1)
    scope('pm', (11, 2), (11, 5))
    pair('hammer', (11, 7), (12, 0), interval=1)
    pair('ss', (12, 1), (12, 2), interval=3)
    pair('ss', (12, 5), (12, 6), interval=-2)
    pair('sl', (13, 0), (13, 1), interval=1)
    pair('sl', (13, 4), (13, 5), interval=-1)
    for event, slide in zip([0, 2, 4, 6], [m.SlideType.intoFromBelow,
                            m.SlideType.intoFromAbove, m.SlideType.outDownwards,
                            m.SlideType.outUpwards]):
        beat(14, event).notes[0].effect.slides = [slide]
        cases.append({'kind': slide.name, 'bar': 14, 'event': event, 'voice': 0})
    pair('ss' if index % 2 == 0 else 'sl', (15, 7), (16, 0), interval=3)
    rhythm = [m.Duration(8, tuplet=m.Tuplet(3, 2)) for _ in range(3)]
    rhythm += [m.Duration(4, True), m.Duration(8), m.Duration(4)]
    fill(17, rhythm)
    fill(18, rhythm)
    pair('tie', (18, 0), (18, 1))
    pair('pull', (18, 3), (18, 4), interval=-1)
    beat(18, 2).notes.clear()
    beat(18, 2).status = m.BeatStatus.rest
    cases.append({'kind': 'triplet_dotted', 'bars': [17, 18]})
    fill(19, [m.Duration(4) for _ in range(4)])
    fill(19, [m.Duration(8, tuplet=m.Tuplet(3, 2)) for _ in range(3)]
         + [m.Duration(4) for _ in range(3)],
         voice_index=1, string=6)
    cases.append({'kind': 'multivoice', 'bars': [19, 28, 29]})
    scope('pm', (20, 3), (20, 3))
    scope('let', (21, 5), (21, 5))
    scope('pm', (22, 2), (22, 5))
    scope('let', (23, 1), (23, 6))
    pair('ss', (23, 7), (24, 0), interval=3)
    pair('sl', (20, 7), (21, 0), interval=-1)
    scope('pm', (24, 4), (27, 2))
    scope('let', (26, 5), (27, 2))
    scope('pm', (27, 4), (27, 7))
    scope('let', (27, 4), (27, 7))
    fill(28, voice_index=1, string=6)
    scope('pm', (28, 0), (28, 5))
    scope('let', (28, 2), (28, 7), voice=1)
    fill(29, rhythm)
    fill(29, [m.Duration(4) for _ in range(4)], voice_index=1, string=6)
    for event in [2, 4]:
        beat(29, event).notes.clear()
        beat(29, event).status = m.BeatStatus.rest
    scope('let', (30, 1), (30, 6))
    pair('tie', (30, 1), (30, 2))
    pair('hammer', (30, 4), (30, 5), interval=1)
    scope('pm', (31, 0), (31, 6))
    pair('ss', (31, 1), (31, 2), interval=2)
    pair('sl', (31, 4), (31, 5), interval=-2)
    song._coverage_cases = cases
    return song, 'guitar', 24, -12, None


SHAPES = [
    ('C', ['x', 3, 2, 0, 1, 0], [None, 3, 2, None, 1, None], []),
    ('G', [3, 2, 0, 0, 0, 3], [2, 1, None, None, None, 3], []),
    ('D', ['x', 'x', 0, 2, 3, 2], [None, None, None, 1, 3, 2], []),
    ('Am', ['x', 0, 2, 2, 1, 0], [None, None, 2, 3, 1, None], []),
    ('Em', [0, 2, 2, 0, 0, 0], [None, 2, 3, None, None, None], []),
    ('A7', ['x', 0, 2, 0, 2, 0], [None, None, 1, None, 2, None], []),
    ('D7', ['x', 'x', 0, 2, 1, 2], [None, None, None, 2, 1, 3], []),
    ('Cmaj7', ['x', 3, 2, 0, 0, 0], [None, 3, 2, None, None, None], []),
    ('Dsus4', ['x', 'x', 0, 2, 3, 3], [None, None, None, 1, 2, 3], []),
    ('C/G', [3, 3, 2, 0, 1, 0], [3, 4, 2, None, 1, None], []),
    ('D/F#', [2, 'x', 0, 2, 3, 2], [0, None, None, 1, 3, 2], []),
    ('F', [1, 3, 3, 2, 1, 1], [1, 3, 4, 2, 1, 1], [[1, 0, 5]]),
    ('Fm', [1, 3, 3, 1, 1, 1], [1, 3, 4, 1, 1, 1], [[1, 0, 5]]),
    ('F7', [1, 3, 1, 2, 1, 1], [1, 3, 1, 2, 1, 1], [[1, 0, 5]]),
    ('Fm7', [1, 3, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1], [[1, 0, 5]]),
    ('B', ['x', 2, 4, 4, 4, 2], [None, 1, 3, 3, 3, 1], [[2, 1, 5], [4, 2, 4]]),
    ('Bm', ['x', 2, 4, 4, 3, 2], [None, 1, 3, 4, 2, 1], [[2, 1, 5]]),
    ('B7', ['x', 2, 4, 2, 4, 2], [None, 1, 3, 1, 4, 1], [[2, 1, 5]]),
]


NAMES = [('C',), ('C#', 'Db'), ('D',), ('D#', 'Eb'), ('E',), ('F',),
         ('F#', 'Gb'), ('G',), ('G#', 'Ab'), ('A',), ('A#', 'Bb'), ('B',)]


def sample_diagram(rng):
    name, frets, fingers, barres = rng.choice(SHAPES)
    frets, fingers, barres = list(frets), list(fingers), [list(b) for b in barres]
    if barres:
        shift = rng.randrange(11)
        base_pc = 5 if name.startswith('F') else 11
        name = rng.choice(NAMES[(base_pc + shift) % 12]) + name[1:]
        frets = [f + shift if type(f) is int else f for f in frets]
        barres = [[f + shift, a, b] for f, a, b in barres]
    base = min((f for f in frets if type(f) is int and f > 0), default=1)
    if base <= 3 or 0 in frets:
        base = 1
    if rng.random() < .55:
        fingers = [None] * len(frets)
    return name, {'base_fret': base, 'frets': frets, 'fingers': fingers, 'barres': barres}


def create_chord_score(index, seed, bars):
    rng = random.Random(seed + index)
    tuning = [64, 59, 55, 50, 45, 40]
    targets = []
    for bar in range(bars):
        measure = parse_measure_target('M time=4/4 key=CMajor | V0{@0:w:r}')
        events = []
        name, diagram = sample_diagram(rng)
        for beat in range(8):
            changed = beat == 0 or (beat in {2, 4, 6} and rng.random() < .5)
            if changed and beat:
                name, diagram = sample_diagram(rng)
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
    return song, 'guitar', 24, -12, None


def create_picking_score(index, seed, bars):
    """Picking signs alone and beside H/P phrases, with unmarked controls."""
    song, instrument, program, shift, instruction = create_technique_score(index * 2, seed, bars)
    rng = random.Random(f'picking:{seed}:{index}')
    track = song.tracks[0]
    track.offset = 0
    for mi, measure in enumerate(track.measures):
        events = measure.voices[0].beats
        group = rng.choice([2, 3])
        for ei, event in enumerate(events):
            if ei % group == 0:
                string, fret = rng.randint(1, len(track.strings)), rng.randint(4, 14)
                direction = rng.choice([-1, 1])
            event.effect = m.BeatEffect()
            event.octave = m.Octave.none
            event.text = ''
            note = m.Note(event, string=string, value=fret + direction * (ei % group),
                          type=m.NoteType.normal)
            event.notes = [note]
            if mi % 4 in (1, 3) and ei % group + 1 < group and ei + 1 < len(events):
                note.effect.hammer = True
            if mi % 4 in (2, 3) and ei % group == 0:
                event.effect.pickStroke = rng.choice([m.BeatStrokeDirection.up, m.BeatStrokeDirection.down])
    return song, instrument, program, shift, None


def build(output, sources=1200, bars=24, seed=20260928, profile='general', format='gp5'):
    output = output.resolve()
    (output / 'sources').mkdir(parents=True, exist_ok=True)
    (output / 'labels').mkdir(exist_ok=True)
    catalog, manifest, native_styles = [], [], []
    for index in range(sources):
        creator = {'general': create_score, 'techniques': create_technique_score, 'picking': create_picking_score,
                   'chords': create_chord_score,
                   'native-coverage': create_native_coverage_score}[profile]
        phrase = index // 3 if profile == 'chords' else index
        song, instrument, program, shift, instruction = creator(phrase, seed, bars)
        if profile == 'chords':
            # Keep identical sounding music across absent/name/grid views.
            # All three inherit one source family and one dataset split.
            view = index % 3
            song.tracks[0].settings.diagramsInScore = view == 2
            song.tracks[0].settings.diagramList = view != 0 and phrase % 3 == 0
            if view == 0:
                for measure in song.tracks[0].measures:
                    for voice in measure.voices:
                        for beat in voice.beats:
                            beat.effect.chord = None
        source_id = f'{seed:08x}{index:08x}'
        # Rotate holdouts so every instrument has independently generated scores.
        split = ('train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'validation', 'test')[(phrase if profile in ('native-coverage', 'chords', 'picking') else index // 4) % 10]
        path = output / 'sources' / f'{source_id}.gp5'
        gp.write(song, str(path))
        if format == 'gp':
            native_styles.append({'source': str(path),
                'diagrams_in_score': song.tracks[0].settings.diagramsInScore,
                'diagram_list': song.tracks[0].settings.diagramList})
        label = analyze_source(path, 0)
        modes = (['tab'] if profile == 'native-coverage' else
                 ['notation', 'both', 'tab'] if instrument in {'guitar', 'bass'} else ['notation'])
        family = f'synthetic-{seed}-{phrase}'
        label.update(source_id=source_id, family=family, modes=modes)
        (output / 'labels' / f'{source_id}.json').write_text(json.dumps(label, ensure_ascii=False))
        row = {'source_id': source_id, 'source_path': str(path), 'family': family, 'split': split,
               'instrument': instrument, 'midi_program': program, 'modes': modes,
               'expected_native_transpose': shift, 'instruction': instruction}
        if profile == 'techniques' and instruction:
            row.update(instruction_kind='capo', capo=song.tracks[0].offset)
        if profile == 'native-coverage':
            row['coverage_cases'] = song._coverage_cases
        catalog.append(row)
        manifest.extend({'document_id': f'{mode}-{source_id}', 'source_family_id': family,
                         'split': 'dev' if split == 'validation' else split, 'instrument_kind': instrument,
                         'source': f'{source_id}.gp5', 'display_mode': mode} for mode in modes)
        if (index + 1) % 100 == 0:
            print('Generated', index + 1, 'scores', flush=True)
    if format == 'gp':
        # GP8's legacy GP5 importer does not preserve in-score diagram style.
        # Save that authored setting in the standard GP format before the
        # native GP renderer creates pixels and observation evidence.
        root = Path(__file__).resolve().parents[2]
        node = shutil.which('node') or str(root / 'tools/node-bin/node')
        module = (root / 'tools/alphatab/package/dist/alphaTab.core.mjs').as_uri()
        script = """
import fs from 'node:fs';
const at = await import(process.argv[1]);
const rows = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const row of rows) {
    const score = at.importer.ScoreLoader.loadScoreFromBytes(new Uint8Array(fs.readFileSync(row.source)));
    score.stylesheet.globalDisplayChordDiagramsInScore = row.diagrams_in_score;
    score.stylesheet.globalDisplayChordDiagramsOnTop = row.diagram_list;
    fs.writeFileSync(row.source.replace(/\\.gp5$/, '.gp'), new at.exporter.Gp7Exporter().export(score));
}
"""
        subprocess.run([node, '--input-type=module', '-e', script, module],
                       input=json.dumps(native_styles), text=True, check=True)
        for row in catalog:
            original = Path(row['source_path'])
            row['source_path'] = str(original.with_suffix('.gp'))
        for row in manifest:
            row['source'] = str(Path(row['source']).with_suffix('.gp'))
    (output / 'source_catalog.json').write_text(json.dumps({'sources': catalog}, indent=2))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    print({'sources': len(catalog), 'documents': len(manifest), 'measures': len(manifest) * bars}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=int, default=1200)
    parser.add_argument('--bars', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--profile', choices=['general', 'techniques', 'picking', 'chords', 'native-coverage'], default='general')
    parser.add_argument('--format', choices=['gp5', 'gp'], default='gp5',
                        help='Use gp to preserve authored chord-diagram placement in GP8')
    build(**vars(parser.parse_args()))
