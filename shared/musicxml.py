"""Optional MusicXML projection of the independent score IR."""

from fractions import Fraction
from pathlib import Path
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from shared.instruments import pitch_reference
from shared.pitch_context import transpose_key
from shared.score_state import key_fifths


def element(parent, tag, text=None, **attributes):
    node = ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})
    if text is not None:
        node.text = str(text)
    return node


def duration_ticks(duration):
    value = Fraction(3840 * duration.get('tuplet_times', 1), duration['value'] * duration.get('tuplet_enters', 1))
    return int(value * (Fraction(7, 4) if duration.get('double_dotted') else Fraction(3, 2) if duration.get('dotted') else 1))


def pitch_name(pitch, flats=False):
    names = [('C', 0), ('D', -1), ('D', 0), ('E', -1), ('E', 0), ('F', 0),
             ('G', -1), ('G', 0), ('A', -1), ('A', 0), ('B', -1), ('B', 0)] if flats else [
             ('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
             ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0)]
    step, alter = names[int(pitch) % 12]
    return step, alter, int(pitch) // 12 - 1


def octave_direction(measure, semitones, voice, staff, *, stop=False):
    direction = element(measure, 'direction', placement='above' if semitones > 0 else 'below')
    element(element(direction, 'direction-type'), 'octave-shift',
            type='stop' if stop else 'down' if semitones > 0 else 'up',
            size=8 if abs(semitones) == 12 else 15, number=(voice - 1) % 16 + 1)
    element(direction, 'voice', voice)
    element(direction, 'staff', staff)


def chord_harmony(measure, name, diagram, staff):
    """Keep chord spelling, slash bass and actual fingering in MusicXML."""
    from shared.chords import chord_key

    match = re.fullmatch(r'([A-Ga-g])((?:##|bb|#|b|x)?)(.*?)(?:/([A-Ga-g])((?:##|bb|#|b|x)?))?', chord_key(name))
    harmony = element(measure, 'harmony')
    alterations = {'#': 1, 'b': -1, '##': 2, 'bb': -2, 'x': 2}
    if match:
        step, accidental, suffix, bass, bass_accidental = match.groups()
        root = element(harmony, 'root')
        element(root, 'root-step', step.upper())
        if accidental:
            element(root, 'root-alter', alterations[accidental])
    else:
        # MusicXML's textual function keeps custom/numbered chord labels and
        # anonymous frames without inventing a root pitch or dropping a frame.
        element(harmony, 'function', name or '')
        suffix, bass, bass_accidental = '', None, None
    kinds = {'': 'major', 'm': 'minor', 'min': 'minor', '7': 'dominant',
             'maj7': 'major-seventh', 'M7': 'major-seventh', 'm7': 'minor-seventh',
             'dim': 'diminished', 'dim7': 'diminished-seventh', 'aug': 'augmented',
             '+': 'augmented', 'sus4': 'suspended-fourth', 'sus2': 'suspended-second',
             '6': 'major-sixth', 'm6': 'minor-sixth', '9': 'dominant-ninth',
             'maj9': 'major-ninth', 'm9': 'minor-ninth', '11': 'dominant-11th',
             '13': 'dominant-13th', 'm7b5': 'half-diminished', '5': 'power'}
    kind = element(harmony, 'kind', kinds.get(suffix, 'other') if match else 'other')
    kind.set('text', suffix)
    if bass:
        node = element(harmony, 'bass')
        element(node, 'bass-step', bass.upper())
        if bass_accidental:
            element(node, 'bass-alter', alterations[bass_accidental])
    if diagram:
        frame = element(harmony, 'frame')
        count = len(diagram['frets'])
        element(frame, 'frame-strings', count)
        maximum = max((f for f in diagram['frets'] if type(f) is int), default=0)
        element(frame, 'frame-frets', max(4, maximum - diagram['base_fret'] + 1))
        element(frame, 'first-fret', diagram['base_fret'])
        positions = {(i, f): diagram['fingers'][i] for i, f in enumerate(diagram['frets']) if type(f) is int}
        for f, low, high in diagram['barres']:
            positions.setdefault((low, f), None)
            positions.setdefault((high, f), None)
        for (i, fret), finger in sorted(positions.items()):
            note = element(frame, 'frame-note')
            element(note, 'string', count - i)
            element(note, 'fret', fret)
            if finger is not None:
                element(note, 'fingering', finger)
            for f, low, high in diagram['barres']:
                if f == fret and i in {low, high}:
                    element(note, 'barre', type='start' if i == low else 'stop')
    element(harmony, 'staff', staff)


def score_musicxml(score):
    root = ET.Element('score-partwise', version='4.0')
    element(element(root, 'work'), 'work-title', score.get('title', ''))
    identification = element(root, 'identification')
    element(identification, 'creator', score.get('artist', ''), type='composer')
    part_list = element(root, 'part-list')
    channels = iter([i for i in range(1, 17) if i != 10])
    for i, part in enumerate(score['parts']):
        pid = f'P{i + 1}'
        header = element(part_list, 'score-part', id=pid)
        element(header, 'part-name', part['name'])
        element(header, 'part-abbreviation', part['name'][:12])
        if part['instrument'] == 'drums':
            pitches = sorted({n['pitch'] for s in part['staves'] for m in s['measures'] for v in m['voices'] for e in v['events'] for n in e.get('notes', [])})
            for pitch in pitches:
                element(element(header, 'score-instrument', id=f'{pid}-D{pitch}'), 'instrument-name', f'Drum {pitch}')
            for pitch in pitches:
                midi = element(header, 'midi-instrument', id=f'{pid}-D{pitch}')
                element(midi, 'midi-channel', 10)
                element(midi, 'midi-unpitched', pitch + 1)
        else:
            instrument_id = pid + '-I1'
            element(element(header, 'score-instrument', id=instrument_id), 'instrument-name', part['name'])
            midi = element(header, 'midi-instrument', id=instrument_id)
            element(midi, 'midi-channel', next(channels, 1))
            element(midi, 'midi-program', part.get('midi_program', 0) + 1)
    count = max((m['index'] + 1 for p in score['parts'] for s in p['staves'] for m in s['measures']), default=0)
    timeline = {m['index']: m for m in score.get('timeline', [])}
    for pi, part in enumerate(score['parts']):
        node = element(root, 'part', id=f'P{pi + 1}')
        staves = [{m['index']: m for m in s['measures']} for s in part['staves']]
        previous, time, contexts, keys = {}, '4/4', {}, {}
        for bi in range(count):
            measure = element(node, 'measure', number=bi + 1, id=f'p{pi + 1}m{bi + 1}')
            rows = [s.get(bi) for s in staves]
            old_time, old_contexts, old_keys = time, dict(contexts), dict(keys)
            time = timeline.get(bi, {}).get('time_signature') or next((r['time_signature'] for r in rows if r and r.get('time_signature')), time)
            numerator, denominator = map(int, time.split('/'))
            length = numerator * 3840 // denominator
            attributes = element(measure, 'attributes')
            if bi == 0:
                element(attributes, 'divisions', 960)
            if bi == 0 or time != old_time:
                signature = element(attributes, 'time')
                element(signature, 'beats', numerator)
                element(signature, 'beat-type', denominator)
            if bi == 0 and len(staves) > 1:
                element(attributes, 'staves', len(staves))
            for si, row in enumerate(rows):
                context = contexts[si] = (row or {}).get('pitch_context', contexts.get(si, {}))
                key = (row or {}).get('key_signature') or keys.get(si, 'CMajor')
                keys[si] = key
                if bi == 0 or old_keys.get(si) != key:
                    key_node = element(attributes, 'key', number=si + 1)
                    element(key_node, 'fifths', key_fifths(transpose_key(key, -(context.get('instrument_transpose') or 0))))
                    element(key_node, 'mode', 'minor' if key.endswith('Minor') else 'major')
                name = context.get('clef') or ('percussion' if part['instrument'] == 'drums' else 'F4' if part['instrument'] == 'bass' else 'G2')
                if bi == 0 or any(context.get(k) != old_contexts.get(si, {}).get(k) for k in ('clef', 'clef_octave')):
                    clef = element(attributes, 'clef', number=si + 1)
                    element(clef, 'sign', name[0] if name in {'G2', 'F4', 'C3', 'C4'} else 'percussion' if name == 'percussion' else 'TAB')
                    if name[-1:].isdigit():
                        element(clef, 'line', name[-1])
                    if context.get('clef_octave'):
                        element(clef, 'clef-octave-change', context['clef_octave'] // 12)
                shift = context.get('instrument_transpose', 0)
                if (bi == 0 and shift) or (bi > 0 and shift != old_contexts.get(si, {}).get('instrument_transpose', 0)):
                    transpose = element(attributes, 'transpose', number=si + 1)
                    octave = int(shift / 12)
                    chromatic = shift - octave * 12
                    diatonic = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}[abs(chromatic)] * (1 if chromatic >= 0 else -1)
                    element(transpose, 'diatonic', diatonic)
                    element(transpose, 'chromatic', chromatic)
                    if octave:
                        element(transpose, 'octave-change', octave)
            if not len(attributes):
                measure.remove(attributes)
            tempo = timeline.get(bi, {}).get('tempo_quarter')
            if tempo:
                direction = element(measure, 'direction', placement='above')
                metronome = element(element(direction, 'direction-type'), 'metronome')
                element(metronome, 'beat-unit', 'quarter')
                element(metronome, 'per-minute', tempo)
                element(direction, 'sound', tempo=tempo)
            bars = {mark for row in rows if row for mark in row.get('bars', [])}
            section = next((row.get('section') for row in rows if row and row.get('section')), None)
            if section:
                direction = element(measure, 'direction', placement='above')
                element(element(direction, 'direction-type'), 'rehearsal', section)
            if 'repeat_open' in bars:
                barline = element(measure, 'barline', location='left')
                element(barline, 'bar-style', 'heavy-light')
                element(barline, 'repeat', direction='forward')
            first_voice = True
            for si, row in enumerate(rows):
                voices = row['voices'] if row else [{'voice': 0, 'events': []}]
                context = contexts[si]
                for voice in voices:
                    if not first_voice:
                        element(element(measure, 'backup'), 'duration', length)
                    first_voice = False
                    cursor = 0
                    voice_number = si * 16 + voice['voice'] + 1
                    current_ottava = 0
                    if not voice['events']:
                        rest = element(measure, 'note')
                        element(rest, 'rest', measure='yes')
                        element(rest, 'duration', length)
                        element(rest, 'voice', voice_number)
                        element(rest, 'staff', si + 1)
                        cursor = length
                    events = voice['events']
                    for ei, event in enumerate(events):
                        start = event.get('start', cursor)
                        if start > cursor:
                            element(element(measure, 'forward'), 'duration', start - cursor)
                        elif start < cursor:
                            element(element(measure, 'backup'), 'duration', cursor - start)
                        duration = duration_ticks(event['duration'])
                        ottava = sum(int(e.partition(':')[2]) for e in event.get('effects', []) if e.startswith('ottava:'))
                        if ottava != current_ottava:
                            if current_ottava:
                                octave_direction(measure, current_ottava, voice_number, si + 1, stop=True)
                            if ottava:
                                octave_direction(measure, ottava, voice_number, si + 1)
                            current_ottava = ottava
                        from shared.chords import event_chord
                        chord_name, chord_diagram = event_chord(event)
                        if chord_name or chord_diagram:
                            chord_harmony(measure, chord_name, chord_diagram, si + 1)
                        for effect in event.get('effects', []):
                            if effect.startswith('text:'):
                                direction = element(measure, 'direction', placement='above')
                                element(element(direction, 'direction-type'), 'words', unquote(effect.partition(':')[2]))
                                element(direction, 'staff', si + 1)
                        notes = event.get('notes') or [None]
                        current = {}
                        for ni, source in enumerate(notes):
                            note = element(measure, 'note', id=f'n{pi + 1}_{si + 1}_{bi + 1}_{voice_number}_{ei}_{ni}')
                            if ni:
                                element(note, 'chord')
                            sounding = None
                            if source is None:
                                element(note, 'rest')
                            else:
                                sounding = source.get('pitch')
                                if sounding is None:
                                    fret = source.get('fret')
                                    if not 1 <= source.get('string', 0) <= len(part['tuning']):
                                        raise ValueError(f"Missing tuning for {part['id']}, bar {bi + 1}, string {source.get('string')}")
                                    sounding = part['tuning'][source['string'] - 1] + (fret if isinstance(fret, int) else 0)
                                if pitch_reference(part['instrument']) == 'before_capo':
                                    sounding += part.get('capo', 0)
                                # MusicXML keeps octave-clef and ottava notes at
                                # performed pitch; those marks alter engraving.
                                # Only instrument transposition changes <pitch>.
                                pitch = sounding - (context.get('instrument_transpose') or 0)
                                step, alter, octave = pitch_name(pitch, key_fifths(transpose_key(keys[si], -(context.get('instrument_transpose') or 0))) < 0)
                                if part['instrument'] == 'drums':
                                    display = {35: ('F', 4), 36: ('F', 4), 37: ('C', 5), 38: ('C', 5), 40: ('C', 5),
                                               41: ('F', 4), 43: ('F', 4), 45: ('A', 4), 47: ('C', 5), 48: ('D', 5), 50: ('E', 5),
                                               42: ('G', 5), 44: ('G', 5), 46: ('G', 5), 49: ('A', 5), 51: ('F', 5), 53: ('F', 5)}
                                    step, octave = display.get(sounding, ('B', 4))
                                    pn = element(note, 'unpitched')
                                    element(pn, 'display-step', step)
                                    element(pn, 'display-octave', octave)
                                else:
                                    pn = element(note, 'pitch')
                                    element(pn, 'step', step)
                                    if alter:
                                        element(pn, 'alter', alter)
                                    element(pn, 'octave', octave)
                            element(note, 'duration', duration)
                            if source is not None and part['instrument'] == 'drums':
                                element(note, 'instrument', id=f'P{pi + 1}-D{sounding}')
                            tied = source is not None and 'tie' in source.get('effects', [])
                            previous_note = previous.get((si, voice['voice']), {}).get(sounding)
                            if tied:
                                element(note, 'tie', type='stop')
                                if previous_note is not None:
                                    previous_note.insert(2, ET.Element('tie', type='start'))
                            element(note, 'voice', voice_number)
                            types = {1: 'whole', 2: 'half', 4: 'quarter', 8: 'eighth', 16: '16th', 32: '32nd', 64: '64th', 128: '128th'}
                            element(note, 'type', types[event['duration']['value']])
                            for _ in range(2 if event['duration'].get('double_dotted') else 1 if event['duration'].get('dotted') else 0):
                                element(note, 'dot')
                            if event['duration'].get('tuplet_enters', 1) != 1:
                                tm = element(note, 'time-modification')
                                element(tm, 'actual-notes', event['duration']['tuplet_enters'])
                                element(tm, 'normal-notes', event['duration']['tuplet_times'])
                            element(note, 'staff', si + 1)
                            # Derive beams within a quarter beat; the IR stores rhythm,
                            # so engraving does not need to become a model token stream.
                            if not ni and source is not None and event['duration']['value'] >= 8:
                                beat = start // 960
                                for level in range(1, event['duration']['value'].bit_length() - 2):
                                    def joins(other):
                                        return (other is not None and other.get('notes') and other['start'] // 960 == beat
                                                and other['duration']['value'] >= 2 ** (level + 2))
                                    left = joins(events[ei - 1] if ei else None)
                                    right = joins(events[ei + 1] if ei + 1 < len(events) else None)
                                    if left or right:
                                        element(note, 'beam', 'continue' if left and right else 'end' if left else 'begin', number=level)
                            if source is not None:
                                notations = element(note, 'notations')
                                enters = event['duration'].get('tuplet_enters', 1)
                                if not ni and enters != 1:
                                    in_group = [j for j, e in enumerate(events) if e['start'] // 960 == start // 960
                                                and e['duration'].get('tuplet_enters', 1) == enters]
                                    if ei == in_group[0]:
                                        element(notations, 'tuplet', type='start', number=1, bracket='yes')
                                    if ei == in_group[-1]:
                                        element(notations, 'tuplet', type='stop', number=1)
                                if tied:
                                    element(notations, 'tied', type='stop', number=voice['voice'] + 1)
                                    if previous_note is not None:
                                        prev_notations = previous_note.find('notations')
                                        if prev_notations is None:
                                            prev_notations = element(previous_note, 'notations')
                                        element(prev_notations, 'tied', type='start', number=voice['voice'] + 1)
                                if source.get('string'):
                                    technical = element(notations, 'technical')
                                    element(technical, 'string', source['string'])
                                    if isinstance(source.get('fret'), int):
                                        element(technical, 'fret', source['fret'])
                                for effect in source.get('effects', []):
                                    if effect in {'stacc', 'accent', 'heavy'}:
                                        element(element(notations, 'articulations'), {'stacc': 'staccato', 'accent': 'accent', 'heavy': 'strong-accent'}[effect])
                                    elif effect.startswith('trill'):
                                        element(element(notations, 'ornaments'), 'trill-mark')
                                    elif effect != 'tie':
                                        element(notations, 'other-notation', effect, type='single')
                                current[sounding] = note
                        previous[(si, voice['voice'])] = current
                        cursor = start + duration
                    if current_ottava:
                        octave_direction(measure, current_ottava, voice_number, si + 1, stop=True)
                    if cursor < length:
                        element(element(measure, 'forward'), 'duration', length - cursor)
            if 'repeat_close' in bars or 'double' in bars:
                barline = element(measure, 'barline', location='right')
                element(barline, 'bar-style', 'light-heavy' if 'repeat_close' in bars else 'light-light')
                if 'repeat_close' in bars:
                    repeats = next((row.get('repeat_count') for row in rows if row and row.get('repeat_count')), 2)
                    element(barline, 'repeat', direction='backward', times=repeats)
    # MusicXML defines child order even when tolerant renderers accept more.
    orders = {
        'attributes': ['divisions', 'key', 'time', 'staves', 'part-symbol', 'instruments', 'clef', 'staff-details', 'transpose'],
        'note': ['grace', 'cue', 'chord', 'pitch', 'unpitched', 'rest', 'duration', 'tie', 'instrument', 'footnote', 'level', 'voice', 'type', 'dot', 'accidental', 'time-modification', 'stem', 'notehead', 'staff', 'beam', 'notations', 'lyric', 'play', 'listen'],
    }
    for node in root.iter():
        if node.tag in orders:
            order = {tag: i for i, tag in enumerate(orders[node.tag])}
            node[:] = sorted(node, key=lambda child: order.get(child.tag, 999))
    ET.indent(root)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def write_musicxml(score, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(score_musicxml(score))
    return output
