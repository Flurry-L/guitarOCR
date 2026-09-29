"""Project the score IR into synchronized GP5 tracks without flattening time."""

from collections import Counter, defaultdict
from copy import deepcopy

from gp5_export.writer import GP5ReadbackError, GP5TimingError, targets_to_song, write_targets_gp5
from shared.m2 import format_measure_target, full_measure_rest_target, parse_measure_target


def notation_rows(rows, tuning):
    """Give a staff that changes display mode one pitch representation."""
    result = []
    for row in rows:
        measure = parse_measure_target(row['target'])
        for voice in measure['voices']:
            for event in voice['events']:
                for note in event.get('notes', []):
                    if 'pitch' not in note:
                        fret = note.get('fret')
                        note['pitch'] = tuning[note['string'] - 1] + (fret if isinstance(fret, int) else 0)
                    effects = []
                    for effect in note.get('effects', []):
                        fields = effect.split(':')
                        if fields[0] in {'grace', 'trill'} and len(fields) > 1 and fields[1] != 'x':
                            from shared.techniques import ornament_position

                            fret, pitch = ornament_position(fields[1])
                            if pitch is None and fret is not None:
                                fields[1] = 'p' + str(tuning[note['string'] - 1] + fret)
                        effects.append(':'.join(fields))
                    note['effects'] = effects
        result.append({**row, 'target': format_measure_target(measure, 'notation', preserve_playback=True)})
    return result


def pitch_partitions(targets, percussion=False):
    """Split wide or dense parts only when seven GP5 slots cannot hold them."""
    from gp5_export.pitched import storage_tuning

    measures = [parse_measure_target(t) for t in targets]
    try:
        storage_tuning(measures, percussion)
        return [targets]
    except ValueError:
        pass
    # A stable pitch range keeps ties on the same track across barlines.
    pitches = sorted({n['pitch'] for m in measures for v in m['voices'] for e in v['events'] for n in e.get('notes', [])})
    if len(pitches) < 2:
        raise ValueError('Repeated unisons exceed GP5 note slots')
    middle = pitches[len(pitches) // 2]
    groups = []
    for upper in (False, True):
        subset = deepcopy(measures)
        for measure in subset:
            for voice in measure['voices']:
                for event in voice['events']:
                    event['notes'] = [n for n in event.get('notes', []) if (n['pitch'] >= middle) == upper]
                    if not event['notes'] and event['status'] == 'normal':
                        event['status'] = 'rest'
        groups.extend(pitch_partitions([format_measure_target(m, 'notation', preserve_playback=True) for m in subset], percussion))
    return groups


def assign_notation_strings(rows, tuning):
    from gp5_export.fingering import _assign_positions, _plan_notation_voice_positions, _tie_reservation_note_ids

    measures = [parse_measure_target(r['target']) for r in rows]
    voice_ids = sorted({v['voice'] for m in measures for v in m['voices']})
    reserved_ids = _tie_reservation_note_ids(measures)
    for voice_id in voice_ids:
        plan = _plan_notation_voice_positions(measures, voice_id, tuning)
        previous, reserved = {}, {}
        for measure in measures:
            voice = next((v for v in measure['voices'] if v['voice'] == voice_id), None)
            if voice is None:
                continue
            for event in voice['events']:
                notes = event.get('notes', [])
                positions = [plan.get(id(n)) for n in notes]
                if any(p is None for p in positions):
                    positions = _assign_positions(notes, tuning, previous, reserved)
                for note, (string, fret) in zip(notes, positions, strict=True):
                    note.update(string=string, fret=fret)
                    if 'dead' in note.get('effects', []):
                        continue
                    previous[string] = (note['pitch'], fret)
                    if id(note) in reserved_ids.get(voice_id, set()):
                        reserved[string] = note['pitch']
                    elif 'tie' in note.get('effects', []):
                        reserved.pop(string, None)
    return [{**row, 'target': format_measure_target(m, 'both', preserve_playback=True)}
            for row, m in zip(rows, measures, strict=True)]


def score_to_song(result):
    groups = defaultdict(list)
    for row in result['records']:
        groups[(row.get('part_id', 'part-1'), row.get('staff_id', 'staff-1'))].append(row)
    timelines = defaultdict(list)
    for rows in groups.values():
        for i, row in enumerate(rows):
            index = row.get('bar_index', i)
            row = {**row, 'bar_index': index}
            timelines[index].append(row)
    total = max(timelines, default=-1) + 1
    meters, meter = [], '4/4'
    for index in range(total):
        candidates = [parse_measure_target(r['target']).get('time_signature') or
                      (r.get('score_state') or {}).get('time') for r in timelines[index]]
        candidates = [s for s in candidates if s]
        meter = Counter(candidates).most_common(1)[0][0] if candidates else meter
        meters.append(meter)
    song, mapping = None, []
    channels = {}
    for (part_id, staff_id), rows in groups.items():
        rows = sorted(rows, key=lambda r: r.get('bar_index', r['measure_number'] - 1))
        first = rows[0]
        instrument = first.get('instrument', result.get('instrument', 'guitar'))
        mode = first.get('mode', result['mode'])
        tuning = first.get('tuning', result['tuning_used'])
        if len({r.get('mode', mode) for r in rows}) > 1:
            rows = notation_rows(rows, tuning)
            mode = 'notation'
        virtual_tuning = False
        if mode == 'notation' and instrument in {'guitar', 'bass'} and not any(r.get('tuning_explicit') for r in rows):
            from gp5_export.fingering import notation_fingering_errors

            virtual_tuning = any(notation_fingering_errors(parse_measure_target(r['target']), tuning) for r in rows)
        if mode == 'notation' and instrument in {'guitar', 'bass'} and len(tuning) > 7 and not virtual_tuning:
            rows = assign_notation_strings(rows, tuning)
            mode = 'both'
        by_index = {r.get('bar_index', i): r for i, r in enumerate(rows)}
        voices = sorted({v['voice'] for r in rows for v in parse_measure_target(r['target'])['voices']})
        voice_groups = [voices[i:i + 2] for i in range(0, len(voices), 2)]
        strings = [list(range(i, min(i + 7, len(tuning)))) for i in range(0, len(tuning), 7)] if instrument in {'guitar', 'bass'} and not virtual_tuning else [[]]
        for voice_ids in voice_groups:
          for string_ids in strings:
            targets, source_rows = [], []
            for index in range(total):
                row = by_index.get(index)
                numerator, denominator = map(int, meters[index].split('/'))
                data = parse_measure_target(row['target'] if row else full_measure_rest_target((numerator, denominator)))
                data['time_signature'] = meters[index]
                data['print_time_signature'] = True
                data['voices'] = [v for v in data['voices'] if v['voice'] in voice_ids]
                for voice in data['voices']:
                    voice['voice'] = voice_ids.index(voice['voice'])
                    if len(strings) > 1 and mode in {'tab', 'both'}:
                        for event in voice['events']:
                            event['notes'] = [n for n in event.get('notes', []) if n.get('string', 0) - 1 in string_ids]
                            for note in event['notes']:
                                note['string'] = string_ids.index(note['string'] - 1) + 1
                            if not event['notes']:
                                event['status'] = 'rest'
                if not data['voices']:
                    data['voices'] = parse_measure_target(full_measure_rest_target((numerator, denominator)))['voices']
                targets.append(format_measure_target(data, mode, preserve_playback=True))
                source_rows.append(row['measure_number'] if row else None)
            partitions = pitch_partitions(targets, instrument == 'drums') if instrument in {'pitched', 'drums'} or virtual_tuning else [targets]
            for partition_index, partition in enumerate(partitions):
                try:
                    local = targets_to_song(partition, mode=mode, title=result.get('title', ''), artist=result.get('artist', ''),
                                            tuning=[tuning[i] for i in string_ids] if string_ids else [],
                                            capo=first.get('capo', result.get('capo', 0)), instrument=instrument,
                                            midi_program=first.get('midi_program', result.get('midi_program')), virtual_tuning=virtual_tuning)
                except GP5TimingError as error:
                    numbers = [source_rows[index - 1] for index in error.measures]
                    raise GP5TimingError([number for number in numbers if number is not None], error.detail) from error
                if song is None:
                    song = local
                    track = song.tracks.pop()
                else:
                    track = local.tracks[0]
                track.song = song
                track.number = len(song.tracks) + 1
                track.name = first.get('part_name') or track.name
                if sum(p == part_id for p, _s in groups) > 1:
                    track.name += ' ' + ('R.H.' if staff_id == 'staff-1' else 'L.H.' if staff_id == 'staff-2' else staff_id)
                if len(voice_groups) > 1:
                    track.name += ' V' + ','.join(str(v + 1) for v in voice_ids)
                if len(strings) > 1:
                    track.name += ' strings ' + ','.join(str(i + 1) for i in string_ids)
                if len(partitions) > 1:
                    track.name += f' range {partition_index + 1}'
                if instrument == 'drums':
                    channel = 9
                else:
                    program = track.channel.instrument
                    if part_id not in channels:
                        used = {value[0] for value in channels.values()}
                        free = [i for i in range(16) if i != 9 and i not in used]
                        if free:
                            channels[part_id] = (free[0], program)
                        else:
                            reuse = next((v for v in channels.values() if v[1] == program), None)
                            if reuse is None:
                                raise ValueError('GP5 MIDI channels cannot preserve every instrument; use the score IR')
                            channels[part_id] = reuse
                    channel = channels[part_id][0]
                track.channel.channel = track.channel.effectChannel = channel
                for measure, header in zip(track.measures, song.measureHeaders, strict=True):
                    measure.track = track
                    measure.header = header
                song.tracks.append(track)
                mapping.append(source_rows)
    if song is None:
        raise ValueError('At least one staff is required')
    return song, mapping


def write_score_gp5(result, output):
    song, mapping = score_to_song(deepcopy(result))
    try:
        return write_targets_gp5([], output, _song=song)
    except GP5ReadbackError as error:
        numbers = [mapping[track][bar - 1] for track, bar in error.locations]
        raise GP5ReadbackError([n for n in numbers if n is not None], error.locations) from error
