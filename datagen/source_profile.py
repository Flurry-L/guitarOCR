"""Recover legacy track metadata from the source instead of assuming guitar."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=8192)
def source_profile(label_path):
    label = json.loads(Path(label_path).read_text())
    track = label['track']
    instrument = track.get('instrument')
    program = track.get('midi_program')
    tuning = list(track.get('tuning_midi_high_to_low') or [])
    legacy = instrument is None
    if instrument is None or program is None:
        import guitarpro

        source = next((Path(value) for key in ('source_path', 'prepared_gp5')
                       if (value := label.get(key)) and Path(value).is_file()), None)
        if source is None:
            raise ValueError(f'Missing source track metadata and source file: {label_path}')
        song = guitarpro.parse(source, encoding=label.get('source_encoding') or 'cp1252')
        native = song.tracks[int(track.get('index', 0))]
        program = int(native.channel.instrument) if program is None else int(program)
        if instrument is None:
            instrument = ('drums' if native.isPercussionTrack else
                          'guitar' if 24 <= program <= 31 else
                          'bass' if 32 <= program <= 39 else 'pitched')
        if not tuning:
            tuning = [string.value for string in native.strings]
    if instrument not in {'guitar', 'bass'}:
        tuning = []
    return {'instrument': instrument, 'midi_program': int(program),
            'tuning': tuning, 'string_count': len(tuning), 'legacy': legacy}


def apply_source_profile(row):
    """Correct prompt metadata while preserving the annotated musical events."""
    if not row.get('label_json'):
        return dict(row)
    profile = source_profile(str(row['label_json']))
    old_instrument = row.get('instrument', 'guitar')
    result = {**row, **{key: profile[key] for key in ('instrument', 'midi_program', 'string_count')},
              'tuning': list(profile['tuning'])}
    if row.get('score_state') is not None:
        result['score_state'] = dict(row['score_state'])
        if profile['tuning']:
            result['score_state']['tuning'] = list(profile['tuning'])
        else:
            result['score_state'].pop('tuning', None)
    # Old GP5 bass sources were given the guitar fallback before written-pitch
    # supervision was built. Both instruments transpose by an octave, so only
    # the incorrect default clef and instrument prompt need correction here.
    if profile['legacy'] and old_instrument == 'guitar' and profile['instrument'] == 'bass':
        if row.get('pitch_context') and row['pitch_context'].get('clef') == 'G2':
            result['pitch_context'] = {**row['pitch_context'], 'clef': 'F4'}
    _align_text(result)
    return result


def apply_composed_profile(row):
    """Keep the printed part identity; recover the source image's legacy clef."""
    result = {**row, 'string_count': len(row.get('tuning') or [])}
    if row.get('label_json'):
        profile = source_profile(str(row['label_json']))
        context = row.get('pitch_context') or {}
        if profile['legacy'] and profile['instrument'] == 'bass' and context.get('clef') == 'G2':
            result['pitch_context'] = {**context, 'clef': 'F4'}
        _align_text(result)
    return result


def _align_text(row):
    if row.get('measure_index') is None:
        return
    from datagen.native_alignment import native_text_target

    for key in ('target', 'sounding_target'):
        if row.get(key):
            row[key] = native_text_target(row[key], row['label_json'], row['mode'],
                                          row.get('source_measure_index', row['measure_index']))
