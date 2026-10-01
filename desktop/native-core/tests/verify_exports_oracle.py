"""Development-only compatibility oracle; never run or packaged by the client.
Run with the project's existing .venv Python and a prebuilt native_export binary.
Checks real legacy/native files through the same independent reader, plus XML.
"""
import copy
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import attr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.gp_io import read_gp
from shared.m2 import format_measure_target
from shared.musicxml import score_musicxml
from gp5_export.score import write_score_gp5
ROOT = Path(__file__).parent

def recognition(score):
    rows = []
    for p in score['parts']:
        for staff in p['staves']:
            for m in staff['measures']:
                m = copy.deepcopy(m)
                mode = m.get('mode', 'both')
                # Canonical fixtures can omit redundant pitches on physical notes.
                for v in m['voices']:
                    for e in v['events']:
                        for n in e['notes']:
                            if 'pitch' not in n and mode in ('both', 'notation'):
                                n['pitch'] = p['tuning'][n['string'] - 1] + (n['fret'] if isinstance(n['fret'], int) else 0)
                row = {'part_id': p['id'], 'staff_id': staff['id'], 'bar_index': m['index'],
                       'measure_number': len(rows) + 1, 'target': format_measure_target(m, mode, preserve_playback=True),
                       'mode': mode, 'instrument': p['instrument'], 'tuning': p['tuning'],
                       'capo': p['capo'], 'midi_program': p['midi_program']}
                if p['name'] != p['instrument']:
                    row['part_name'] = p['name']
                rows.append(row)
    first = score['parts'][0]
    return {'title': score['title'], 'artist': score['artist'], 'mode': rows[0]['mode'],
            'instrument': first['instrument'], 'tuning_used': first['tuning'], 'capo': first['capo'],
            'midi_program': first['midi_program'], 'records': rows}

def effect(note, track):
    result = attr.asdict(note.effect)
    result['slides'] = sorted(x.value for x in note.effect.slides)
    # Virtual slot numbering is invisible in notation-only tracks; compare
    # ornament pitches rather than its auxiliary storage fret.
    if not track.settings.tablature:
        base = track.strings[note.string - 1].value
        for name in ('grace', 'trill'):
            if result[name] is not None:
                result[name]['fret'] += base
    return result

def note_snapshot(song):
    result = []
    for ti, track in enumerate(song.tracks):
        for mi, measure in enumerate(track.measures):
            for vi, voice in enumerate(measure.voices):
                for beat in voice.beats:
                    notes = []
                    for n in beat.notes:
                        pitch = track.strings[n.string - 1].value + n.value
                        notes.append((pitch, n.type.name, n.velocity, n.swapAccidentals,
                                      n.string if track.settings.tablature else None,
                                      effect(n, track)))
                    notes.sort(key=lambda n: (n[0], n[1], n[4] or 0))
                    be = beat.effect
                    result.append((ti, mi, vi, beat.start - measure.header.start,
                                   beat.duration.time, beat.status.name, notes, beat.text,
                                   be.fadeIn, be.hasRasgueado, be.pickStroke, be.slapEffect,
                                   be.stroke, beat.octave,
                                   be.mixTableChange.tempo.value if be.mixTableChange and be.mixTableChange.tempo else None))
    return result

def xml_tree(node):
    text = node.text if node.text and node.text.strip() else None
    children = [xml_tree(c) for c in node]
    # Both placements are legal, and legacy inserts forward ties retrospectively.
    if node.tag == 'notations':
        children.sort(key=repr)
    return (node.tag, sorted(node.attrib.items()), text, children)

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    for source in sorted((ROOT / 'fixtures/music-exports').glob('*.json')):
        score = json.loads(source.read_text())
        actual = directory / (source.stem + '.gp5')
        legacy = directory / (source.stem + '-legacy.gp5')
        native_run = subprocess.run([sys.argv[1], str(source), 'gp5', str(actual)], text=True, capture_output=True, check=True)
        report = json.loads(native_run.stdout)['report']
        write_score_gp5(recognition(score), legacy)
        before, after = read_gp(legacy, encoding='cp936'), read_gp(actual, encoding='cp936')
        assert report['readback_verified']
        assert before.title == after.title
        assert len(before.tracks) == len(after.tracks), (source.name, 'track count', len(before.tracks), len(after.tracks))
        for left, right in zip(before.tracks, after.tracks, strict=True):
            assert (left.name, left.offset, left.fretCount, left.channel.instrument, left.channel.channel,
                    left.settings.tablature, left.settings.notation, left.clefTranspose) == (
                        right.name, right.offset, right.fretCount, right.channel.instrument, right.channel.channel,
                        right.settings.tablature, right.settings.notation, right.clefTranspose), (source.name, 'track settings', left, right)
        assert note_snapshot(before) == note_snapshot(after), (source.name, 'note/effect/timing projection', note_snapshot(before), note_snapshot(after))
        for left, right in zip(before.measureHeaders, after.measureHeaders, strict=True):
            for field in ('isRepeatOpen', 'repeatClose', 'repeatAlternative', 'tripletFeel', 'direction', 'fromDirection', 'keySignature'):
                assert getattr(left, field) == getattr(right, field), (source.name, field, getattr(left, field), getattr(right, field))
        if source.stem == 'diagram':
            a = before.tracks[0].measures[0].voices[0].beats[0].effect.chord
            b = after.tracks[0].measures[0].voices[0].beats[0].effect.chord
            assert attr.asdict(a) == attr.asdict(b), (a, b)
        xmlpath = actual.with_suffix('.musicxml')
        subprocess.run([sys.argv[1], str(source), 'musicxml', str(xmlpath)], text=True, capture_output=True, check=True)
        old_tree, new_tree = xml_tree(ET.fromstring(score_musicxml(score))), xml_tree(ET.parse(xmlpath).getroot())
        assert old_tree == new_tree, (source.name, 'MusicXML differs from legacy projection', old_tree, new_tree)
        print(source.stem, 'PASS legacy GP5+MusicXML compatibility')
