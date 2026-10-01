"""Recover fretted tuning from independently read notation and TAB evidence."""

from collections import Counter, defaultdict

from scorelib.m2 import parse_measure_target
from scorelib.pitch_context import convert_pitch_target
from scorelib.score_state import resolve_fretted_pitches


def reconcile_score_tuning(records):
    parts = defaultdict(list)
    for row in records:
        if row.get('mode') == 'both' and row.get('visual_pitch'):
            parts[row.get('part_id', 'part-1')].append(row)
    for rows in parts.values():
        evidence = defaultdict(Counter)
        bars, frets_seen = defaultdict(set), defaultdict(set)
        for row in rows:
            if row.get('fallback_reason'):
                continue
            target = row['target']
            if row.get('written_target') and not row.get('manually_edited'):
                try:
                    target = convert_pitch_target(row['written_target'], row.get('pitch_context') or {}, mode='both')
                except (ValueError, KeyError, TypeError):
                    continue
            measure = parse_measure_target(target)
            for voice in measure['voices']:
                for event in voice['events']:
                    for note in event.get('notes', []):
                        effects = note.get('effects', [])
                        if any(effect in effects for effect in ('dead', 'tie')) or any(effect.startswith('harm:') for effect in effects):
                            # Harmonic notation can show the overtone or touch
                            # point. Subtracting its TAB fret does not identify
                            # the open string; keep the printed/default tuning.
                            continue
                        string, fret, pitch = (note.get(key) for key in ('string', 'fret', 'pitch'))
                        if all(type(v) is int for v in (string, fret, pitch)) and 0 <= pitch - fret <= 127:
                            evidence[string][pitch - fret] += 1
                            bars[(string, pitch - fret)].add(row.get('bar_index', row['measure_number']))
                            frets_seen[(string, pitch - fret)].add(fret)
        tuning = list(rows[0].get('tuning') or [])
        if not tuning:
            continue
        explicit = any(row.get('tuning_explicit') for row in rows)
        conflicts, confirmed, altered = [], set(), False
        unusual = rows[0].get('instrument') == 'guitar' and len(tuning) in {4, 5}
        if unusual and not explicit:
            from scorelib.instruments import STANDARD_TUNINGS

            reliable = {string: counts.most_common(1)[0][0] for string, counts in evidence.items()
                        if 1 <= string <= len(tuning) and counts.most_common(1)[0][1] >= 6
                        and counts.most_common(1)[0][1] / counts.total() >= .85
                        and len(bars[(string, counts.most_common(1)[0][0])]) >= 3
                        and len(frets_seen[(string, counts.most_common(1)[0][0])]) >= 2}
            candidates = {tuple(value) for value in STANDARD_TUNINGS.values()
                          if len(value) == len(tuning) and len(reliable) >= 2
                          and all(value[string - 1] == pitch for string, pitch in reliable.items())}
            if len(candidates) == 1:
                inferred = list(next(iter(candidates)))
                altered = tuning != inferred
                tuning = inferred
                confirmed.update(reliable)
        shifts = defaultdict(list)
        for string, counts in evidence.items():
            pitch, count = counts.most_common(1)[0]
            if (1 <= string <= len(tuning) and count >= 6 and count / counts.total() >= .85
                    and len(bars[(string, pitch)]) >= 3 and len(frets_seen[(string, pitch)]) >= 2):
                shifts[pitch - tuning[string - 1]].append(string)
        uniform_conflicts = {s for shift, strings in shifts.items() if shift and len(strings) >= 3 for s in strings}
        for string, counts in evidence.items():
            pitch, count = counts.most_common(1)[0]
            if not 1 <= string <= len(tuning):
                conflicts.append(string)
                continue
            reliable = (count >= 6 and count / counts.total() >= .85
                        and len(bars[(string, pitch)]) >= 3
                        and len(frets_seen[(string, pitch)]) >= 2)
            if explicit:
                confirmed.add(string)
            elif string in uniform_conflicts:
                # A uniform shift of several strings is ambiguous between a
                # detuned instrument and a wrong global pitch context.
                conflicts.append(string)
            elif reliable and abs(pitch - tuning[string - 1]) <= 7:
                # An octave-wide discrepancy is much more likely a clef or
                # transposition error than a newly discovered string tuning.
                altered |= tuning[string - 1] != pitch
                tuning[string - 1] = pitch
                confirmed.add(string)
            elif reliable or count >= 6 and count / counts.total() < .65:
                conflicts.append(string)
        for row in records:
            if row.get('part_id', 'part-1') != rows[0].get('part_id', 'part-1'):
                continue
            row['tuning'] = tuning
            row['tuning_source'] = (row.get('tuning_source', 'explicit') if explicit else
                                    'notation_tab_consensus' if altered else row.get('tuning_source', 'default'))
            row['tuning_evidence'] = {str(k): dict(v) for k, v in evidence.items()}
            if row.get('mode') == 'both' and not row.get('manually_edited'):
                if not conflicts and (explicit or len(confirmed) >= 2):
                    row['target'], changed = resolve_fretted_pitches(
                        row['target'], tuning, verified_strings=None if explicit else confirmed)
                    if changed:
                        row['pitch_reconciled'] = True
                elif conflicts:
                    row.update(needs_review=True, tuning_needs_review=conflicts)
    return records
