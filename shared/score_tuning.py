"""Recover fretted tuning from independently read notation and TAB evidence."""

from collections import Counter, defaultdict

from shared.m2 import parse_measure_target
from shared.pitch_context import convert_pitch_target
from shared.score_state import resolve_fretted_pitches


def reconcile_score_tuning(records):
    parts = defaultdict(list)
    for row in records:
        if row.get('mode') == 'both' and row.get('visual_pitch'):
            parts[row.get('part_id', 'part-1')].append(row)
    for rows in parts.values():
        evidence = defaultdict(Counter)
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
        tuning = list(rows[0].get('tuning') or [])
        if not tuning:
            continue
        explicit = any(row.get('tuning_explicit') for row in rows)
        conflicts = []
        for string, counts in evidence.items():
            pitch, count = counts.most_common(1)[0]
            if not 1 <= string <= len(tuning):
                conflicts.append(string)
                continue
            if not explicit and count / counts.total() >= .65:
                tuning[string - 1] = pitch
            elif not explicit:
                conflicts.append(string)
        for row in records:
            if row.get('part_id', 'part-1') != rows[0].get('part_id', 'part-1'):
                continue
            row['tuning'] = tuning
            row['tuning_source'] = 'manual' if explicit else 'notation_tab_consensus'
            row['tuning_evidence'] = {str(k): dict(v) for k, v in evidence.items()}
            if row.get('mode') == 'both' and not row.get('manually_edited'):
                if not conflicts:
                    row['target'], changed = resolve_fretted_pitches(row['target'], tuning)
                    if changed:
                        row['pitch_reconciled'] = True
                else:
                    row.update(needs_review=True, tuning_needs_review=conflicts)
    return records
