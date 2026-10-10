"""Evaluate single-image observations and their assembly into written music."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from urllib.parse import unquote

import numpy as np
from scipy.optimize import linear_sum_assignment
from scorelib.score import duration_ticks, parse_measure_target

from research.inference.run import (
    assemble_predictions,
    observation_record,
    predict_observations,
    read_auxiliary,
    read_rows,
    select_rows,
)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def f1(expected, predicted, correct):
    return {'expected': expected, 'predicted': predicted, 'correct': correct,
            'precision': ratio(correct, predicted), 'recall': ratio(correct, expected),
            'f1': ratio(2 * correct, expected + predicted)}


def ticks(duration):
    if duration is None:
        return None
    try:
        if isinstance(duration, str):
            duration = parse_measure_target(f'M | V0{{@0:{duration}:r}}')['voices'][0]['events'][0]['duration']
        return float(duration_ticks(duration))
    except (ValueError, TypeError, KeyError):
        return None


def events(measure):
    return [{**event, 'at': (voice['voice'], i), 'start': event.get('start'),
             'duration': ticks(event.get('duration')), 'silent': not event.get('notes')}
            for voice in (measure or {}).get('voices', [])
            for i, event in enumerate(voice['events'])]


def notes(measure):
    return [{'at': (*event['at'], i), 'start': event.get('start'),
             'duration': event['duration'], **note}
            for event in events(measure) for i, note in enumerate(event.get('notes') or [])]


def near(gold, actual, tolerance, duration=False):
    fields = ('start', 'duration') if duration else ('start',)
    return all(isinstance(gold.get(field), (int, float))
               and isinstance(actual.get(field), (int, float))
               and abs(gold[field] - actual[field]) <= tolerance for field in fields)


def pairs(gold, actual, matches, *, prefer=None):
    if not gold or not actual:
        return []
    scores = np.array([[int(matches(a, b)) for b in actual] for a in gold], dtype=float)
    if prefer is not None:
        # Preferences in [0, 1] break ties without sacrificing a valid match.
        scores += scores * np.array([[prefer(a, b) for b in actual] for a in gold]) / (min(scores.shape) + 1)
    left, right = linear_sum_assignment(scores, maximize=True)
    return [(int(a), int(b)) for a, b in zip(left, right) if scores[a, b]]


def visible_equal(gold, actual, mode, *, ignore_fret=False):
    fields = ([] if mode == 'tab' else ['staff_position', 'accidental'])
    if mode != 'notation':
        fields += ['string'] + ([] if ignore_fret else ['fret'])
    return all(gold.get(field) == actual.get(field, 'unknown' if field == 'accidental' else None) for field in fields
               if gold.get(field) != 'unknown' and (gold.get(field) is not None or field == 'accidental'))


def endpoint(value, mapping=None):
    if not isinstance(value, list):
        return value
    key = tuple(value)
    return mapping.get(key, ('unmatched', *key)) if mapping is not None else key


def relation_counts(observation, mapping=None):
    from research.inference.assembly import relation_voice

    counts = Counter()
    for relation in observation.get('relations', []):
        start, end = relation.get('start'), relation.get('end')
        voice = None
        if not isinstance(start, list) and not isinstance(end, list):
            # Edge-to-edge scopes have no local endpoint carrying their voice.
            voice = relation_voice(relation)
            if mapping is not None and voice is not None:
                matches = {gold[0] for actual, gold in mapping.items() if actual[0] == voice}
                voice = next(iter(matches)) if len(matches) == 1 else ('unmatched', voice)
        shape = relation.get('shape') if relation.get('kind') is None else None
        counts[relation.get('kind'), relation.get('value'), endpoint(start, mapping), endpoint(end, mapping), voice, shape] += 1
    return counts


def effect_counts(observation, mapping=None):
    return Counter((effect, endpoint(list(item['at']), mapping))
                   for item in [*events(observation), *notes(observation)]
                   for effect in item.get('effects', []))


def add_counts(counts, name, gold, actual, correct):
    counts[name + '_expected'] += gold
    counts[name + '_predicted'] += actual
    counts[name + '_correct'] += correct


def add_relation_counts(counts, name, gold, actual):
    add_counts(counts, name, gold.total(), actual.total(), (gold & actual).total())
    for kind in ('tie', 'hammer', 'sl', 'ss', 'pm', 'let', 'ottava', None):
        expected = Counter({key: n for key, n in gold.items() if key[0] == kind})
        predicted = Counter({key: n for key, n in actual.items() if key[0] == kind})
        add_counts(counts, name + '_' + (kind or 'unknown'), expected.total(), predicted.total(),
                   (expected & predicted).total())


def ordered_event_pairs(gold, actual):
    """Unknown timing cannot locate a repeated note by its musical time.

    Use the same local voice and event ordinal only when its event count
    agrees. Do not choose a correspondence based on the field being scored.
    A missing/extra event makes that voice's positional evidence ambiguous.
    """
    expected_sizes = Counter(event['at'][0] for event in gold)
    actual_sizes = Counter(event['at'][0] for event in actual)
    positions = {event['at']: i for i, event in enumerate(actual)}
    return [(i, positions[event['at']]) for i, event in enumerate(gold)
            if expected_sizes[event['at'][0]] == actual_sizes[event['at'][0]]
            and event['at'] in positions]


def count_rhythm_fields(counts, gold, actual, aligned, tolerance):
    alignment = dict(aligned)
    assigned = {b for _, b in aligned}
    for field in ('start', 'duration'):
        known = [(i, event) for i, event in enumerate(gold) if event[field] is not None]
        unknown = [i for i, event in enumerate(gold) if event[field] is None]
        predicted = sum(actual[alignment[i]][field] is not None for i, _ in known if i in alignment)
        predicted += sum(event[field] is not None for i, event in enumerate(actual) if i not in assigned)
        correct = sum(i in alignment and actual[alignment[i]][field] is not None
                      and abs(event[field] - actual[alignment[i]][field]) <= tolerance for i, event in known)
        add_counts(counts, 'known_' + field, len(known), predicted, correct)
        matched = [actual[alignment[i]][field] for i in unknown if i in alignment]
        counts['unknown_' + field + '_expected'] += len(unknown)
        counts['unknown_' + field + '_matched'] += len(matched)
        counts['unknown_' + field + '_preserved'] += sum(value is None for value in matched)
        counts['unknown_' + field + '_invented'] += sum(value is not None for value in matched)


def musical_counts(counts, prefix, gold, resolved, mode, tolerance):
    expected, actual = notes(gold), notes(resolved)
    fields = ['string', 'fret'] if mode == 'tab' else ['pitch']
    if mode == 'both':
        fields += ['string', 'fret']
    complete = [note for note in actual if all(note.get(field) is not None for field in fields)]
    matches = pairs(expected, complete, lambda a, b: near(a, b, tolerance, True)
                    and all(a.get(field) == b.get(field) for field in fields))
    add_counts(counts, prefix + '_notes', len(expected), len(complete), len(matches))
    counts[prefix + '_note_slots'] += len(actual)
    counts[prefix + '_determinate_notes'] += len(complete)


def summarize(rows, predictions, assembled, auxiliary=None, tolerance=120):
    from research.inference.assembly import parse_observation

    counts, failures, maps, gold_predictions = Counter(), [], {}, []
    before = [assemble_predictions([prediction], [auxiliary[i]] if auxiliary is not None else None)['records'][0]
              for i, prediction in enumerate(predictions)]
    for index, (row, prediction, after) in enumerate(zip(rows, predictions, assembled['records'], strict=True)):
        gold = parse_observation(row['observation'])
        actual = prediction.get('observation') or {'voices': [], 'relations': []}
        gold_predictions.append({**row, 'observation': gold})
        counts['measures'] += 1
        counts['valid_observations'] += prediction.get('observation') is not None
        expected_events, actual_events = events(gold), events(actual)
        expected_notes, actual_notes = notes(gold), notes(actual)
        positional = ordered_event_pairs(expected_events, actual_events)
        count_rhythm_fields(counts, expected_events, actual_events, positional, tolerance)
        partial = {i for i, event in enumerate(expected_events)
                   if event['start'] is None or event['duration'] is None}
        partial_pairs = [(a, b) for a, b in positional if a in partial
                         and expected_events[a]['silent'] == actual_events[b]['silent']
                         and (expected_events[a]['start'] is None
                              or near(expected_events[a], actual_events[b], tolerance))]
        partial_map = {expected_events[a]['at']: actual_events[b]['at'] for a, b in partial_pairs}
        partial_actual = set(partial_map.values())
        unknown_starts = {event['at'] for event in expected_events if event['start'] is None}
        assigned_unknown_starts = {actual for expected, actual in partial_map.items() if expected in unknown_starts}

        def note_time_matches(a, b):
            if a['at'][:2] in unknown_starts:
                return partial_map.get(a['at'][:2]) == b['at'][:2]
            return b['at'][:2] not in assigned_unknown_starts and near(a, b, tolerance)

        note_pairs = pairs(expected_notes, actual_notes,
                           lambda a, b: note_time_matches(a, b) and visible_equal(a, b, row['mode']))
        shared_notes = Counter((expected_notes[a]['at'][:2], actual_notes[b]['at'][:2])
                               for a, b in note_pairs)
        event_pairs = pairs(expected_events, actual_events,
                            lambda a, b: b['at'] not in partial_actual
                            and near(a, b, tolerance, True) and a['silent'] == b['silent'],
                            prefer=lambda a, b: shared_notes[a['at'], b['at']]
                            / max(len(a.get('notes') or []), len(b.get('notes') or []), 1))
        add_counts(counts, 'rhythm', len(expected_events) - len(partial),
                   len(actual_events) - len(partial_pairs), len(event_pairs))
        add_counts(counts, 'visible_notes', len(expected_notes), len(actual_notes), len(note_pairs))
        mapping = {actual_events[b]['at']: expected_events[a]['at'] for a, b in [*event_pairs, *partial_pairs]}
        mapping.update({actual_notes[b]['at']: expected_notes[a]['at'] for a, b in note_pairs})
        maps[index] = mapping
        gold_relations, actual_relations = relation_counts(gold), relation_counts(actual, mapping)
        add_relation_counts(counts, 'local_relations', gold_relations, actual_relations)
        gold_effects, actual_effects = effect_counts(gold), effect_counts(actual, mapping)
        add_counts(counts, 'visible_effects', gold_effects.total(), actual_effects.total(),
                   (gold_effects & actual_effects).total())
        # A chord's name and onset are independent of the played note's length.
        # Equivalent percent encodings describe the same printed name.
        gold_chords, actual_chords = ([{**event, 'name': unquote(effect[6:])}
            for event in values for effect in event.get('effects', []) if effect.startswith('chord:')]
            for values in (expected_events, actual_events))
        positions = {expected_events[a]['at']: actual_events[b]['at'] for a, b in positional}
        chord_pairs = pairs(gold_chords, actual_chords, lambda a, b:
            a['name'] == b['name'] and (near(a, b, tolerance) if a['start'] is not None
                else positions.get(a['at']) == b['at']))
        add_counts(counts, 'visible_chords', len(gold_chords), len(actual_chords), len(chord_pairs))
        if row['mode'] != 'notation':
            unknown = [note for note in expected_notes if note.get('fret') is None]
            aligned = pairs(unknown, actual_notes, lambda a, b: note_time_matches(a, b)
                            and visible_equal(a, b, row['mode'], ignore_fret=True))
            counts['unknown_frets'] += len(unknown)
            counts['matched_unknown_frets'] += len(aligned)
            counts['invented_frets'] += sum(actual_notes[b].get('fret') is not None for _, b in aligned)
        if row.get('reference_target'):
            reference = parse_measure_target(row['reference_target'])
            for prefix, record in [('before', before[index]), ('after', after)]:
                musical_counts(counts, prefix, reference, record.get('resolved'), row['mode'], tolerance)
                counts[prefix + '_complete_measures'] += record.get('ir') is not None
            counts['reference_measures'] += 1
        differences = []
        if prediction.get('error'):
            differences.append('invalid_observation')
        if len(note_pairs) != len(expected_notes) or len(note_pairs) != len(actual_notes):
            differences.append('visible_notes')
        if (len(event_pairs) != len(expected_events) - len(partial)
                or len(event_pairs) != len(actual_events) - len(partial_pairs)):
            differences.append('rhythm')
        if any(expected_events[a][field] is None and actual_events[b][field] is not None
               for a, b in positional for field in ('start', 'duration')):
            differences.append('invented_rhythm')
        if len(partial_pairs) != len(partial):
            differences.append('unmatched_unknown_rhythm')
        if any(expected_events[a][field] is not None
               and (actual_events[b][field] is None
                    or abs(expected_events[a][field] - actual_events[b][field]) > tolerance)
               for a, b in positional for field in ('start', 'duration')):
            differences.append('known_rhythm_fields')
        if gold_relations != actual_relations:
            differences.append('local_relations')
        if gold_effects != actual_effects:
            differences.append('visible_effects')
        state_fields = gold.get('state', {}).keys() | actual.get('state', {}).keys()
        state_errors = [key for key in sorted(state_fields)
                        if gold.get('state', {}).get(key) != actual.get('state', {}).get(key)]
        if state_errors:
            differences.append('state')
        if differences:
            failures.append({'id': row['id'], 'image': row['image'], 'error': prediction.get('error'),
                             'differences': differences, 'state_fields': state_errors,
                             'expected_notes': len(expected_notes), 'predicted_notes': len(actual_notes),
                             'correct_visible_notes': len(note_pairs), 'raw': prediction.get('raw')})

    # Labelled relations define expected coverage. Auxiliary labels here only
    # identify voices in the reference; they never enter model predictions.
    truth_auxiliary = [row.get('oracle_auxiliary', {}) for row in rows]
    expected_assembly = assemble_predictions(gold_predictions, truth_auxiliary)

    def coverage(relations, predicted=False):
        values = Counter()
        for relation in relations:
            for record, voice, event in relation.get('coverage', []):
                at = (voice, event)
                if predicted:
                    at = maps[record].get(at, ('unmatched', *at))
                values[(relation['kind'], relation.get('value'), record, *at)] += 1
        return values

    expected, actual = coverage(expected_assembly['relations']), coverage(assembled['relations'], True)
    add_relation_counts(counts, 'span_events', expected, actual)

    def chains(relations, predicted=False):
        def anchor(value):
            if not isinstance(value, list):
                return value
            record, *local = value
            return (record, *endpoint(local, maps[record] if predicted else None))
        return Counter((r['kind'], r.get('value'), anchor(r['start']), anchor(r['end']))
                       for r in relations if r['status'] == 'resolved' and r['kind'] is not None)

    expected, actual = chains(expected_assembly['relations']), chains(assembled['relations'], True)
    add_counts(counts, 'resolved_relations', expected.total(), actual.total(), (expected & actual).total())
    metrics = {name: f1(*(counts[name + suffix] for suffix in ('_expected', '_predicted', '_correct')))
               for name in ('visible_notes', 'rhythm', 'visible_effects', 'visible_chords', 'local_relations', 'resolved_relations', 'span_events')}
    for name in ('local_relations', 'span_events'):
        metrics[name + '_by_kind'] = {
            kind: f1(*(counts[f'{name}_{kind}{suffix}'] for suffix in ('_expected', '_predicted', '_correct')))
            for kind in ('tie', 'hammer', 'sl', 'ss', 'pm', 'let', 'ottava', 'unknown')}
    metrics['valid_observation_fraction'] = ratio(counts['valid_observations'], counts['measures'])
    metrics['known_rhythm_fields'] = {
        field: f1(*(counts['known_' + field + suffix] for suffix in ('_expected', '_predicted', '_correct')))
        for field in ('start', 'duration')}
    metrics['unknown_rhythm'] = {}
    for field in ('start', 'duration'):
        values = {name: counts['unknown_' + field + '_' + name]
                  for name in ('expected', 'matched', 'preserved', 'invented')}
        values.update(missing=values['expected'] - values['matched'],
                      preserved_fraction=ratio(values['preserved'], values['expected']),
                      invented_fraction=ratio(values['invented'], values['matched']))
        metrics['unknown_rhythm'][field] = values
    metrics['unknown_frets'] = {'expected': counts['unknown_frets'], 'matched': counts['matched_unknown_frets'],
                                'missing': counts['unknown_frets'] - counts['matched_unknown_frets'],
                                'matched_fraction': ratio(counts['matched_unknown_frets'], counts['unknown_frets']),
                                'invented': counts['invented_frets'],
                                'invented_fraction': ratio(counts['invented_frets'], counts['matched_unknown_frets'])}
    for prefix in ('before', 'after'):
        metrics[prefix + '_assembly'] = {
            'written_notes': f1(*(counts[prefix + '_notes' + suffix]
                                  for suffix in ('_expected', '_predicted', '_correct'))),
            'determinate_notes': counts[prefix + '_determinate_notes'],
            'predicted_note_slots': counts[prefix + '_note_slots'],
            'determinate_fraction': ratio(counts[prefix + '_determinate_notes'], counts[prefix + '_note_slots']),
            'm_text_available_fraction': ratio(counts[prefix + '_complete_measures'], counts['reference_measures']),
        }
    return {'metrics': metrics, 'counts': dict(counts), 'failures': failures,
            'rhythm_denominators': {
                'rhythm': 'Events with both reference start and duration known; aligned partial events are excluded.',
                'known_rhythm_fields': 'Known reference fields; predictions at reference-unknown fields are excluded. Unmatched numeric predictions count as extras.',
                'unknown_rhythm': 'Reference null fields; preserved_fraction includes missing events in its denominator, invented_fraction uses matched fields.',
                'field_alignment': 'Same local voice and event ordinal, requiring equal event counts per voice. Ambiguous voices remain unmatched; repeated notes are never rearranged to improve field scores.',
            },
            'timing_tolerance_ticks': tolerance, 'quarter_note_ticks': 960,
            'visible_chords_matching': 'Printed name after percent decoding and onset within tolerance; duration is scored separately. Unknown reference onsets use unambiguous local event order.',
            'pitch_reference': 'written',
            'reference_scope': 'reference_target for music; labelled observations for relation coverage'}


def run(manifest, output, model=None, device='cuda:0', limit=0, batch_size=16, max_tokens=4096,
        observations=None, auxiliary=None, oracle_auxiliary=False, tolerance=120):
    if auxiliary and oracle_auxiliary:
        raise ValueError('Choose predicted auxiliary input or oracle auxiliary labels')
    if tolerance < 0:
        raise ValueError('Timing tolerance must be nonnegative')
    rows = select_rows(read_rows(manifest), limit)
    if not rows:
        raise ValueError('No observation references to evaluate')
    if observations is None and model is None:
        raise ValueError('Provide a merged model or previously predicted observations')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'selection.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    started = time.monotonic()
    if observations:
        saved = {row['id']: row for row in read_rows(observations)}
        stream = (observation_record(row, saved.get(row['id'], {}).get('raw',
                    saved.get(row['id'], {}).get('observation')), saved.get(row['id'], {}).get('tokens')) for row in rows)
    else:
        stream = predict_observations(rows, model, device, batch_size, max_tokens)
    predictions = []
    with (output / 'predictions.jsonl').open('w') as handle:
        for prediction in stream:
            predictions.append(prediction)
            handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            handle.flush()
    extra = (read_auxiliary(auxiliary, rows) if auxiliary else
             [row.get('oracle_auxiliary', {}) for row in rows] if oracle_auxiliary else None)
    assembled = assemble_predictions(predictions, extra)
    result = summarize(rows, predictions, assembled, extra, tolerance)
    for field in ('mode', 'renderer'):
        groups = result['by_' + field] = {}
        for value in sorted({row.get(field) or 'unknown' for row in rows}):
            indices = [i for i, row in enumerate(rows) if (row.get(field) or 'unknown') == value]
            references = [rows[i] for i in indices]
            values = [predictions[i] for i in indices]
            assistance = [extra[i] for i in indices] if extra is not None else None
            grouped = summarize(references, values, assemble_predictions(values, assistance), assistance, tolerance)
            groups[value] = {'samples': len(indices), 'metrics': grouped['metrics'], 'counts': grouped['counts']}
    result['run'] = {'scope': 'research_single_image_observation_and_assembly',
                     'inference': 'saved_observations' if observations else 'vllm',
                     'model': str(model) if model else None, 'device': device if model else None,
                     'samples': len(rows), 'requested_soft_limit': limit,
                     'auxiliary': 'oracle' if oracle_auxiliary else 'predicted' if auxiliary else 'none',
                     'seconds': round(time.monotonic() - started, 3)}
    (output / 'assembled.json').write_text(json.dumps(assembled, ensure_ascii=False, indent=2) + '\n')
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Merged observation OCR model')
    parser.add_argument('--observations', type=Path, help='Evaluate saved predictions without a GPU')
    parser.add_argument('--auxiliary', type=Path, help='Predicted state/voice mappings keyed by id')
    parser.add_argument('--oracle-auxiliary', action='store_true', help='Report a separate oracle-assisted result')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0, help='Soft measure limit; complete documents are retained')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--tolerance', type=int, default=120, help='Onset/duration tolerance in ticks; quarter note = 960')
    report = run(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ('run', 'metrics', 'by_mode', 'by_renderer')}, ensure_ascii=False, indent=2))
