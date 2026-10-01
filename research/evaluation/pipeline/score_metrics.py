"""Score notes at their actual part, staff and absolute bar positions."""

from collections import Counter

from scipy.optimize import linear_sum_assignment
import numpy as np

from scorelib.chords import chord_key
from scorelib.chords import event_chord
from scorelib.chords import normalize_diagram
from scorelib.m2 import parse_measure_target
from scorelib.musicxml import duration_ticks


def staff_mapping(truth, predictions, matched):
    gold = sorted({(r['part_id'], r['staff_id']) for r in truth})
    predicted = sorted({(r.get('part_id'), r.get('staff_id')) for r in predictions}, key=str)
    votes = np.zeros((len(predicted), len(gold)), dtype=np.int64)
    for i, j in matched.items():
        a, b = predictions[j], truth[i]
        votes[predicted.index((a.get('part_id'), a.get('staff_id'))), gold.index((b['part_id'], b['staff_id']))] += 1
    # Match instruments first: independently matching staves would forgive a
    # piano's two staves being wrongly exported as two unrelated instruments.
    gold_parts = sorted({p for p, _ in gold}, key=str)
    predicted_parts = sorted({p for p, _ in predicted}, key=str)
    part_votes = np.zeros((len(predicted_parts), len(gold_parts)), dtype=np.int64)
    for i, (part, _) in enumerate(predicted):
        for j, (reference, _) in enumerate(gold):
            part_votes[predicted_parts.index(part), gold_parts.index(reference)] += votes[i, j]
    a, b = linear_sum_assignment(part_votes, maximize=True)
    result = {}
    for i, j in zip(a, b):
        if not part_votes[i, j]:
            continue
        rows = [k for k, (part, _) in enumerate(predicted) if part == predicted_parts[i]]
        columns = [k for k, (part, _) in enumerate(gold) if part == gold_parts[j]]
        local_a, local_b = linear_sum_assignment(votes[np.ix_(rows, columns)], maximize=True)
        result.update({predicted[rows[x]]: gold[columns[y]] for x, y in zip(local_a, local_b)
                       if votes[rows[x], columns[y]]})
    return result


def score_counts(truth, predictions, matched):
    mapping = staff_mapping(truth, predictions, matched)

    def contents(rows, predicted=False):
        notes, chords, techniques, fingerings, rhythm = (Counter() for _ in range(5))
        for row in rows:
            staff = (row.get('part_id'), row.get('staff_id'))
            staff = mapping.get(staff, ('unmatched', str(staff))) if predicted else staff
            try:
                text = row.get('target', '') if predicted else row.get('sounding_target', row['target'])
                measure = parse_measure_target(text)
            except (ValueError, KeyError, TypeError):
                continue
            for voice in measure['voices']:
                for event in voice['events']:
                    at = (*staff, row.get('bar_index'), voice['voice'], event['start'])
                    duration = duration_ticks(event['duration'])
                    rhythm[(*at, duration, not bool(event.get('notes')))] += 1
                    name, _diagram = event_chord(event)
                    if name:
                        chords[(*at, chord_key(name))] += 1
                    for note in event.get('notes', []):
                        dead = note.get('fret') == 'x' or 'dead' in note.get('effects', [])
                        identity = ('dead', note.get('string') if row['mode'] == 'tab' else None) if dead else (
                            ('position', note.get('string'), note.get('fret')) if row['mode'] == 'tab' else ('pitch', note.get('pitch')))
                        notes[(*at, duration, identity)] += 1
                        if row['mode'] in {'tab', 'both'}:
                            fingerings[(*at, duration, note.get('string'), note.get('fret'))] += 1
                        for effect in note.get('effects', []):
                            techniques[(*at, identity, effect)] += 1
                    for effect in event.get('effects', []):
                        if not effect.startswith(('chord:', 'diagram:')):
                            techniques[(*at, 'beat', effect)] += 1
        return notes, chords, techniques, fingerings, rhythm

    result = Counter()
    for name, gold, actual in zip(('score_notes', 'score_chords', 'score_techniques', 'score_fingerings', 'score_rhythm'), contents(truth), contents(predictions, True)):
        result[name + '_expected'] = gold.total()
        result[name + '_predicted'] = actual.total()
        result[name + '_correct'] = (gold & actual).total()
    result['placed_bars'] = sum(
        mapping.get((predictions[j].get('part_id'), predictions[j].get('staff_id'))) == (truth[i]['part_id'], truth[i]['staff_id'])
        and predictions[j].get('bar_index') == truth[i]['bar_index'] for i, j in matched.items())
    return result


def readable_metrics(counts):
    values = {}
    for name in ('score_notes', 'score_chords', 'score_techniques', 'score_fingerings', 'score_rhythm',
                 'chord_symbols', 'diagram_strings', 'diagram_fingers', 'diagram_barres'):
        if name + '_expected' not in counts:
            continue
        expected, predicted, correct = (counts.get(name + suffix, 0) for suffix in ('_expected', '_predicted', '_correct'))
        values[name] = {'precision': correct / predicted if predicted else 0,
                        'recall': correct / expected if expected else 0,
                        'f1': 2 * correct / (expected + predicted) if expected + predicted else 0,
                        'missed_per_100': 100 * (expected - correct) / expected if expected else None,
                        'extra_per_100': 100 * (predicted - correct) / expected if expected else None}
    values['bar_placement_accuracy'] = counts.get('placed_bars', 0) / counts['expected_bars'] if counts.get('expected_bars') else 0
    if counts.get('nonpitch_annotations'):
        values['false_pitch_annotation_rate'] = counts.get('false_pitch_annotations', 0) / counts['nonpitch_annotations']
    return values


def annotation_counts(truth, predictions):
    """Include missed and extra printed diagrams, not just successful OCR crops."""
    from research.evaluation.pipeline.evaluate_scores import overlap

    costs = np.zeros((len(truth), len(predictions)), dtype=float)
    for i, gold in enumerate(truth):
        for j, actual in enumerate(predictions):
            if gold['page'] == actual.get('page') and gold['part_id'] == actual.get('part_id', 'part-1'):
                costs[i, j] = overlap(gold['bbox'], actual['bbox'])
    a, b = linear_sum_assignment(costs, maximize=True)
    pairs = {i: j for i, j in zip(a, b) if costs[i, j] >= .3}

    def contents(rows):
        result = []
        for row in rows:
            value = row.get('parsed', {})
            chord = value.get('kind') in {'chord', 'chord_diagram'}
            shape = normalize_diagram(value.get('diagram')) if chord else None
            names = Counter([chord_key(value['text'])]) if chord and value.get('text') else Counter()
            result.append((names,
                Counter(enumerate(shape['frets'])) if shape else Counter(),
                Counter((i, finger) for i, finger in enumerate(shape['fingers']) if finger is not None) if shape else Counter(),
                Counter(tuple(barre) for barre in shape['barres']) if shape else Counter()))
        return result

    gold, actual = contents(truth), contents(predictions)
    result = Counter()
    for i, annotation in enumerate(truth):
        if annotation.get('parsed', {}).get('kind') in {'chord', 'chord_diagram', 'technique', 'tempo', 'other'}:
            result['nonpitch_annotations'] += 1
            if i in pairs:
                value = predictions[pairs[i]].get('parsed', {})
                result['false_pitch_annotations'] += value.get('semitones') is not None or value.get('capo') is not None
    for index, name in enumerate(('chord_symbols', 'diagram_strings', 'diagram_fingers', 'diagram_barres')):
        result[name + '_expected'] = sum(value[index].total() for value in gold)
        result[name + '_predicted'] = sum(value[index].total() for value in actual)
        result[name + '_correct'] = sum((gold[i][index] & actual[j][index]).total() for i, j in pairs.items())
    return result
