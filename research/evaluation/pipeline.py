"""Evaluate full documents with the application's native recognition pipeline."""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scorelib.chords import chord_key, event_chord, normalize_diagram
from scorelib.score import duration_ticks, parse_measure_target

from research.common.images import MODEL_RENDER_DPI
from research.evaluation.measure_metrics import MeasureSequenceMetrics
from research.inference.run import NativeClient


def overlap(a, b):
    x, y, w, h = a
    u, v, s, t = b
    area = max(0., min(x + w, u + s) - max(x, u)) * max(0., min(y + h, v + t) - max(y, v))
    return area / max(1., w * h + s * t - area)

def canonical_rows(rows):
    systems, parts, staves = {}, {}, defaultdict(dict)
    result = []
    for system, part, staff in rows:
        result.append([systems.setdefault(system, len(systems)), parts.setdefault(part, len(parts)),
                       staves[part].setdefault(staff, len(staves[part]))])
    return result

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
                    duration = int(duration_ticks(event['duration']))
                    rhythm[(*at, duration, not bool(event.get('notes')))] += 1
                    name, _diagram = event_chord(event)
                    if name:
                        chords[(*at, chord_key(name))] += 1
                    visible_spans = set()
                    for note in event.get('notes', []):
                        dead = note.get('fret') == 'x' or 'dead' in note.get('effects', [])
                        identity = ('dead', note.get('string') if row['mode'] == 'tab' else None) if dead else (
                            ('position', note.get('string'), note.get('fret')) if row['mode'] == 'tab' else ('pitch', note.get('pitch')))
                        notes[(*at, duration, identity)] += 1
                        if row['mode'] in {'tab', 'both'}:
                            fingerings[(*at, duration, note.get('string'), note.get('fret'))] += 1
                        for effect in note.get('effects', []):
                            if effect in {'pm', 'let'}:
                                visible_spans.add(effect)
                            else:
                                techniques[(*at, identity, effect)] += 1
                    for effect in event.get('effects', []):
                        if effect in {'pm', 'let'}:
                            visible_spans.add(effect)
                        elif not effect.startswith(('chord:', 'diagram:')):
                            techniques[(*at, 'beat', effect)] += 1
                    for effect in visible_spans:
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


def documents(paths):
    """Read generated score.json files or the native measure generator's JSONL."""
    result = []
    for path in paths:
        sources = sorted(path.rglob("score.json")) if path.is_dir() else [path]
        for source in sources:
            if source.suffix == ".jsonl":
                groups = defaultdict(list)
                for line in source.read_text().splitlines():
                    row = json.loads(line)
                    groups[row["source_pdf"]].append({
                        **row, "bbox": [v * MODEL_RENDER_DPI / 25.4 for v in row["bbox_mm"]],
                    })
                for pdf, rows in groups.items():
                    expected = rows[0]['source_measures']
                    if (len(rows) != expected or
                            sorted(r['bar_index'] for r in rows) != list(range(expected))):
                        raise ValueError(f'Full-document evaluation requires every bar: {pdf}')
                scores = [{"id": rows[0]["source_id"] + "-" + rows[0]["mode"],
                           "pdf": pdf, "records": rows} for pdf, rows in groups.items()]
            else:
                scores = [json.loads(source.read_text())]
            for score in scores:
                pdf = Path(score["pdf"])
                if not pdf.is_absolute():
                    pdf = source.parent / pdf
                if not pdf.is_file():
                    raise FileNotFoundError(pdf)
                result.append({**score, "pdf": pdf, "source": str(source)})
    if not result:
        raise ValueError("No complete scores to evaluate")
    return result


def compare(truth, project):
    rows = project.get("measures", [])
    pairs = [(overlap(gold["bbox"], row["bbox"]), i, j)
             for i, gold in enumerate(truth["records"]) for j, row in enumerate(rows)
             if gold["page"] == row["page"]]
    matched, used = {}, set()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou >= .5 and i not in matched and j not in used:
            matched[i] = j
            used.add(j)
    counts = Counter(scores=1, expected_bars=len(truth["records"]), detected_bars=len(rows),
                     matched_bars=len(matched), extra_bars=len(rows) - len(used),
                     failed=bool(project.get("error")))
    counts.update(score_counts(truth["records"], rows, matched))
    if "annotations" in truth:
        metadata = project.get("metadata", {})
        annotations = [row for part in metadata.get("parts", [metadata])
                       for kind in ("score_annotations", "pitch_instructions")
                       for row in part.get("document_metadata", {}).get(kind, [])]
        counts.update(annotation_counts(truth["annotations"], annotations))
    predictions, gold_groups, predicted_groups = [], [], []
    for i, gold in enumerate(truth["records"]):
        row = rows[matched[i]] if i in matched else {}
        if row:
            gold_groups.append([gold["bar_index"], gold["part_id"], gold["staff_id"]])
            predicted_groups.append([row.get("bar_index"), row.get("part_id"), row.get("staff_id")])
        counts["aligned_bars"] += bool(row) and row.get("bar_index") == gold["bar_index"]
        counts["mode_correct"] += row.get("mode") == gold["mode"]
        counts["instrument_correct"] += row.get("instrument") == gold["instrument"]
        predictions.append({
            **{key: gold.get(key) for key in ("id", "mode", "instrument", "tuning", "string_count")},
            "score": truth["id"], "expected": gold.get("sounding_target", gold["target"]),
            "predicted": row.get("target", ""), "matched": i in matched,
            "needs_review": row.get("needs_review", True),
        })
    counts["grouping_exact"] += (len(matched) == len(truth["records"]) == len(rows)
                                and canonical_rows(gold_groups) == canonical_rows(predicted_groups)
                                and counts["aligned_bars"] == len(rows))
    return counts, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", type=Path, required=True,
                        help="score.json files, directories containing them, or native measure JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    scores = documents(args.scores)
    args.output.mkdir(parents=True, exist_ok=False)
    counts, metrics, timings = Counter(), MeasureSequenceMetrics(), []
    with NativeClient(args.endpoint) as client, \
            (args.output / "predictions.jsonl").open("w") as predictions, \
            (args.output / "scores.jsonl").open("w") as reports:
        for index, truth in enumerate(scores):
            started = time.monotonic()
            try:
                project = client.recognize([truth["pdf"]], args.output / "scores" / str(index))
            except (RuntimeError, OSError, httpx.HTTPError) as error:
                project = {"error": str(error)}
            seconds = time.monotonic() - started
            timings.append(seconds)
            found, rows = compare(truth, project)
            counts.update(found)
            for row in rows:
                metrics.update(row["expected"], row["predicted"], row["mode"],
                               tuning=row["tuning"], string_count=row["string_count"],
                               instrument=row["instrument"])
                predictions.write(json.dumps(row, ensure_ascii=False) + "\n")
            report = {"score": truth["id"], "source": truth["source"], "counts": dict(found),
                      "seconds": seconds, "project": project.get("id"), "error": project.get("error")}
            reports.write(json.dumps(report, ensure_ascii=False) + "\n")
            reports.flush()
            predictions.flush()
            print(json.dumps(report, ensure_ascii=False), flush=True)
    result = {"scope": "native_application_full_documents", "counts": dict(counts),
              "score_metrics": readable_metrics(counts), "metrics": metrics.result(),
              "latency": {"sum_seconds": sum(timings), "median_seconds": statistics.median(timings)}}
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
