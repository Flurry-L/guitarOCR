"""Evaluate and calibrate the exported visible-technique models."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image
from scorelib.techniques import EFFECTS

from research.common.techniques import (
    EFFECT_LABELS,
    EFFECT_POLICY,
    HEIGHT,
    MAX_EVENTS,
    WIDTH,
    effect_decisions,
    effect_probabilities,
    event_features,
    located_queries,
    metrics,
    pair_points,
    prepare_image,
    prepare_images,
)
from research.common.techniques import TECHNIQUE_LABELS as LABELS


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def evaluate(manifest, model, output, grounding=None):
    output.mkdir(parents=True, exist_ok=False)
    read_relation = EventSpanRelationReader(model).predict_paths
    locator = EventGroundingReader(grounding) if grounding else None
    rows, predictions, groups = read_rows(manifest), [], defaultdict(lambda: [[], []])
    for index, row in enumerate(rows):
        queries = [{k: e[k] for k in ('measure_index', 'voice', 'onset', 'pixel')} for e in row['events']]
        values = {'gold_pixels': read_relation(row['image'], queries, row.get('previous_system_image'))}
        if locator:
            values['learned_pixels'] = read_relation(row['image'], located_queries(row, locator), row.get('previous_system_image'))
        predictions.append(dict(id=row['id'], **values))
        for condition, actual in values.items():
            strata = ('all', 'mode/' + row['mode'], 'native' if row.get('renderer') == 'guitar_pro' else 'authored',
                      'continuation' if row.get('intervention') else 'original')
            for stratum in strata:
                logits, labels = groups[condition + '/' + stratum]
                logits.extend([e['probabilities'][label] for label in LABELS] for e in actual)
                labels.extend(LABELS.index(e['label']) for e in row['events'])
        if index % 100 == 0:
            print(json.dumps(dict(evaluated=index + 1, cases=len(rows))), flush=True)
    result = {key: metrics(np.asarray(logits), np.asarray(labels)) for key, (logits, labels) in groups.items()}
    result['provenance'] = dict(manifest=str(manifest), model=str(model), runtime='onnx',
                               query='class-free pixel coordinates; reference music only for learned grounding',
                               source_owner_inference=False, actual_ocr=False)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    return result


class EventSpanRelationReader:
    def __init__(self, path, threads=2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        contract = json.loads(metadata.get("guitarocr.event_span_relation", "{}"))
        expected = {
            "version": 1, "width": WIDTH, "height": HEIGHT * 2,
            "system_height": HEIGHT, "max_events": MAX_EVENTS,
            "resize": "contain_lanczos", "pair_order": ["previous", "current"],
            "normalization": "RGB/127.5-1; channel4=0",
            "event_coordinates": "normalized pair canvas; current system in lower half",
            "labels": list(LABELS),
        }
        self.outputs = ["log_probabilities", "span_geometry", "presence_logits", "coverage_logits"]
        if (any(contract.get(k) != v for k, v in expected.items())
                or not {"images", "events", "padding"} <= {x.name for x in self.session.get_inputs()}
                or not set(self.outputs) <= {x.name for x in self.session.get_outputs()}):
            raise ValueError("Unsupported span relation model contract")

    def predict(self, image, events, previous_image=None):
        if len(events) > MAX_EVENTS:
            raise ValueError("Too many event relations; refusing to truncate")
        if not events:
            return []
        pixels = np.asarray([e["pixel"] for e in events], dtype=np.float64)
        if pixels.shape != (len(events), 2) or not np.isfinite(pixels).all():
            raise ValueError("Invalid event pixel")
        inside = ((pixels >= 0).all(-1)
                  & (pixels < np.asarray(image.size)).all(-1))
        tensor, transform = prepare_images(image, previous_image)
        points = np.zeros((1, MAX_EVENTS, 2), dtype=np.float32)
        points[0, :len(events)] = pair_points(transform, pixels)
        padding = np.arange(MAX_EVENTS)[None] >= len(events)
        values = self.session.run(self.outputs, {
            "images": tensor[None], "events": points, "padding": padding,
        })
        shapes = [(1, MAX_EVENTS, 5), (1, MAX_EVENTS, 2, 8),
                  (1, MAX_EVENTS, 2), (1, MAX_EVENTS, 2)]
        if any(value.shape != shape or not np.isfinite(value).all()
               for value, shape in zip(values, shapes, strict=True)):
            raise ValueError("Invalid span relation output")
        logits, geometry, presence, coverage = [v[0, :len(events)] for v in values[:4]]
        probabilities = np.exp(logits - logits.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        presence = np.exp(-np.logaddexp(0, -presence))
        coverage = np.exp(-np.logaddexp(0, -coverage))
        result = [{
            **event, "label": LABELS[int(p.argmax())],
            "probabilities": dict(zip(LABELS, p.astype(float).tolist(), strict=True)),
            "span_evidence": {kind: {
                "geometry_pair_normalized": g[i].astype(float).tolist(),
                "present_probability": float(ex[i]), "coverage_probability": float(cv[i]),
            } for i, kind in enumerate(("pm", "let"))},
            "status": "uncalibrated_event_relation", "scope": "visible_event_only",
        } for event, p, g, ex, cv in zip(events, probabilities, geometry, presence, coverage, strict=True)]
        for index, (row, valid) in enumerate(zip(result, inside, strict=True)):
            if not valid:
                row.update(label='unknown', probabilities=dict(zip(LABELS, (0., 0., 0., 0., 1.), strict=True)),
                           reason='event_pixel_outside_current_system')
                row.pop('span_evidence', None)
        return result

    def predict_paths(self, path, events, previous_path=None):
        """Evaluate the deployed reader from immutable research image paths."""
        with ExitStack() as stack:
            current = stack.enter_context(Image.open(path)).convert('RGB')
            previous = (stack.enter_context(Image.open(previous_path)).convert('RGB')
                        if previous_path else None)
            return self.predict(current, events, previous)


def _session(path, key, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    metadata = json.loads(session.get_modelmeta().custom_metadata_map[key])
    if (
        metadata["width"] != WIDTH
        or metadata["height"] != HEIGHT
        or metadata["resize"] != "contain_lanczos"
    ):
        raise ValueError("Unsupported technique model image contract")
    return session, metadata


class EventGroundingReader:
    def __init__(self, path: Path, threads=2):
        self.session, self.metadata = _session(
            path, "guitarocr.event_grounding", threads
        )

    def predict(self, image, measure, mode):
        features, anchors = event_features(measure, mode)
        if len(anchors) > MAX_EVENTS:
            raise ValueError("Too many event queries; refusing to truncate identities")
        if not anchors:
            return []
        if len(set(map(tuple, anchors))) != len(anchors):
            raise ValueError("Duplicate event identities")
        tensor, transform = prepare_image(image)
        queries = np.zeros((1, MAX_EVENTS, features.shape[1]), dtype=np.float32)
        queries[0, : len(features)] = features
        padding = np.arange(MAX_EVENTS)[None] >= len(anchors)
        positions, scores = self.session.run(
            None, {"images": tensor[None], "events": queries, "padding": padding}
        )
        points = transform.to_pixels(positions[0, : len(anchors)])
        logits = scores[0, : len(anchors)]
        probability = np.exp(logits - logits.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        entropy = -(probability * np.log(probability.clip(1e-30))).sum(-1)
        return [
            {
                "voice": int(voice),
                "onset": int(onset),
                "pixel": point.tolist(),
                "normalized_canvas_xy": normalized.tolist(),
                "attention_entropy": float(ent),
                "inside_image": bool(
                    0 <= point[0] < image.width and 0 <= point[1] < image.height
                ),
                "status": "uncalibrated_pixel_evidence",
            }
            for (voice, onset), point, normalized, ent in zip(
                anchors, points, positions[0, : len(anchors)], entropy, strict=True
            )
        ]


GRID = (.8, .9, .95, .98, .99, .995, .997, .999, .9995, .9999)


def field_truth(label, fields, effect):
    if isinstance(fields, dict):
        return fields.get(effect)
    return effect in EFFECTS[label] if label in EFFECTS else None


def apply_events(events, thresholds):
    result = []
    for event in events:
        decisions = effect_decisions(event.get('relation', {}).get('probabilities', {}), thresholds)
        baseline = EFFECTS.get(event.get('baseline_label'), frozenset())
        resolved = frozenset(effect for effect, value in decisions.items()
                             if value is True or value is None and effect in baseline)
        label = EFFECT_LABELS[resolved] if any(v is not None for v in decisions.values()) else 'unknown'
        result.append({**event, 'effect_decisions': decisions,
                       'relation': {'label': label, 'probabilities': {label: 1.}}})
    return result


def summarize_effects(events, thresholds):
    counts = Counter()
    for event in apply_events(events, thresholds):
        for effect, decision in event['effect_decisions'].items():
            truth = field_truth(event['expected_label'], event.get('expected_effects'), effect)
            counts['fields'] += 1
            counts['unobservable_fields'] += truth is None
            accepted = event['ocr_present'] and decision is not None
            counts['accepted_fields'] += accepted
            if not accepted:
                continue
            correct = event['expected_present'] and truth is not None and decision == truth
            baseline = effect in EFFECTS.get(event['baseline_label'], frozenset())
            counts['accepted_correct_fields'] += correct
            counts['accepted_wrong_or_unobservable_fields'] += not correct
            counts['corrected_fields'] += correct and decision != baseline
            counts['regressed_fields'] += truth is not None and baseline == truth and decision != truth
    return dict(counts)


def summarize(events):
    """Score the identity union, plus effect confusion and exact change counts."""
    counts = Counter()
    confusion = {effect: Counter() for effect in ('pm', 'let')}
    changes = {effect: Counter() for effect in ('pm', 'let')}
    for event in events:
        expected = event.get('expected_label')
        previous = event.get('baseline_label')
        label = event.get('relation', {}).get('label', 'unknown')
        accepted = event['ocr_present'] and label in EFFECTS
        actual = (label if accepted else previous) if event['ocr_present'] else None
        present = event['expected_present'] and event['ocr_present']
        previous_correct = present and previous == expected
        actual_correct = present and actual == expected
        counts['union_events'] += 1
        counts['expected_events'] += event['expected_present']
        counts['ocr_events'] += event['ocr_present']
        counts['matched_identities'] += present
        counts['missing_ocr_events'] += event['expected_present'] and not event['ocr_present']
        counts['extra_ocr_events'] += event['ocr_present'] and not event['expected_present']
        if present and expected == 'unknown':
            # An unobservable playback bit is not a visual negative. Preserve
            # its identity and acceptance count, but do not fabricate truth.
            counts['unobservable_reference_events'] += 1
            counts['accepted_events'] += accepted
            counts['accepted_unscored_events'] += accepted
            continue
        counts['joint_exact_correct'] += actual_correct
        counts['joint_errors'] += not actual_correct
        counts['matched_label_errors'] += present and actual != expected
        counts['accepted_events'] += accepted
        counts['accepted_positive_events'] += accepted and actual in ('pm', 'let', 'both')
        counts['accepted_source_agreements'] += accepted and actual_correct
        counts['accepted_source_disagreements'] += accepted and not actual_correct
        changed = event['ocr_present'] and actual != previous
        counts['accepted_unchanged_events'] += accepted and not changed
        counts['accepted_unchanged_source_agreements'] += accepted and not changed and actual_correct
        counts['accepted_unchanged_source_disagreements'] += accepted and not changed and not actual_correct
        counts['changed_events'] += changed
        counts['corrected_events'] += changed and actual_correct and not previous_correct
        counts['regressed_events'] += changed and previous_correct and not actual_correct
        counts['changed_still_wrong_events'] += changed and not previous_correct and not actual_correct
        truth_effects = EFFECTS.get(expected, frozenset())
        before_effects = EFFECTS.get(previous, frozenset())
        after_effects = EFFECTS.get(actual, frozenset())
        for effect in confusion:
            truth, before, after = effect in truth_effects, effect in before_effects, effect in after_effects
            confusion[effect]['tp' if truth and after else 'fn' if truth else 'fp' if after else 'tn'] += 1
            if after and not before:
                changes[effect]['added'] += 1
                changes[effect]['correct_additions' if truth else 'incorrect_additions'] += 1
            if before and not after:
                changes[effect]['dropped'] += 1
                changes[effect]['correct_drops' if not truth else 'incorrect_drops'] += 1
    result = dict(counts)
    result['scored_union_events'] = counts['union_events'] - counts['unobservable_reference_events']
    result['joint_exact_accuracy'] = counts['joint_exact_correct'] / max(1, result['scored_union_events'])
    result['effects'] = {}
    for effect, values in confusion.items():
        tp, fp, fn = (values[key] for key in ('tp', 'fp', 'fn'))
        result['effects'][effect] = {**{key: values[key] for key in ('tp', 'fp', 'fn', 'tn')},
                                     'precision': tp / max(1, tp + fp),
                                     'recall': tp / max(1, tp + fn),
                                     'f1': 2 * tp / max(1, 2 * tp + fp + fn),
                                     'changes': {key: changes[effect][key] for key in (
                                         'added', 'dropped', 'correct_additions', 'incorrect_additions',
                                         'correct_drops', 'incorrect_drops')}}
    return result


def calibrate(manifest, predictions, actual, output):
    if output.exists():
        raise FileExistsError('Choose a new output path for calibration')
    rows, values = read_rows(manifest), read_rows(predictions)
    expected = {r['id']: r for r in rows}
    if set(expected) != {r['id'] for r in values}:
        raise ValueError('Calibration case identity mismatch')
    pool = [(e, effect_probabilities(p.get('probabilities', {}))) for r in values
            for e,p in zip(expected[r['id']]['events'], r['learned_pixels'], strict=True)]
    cases = read_rows(actual)
    events = [e for r in cases for e in r['events']]
    scores = [(e, effect_probabilities(e.get('relation', {}).get('probabilities', {}))) for e in events]
    thresholds = {effect: {} for effect in ('pm', 'let')}
    curves = {}
    for effect in thresholds:
        for state in ('present', 'absent'):
            positive = state == 'present'
            curve = []
            for threshold in GRID:
                chosen = [truth for truth, p in pool if p.get(effect, {}).get(state, 0) >= threshold]
                errors = sum(field_truth(truth['label'], truth.get('visible_effect_labels'), effect) != positive for truth in chosen)
                actual_chosen = [e for e,p in scores if e['ocr_present'] and p.get(effect, {}).get(state,0) >= threshold]
                actual_errors = sum(not e['expected_present']
                                    or field_truth(e['expected_label'], e.get('expected_effects'), effect) != positive for e in actual_chosen)
                corrected = sum(field_truth(e['expected_label'], e.get('expected_effects'), effect) == positive
                                and (effect in EFFECTS.get(e['baseline_label'], frozenset())) != positive for e in actual_chosen)
                curve.append(dict(threshold=threshold, accepted=len(chosen), errors=errors,
                    precision=(len(chosen)-errors)/max(1,len(chosen)), actual_accepted=len(actual_chosen),
                    actual_errors=actual_errors, corrected_effects=corrected))
            thresholds[effect][state] = next((c['threshold'] for c in curve
                if c['accepted'] and c['precision'] >= .995 and not c['actual_errors']), None)
            curves[f'{effect}_{state}'] = curve
    selection = 'lowest predeclared threshold with >=99.5% full validation effect precision and zero accepted actual OCR field errors; unobservable truth is not counted as correct'
    analysis = dict(selection=selection, grid=GRID, curves=curves,
                    actual_effects=summarize_effects(events, thresholds),
                    actual_ocr={mode: summarize(apply_events([e for c in cases if mode == 'all' or c['mode']==mode
                                                              for e in c['events']], thresholds))
                                for mode in ('all','tab','notation','both')})
    analysis_path = output.with_name(output.stem + '-analysis.json')
    analysis_path.write_text(json.dumps(analysis, indent=2))
    config = dict(protocol='visible_technique_relation',
        supported_modes=['tab','both','notation'],
        application_policy=EFFECT_POLICY,
        input=dict(require_full_system_context=True, require_actual_ocr_event_queries=True,
                   previous_system='trusted same-part/staff order; white when unavailable'),
        acceptance=dict(rewrite_note_effects=False), effect_thresholds=thresholds,
        calibration=dict(split='validation',analysis=str(analysis_path),selection=selection,
                         actual_ocr_cases=len(cases),complete_events=len(pool)),
        independent_test_opened=False, scope='visible effect overlay only; unknown fields retain baseline; source note ownership unchanged')
    output.write_text(json.dumps(config,indent=2))
    return dict(calibration=str(output),thresholds=thresholds,actual_ocr=analysis['actual_ocr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--model', type=Path, required=True)
    evaluation.add_argument('--grounding', type=Path)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--predictions', type=Path, required=True)
    calibration.add_argument('--actual', type=Path, required=True)
    for command in (evaluation, calibration):
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    print(json.dumps((evaluate if command == 'evaluate' else calibrate)(**args), indent=2))
