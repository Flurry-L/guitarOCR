"""Evaluate model tasks on reference crops, without raster answers.

This measures the semantic OCR model and its declared task adapter. It does not
include layout detection or the production auxiliary evidence/relationship heads.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from research.common.ocr import INFORMATION_TASKS
from research.data.pack_score import samples
from research.inference.vllm import VllmBackend


def compare(expected, predicted):
    counts = Counter(samples=1)
    for key, value in expected.items():
        counts[key + '_total'] += 1
        counts[key + '_correct'] += key in predicted and type(predicted[key]) is type(value) and predicted[key] == value
    counts['exact'] += expected == predicted
    return counts


def character_edits(reference, predicted):
    previous = list(range(len(predicted) + 1))
    for i, letter in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (letter != actual)))
        previous = current
    return previous[-1]


def score(records):
    counts = defaultdict(lambda: [0, 0])
    text_counts = defaultdict(Counter)
    annotation_counts = defaultdict(int)
    confusion = Counter()
    for row in records:
        mode = row.get("provenance", {}).get("mode", "unknown")
        groups = ["overall", mode]
        if 'title' in row['expected']:
            visible = (row['expected'].get('title') or '') + (row['expected'].get('artist') or '')
            language = 'zh' if any('\u4e00' <= c <= '\u9fff' for c in visible) else 'other'
            source = 'header/rendered' if 'header_images' in Path(row['image']).parts else 'header/native'
            text_groups = ('overall', 'language/' + language, source, source + '/' + language)
            groups.extend(text_groups[1:])
            for key in ('title', 'artist', 'tuning_name'):
                expected = row['expected'].get(key) or ''
                actual = row['predicted'].get(key)
                actual = actual if isinstance(actual, str) else ''
                edits = character_edits(expected, actual)
                for group in text_groups:
                    text_counts[group + '/' + key].update(expected_characters=len(expected),
                        predicted_characters=len(actual), character_edits=edits)
        if "instrument" in row["expected"]:
            groups.append("instrument/" + str(row["expected"]["instrument"]))
        if "kind" in row["expected"]:
            groups.append("instruction/" + str(row["expected"]["kind"]))
            from scorelib.chords import chord_key, normalize_diagram
            expected, predicted = row['expected'], row['predicted']
            confusion[f"{expected.get('kind')} -> {predicted.get('kind')}"] += 1
            annotation_counts['classified_samples'] += 1
            annotation_counts['kind_correct'] += predicted.get('kind') == expected.get('kind')
            is_chord = predicted.get('kind') in ('chord', 'chord_diagram')
            annotation_counts['chord_names_predicted'] += is_chord and bool(predicted.get('text'))
            nonpitch = expected.get('kind') not in {'instrument', 'ottava', 'capo'}
            annotation_counts['nonpitch_samples'] += nonpitch
            annotation_counts['false_pitch'] += nonpitch and (predicted.get('semitones') is not None or predicted.get('capo') is not None)
            if expected.get('semitones') is not None or expected.get('capo') is not None:
                annotation_counts['pitch_samples'] += 1
                annotation_counts['pitch_correct'] += all(
                    k in predicted and type(predicted[k]) is type(expected.get(k)) and predicted[k] == expected.get(k)
                    for k in ('kind', 'semitones', 'capo'))
            if expected.get('kind') in {'chord', 'chord_diagram'}:
                annotation_counts['chord_names'] += 1
                annotation_counts['chord_names_correct'] += (is_chord and isinstance(predicted.get('text'), str)
                    and chord_key(expected.get('text')) == chord_key(predicted['text']))
            actual = normalize_diagram(predicted.get('diagram')) or {}
            annotation_counts['diagrams_predicted'] += bool(actual)
            annotation_counts['diagram_strings_predicted'] += len(actual.get('frets', []))
            annotation_counts['finger_labels_predicted'] += sum(f is not None for f in actual.get('fingers', []))
            annotation_counts['barres_predicted'] += len(actual.get('barres', []))
            if expected.get('diagram'):
                gold = expected['diagram']
                annotation_counts['diagrams_expected'] += 1
                annotation_counts['diagrams_valid'] += bool(actual)
                annotation_counts['diagram_base_fret_correct'] += gold['base_fret'] == actual.get('base_fret')
                annotation_counts['diagram_string_count_correct'] += len(gold['frets']) == len(actual.get('frets', []))
                annotation_counts['diagram_strings'] += len(gold['frets'])
                annotation_counts['diagram_strings_correct'] += sum(a == b for a, b in zip(gold['frets'], actual.get('frets', [])))
                annotation_counts['finger_labels'] += sum(f is not None for f in gold['fingers'])
                annotation_counts['finger_labels_correct'] += sum(a is not None and a == b for a, b in zip(gold['fingers'], actual.get('fingers', [])))
                a, b = {tuple(v) for v in gold['barres']}, {tuple(v) for v in actual.get('barres', [])}
                annotation_counts['barres_expected'] += len(a)
                annotation_counts['barres_correct'] += len(a & b)
        if "clef" in row["expected"]:
            groups.append("clef/" + str(row["expected"]["clef"]))
        for key, expected in row["expected"].items():
            for group in groups:
                pair = counts[(group, key)]
                pair[0] += int(key in row["predicted"] and type(row["predicted"][key]) is type(expected) and row["predicted"][key] == expected)
                pair[1] += 1
    result = {}
    for (group, key), (correct, total) in sorted(counts.items()):
        result.setdefault(group, {})[key] = {"correct": correct, "total": total, "accuracy": correct / total}
    result['annotation_counts'] = dict(annotation_counts)
    result['annotation_metrics'] = {
        metric: annotation_counts[numerator] / annotation_counts[denominator] if annotation_counts[denominator] else None
        for metric, numerator, denominator in (
            ('kind_accuracy', 'kind_correct', 'classified_samples'),
            ('false_pitch_rate', 'false_pitch', 'nonpitch_samples'),
            ('pitch_instruction_recall', 'pitch_correct', 'pitch_samples'),
            ('chord_name_accuracy', 'chord_names_correct', 'chord_names'),
            ('chord_name_precision', 'chord_names_correct', 'chord_names_predicted'),
            ('diagram_validity', 'diagrams_valid', 'diagrams_expected'),
            ('diagram_base_fret_accuracy', 'diagram_base_fret_correct', 'diagrams_expected'),
            ('diagram_string_count_accuracy', 'diagram_string_count_correct', 'diagrams_expected'),
            ('diagram_string_accuracy', 'diagram_strings_correct', 'diagram_strings'),
            ('diagram_string_precision', 'diagram_strings_correct', 'diagram_strings_predicted'),
            ('printed_finger_accuracy', 'finger_labels_correct', 'finger_labels'),
            ('printed_finger_precision', 'finger_labels_correct', 'finger_labels_predicted'),
            ('barre_recall', 'barres_correct', 'barres_expected'),
            ('barre_precision', 'barres_correct', 'barres_predicted'),
        )}
    result['annotation_kind_confusion'] = dict(confusion)
    result['text_metrics'] = {
        name: {**value, 'character_error_rate': value['character_edits'] / value['expected_characters']
               if value['expected_characters'] else None}
        for name, value in text_counts.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    kinds = {
        'header': 'Read only visible song title',
        'tempo': 'Read the printed quarter-note tempo.',
        'staff': 'Read the first staff and its instrument label.',
        'clef': 'Read the visible clef.',
        'annotation': 'Classify and read the visible score annotation.',
    }
    tasks = []
    for path in args.dataset:
        for index, row in enumerate(samples(path)):
            if not row['messages'][-1]['content'].lstrip().startswith('{'):
                continue
            prompt = row['messages'][0]['content'].replace('<image>', '')
            kind = next((kind for kind, prefix in kinds.items() if prompt.startswith(prefix)), None)
            if kind is None:
                raise ValueError(f'Unknown information prompt: {path}:{index + 1}')
            image = Path(row['images'][0])
            image = image if image.is_absolute() else path.resolve().parent / image
            tasks.append({'id': str(path) + ':' + str(index), 'corpus': str(path.parent), 'kind': kind,
                          'image': str(image), 'prompt': prompt,
                          'provenance': row.get('provenance', {}),
                          'expected': json.loads(row['messages'][-1]['content'])})
    if not tasks:
        raise ValueError('No information samples to evaluate')
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / 'predictions.jsonl'
    records, metrics = [], defaultdict(Counter)
    loading = time.perf_counter()
    engine = VllmBackend(args.model, args.device, options={'max_model_len': 8192, 'max_num_seqs': args.batch_size,
                        'kv_cache_memory_bytes': 4 * 1024 ** 3})
    started = time.perf_counter()
    try:
        with destination.open('w') as output:
            for kind in kinds:
                subset = [row for row in tasks if row['kind'] == kind]
                for offset in range(0, len(subset), args.batch_size):
                    batch = subset[offset:offset + args.batch_size]
                    messages = [[{'role': 'user', 'content': [{'type': 'image', 'url': row['image']},
                                 {'type': 'text', 'text': row['prompt']}]}] for row in batch]
                    task = kind if kind in INFORMATION_TASKS else None
                    results = engine.generate_batch(messages, 512, task=task)
                    for row, (raw, tokens) in zip(batch, results, strict=True):
                        error = None
                        try:
                            predicted = json.loads(raw)
                        except ValueError:
                            predicted = {}
                        if not isinstance(predicted, dict):
                            predicted = {}
                        if not predicted:
                            error = 'Expected a JSON object'
                        value = {**row, 'predicted': predicted,
                                 'raw': raw, 'tokens': tokens, 'error': error, 'inference_task': task,
                                 'task_adapter_applied': engine.task_adapter_applied(task)}
                        sample = compare(row['expected'], predicted)
                        for group in (kind, row['corpus'] + '/' + kind):
                            metrics[group].update(sample)
                        records.append(value)
                        output.write(json.dumps(value, ensure_ascii=False) + '\n')
                    output.flush()
                    print(json.dumps({'completed': len(records), 'total': len(tasks), 'kind': kind,
                                      'seconds': time.perf_counter() - started}), flush=True)
    finally:
        engine.close()
        (args.output / 'runtime.json').write_text(json.dumps({
            'engine_startup_seconds': started - loading,
            'scope': 'Semantic OCR on reference crops; excludes page detection and auxiliary heads.',
            'worker_seconds': time.perf_counter() - started,
        }, indent=2))
    metrics['fields'] = score(records)
    (args.output / 'metrics.json').write_text(json.dumps(dict(metrics), ensure_ascii=False, indent=2))
    print(json.dumps(dict(metrics), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
