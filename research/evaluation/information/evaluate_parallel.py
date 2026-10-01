"""Evaluate document metadata on fixed samples using independent GPU workers."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from research.defaults import INFO_ADAPTER
from research.defaults import MODEL


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
            from research.inference.information.image_ocr import parse_info_response
            from scorelib.chords import chord_key
            from scorelib.chords import normalize_diagram
            expected, predicted = row['expected'], row['predicted']
            confusion[f"{expected.get('kind')} -> {predicted.get('kind')}"] += 1
            parsed = parse_info_response(json.dumps(predicted), 'annotation', row['image'])
            annotation_counts['classified_samples'] += 1
            annotation_counts['kind_correct'] += parsed.get('kind') == expected.get('kind')
            is_chord = parsed.get('kind') in {'chord', 'chord_diagram'}
            annotation_counts['chord_names_predicted'] += is_chord and bool(parsed.get('text'))
            nonpitch = expected.get('kind') not in {'instrument', 'ottava', 'capo'}
            annotation_counts['nonpitch_samples'] += nonpitch
            annotation_counts['raw_false_pitch'] += nonpitch and (predicted.get('semitones') is not None or predicted.get('capo') is not None)
            annotation_counts['applied_false_pitch'] += nonpitch and (parsed.get('semitones') is not None or parsed.get('capo') is not None)
            if expected.get('semitones') is not None or expected.get('capo') is not None:
                annotation_counts['pitch_samples'] += 1
                annotation_counts['pitch_correct'] += all(parsed.get(k) == expected.get(k) for k in ('kind', 'semitones', 'capo'))
            if expected.get('kind') in {'chord', 'chord_diagram'}:
                annotation_counts['chord_names'] += 1
                annotation_counts['chord_names_correct'] += is_chord and chord_key(expected.get('text')) == chord_key(parsed.get('text'))
            actual = normalize_diagram(parsed.get('diagram')) or {}
            annotation_counts['diagrams_predicted'] += bool(actual)
            annotation_counts['diagram_strings_predicted'] += len(actual.get('frets', []))
            annotation_counts['finger_labels_predicted'] += sum(f is not None for f in actual.get('fingers', []))
            annotation_counts['barres_predicted'] += len(actual.get('barres', []))
            if expected.get('diagram'):
                gold = expected['diagram']
                model_diagram = normalize_diagram(predicted.get('diagram')) or {}
                annotation_counts['model_diagram_strings_correct'] += sum(a == b for a, b in zip(gold['frets'], model_diagram.get('frets', [])))
                annotation_counts['model_barres_correct'] += len({tuple(b) for b in gold['barres']} & {tuple(b) for b in model_diagram.get('barres', [])})
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
            ('raw_false_pitch_rate', 'raw_false_pitch', 'nonpitch_samples'),
            ('applied_false_pitch_rate', 'applied_false_pitch', 'nonpitch_samples'),
            ('pitch_instruction_recall', 'pitch_correct', 'pitch_samples'),
            ('chord_name_accuracy', 'chord_names_correct', 'chord_names'),
            ('chord_name_precision', 'chord_names_correct', 'chord_names_predicted'),
            ('diagram_validity', 'diagrams_valid', 'diagrams_expected'),
            ('diagram_base_fret_accuracy', 'diagram_base_fret_correct', 'diagrams_expected'),
            ('diagram_string_count_accuracy', 'diagram_string_count_correct', 'diagrams_expected'),
            ('diagram_string_accuracy', 'diagram_strings_correct', 'diagram_strings'),
            ('model_diagram_string_accuracy', 'model_diagram_strings_correct', 'diagram_strings'),
            ('diagram_string_precision', 'diagram_strings_correct', 'diagram_strings_predicted'),
            ('printed_finger_accuracy', 'finger_labels_correct', 'finger_labels'),
            ('printed_finger_precision', 'finger_labels_correct', 'finger_labels_predicted'),
            ('barre_recall', 'barres_correct', 'barres_expected'),
            ('model_barre_recall', 'model_barres_correct', 'barres_expected'),
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        parser.error("GPU indexes must be unique integers")
    rows = ([json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
            if args.dataset.suffix == '.jsonl' else json.loads(args.dataset.read_text()))
    rows = [row for row in rows if row['messages'][-1]['content'].lstrip().startswith('{')]
    if len(rows) < len(devices):
        parser.error("Use no more GPUs than samples")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(len(devices)):
        (args.output / f"input-{index}.json").write_text(json.dumps(rows[index::len(devices)], ensure_ascii=False))

    def worker(item):
        index, gpu = item
        with (args.output / f"shard-{index}.log").open("w") as log:
            subprocess.run([
                sys.executable, "-m", 'research.evaluation.information.evaluate',
                "--dataset", str(args.output / f"input-{index}.json"),
                "--model", str(args.model), "--adapter", str(args.adapter),
                "--output", str(args.output / f"shard-{index}.jsonl"),
                '--batch-size', str(args.batch_size),
            ], check=True, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1"})
        print(f"Finished GPU {gpu}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, enumerate(devices)))
    records = [json.loads(line) for index in range(len(devices))
               for line in (args.output / f"shard-{index}.jsonl").read_text().splitlines() if line]
    if len(records) != len(rows) or sorted(r["image"] for r in records) != sorted(r["images"][0] for r in rows):
        raise ValueError("Evaluation has missing or duplicate samples")
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    metrics = score(records)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
