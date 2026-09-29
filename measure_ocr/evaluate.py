from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from shared.defaults import MODEL, MEASURE_ADAPTER
from typing import Any

from measure_ocr.prompts import recognition_prompt
from shared.artifacts import write_json
from datagen.sampling import load_measure_rows
from shared.pitch_context import convert_pitch_target
from shared.constraints import validate_measure_target
from measure_ocr.metrics import MeasureSequenceMetrics
from shared.glm_backend import GlmBackend
from shared.m2 import format_history_context
from measure_ocr.recognizer import _rest_fallback_target


def _artifact_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    identity: dict[str, Any] = {"path": str(resolved)}
    if resolved.is_file():
        stat = resolved.stat()
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return identity
    if not resolved.is_dir():
        identity["missing"] = True
        return identity
    artifacts = []
    for name in (
        "adapter_config.json",
        "capabilities.json",
        "adapter_model.safetensors",
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        *(p.name for p in sorted(resolved.glob('model-*-of-*.safetensors'))),
    ):
        candidate = resolved / name
        if candidate.is_file():
            stat = candidate.stat()
            artifacts.append(
                {
                    "name": name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    identity["artifacts"] = artifacts
    return identity


def _run_signature(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    manifest = args.manifest.resolve()
    manifest_stat = manifest.stat()
    value = {
        "schema": 2,
        "fallback_policy": "preserve_previous_bar_feel",
        "prompt_policy": "instrument_and_visible_pitch_context",
        "manifest": {
            "path": str(manifest),
            "size": manifest_stat.st_size,
            "mtime_ns": manifest_stat.st_mtime_ns,
        },
        "selected_ids": [row["id"] for row in rows],
        "model": _artifact_identity(args.model),
        "adapter": _artifact_identity(args.adapter),
        "max_new_tokens": args.max_new_tokens,
        "maximum_attempts": args.maximum_attempts,
        "image_ablation": args.image_ablation,
        "seed": args.seed,
        "context_source": getattr(args, "context_source", "gold"),
        "device": args.device,
    }
    if getattr(args, "batch_size", 1) != 1:
        value["batch_size"] = args.batch_size
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _inference_images(
    rows: list[dict[str, Any]], image_ablation: str
) -> dict[str, str]:
    if image_ablation == "none":
        return {row["id"]: row["image"] for row in rows}
    if image_ablation != "shuffled":
        raise ValueError(f"Unsupported image ablation: {image_ablation}")

    result: dict[str, str] = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        if len(mode_rows) < 2:
            raise ValueError(f"Need at least two {mode} rows for shuffled images")
        for index, row in enumerate(mode_rows):
            replacement = None
            for offset in range(1, len(mode_rows)):
                candidate = mode_rows[(index + offset) % len(mode_rows)]
                if candidate["source_id"] != row["source_id"]:
                    replacement = candidate
                    break
            if replacement is None:
                raise ValueError(
                    f"Need at least two independent {mode} sources for shuffled images"
                )
            result[row["id"]] = replacement["image"]
    return result


def _extract_m2(text: str) -> str:
    value = text.strip()
    start = value.find("M2")
    if start >= 0:
        value = value[start:]
    for marker in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        value = value.partition(marker)[0]
    return value.strip().strip("`").strip()


def _shard_rows(rows: list[dict], index: int, count: int, context_source: str) -> list[dict]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("Require shards >= 1 and 0 <= shard-index < shards")
    if context_source == "predicted":
        # A whole source stays on one worker so predicted context is never lost.
        sources = sorted({row["source_id"] for row in rows})
        assigned = set(sources[index::count])
        return [row for row in rows if row["source_id"] in assigned]
    return rows[index::count]


def run_inference(args: argparse.Namespace) -> None:
    context_source = getattr(args, "context_source", "gold")
    batch_size = getattr(args, "batch_size", 1)
    if batch_size < 1 or context_source == "predicted" and batch_size != 1:
        raise ValueError("Use a positive batch size; predicted context requires batch-size 1")
    if context_source == "predicted" and args.max_samples:
        raise ValueError(
            "Predicted context requires complete sequences; use --max-samples 0 and optionally --max-sources"
        )
    rows = load_measure_rows(args.manifest, args.max_samples, args.seed)
    if context_source == "predicted":
        rows.sort(key=lambda row: (row["source_id"], row["mode"], row["measure_index"]))
        sources = sorted({row["source_id"] for row in rows})
        if getattr(args, "max_sources", 0):
            sources = sources[: args.max_sources]
            rows = [row for row in rows if row["source_id"] in sources]
        indexes = {}
        for row in rows:
            key = (row["source_id"], row["mode"])
            expected_index = indexes.get(key, 0)
            if row["measure_index"] != expected_index:
                raise ValueError(
                    f"Sequence {key} has missing or duplicate measures; expected index {expected_index}"
                )
            indexes[key] = expected_index + 1
    inference_images = _inference_images(rows, args.image_ablation)
    signature = _run_signature(args, rows)
    rows = _shard_rows(rows, getattr(args, "shard_index", 0), getattr(args, "shards", 1), context_source)
    completed: dict[str, dict[str, Any]] = {}
    if args.predictions.is_file() and not args.resume:
        args.predictions.unlink()
    if args.predictions.is_file():
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                if record.get("run_signature") != signature:
                    raise ValueError(
                        "Existing predictions were produced by a different "
                        "manifest/model/adapter selection. Remove the file or "
                        "run without --resume to overwrite it."
                    )
                completed[record["id"]] = record

    backend = None
    batch_predictions = {}
    label_cache: dict[str, dict[str, Any]] = {}
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    histories = {}
    capabilities = args.adapter / "capabilities.json" if args.adapter else None
    supports_pitch = bool(capabilities and capabilities.is_file() and json.loads(capabilities.read_text()).get("pitch_context"))
    written_pitch = bool(capabilities and capabilities.is_file() and json.loads(capabilities.read_text()).get("written_pitch"))

    def track_metadata(row):
        path = str(row.get("label_json") or "")
        if path and path not in label_cache:
            label_cache[path] = json.loads(Path(path).read_text())
        return label_cache.get(path, {}).get("track", {})

    def make_messages(row, context):
        pitch_context = row.get("pitch_context") if supports_pitch else None
        if pitch_context is not None:
            pitch_context = {**pitch_context, "capo": track_metadata(row).get("capo", 0)}
        return [{"role": "user", "content": [
            {"type": "image", "url": inference_images[row["id"]]},
            {"type": "text", "text": recognition_prompt(row["mode"], context, row.get("instrument", "guitar"),
                                                         pitch_context,
                                                         written_pitch=written_pitch)},
        ]}]

    with args.predictions.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            history = histories.setdefault((row["source_id"], row["mode"]), [])
            if row["id"] in completed:
                history.append(completed[row["id"]]["predicted"])
                continue
            context = (
                row.get("previous_context")
                if context_source == "gold"
                else (
                    format_history_context(history, row["mode"])
                    if history
                    else "START"
                )
            )
            backend = backend or GlmBackend(args.model, args.adapter, args.device)
            messages = make_messages(row, context)
            track = track_metadata(row)
            tuning = track.get("tuning_midi_high_to_low")
            string_count = track.get("string_count")
            raw = ""
            predicted = ""
            constraint_errors: list[str] = []
            for attempt in range(1, args.maximum_attempts + 1):
                if batch_size > 1 and attempt == 1:
                    if row["id"] not in batch_predictions:
                        pending = [r for r in rows[index - 1:] if r["id"] not in completed][:batch_size]
                        results = backend.generate_batch(
                            [make_messages(r, r.get("previous_context")) for r in pending],
                            args.max_new_tokens, skip_special_tokens=False,
                        )
                        batch_predictions.update((r["id"], result) for r, result in zip(pending, results, strict=True))
                    raw, _count = batch_predictions.pop(row["id"])
                else:
                    raw, _count = backend.generate(
                        messages, args.max_new_tokens, skip_special_tokens=False,
                    )
                predicted = _extract_m2(raw)
                model_prediction = predicted
                pitch_errors = []
                if written_pitch and supports_pitch and row.get("pitch_context") is not None and row["mode"] == "notation" and row.get("instrument", "guitar") != "drums":
                    try:
                        predicted = convert_pitch_target(predicted, {**row["pitch_context"], "capo": track.get("capo", 0)})
                    except (ValueError, KeyError, TypeError) as error:
                        pitch_errors = [f"pitch_conversion: {error}"]
                _parsed, constraint_errors = validate_measure_target(
                    predicted,
                    row["mode"],
                    tuning=tuning,
                    string_count=string_count,
                )
                constraint_errors.extend(pitch_errors)
                if not constraint_errors or attempt >= args.maximum_attempts:
                    break
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": model_prediction}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Correct the M2 using the same image. Constraint errors: "
                                        + "; ".join(constraint_errors[:8])
                                        + ". Return only one corrected M2 fragment."
                                    ),
                                }
                            ],
                        },
                    ]
                )
            raw_prediction = predicted
            if context_source == "predicted" and constraint_errors:
                predicted = _rest_fallback_target(history)
            history.append(predicted)
            record = {
                "context_source": context_source,
                "previous_context": context,
                "raw_prediction": raw_prediction,
                "needs_review": bool(constraint_errors),
                "run_signature": signature,
                "id": row["id"],
                "source_id": row["source_id"],
                "mode": row["mode"],
                "image": row["image"],
                "inference_image": inference_images[row["id"]],
                "image_ablation": args.image_ablation,
                "expected": row["target"],
                "predicted": predicted,
                "raw": raw,
                "tuning": tuning,
                "string_count": string_count,
                "instrument": row.get("instrument", "guitar"),
                "midi_program": track.get("midi_program"),
                "pitch_context": row.get("pitch_context"),
                "recognition_attempts": attempt,
                "constraint_errors": constraint_errors,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[row["id"]] = record
            print(f"[{index}/{len(rows)}] {row['id']}", flush=True)


def evaluate(predictions: Path, metrics_path: Path | None) -> dict[str, Any]:
    metrics = MeasureSequenceMetrics()
    by_mode: dict[str, MeasureSequenceMetrics] = {}
    raw_metrics = MeasureSequenceMetrics()
    raw_by_mode: dict[str, MeasureSequenceMetrics] = {}
    by_instrument: dict[str, MeasureSequenceMetrics] = {}
    by_strings: dict[str, MeasureSequenceMetrics] = {}
    by_pitched_family: dict[str, MeasureSequenceMetrics] = {}
    by_program: dict[str, MeasureSequenceMetrics] = {}
    by_voices: dict[str, MeasureSequenceMetrics] = {}
    by_corpus: dict[str, MeasureSequenceMetrics] = {}
    by_transpose: dict[str, MeasureSequenceMetrics] = {}
    by_octave_marking: dict[str, MeasureSequenceMetrics] = {}
    conditions, signatures, review = set(), set(), 0
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        conditions.add(row.get("context_source", "gold"))
        signatures.add(row.get("run_signature"))
        review += int(row.get("needs_review", bool(row.get("constraint_errors"))))
        sample = MeasureSequenceMetrics()
        sample.update(
            row["expected"],
            row["predicted"],
            row["mode"],
            tuning=row.get("tuning"),
            string_count=row.get("string_count"),
            instrument=row.get('instrument'),
        )
        metrics.merge(sample)
        by_mode.setdefault(row["mode"], MeasureSequenceMetrics()).merge(sample)
        raw_prediction = row.get("raw_prediction", row["predicted"])
        instrument = row.get("instrument", "guitar")
        groups = [by_instrument.setdefault(instrument, MeasureSequenceMetrics())]
        groups.append(by_voices.setdefault(str(row['expected'].count('||') + 1), MeasureSequenceMetrics()))
        if row.get('corpus'):
            groups.append(by_corpus.setdefault(row['corpus'], MeasureSequenceMetrics()))
        if type(row.get('midi_program')) is int:
            groups.append(by_program.setdefault(str(row['midi_program']), MeasureSequenceMetrics()))
        if instrument in {"guitar", "bass"}:
            key = f"{instrument}/{row.get('string_count')}/{row['mode']}"
            groups.append(by_strings.setdefault(key, MeasureSequenceMetrics()))
        elif instrument == "pitched":
            # GM 0..7 includes acoustic/electric pianos, harpsichord and clavinet.
            program = row.get("midi_program")
            family = ("unknown" if type(program) is not int else
                      "piano_keyboard" if 0 <= program <= 7 else "other_melodic")
            groups.append(by_pitched_family.setdefault(family, MeasureSequenceMetrics()))
        if row.get("pitch_context") and row["mode"] != "tab" and instrument != "drums":
            pitch = row["pitch_context"]
            groups.append(by_transpose.setdefault(str(pitch.get("instrument_transpose")), MeasureSequenceMetrics()))
            marking = "present" if pitch.get("octave_spans") else "absent"
            groups.append(by_octave_marking.setdefault(marking, MeasureSequenceMetrics()))
        for accumulator in groups:
            accumulator.merge(sample)
        raw_sample = sample
        if raw_prediction != row["predicted"]:
            raw_sample = MeasureSequenceMetrics()
            raw_sample.update(
                row["expected"], raw_prediction, row["mode"],
                tuning=row.get("tuning"), string_count=row.get("string_count"),
                instrument=row.get('instrument'),
            )
        for accumulator in (raw_metrics, raw_by_mode.setdefault(row["mode"], MeasureSequenceMetrics())):
            accumulator.merge(raw_sample)
    if len(conditions) > 1 or len(signatures) > 1:
        raise ValueError("Evaluate each model/run and context condition separately")
    result = {
        "context_source": next(iter(conditions), None),
        "run_signature": next(iter(signatures), None),
        "needs_review": review,
        "scope": "ordered_crops"
        if conditions == {"predicted"}
        else "ground_truth_crops_and_context",
        "overall": metrics.result(),
        "by_mode": {mode: value.result() for mode, value in sorted(by_mode.items())},
        "by_instrument": {key: value.result() for key, value in sorted(by_instrument.items())},
        "by_strings": {key: value.result() for key, value in sorted(by_strings.items())},
        "by_pitched_family": {key: value.result() for key, value in sorted(by_pitched_family.items())},
        "by_program": {key: value.result() for key, value in sorted(by_program.items())},
        "by_voices": {key: value.result() for key, value in sorted(by_voices.items())},
        "by_corpus": {key: value.result() for key, value in sorted(by_corpus.items())},
        "by_transpose": {key: value.result() for key, value in sorted(by_transpose.items())},
        "by_octave_marking": {key: value.result() for key, value in sorted(by_octave_marking.items())},
        # Rest placeholders keep the sequence usable but are not valid OCR outputs.
        "raw_overall": raw_metrics.result(),
        "raw_by_mode": {mode: value.result() for mode, value in sorted(raw_by_mode.items())},
    }
    if metrics_path is not None:
        write_json(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer and score GLM-OCR M2 measure sequences."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("database/scores/datasets/measure_ocr/manifests/test.jsonl"),
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=MEASURE_ADAPTER,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/glm_ocr_measure_sequence_test_predictions.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/glm_ocr_measure_sequence_test_metrics.json"),
    )
    parser.add_argument(
        "--context-source", choices=("gold", "predicted"), default="gold"
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="Limit complete sources in predicted-context mode",
    )
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Independent gold-context crops per generation batch")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--image-ablation",
        choices=("none", "shuffled"),
        default="none",
        help="Replace each image with another held-out image of the same layout.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when every existing record has the same run signature.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not args.evaluate_only:
        run_inference(args)
    result = evaluate(args.predictions, args.metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
