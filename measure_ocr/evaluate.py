from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from shared.defaults import MODEL, MEASURE_ADAPTER
from typing import Any

from measure_ocr.prompts import recognition_prompt
from shared.constraints import validate_measure_target
from measure_ocr.metrics import MeasureSequenceMetrics
from shared.glm_backend import GlmBackend
from shared.m2 import format_history_context
from measure_ocr.recognizer import _rest_fallback_target


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_rows(manifest: Path, maximum: int | None, seed: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]

    def stable(value):
        return sha256(f"{seed}:{value}".encode("utf-8")).digest()

    if not maximum:
        return sorted(rows, key=lambda row: stable(row["id"]))

    modes = sorted({row["mode"] for row in rows})
    quotas = {mode: maximum // len(modes) for mode in modes}
    for mode in modes[: maximum % len(modes)]:
        quotas[mode] += 1
    selected = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_selected: dict[str, dict[str, Any]] = {}

        # Reserve a small, source-diverse slice for every represented semantic
        # class before filling the quota. A purely uniform sample can contain
        # almost no bends, harmonics or multi-voice measures and therefore
        # cannot support the per-technique release gate.
        tags = sorted(
            {str(tag) for row in mode_rows for tag in row.get("semantic_tags", [])}
        )
        coverage_per_tag = min(20, max(4, quotas[mode] // 50))
        for tag in tags:
            candidates = sorted(
                (row for row in mode_rows if tag in row.get("semantic_tags", [])),
                key=lambda row: stable(f"technique:{mode}:{tag}:{row['id']}"),
            )
            tag_sources: set[str] = set()
            for row in candidates:
                if (
                    len(mode_selected) >= quotas[mode]
                    or len(tag_sources) >= coverage_per_tag
                ):
                    break
                if row["id"] in mode_selected or row["source_id"] in tag_sources:
                    continue
                mode_selected[row["id"]] = row
                tag_sources.add(row["source_id"])

        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in mode_rows:
            if row["id"] not in mode_selected:
                by_source.setdefault(row["source_id"], []).append(row)
        for values in by_source.values():
            values.sort(key=lambda row: stable(row["id"]))
        source_ids = sorted(by_source, key=lambda value: stable(f"{mode}:{value}"))
        while len(mode_selected) < quotas[mode]:
            progressed = False
            for source_id in source_ids:
                if not by_source[source_id]:
                    continue
                row = by_source[source_id].pop()
                mode_selected[row["id"]] = row
                progressed = True
                if len(mode_selected) >= quotas[mode]:
                    break
            if not progressed:
                break
        selected.extend(mode_selected.values())
    return sorted(selected, key=lambda row: stable(row["id"]))


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
        "adapter_model.safetensors",
        "config.json",
        "model.safetensors",
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
    rows = _load_rows(args.manifest, args.max_samples, args.seed)
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

    def make_messages(row, context):
        return [{"role": "user", "content": [
            {"type": "image", "url": inference_images[row["id"]]},
            {"type": "text", "text": recognition_prompt(row["mode"], context)},
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
            label_path = str(row.get("label_json") or "")
            if label_path and label_path not in label_cache:
                label_cache[label_path] = json.loads(
                    Path(label_path).read_text(encoding="utf-8")
                )
            track = label_cache.get(label_path, {}).get("track", {})
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
                _parsed, constraint_errors = validate_measure_target(
                    predicted,
                    row["mode"],
                    tuning=tuning,
                    string_count=string_count,
                )
                if not constraint_errors or attempt >= args.maximum_attempts:
                    break
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": predicted}],
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
                "recognition_attempts": attempt,
                "constraint_errors": constraint_errors,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[row["id"]] = record
            print(f"[{index}/{len(rows)}] {row['id']}", flush=True)


def evaluate(predictions: Path, metrics_path: Path) -> dict[str, Any]:
    metrics = MeasureSequenceMetrics()
    by_mode: dict[str, MeasureSequenceMetrics] = {}
    raw_metrics = MeasureSequenceMetrics()
    raw_by_mode: dict[str, MeasureSequenceMetrics] = {}
    conditions, signatures, review = set(), set(), 0
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        conditions.add(row.get("context_source", "gold"))
        signatures.add(row.get("run_signature"))
        review += int(row.get("needs_review", bool(row.get("constraint_errors"))))
        metrics.update(
            row["expected"],
            row["predicted"],
            row["mode"],
            tuning=row.get("tuning"),
            string_count=row.get("string_count"),
        )
        by_mode.setdefault(row["mode"], MeasureSequenceMetrics()).update(
            row["expected"],
            row["predicted"],
            row["mode"],
            tuning=row.get("tuning"),
            string_count=row.get("string_count"),
        )
        raw_prediction = row.get("raw_prediction", row["predicted"])
        for accumulator in (raw_metrics, raw_by_mode.setdefault(row["mode"], MeasureSequenceMetrics())):
            accumulator.update(
                row["expected"], raw_prediction, row["mode"],
                tuning=row.get("tuning"), string_count=row.get("string_count"),
            )
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
        # Rest placeholders keep the sequence usable but are not valid OCR outputs.
        "raw_overall": raw_metrics.result(),
        "raw_by_mode": {mode: value.result() for mode, value in sorted(raw_by_mode.items())},
    }
    _write_json(metrics_path, result)
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
