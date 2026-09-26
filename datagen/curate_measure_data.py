"""Audit M2 labels and build clean, hard-case and degraded OCR training sets."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from datagen.build_measure_data import _balanced_hardcase_rows
from datagen.scan_augment import _save
from measure_ocr.evaluate import _load_rows
from measure_ocr.prompts import recognition_prompt
from shared.constraints import validate_measure_target


def _chat(row):
    return {
        "messages": [
            {"role": "user", "content": "<image>" + recognition_prompt(row["mode"], row["previous_context"])},
            {"role": "assistant", "content": row["target"]},
        ],
        "images": [row["image"]],
    }


def curate(source: Path, output: Path, seed: int = 20260927, workers: int = 16):
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    llama = output / "llamafactory"
    llama.mkdir(exist_ok=True)
    catalog = {r["source_id"]: r for r in json.loads((source / "source_catalog.json").read_text())["sources"]}
    labels = {sid: json.loads((source / "labels" / f"{sid}.json").read_text())["track"] for sid in catalog}
    reports, info = {}, {}

    def write_chat(name, rows):
        with (llama / f"{name}.json").open("w") as handle:
            handle.write("[")
            for index, row in enumerate(rows):
                if index:
                    handle.write(",")
                json.dump(_chat(row), handle, ensure_ascii=False)
            handle.write("]\n")
        info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt",
                      "columns": {"messages": "messages", "images": "images"},
                      "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}

    groups = {}
    for split in ("train", "validation", "test"):
        path = source / "manifests" / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        valid, rejected = [], []
        invalid_targets = 0
        invalid_contexts = 0
        context_checks = {}
        for row in rows:
            assignment = catalog[row["source_id"]]
            if assignment["split"] != split or row["split"] != split:
                raise ValueError("Source assignment changed")
            row["family"] = assignment["family"]
            track = labels[row["source_id"]]
            _, errors = validate_measure_target(row["target"], row["mode"], tuning=track["tuning_midi_high_to_low"], string_count=track["string_count"])
            context = row["previous_context"]
            context_key = (context, row["mode"], tuple(track["tuning_midi_high_to_low"]))
            if context_key not in context_checks:
                context_checks[context_key] = [] if context == "START" else validate_measure_target(
                    context.replace("C2", "M2", 1), row["mode"],
                    tuning=track["tuning_midi_high_to_low"], string_count=track["string_count"],
                )[1]
            context_errors = context_checks[context_key]
            invalid_targets += bool(errors)
            invalid_contexts += bool(context_errors)
            if errors or context_errors:
                rejected.append({"id": row["id"], "source_id": row["source_id"],
                                 "errors": errors + ["previous_context:" + e for e in context_errors]})
            else:
                valid.append(row)
        # Held-out sequences remain complete for predicted-context evaluation.
        bad_sources = {r["source_id"] for r in rejected} if split != "train" else set()
        valid = [r for r in valid if r["source_id"] not in bad_sources]
        groups[split] = {r["family"] for r in valid}
        destination = output / "manifests" / f"{split}.jsonl"
        manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid)
        # Preserve identity for resumable evaluation when held-out data did not change.
        if not destination.exists() or destination.read_text() != manifest_text:
            destination.write_text(manifest_text)
        (output / f"rejected_{split}.json").write_text(json.dumps(rejected, indent=2) + "\n")
        reports[split] = {"input_samples": len(rows), "samples": len(valid), "families": len(groups[split]),
                          "by_mode": dict(Counter(r["mode"] for r in valid)), "invalid_targets": invalid_targets,
                          "invalid_contexts": invalid_contexts, "rejected_samples": len(rejected),
                          "excluded_sequence_sources": sorted(bad_sources), "manifest_sha256": sha256(destination.read_bytes()).hexdigest()}
        write_chat("measure_" + split, valid)
        if split == "train":
            hardcases, coverage = _balanced_hardcase_rows(valid, seed)
            write_chat("measure_train_hardcases", hardcases)
            augmented = []
            for row in valid:
                if int.from_bytes(sha256(f"{seed}:{row['id']}".encode()).digest()[:4], "big") % 4:
                    continue
                path = output / "images" / row["source_id"] / (row["id"] + "_scan.png")
                augmented.append({**row, "image": str(path.resolve()), "original_image": row["image"]})
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for index, _ in enumerate(pool.map(_save, ((Path(r["original_image"]), Path(r["image"]), f"{seed}:{r['id']}", True) for r in augmented), chunksize=64), 1):
                    if index % 10000 == 0:
                        print(f"Augmented {index}/{len(augmented)} training crops", flush=True)
            write_chat("measure_train_scan", augmented)
            reports[split].update(hardcases=len(hardcases), hardcase_coverage=coverage, augmented_samples=len(augmented))
        else:
            quick = _load_rows(destination, 900, seed)
            write_chat("measure_" + split + "_quick", quick)
        print(split, {k:v for k,v in reports[split].items() if k != "hardcase_coverage"}, flush=True)
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
        raise ValueError("Family leakage across OCR splits")
    (llama / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    reports["source_catalog_sha256"] = sha256((source / "source_catalog.json").read_bytes()).hexdigest()
    reports["seed"] = seed
    reports["augmentation"] = "scan_v1: train only; held-out images unchanged"
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    curate(args.source, args.output, args.seed, args.workers)


if __name__ == "__main__":
    main()
