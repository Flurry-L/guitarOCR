"""Evaluate document metadata on fixed samples using independent GPU workers."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from shared.defaults import INFO_ADAPTER, MODEL


def score(records):
    counts = defaultdict(lambda: [0, 0])
    for row in records:
        mode = row.get("provenance", {}).get("mode", "unknown")
        for key, expected in row["expected"].items():
            for group in ("overall", mode):
                pair = counts[(group, key)]
                pair[0] += int(key in row["predicted"] and row["predicted"][key] == expected)
                pair[1] += 1
    result = {}
    for (group, key), (correct, total) in sorted(counts.items()):
        result.setdefault(group, {})[key] = {"correct": correct, "total": total, "accuracy": correct / total}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        parser.error("GPU indexes must be unique integers")
    rows = json.loads(args.dataset.read_text())
    if len(rows) < len(devices):
        parser.error("Use no more GPUs than samples")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(len(devices)):
        (args.output / f"input-{index}.json").write_text(json.dumps(rows[index::len(devices)], ensure_ascii=False))

    def worker(item):
        index, gpu = item
        with (args.output / f"shard-{index}.log").open("w") as log:
            subprocess.run([
                sys.executable, "-m", "document_info.evaluate",
                "--dataset", str(args.output / f"input-{index}.json"),
                "--model", str(args.model), "--adapter", str(args.adapter),
                "--output", str(args.output / f"shard-{index}.jsonl"),
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
    metrics["artifacts"] = {
        "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest(),
        "adapter_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(args.adapter.glob("adapter*")) if p.is_file()},
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
