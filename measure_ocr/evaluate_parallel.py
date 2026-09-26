"""Evaluate identical selected samples on several GPUs, then score one run."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from measure_ocr.evaluate import _load_rows, evaluate
from shared.defaults import MEASURE_ADAPTER, MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=MEASURE_ADAPTER)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--context-source", choices=("gold", "predicted"), default="gold")
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--maximum-attempts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if len(devices) != len(set(devices)) or any(not d.isdigit() for d in devices):
        parser.error("GPU indexes must be unique integers")
    if args.batch_size < 1 or args.context_source == "predicted" and args.batch_size != 1:
        parser.error("Use a positive batch size; predicted context requires batch-size 1")
    args.output.mkdir(parents=True, exist_ok=True)

    def worker(item):
        index, gpu = item
        command = [sys.executable, "-m", "measure_ocr.evaluate"]
        for name in ("manifest", "model", "adapter", "context_source", "max_samples", "max_sources", "max_new_tokens", "maximum_attempts", "batch_size", "seed"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend([
            "--shards", str(len(devices)), "--shard-index", str(index),
            "--predictions", str(args.output / f"shard-{index}.jsonl"),
            "--metrics", str(args.output / f"shard-{index}.metrics.json"),
        ])
        if args.resume:
            command.append("--resume")
        with (args.output / f"shard-{index}.log").open("a" if args.resume else "w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1"})
        print(f"Finished GPU {gpu}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, enumerate(devices)))
    rows = _load_rows(args.manifest, args.max_samples, args.seed)
    if args.context_source == "predicted" and args.max_sources:
        selected = set(sorted({r["source_id"] for r in rows})[:args.max_sources])
        rows = [r for r in rows if r["source_id"] in selected]
    expected = {row["id"] for row in rows}
    records = [json.loads(line) for index in range(len(devices))
               for line in (args.output / f"shard-{index}.jsonl").read_text().splitlines() if line]
    if len(records) != len(expected) or {r["id"] for r in records} != expected:
        raise ValueError("Sharded evaluation has missing or duplicate samples")
    predictions = args.output / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(records, key=lambda r: r["id"])))
    result = evaluate(predictions, args.output / "metrics.json")
    result["artifacts"] = {
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "adapter_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(args.adapter.glob("adapter*")) if p.is_file()},
        "selected_ids_sha256": sha256("\n".join(sorted(expected)).encode()).hexdigest(),
        "gpus": devices,
    }
    result["inference"] = {name: getattr(args, name) for name in (
        "context_source", "max_samples", "max_sources", "max_new_tokens",
        "maximum_attempts", "batch_size", "seed",
    )}
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"overall": result["overall"], "context_source": args.context_source}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
