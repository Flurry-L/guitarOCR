from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from shared.defaults import MODEL, INFO_ADAPTER

from shared.glm_backend import create_backend


def evaluate(dataset: Path, model_path: Path, adapter_path: Path, output: Path, batch_size: int = 8) -> dict:
    rows = ([json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
            if dataset.suffix == '.jsonl' else json.loads(dataset.read_text(encoding="utf-8")))
    rows = [row for row in rows if row['messages'][-1]['content'].lstrip().startswith('{')]
    started = time.perf_counter()
    backend = create_backend(model_path, adapter_path, "cuda")
    loaded = time.perf_counter()
    correct = {}
    total = {}
    generated_tokens = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    def messages(row):
        prompt = row["messages"][0]["content"].removeprefix("<image>")
        return [{
            "role": "user",
            "content": [
                {"type": "image", "url": row["images"][0]},
                {"type": "text", "text": prompt},
            ],
        }]
    def generated():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            limit = 2048 if any('score structure' in row['messages'][0]['content'] for row in batch) else 512
            yield from zip(batch, backend.generate_batch([messages(row) for row in batch], limit), strict=True)

    with output.open("w", encoding="utf-8") as handle:
        for row, (raw, _count) in generated():
            generated_tokens += _count
            raw = raw.strip()
            expected = json.loads(row["messages"][1]["content"])
            try:
                predicted = json.loads(raw)
            except json.JSONDecodeError:
                predicted = {}
            if not isinstance(predicted, dict):
                predicted = {}
            for key, value in expected.items():
                total[key] = total.get(key, 0) + 1
                correct[key] = correct.get(key, 0) + (key in predicted and type(predicted[key]) is type(value) and predicted[key] == value)
            handle.write(json.dumps({
                "image": row["images"][0], "expected": expected,
                "predicted": predicted, "raw": raw,
                "provenance": row.get("provenance", {}),
            }, ensure_ascii=False) + "\n")
            handle.flush()
    output.with_suffix('.runtime.json').write_text(json.dumps({
        'samples':len(rows), 'generated_tokens':generated_tokens,
        'startup_seconds':loaded-started, 'inference_seconds':time.perf_counter()-loaded,
        'speculation':getattr(backend, 'metrics', []),
    }, indent=2))
    if hasattr(backend, 'close'):
        backend.close()
    return {
        key: {"correct": correct[key], "total": count, "accuracy": correct[key] / count}
        for key, count in total.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document information generation")
    parser.add_argument("--dataset", type=Path, default=Path("database/headers/datasets/info_mixed/document_info_validation.json"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.model, args.adapter, args.output, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
