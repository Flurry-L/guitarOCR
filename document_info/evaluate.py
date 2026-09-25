from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.defaults import MODEL, INFO_ADAPTER

from shared.glm_backend import GlmBackend


def evaluate(dataset: Path, model_path: Path, adapter_path: Path, output: Path) -> dict:
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    backend = GlmBackend(model_path, adapter_path, "cuda")
    correct = {}
    total = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = row["messages"][0]["content"].removeprefix("<image>")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": row["images"][0]},
                    {"type": "text", "text": prompt},
                ],
            }]
            raw, _count = backend.generate(messages, 128)
            raw = raw.strip()
            expected = json.loads(row["messages"][1]["content"])
            try:
                predicted = json.loads(raw)
            except json.JSONDecodeError:
                predicted = {}
            for key, value in expected.items():
                total[key] = total.get(key, 0) + 1
                correct[key] = correct.get(key, 0) + (predicted.get(key) == value)
            handle.write(json.dumps({
                "image": row["images"][0], "expected": expected,
                "predicted": predicted, "raw": raw,
            }, ensure_ascii=False) + "\n")
            handle.flush()
    return {
        key: {"correct": correct[key], "total": count, "accuracy": correct[key] / count}
        for key, count in total.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document information generation")
    parser.add_argument("--dataset", type=Path, default=Path("database/gp8_measure_sequence_v2/datasets/document_info/llamafactory/document_info_validation.json"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.model, args.adapter, args.output), indent=2))


if __name__ == "__main__":
    main()
