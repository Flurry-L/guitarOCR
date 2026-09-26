"""Expand a two-class DocLayout checkpoint to three measure types plus tempo."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from shared.layout_labels import TYPED_CATEGORIES


def initialize(source: Path, output: Path) -> dict:
    import paddle

    if output.exists():
        raise FileExistsError(output)
    paddle.set_device("cpu")
    state = paddle.load(str(source))
    # Paddle linear weights are [input_features, output_classes]. Copy the
    # learned measure projection into each new class and preserve tempo.
    axes = {
        "transformer.score_head.weight": 1,
        "transformer.score_head.bias": 0,
        "transformer.denoising_class_embed.weight": 0,
    }
    indices = paddle.to_tensor([0, 0, 0, 1], dtype="int64")
    shapes = {}
    for name, axis in axes.items():
        tensor = state[name]
        if tensor.shape[axis] != 2:
            raise ValueError(f"Expected a two-class checkpoint: {name} {tensor.shape}")
        state[name] = paddle.index_select(tensor, indices, axis=axis)
        shapes[name] = {"before": list(tensor.shape), "after": list(state[name].shape)}
    output.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(state, str(output))
    record = {
        "source": str(source.resolve()),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "output_sha256": sha256(output.read_bytes()).hexdigest(),
        "old_labels": ["measure", "tempo_region"],
        "new_labels": list(TYPED_CATEGORIES),
        "expanded_tensors": shapes,
        "unchanged_tensors": len(state) - len(axes),
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
