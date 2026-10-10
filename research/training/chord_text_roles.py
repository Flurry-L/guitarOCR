"""Train bounded contextual text roles with explicit held-out names and fonts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader

from research.data.chord_text_roles import FONT_FAMILIES, SHAPES, SIZE, RoleData
from research.models.auxiliary import ChordTextRoles


def evaluate(model, loader):
    probabilities, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities.extend(model(images.cuda()).softmax(-1)[:, 1].cpu().tolist())
            targets.extend(labels.tolist())
    return np.asarray(probabilities), np.asarray(targets)


def metrics(probabilities, targets, threshold):
    accepted = probabilities >= threshold
    return {
        "total": len(targets),
        "correct": int(((probabilities >= 0.5) == targets).sum()),
        "name_accepted": int(accepted.sum()),
        "name_correct": int((accepted & (targets == 1)).sum()),
        "name_total": int((targets == 1).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(71403)
    model = ChordTextRoles().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loaders = {
        split: DataLoader(
            RoleData(split, count),
            batch_size=64 if split == "train" else 128,
            shuffle=split == "train",
            num_workers=4,
            persistent_workers=True,
        )
        for split, count in [("train", 12000), ("validation", 1000), ("test", 1000)]
    }
    started = time.monotonic()
    step = 0
    while step < args.steps:
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(
                    {
                        "step": step,
                        "loss": float(loss),
                        "seconds": time.monotonic() - started,
                    },
                    flush=True,
                )
            if step >= args.steps:
                break
    model.eval()
    val_prob, val_targets = evaluate(model, loaders["validation"])
    # Threshold selection uses validation only. Test and user images cannot
    # choose the checkpoint, stopping step, threshold or any training sample.
    choices = []
    for threshold in np.arange(0.5, 1.0, 0.005):
        score = metrics(val_prob, val_targets, threshold)
        if (
            score["name_accepted"]
            and score["name_correct"] / score["name_accepted"] >= 0.995
        ):
            choices.append((score["name_correct"], -float(threshold)))
    threshold = -max(choices)[1] if choices else 1.0
    test_prob, test_targets = evaluate(model, loaders["test"])
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": step,
        "seconds": time.monotonic() - started,
        "threshold": threshold,
        "seed": 71403,
        "validation": metrics(val_prob, val_targets, threshold),
        "test": metrics(test_prob, test_targets, threshold),
        "font_families": FONT_FAMILIES,
        "shape_names": {
            split: [shape[0] for shape in shapes] for split, shapes in SHAPES.items()
        },
        "split": "disjoint chord names, voicings, fonts, and random seeds; authored pitch classes and shared fingers validated",
    }
    for split, prob, target in [
        ("validation", val_prob, val_targets),
        ("test", test_prob, test_targets),
    ]:
        np.savez(args.output / f"{split}.npz", probabilities=prob, targets=target)
    model.cpu()
    torch.save(model.state_dict(), args.output / "model.pt")
    contract = {
        "version": 1,
        "input_size": SIZE,
        "resize": "contain_lanczos_center_white",
        "normalization": "rgb_minus1_plus1_query_mask",
        "labels": ["other", "chord_name"],
        "name_threshold": threshold,
    }
    path = args.output / "chord_text_roles.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 4, SIZE, SIZE),
        str(path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.helper.set_model_props(
        graph, {"guitarocr.chord_text_roles": json.dumps(contract)}
    )
    onnx.save(graph, path)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
