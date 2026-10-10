"""Calibrate a signature candidate and measure isolated-work selective accuracy."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def collect(model, dataset, device, batch_size=128, workers=8):
    import torch
    from torch.utils.data import DataLoader

    outputs = [[], [], []]
    labels = []
    start = time.perf_counter()
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=True
    )
    with torch.inference_mode():
        for images, target in loader:
            predictions = model(images.to(device))
            for collected, predicted in zip(outputs, predictions):
                collected.append(predicted.float().cpu().numpy())
            labels.append(target.numpy())
    return (
        [np.concatenate(v) for v in outputs],
        np.concatenate(labels),
        time.perf_counter() - start,
    )


def fit_temperature(logits, target):
    import torch
    from torch.nn import functional as F

    values = torch.tensor(logits, dtype=torch.float64)
    labels = torch.tensor(target)
    log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=60)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(values / log_temperature.exp().clamp(0.1, 10), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.1, 10).detach())


def probability(logits, temperature):
    value = logits.astype(np.float64) / temperature
    value -= value.max(-1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(-1, keepdims=True)


def metrics(outputs, labels, calibration):
    from research.common.signature_image import predictions_from_logits

    predictions = predictions_from_logits(outputs, calibration)
    stats = Counter()
    raw = [v.argmax(-1) for v in outputs]
    for i, prediction in enumerate(predictions):
        for field, expected, candidate in [
            (
                "key",
                int(labels[i, 0]) - 7 if labels[i, 0] < 15 else None,
                int(raw[0][i]) - 7 if raw[0][i] < 15 else None,
            ),
            (
                "time",
                f"{labels[i, 1]}/{(None, 1, 2, 4, 8, 16, 32, 64)[labels[i, 2]]}"
                if labels[i, 1] and labels[i, 2]
                else None,
                f"{raw[1][i]}/{(None, 1, 2, 4, 8, 16, 32, 64)[raw[2][i]]}"
                if raw[1][i] and raw[2][i]
                else None,
            ),
        ]:
            kind = "positive" if expected is not None else "negative"
            stats[f"{field}_{kind}"] += 1
            stats[f"{field}_{kind}_raw_correct"] += candidate == expected
            evidence = prediction["evidence"][field]
            accepted = evidence["status"] != "unknown"
            stats[f"{field}_{kind}_accepted"] += accepted
            stats[f"{field}_{kind}_accepted_correct"] += (
                accepted and evidence["value"] == expected
            )
            stats[f"{field}_false_change"] += (
                accepted
                and evidence["value"] is not None
                and evidence["value"] != expected
            )
    result = dict(stats)
    for field in ("key", "time"):
        for kind in ("positive", "negative"):
            prefix = f"{field}_{kind}"
            n = stats[prefix]
            a = stats[prefix + "_accepted"]
            result[prefix + "_coverage"] = a / n if n else None
            result[prefix + "_accepted_accuracy"] = (
                stats[prefix + "_accepted_correct"] / a if a else None
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--target-precision", type=float, default=0.995)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file

    from research.common.signature_image import signature_config
    from research.models.auxiliary import make_signature_network
    from research.training.state import Signatures

    torch.set_num_threads(1)
    config = signature_config(args.model)
    model = make_signature_network(config).to(args.device).eval()
    model.load_state_dict(load_file(str(args.model / "state.safetensors")))
    validation = Signatures(args.data, "validation", config=config, limit=args.limit)
    test = Signatures(args.data, "test", config=config, limit=args.limit)
    outputs, labels, seconds = collect(model, validation, args.device)
    temperatures = [fit_temperature(v, labels[:, i]) for i, v in enumerate(outputs)]
    probabilities = [probability(v, t) for v, t in zip(outputs, temperatures)]
    winners = [p.argmax(-1) for p in probabilities]
    confidence = [p.max(-1) for p in probabilities]
    fields = {
        "key": (winners[0] != 15, winners[0] == labels[:, 0], confidence[0]),
        "time": (
            (winners[1] != 0) & (winners[2] != 0),
            (winners[1] == labels[:, 1]) & (winners[2] == labels[:, 2]),
            confidence[1] * confidence[2],
        ),
    }
    thresholds, selection = {}, {}
    for field, (present, correct, conf) in fields.items():
        for kind, mask in [("present", present), ("absent", ~present)]:
            if field == "time":
                mask = mask & ((winners[1] != 0) == (winners[2] != 0))
            indexes = np.flatnonzero(mask)
            indexes = indexes[np.argsort(-conf[indexes], kind="stable")]
            counts = np.arange(1, len(indexes) + 1)
            accuracy = np.cumsum(correct[indexes]) / counts
            eligible = np.flatnonzero(
                (counts >= 30) & (accuracy >= args.target_precision)
            )
            # Include tied probabilities together so the claimed precision is
            # calculated on exactly the observations accepted by the threshold.
            if len(indexes):
                ends = np.r_[conf[indexes][:-1] != conf[indexes][1:], True]
                eligible = np.flatnonzero(
                    (counts >= 30) & (accuracy >= args.target_precision) & ends
                )
            count = int(eligible[-1] + 1) if len(eligible) else 0
            thresholds[f"{field}_{kind}"] = (
                float(conf[indexes[count - 1]]) if count else 1.01
            )
            selection[f"{field}_{kind}"] = {
                "accepted": count,
                "available": len(indexes),
                "accuracy": float(accuracy[count - 1]) if count else None,
            }
    calibration = {
        "version": 1,
        "calibrated": True,
        "temperatures": temperatures,
        "thresholds": thresholds,
        "selection": selection,
        "target_precision": args.target_precision,
        "source": "validation_only",
        "validation_samples": len(validation),
        "validation_sources": len(set(validation.sources)),
    }
    (args.model / "calibration.json").write_text(json.dumps(calibration, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "validation_logits.npz",
        key=outputs[0],
        numerator=outputs[1],
        denominator=outputs[2],
        labels=labels,
    )
    result = {
        "calibration": calibration,
        "validation": metrics(outputs, labels, calibration),
        "validation_seconds": seconds,
    }
    outputs, labels, seconds = collect(model, test, args.device)
    np.savez_compressed(
        args.output / "test_logits.npz",
        key=outputs[0],
        numerator=outputs[1],
        denominator=outputs[2],
        labels=labels,
    )
    result.update(
        test=metrics(outputs, labels, calibration),
        test_seconds=seconds,
        test_sources=len(set(test.sources)),
        source_overlap=len(set(validation.sources) & set(test.sources)),
        parameters=sum(p.numel() for p in model.parameters()),
    )
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    (args.output / "test_selection.jsonl").write_text(
        "".join(
            json.dumps({"image": p, "label": label, "source": source}) + "\n"
            for (p, label), source in zip(test.rows, test.sources)
        )
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
