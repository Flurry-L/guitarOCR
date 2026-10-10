"""Train, validate and export event positions from printed musical queries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from research.common.techniques import FEATURE_DIM, HEIGHT, WIDTH, image_tensor
from research.models.technique_event_grounding import EventGroundingNetwork

MAX_EVENTS = 32


class GroundingDataset(Dataset):
    def __init__(self, manifests):
        self.rows = [
            json.loads(line) for manifest in manifests for line in Path(manifest).read_text().splitlines()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            tensor, transform = image_tensor(image, None)
            positions = [
                transform(
                    [
                        x * image.width,
                        y * image.height,
                        x * image.width,
                        y * image.height,
                    ]
                )[:2]
                for x, y in row["positions"]
            ]
        count = len(positions)
        if not 0 < count <= MAX_EVENTS:
            raise ValueError("Unsupported event count; do not truncate anchors")
        features = torch.zeros(MAX_EVENTS, FEATURE_DIM)
        features[:count] = torch.tensor(row["features"])
        target = torch.zeros(MAX_EVENTS, 2)
        target[:count] = torch.tensor(positions)
        padding = torch.arange(MAX_EVENTS) >= count
        return tensor, features, padding, target


def measure(model, loader, *, device="cuda"):
    model.eval()
    errors = []
    outputs = []
    with torch.inference_mode():
        for images, events, padding, target in loader:
            prediction, _ = model(
                images.to(device), events.to(device), padding.to(device)
            )
            prediction = prediction.cpu()
            errors.append((prediction - target)[~padding])
            outputs.append(prediction)
    error = torch.cat(errors)
    return {
        "events": len(error),
        "mean_abs_xy_normalized": error.abs().mean(0).tolist(),
        "median_abs_x_normalized": float(error[:, 0].abs().median()),
        "x_within_0_02": float((error[:, 0].abs() < 0.02).float().mean()),
        "xy_within_0_03": float((error.abs().max(1).values < 0.03).float().mean()),
    }, torch.cat(outputs)


def run(data, output, steps=1200, max_seconds=600, initial=None, learning_rate=0.001):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261004)
    torch.set_num_threads(2)
    train_data = GroundingDataset([root / "manifest_train.jsonl" for root in data])
    validation_data = GroundingDataset([root / "manifest_validation.jsonl" for root in data])
    train = DataLoader(
        train_data, batch_size=24, shuffle=True, num_workers=4, persistent_workers=True
    )
    valid = DataLoader(
        validation_data, batch_size=24, num_workers=2, persistent_workers=True
    )
    model = EventGroundingNetwork().cuda()
    if initial:
        model.load_state_dict(load_file(str(initial)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    provenance = {
        "data": [str(root.resolve()) for root in data],
        "device": torch.cuda.get_device_name(),
        "steps_limit": steps,
        "seconds_limit": max_seconds,
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_opened": False,
        "gold_at_inference": False,
        "query_scope": "oracle printed music; no reference pixel at inference",
        "initial_checkpoint": str(initial) if initial else None,
        "learning_rate": learning_rate,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, HEIGHT // 8, device="cuda"),
        torch.linspace(0, 1, WIDTH // 8, device="cuda"),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), -1).reshape(1, 1, -1, 2)
    sigma = torch.tensor([0.018, 0.04], device="cuda")
    started, step, best, history = time.monotonic(), 0, float("inf"), []
    while step < steps and time.monotonic() - started < max_seconds:
        for images, events, padding, target in train:
            model.train()
            step += 1
            images, events, padding, target = [
                x.cuda() for x in (images, events, padding, target)
            ]
            optimizer.zero_grad(set_to_none=True)
            positions, scores = model(images, events, padding)
            target_heat = (
                -0.5 * ((grid - target[:, :, None]) / sigma).square().sum(-1)
            ).softmax(-1)
            cross_entropy = (
                -(target_heat * F.log_softmax(scores, -1)).sum(-1)[~padding].mean()
            )
            coordinate_loss = (positions - target).square().sum(-1)[~padding].mean()
            loss = cross_entropy + 15 * coordinate_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if (
                step % 200 == 0
                or step >= steps
                or time.monotonic() - started >= max_seconds
            ):
                result, outputs = measure(model, valid)
                result.update(step=step, seconds=time.monotonic() - started)
                print(json.dumps(result), flush=True)
                history.append(result)
                (output / "validation-history.json").write_text(
                    json.dumps(history, indent=2)
                )
                criterion = sum(result["mean_abs_xy_normalized"])
                if criterion < best:
                    best = criterion
                    save_file(
                        {
                            k: v.detach().cpu().contiguous()
                            for k, v in model.state_dict().items()
                        },
                        str(output / "model.safetensors"),
                    )
                    (output / "selected-validation.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    np.save(output / "validation-positions.npy", outputs.numpy())
            if step >= steps or time.monotonic() - started >= max_seconds:
                break
    print(json.dumps(dict(finished=True, steps=step, best_error=best)), flush=True)


def export(candidate, data):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    model = EventGroundingNetwork().eval()
    checkpoint = candidate / "model.safetensors"
    model.load_state_dict(load_file(str(checkpoint)))
    row = GroundingDataset([root / "manifest_validation.jsonl" for root in data])[0]
    inputs = tuple(value[None] for value in row[:3])
    output = candidate / "event_grounding.onnx"
    config = {
        "version": 1,
        "width": WIDTH,
        "height": HEIGHT,
        "resize": "contain_lanczos",
        "feature_dim": 32,
        "max_events": MAX_EVENTS,
        "mode_order": ["tab", "notation", "both"],
        "input_channels": ["red", "green", "blue", "empty_query_mask"],
        "scope": "event pixel evidence; see data provenance for validated sources and modes",
        "data_provenance": [json.loads((root / "provenance.json").read_text()) for root in data],
        "calibrated": False,
        "query_source": "existing OCR music with no technique features",
        "output_coordinates": "xy normalized to the entire contained 960x256 canvas",
    }
    torch.onnx.export(
        model,
        inputs,
        str(output),
        input_names=["images", "events", "padding"],
        output_names=["positions", "scores"],
        opset_version=17,
        dynamic_axes={
            name: {0: "batch"}
            for name in ("images", "events", "padding", "positions", "scores")
        },
        dynamo=False,
    )
    exported = onnx.load(output)
    onnx.helper.set_model_props(
        exported, {"guitarocr.event_grounding": json.dumps(config)}
    )
    onnx.save(exported, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    runtime = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    errors, timings, attention_errors = [], [], []
    dataset = GroundingDataset([root / "manifest_validation.jsonl" for root in data])
    indices = sorted({0, len(dataset)//3, len(dataset)//2, len(dataset)-1})
    for index in indices:
        values = tuple(value[None] for value in dataset[index][:3])
        with torch.inference_mode():
            expected = model(*values)
        started = time.perf_counter()
        actual = runtime.run(
            None,
            dict(
                zip(
                    ("images", "events", "padding"),
                    (v.numpy() for v in values),
                    strict=True,
                )
            ),
        )
        timings.append(time.perf_counter() - started)
        errors.append(
            [
                float(np.max(np.abs(ref.numpy() - got)))
                for ref, got in zip(expected, actual, strict=True)
            ]
        )
        attention_errors.append(
            float(
                np.max(
                    np.abs(
                        expected[1].softmax(-1).numpy()
                        - torch.from_numpy(actual[1]).softmax(-1).numpy()
                    )
                )
            )
        )
    maximum = np.max(errors, axis=0)
    # Logit magnitude is unconstrained. Compare the normalized spatial
    # distribution and final coordinates, while still recording logit error.
    if maximum[0] > 1e-5 or max(attention_errors) > 1e-5:
        raise AssertionError(f"ONNX export failed CPU parity: {maximum}")
    report = {
        "file": str(output),
        "bytes": output.stat().st_size,
        "maximum_abs_error_positions_scores": maximum.tolist(),
        "maximum_abs_attention_probability_error": max(attention_errors),
        "cpu_threads": 2,
        "mean_seconds_per_measure": float(np.mean(timings)),
        "validation_samples": len(indices),
        "test_opened": False,
        "metadata": config,
    }
    (candidate / "export.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--initial', type=Path)
    train.add_argument('--steps', type=int, default=1200)
    train.add_argument('--max-seconds', type=float, default=600)
    train.add_argument('--learning-rate', type=float, default=0.001)
    conversion = commands.add_parser('export')
    conversion.add_argument('--candidate', type=Path, required=True)
    for command in (train, conversion):
        command.add_argument('--data', type=Path, nargs='+', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = (run if command == 'train' else export)(**args)
    if result is not None:
        print(json.dumps(result, indent=2))
