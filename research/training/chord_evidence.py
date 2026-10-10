"""Train/evaluate the compact diagram reader; checkpoints are candidate artifacts."""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader

from research.data.chord_evidence import ChordEvidenceData
from research.models.chord_evidence import OUTPUTS, ChordEvidenceNetwork


def atomic_checkpoint(weights, path):
    """Replace the previous checkpoint after the new weights finish writing."""
    temporary = path.with_suffix(".tmp")
    save_file(weights, str(temporary))
    temporary.replace(path)


def evaluate(model, loader, device):
    model.eval()
    stats = Counter()
    with torch.inference_mode():
        for images, y in loader:
            y = {k: v.to(device) for k, v in y.items()}
            out = model(images.to(device))
            pred = {name: v.float().argmax(-1) for name, v in zip(OUTPUTS[:5], out[:5])}
            bars = out[5].float().sigmoid() > 0.5
            positive = y["kind"] == 0
            stats["samples"] += len(images)
            stats["kind_correct"] += int((pred["kind"] == y["kind"]).sum())
            stats["positive"] += int(positive.sum())
            stats["false_diagram"] += int(((pred["kind"] == 0) & ~positive).sum())
            stats["negative"] += int((~positive).sum())
            if positive.any():
                stats["base_nll_sum"] += float(
                    nn.functional.cross_entropy(
                        out[2][positive].float(), y["base"][positive], reduction="sum"
                    )
                )
                valid = y["frets"] != -100
                fret_ok = ((pred["frets"] == y["frets"]) | ~valid).all(-1)
                finger_ok = ((pred["fingers"] == y["fingers"]) | ~valid).all(-1)
                edge = torch.arange(11, device=device)[None, None, :] < y["strings"][:, None, None] - 1
                barre_ok = ((bars == y["barres"].bool()) | ~edge).flatten(1).all(-1)
                base_ok = pred["base"] == y["base"]
                count_ok = pred["strings"] == y["strings"]
                kind_ok = pred["kind"] == 0
                for name, ok in [
                    ("fret", fret_ok),
                    ("finger", finger_ok),
                    ("barre", barre_ok),
                    ("base", base_ok),
                    ("strings", count_ok),
                    ("geometry", fret_ok & barre_ok & base_ok & count_ok & kind_ok),
                    ("exact", fret_ok & finger_ok & barre_ok & base_ok & count_ok & kind_ok),
                ]:
                    stats[name + "_correct"] += int((ok & positive).sum())
                stats["barre_positive"] += int((y["barres"].flatten(1).any(-1) & positive).sum())
                stats["barre_positive_correct"] += int(
                    (barre_ok & y["barres"].flatten(1).any(-1) & positive).sum()
                )
    stats["base_nll"] = stats["base_nll_sum"] / max(1, stats["positive"])
    return dict(stats)


def continuous_location_targets(positions, visible, side=64):
    """Bilinear heatmap labels retain subcell glyph-center coordinates."""
    coordinates = positions * side - 0.5
    lower = coordinates.floor().long()
    fraction = coordinates - lower
    targets = positions.new_zeros((len(positions), side * side + 1))
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        x = (lower[:, 0] + dx).clamp(0, side - 1)
        y = (lower[:, 1] + dy).clamp(0, side - 1)
        weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (
            fraction[:, 1] if dy else 1 - fraction[:, 1]
        )
        targets.scatter_add_(1, (y * side + x)[:, None], (weight * visible)[:, None])
    targets[:, -1] = ~visible
    return targets


def training_loss(model, images, targets):
    """Train musical fields, visible character locations and glyph reading jointly."""
    y = targets
    device = images.device
    local_readers = hasattr(model, "base_reader")
    if local_readers:
        regions = None
        visible = y["base_supervised"].bool() & (y["base_position"][:, 0] >= 0)
        regions = (y["base_position"], y["base_extent"], visible)
        out, locations, base_detail, base_locations, base_extent = model.forward_with_auxiliary(
            images, regions
        )
    else:
        out = model(images)
    positive = y["kind"] == 0
    loss = nn.functional.cross_entropy(out[0].float(), y["kind"])
    for i, key in [(1, "strings"), (2, "base")]:
        loss = loss + nn.functional.cross_entropy(out[i].float(), y[key])
    if positive.any():
        loss = loss + 2 * nn.functional.cross_entropy(
            out[3][positive].float().flatten(0, 1), y["frets"][positive].flatten()
        )
        loss = loss + nn.functional.cross_entropy(
            out[4][positive].float().flatten(0, 1), y["fingers"][positive].flatten()
        )
        edge = torch.arange(11, device=device)[None, None, :] < y["strings"][positive, None, None] - 1
        bce = nn.functional.binary_cross_entropy_with_logits(
            out[5][positive].float(),
            y["barres"][positive],
            reduction="none",
            pos_weight=torch.tensor(5.0, device=device),
        )
        loss = loss + 2 * (bce * edge).sum() / edge.expand_as(bce).sum()
    if local_readers:
        loss = loss + 0.5 * nn.functional.cross_entropy(base_detail.float(), y["base"])
        known = y["base_supervised"].bool()
        if known.any():
            position = y["base_position"][known]
            visible = position[:, 0] >= 0
            coordinates = (position * 64).long().clamp(0, 63)
            indexes = torch.where(visible, coordinates[:, 1] * 64 + coordinates[:, 0], 4096)
            location_labels = continuous_location_targets(position, visible)
            loss = loss + 0.25 * nn.functional.cross_entropy(base_locations[known].float(), location_labels)
            if base_extent is not None and visible.any():
                expected = y["base_extent"][known][visible].float()
                loss = loss + nn.functional.smooth_l1_loss(
                    base_extent[known][visible].float().log(), expected.log()
                )
        visible = known & (y["base_text_length"] > 0)
        if visible.any():
            characters = model.base_reader.region_characters(
                images[visible], y["base_position"][visible], y["base_extent"][visible]
            )
            lengths = torch.full((int(visible.sum()),), characters.shape[1], dtype=torch.long, device=device)
            loss = loss + nn.functional.ctc_loss(
                characters.float().log_softmax(-1).transpose(0, 1),
                y["base_tokens"][visible],
                lengths,
                y["base_text_length"][visible],
                zero_infinity=True,
            )
    if local_readers:
        positions = y["finger_positions"]
        visible = positions[..., 0] >= 0
        if visible.any():
            coordinates = (positions[visible] * 64).long().clamp(0, 63)
            indexes = coordinates[:, 1] * 64 + coordinates[:, 0]
            loss = loss + 0.25 * nn.functional.cross_entropy(locations[visible].float(), indexes)
        if locations.shape[-1] == 4097:
            absent = y["fingers"] == 0
            if absent.any():
                loss = loss + 0.25 * nn.functional.cross_entropy(
                    locations[absent].float(),
                    torch.full((int(absent.sum()),), 4096, device=device, dtype=torch.long),
                )
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("database/score"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", type=Path)
    p.add_argument("--reader", choices=("primary", "secondary"), default="primary")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--synthetic", type=int, default=24000)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--font", action="append")
    args = p.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        p.error("epochs/batch-size must be positive and workers nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261003)
    random.seed(20261003)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ChordEvidenceNetwork(pretrained=args.initial is None, local_readers=args.reader == "primary").to(
        device
    )
    if args.initial:
        model.load_state_dict(load_file(str(args.initial)))
    font_options = {"fonts": args.font} if args.font else {}
    train = ChordEvidenceData(args.data, "train", args.synthetic, True, **font_options)
    validation = ChordEvidenceData(args.data, "validation", 2000, **font_options)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val = DataLoader(
        validation,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    metadata = {
        "reader": args.reader,
        "arguments": json.loads(json.dumps(vars(args), default=str)),
        "train": len(train),
        "validation": len(validation),
        "initial_checkpoint": str(args.initial) if args.initial else None,
        "selection_metric": "complete_diagram_minus_false_positive_rate",
    }
    (args.output / "training.json").write_text(json.dumps(metadata, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best, step = (-float("inf"), 0)
    started = time.monotonic()
    total = args.epochs * len(loader)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels in loader:
            step += 1
            factor = min(1.0, step / 60) * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            images = images.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = training_loss(model, images, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        stats = evaluate(model, val, device)
        score = stats["exact_correct"] / max(1, stats["positive"]) - stats["false_diagram"] / max(
            1, stats["negative"]
        )
        report = {
            "epoch": epoch + 1,
            "loss": sum(losses) / max(1, len(losses)),
            "score": score,
            "seconds": time.monotonic() - started,
            **stats,
        }
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        atomic_checkpoint(weights, args.output / "last.safetensors")
        if score > best:
            best = score
            atomic_checkpoint(
                weights, args.output / "chord-evidence.safetensors"
            )
            (args.output / "selected-checkpoint.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
