"""Train a compact whole-image signature model on isolated musical works."""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from research.data.prompts import parse_signature

from research.common.signature_image import (
    DENOMINATORS,
    SPATIAL_CONFIG,
    full_signature_array,
    validate_signature_config,
)


def signature_source(path):
    path = Path(path)
    # Native crops encode the original work's ID independently of display mode.
    ids = re.findall(r'(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])', str(path))
    if ids:
        return 'native:' + ids[-1]
    if 'engraved_scores' in path.parts:
        index = path.parts.index('engraved_scores')
        return '/'.join(path.parts[index:index + 3])
    raise ValueError(f'Cannot establish signature source identity: {path}')


class Signatures:
    def __init__(self, roots, split, augment=False, *, config=None, limit=0):
        self.rows, self.sources, self.augment = [], [], augment
        self.config = validate_signature_config(SPATIAL_CONFIG if config is None else config)
        seen = {}
        sources = {}
        for root in roots:
            path = root / f'state_{split}.jsonl'
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    if row.get('key_evidence', 'printed') != 'printed':
                        raise ValueError('Signature label semantics differ from the model contract')
                    state = parse_signature(row['messages'][1]['content'])
                    n, d = map(int, state['time'].split('/')) if state['time'] else (0, None)
                    label = [state['key'] + 7 if state['key'] is not None else 15, n, DENOMINATORS.index(d)]
                    image = row['images'][0]
                    if image in seen:
                        if seen[image] != label:
                            raise ValueError(f'Conflicting printed-signature labels: {image}')
                        continue
                    seen[image] = label
                    sources[image] = signature_source(image) if not row.get('source_id') else 'native:' + str(row['source_id'])
                    self.rows.append((image, label))
        if limit and len(self.rows) > limit:
            self.rows = random.Random(20261003).sample(sorted(self.rows), limit)
        self.sources = [sources[path] for path, _ in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        path, label = self.rows[index]
        shape = self.config['input']
        image = torch.from_numpy(full_signature_array(path, self.augment, width=shape['width'], height=shape['height']))
        return image, torch.tensor(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file, save_file
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from research.common.signature_image import SPATIAL_CONFIG, signature_config
    from research.models.auxiliary import make_signature_network

    torch.set_num_threads(1)
    torch.manual_seed(20261003)
    config = signature_config(args.initial) if args.initial else dict(SPATIAL_CONFIG)
    if args.input_width or args.input_height:
        if (
            not args.input_width
            or not args.input_height
            or min(args.input_width, args.input_height) < 32
            or args.input_width % 32
            or args.input_height % 32
        ):
            parser.error("Provide positive whole-image dimensions divisible by 32")
        config = {
            **config,
            "input": {
                **config["input"],
                "width": args.input_width,
                "height": args.input_height,
            },
        }
    model = make_signature_network(config, pretrained=args.initial is None).to(
        args.device
    )
    if args.initial:
        model.load_state_dict(load_file(str(args.initial)))
    training = Signatures(args.data, "train", True, config=config, limit=args.max_train)
    validation = Signatures(args.data, "validation", config=config, limit=args.max_eval)
    # Source isolation is checked on every declared example, not just sampled rows.
    full_sources = {
        split: set(Signatures(args.data, split, config=config).sources)
        for split in ("train", "validation", "test")
    }
    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = full_sources[left] & full_sources[right]
        if overlap:
            raise ValueError(
                f"Signature work leakage {left}/{right}: {sorted(overlap)[:10]}"
            )
    counts = [Counter(label[head] for _, label in training.rows) for head in range(3)]
    # Presence is rare. Moderate sampling exposes all printed signatures while
    # retaining abundant negative crops, without treating repeats as new works.
    groups = Counter((label[0] != 15, label[1] != 0) for _, label in training.rows)
    sample_weights = [
        1.0 / math.sqrt(groups[(label[0] != 15, label[1] != 0)])
        for path, label in training.rows
    ]
    sampler = WeightedRandomSampler(sample_weights, len(training), replacement=True)
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    losses = []
    for head, size in enumerate((16, 33, 8)):
        weights = torch.tensor(
            [1 / max(1, counts[head][i]) ** 0.25 for i in range(size)],
            device=args.device,
        )
        losses.append(nn.CrossEntropyLoss(weight=weights / weights.mean()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    provenance = {
        "initial": str(args.initial)
        if args.initial
        else "torchvision MobileNet_V3_Small_Weights.DEFAULT",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": sys.argv,
        "seed": 20261003,
        "architecture": config["architecture"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "data": [str(p) for p in args.data],
        "source_counts": {s: len(v) for s, v in full_sources.items()},
        "source_overlap": 0,
        "training": len(training),
        "validation": len(validation),
        "device": args.device,
        "augmentation": "whole-image contain; up to 1.2deg rotation, 6% whitespace per side, readable contrast; no cropping",
    }
    (args.output / "training.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)
    best, step, started = -1.0, 0, time.perf_counter()
    total_steps = args.epochs * len(loader)
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for images, labels in loader:
            step += 1
            factor = min(1.0, step / 100) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            images, labels = (
                images.to(args.device, non_blocking=True),
                labels.to(args.device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
            ):
                outputs = model(images)
                loss = sum(
                    criterion(output.float(), labels[:, head])
                    for head, (criterion, output) in enumerate(zip(losses, outputs))
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
            if step % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            step=step,
                            epoch=epoch + 1,
                            loss=loss_sum / min(100, len(loader)),
                            seconds=round(time.perf_counter() - started, 1),
                        )
                    ),
                    flush=True,
                )
                loss_sum = 0.0
        model.eval()
        stats = Counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
            ),
        ):
            for images, labels in val_loader:
                labels = labels.to(args.device, non_blocking=True)
                outputs = model(images.to(args.device, non_blocking=True))
                predicted = torch.stack([v.argmax(-1) for v in outputs], -1)
                absent = (predicted[:, 1] == 0) | (predicted[:, 2] == 0)
                predicted[absent, 1:] = 0
                for name, mask, correct in [
                    ("key", labels[:, 0] != 15, predicted[:, 0] == labels[:, 0]),
                    (
                        "time",
                        labels[:, 1] != 0,
                        (predicted[:, 1:] == labels[:, 1:]).all(-1),
                    ),
                ]:
                    for kind, selected in [("positive", mask), ("negative", ~mask)]:
                        stats[name + "_" + kind] += int(selected.sum())
                        stats[name + "_" + kind + "_correct"] += int(
                            correct[selected].sum()
                        )
        score = (
            sum(
                stats[f"{name}_{kind}_correct"] / max(1, stats[f"{name}_{kind}"])
                for name in ("key", "time")
                for kind in ("positive", "negative")
            )
            / 4
        )
        report = dict(
            epoch=epoch + 1,
            score=score,
            seconds=round(time.perf_counter() - started, 1),
            **stats,
        )
        print(json.dumps(report), flush=True)
        with (args.output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(report) + "\n")
        state = {
            k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
        }
        save_file(state, args.output / "last.safetensors")
        if score > best:
            best = score
            save_file(state, args.output / "state.safetensors")
            (args.output / "best.json").write_text(json.dumps(report, indent=2))
    print("Signature classifier training complete", flush=True)

if __name__ == "__main__":
    main()
