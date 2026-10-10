"""Train explicit technique span/label geometry and event membership jointly."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from research.common.techniques import (
    HEIGHT,
    MAX_EVENTS,
    WIDTH,
    metrics,
    pair_points,
    prepare_pair,
    prepared,
)
from research.common.techniques import TECHNIQUE_LABELS as LABELS
from research.models.technique_span_relations import TechniqueSpanRelationNetwork


class SpanDataset(Dataset):
    def __init__(self, manifests, training=False):
        self.rows = [json.loads(line) for path in manifests for line in Path(path).read_text().splitlines()]
        self.training = training
        for row in self.rows:
            if not 0 < len(row['events']) <= MAX_EVENTS:
                raise ValueError(f'Event query count must be 1..{MAX_EVENTS}: {row["id"]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image, transform = prepare_pair(row)
        points = np.zeros((MAX_EVENTS, 2), dtype=np.float32)
        gold_points = np.zeros_like(points)
        geometry = np.zeros((MAX_EVENTS, 2, 8), dtype=np.float32)
        present = np.zeros((MAX_EVENTS, 2), dtype=np.float32)
        effect_labels = np.full((MAX_EVENTS, 2), -1., dtype=np.float32)
        labels = np.full(MAX_EVENTS, -100, dtype=np.int64)
        for i, event in enumerate(row['events']):
            gold_points[i] = pair_points(transform, event['pixel'])
            points[i] = pair_points(transform, event.get('query_pixel', event['pixel']))
            for effect in range(2):
                geometry[i, effect] = pair_points(transform, np.asarray(event['geometry'][effect]).reshape(4, 2)).reshape(8)
                if event.get('anchor_previous', [False, False])[effect]:
                    previous_transform = prepared(row['previous_system_image'])[1]
                    anchor = previous_transform.to_normalized(np.asarray(event['geometry'][effect][4:]).reshape(2, 2))
                    anchor[..., 1] /= 2
                    geometry[i, effect, 4:] = anchor.reshape(4)
            present[i] = event['span_present']
            labels[i] = LABELS.index(event['label'])
            fields = event.get('visible_effect_labels')
            for k, kind in enumerate(('pm', 'let')):
                value = (fields.get(kind) if fields is not None else
                         event['label'] in (kind, 'both') if event['label'] != 'unknown' else None)
                effect_labels[i, k] = -1. if value is None else float(value)
        padding = labels == -100
        if self.training:
            if random.random() < .5:
                # Modest independent white margins teach the detector-crop
                # coordinate frame. Every pixel and label moves by the same
                # proportional affine map; no image axis is stretched.
                original_geometry = geometry.copy()
                for half in range(2):
                    factor = random.uniform(.9, 1.)
                    width, height = round(WIDTH*factor), round(HEIGHT*factor)
                    ox, oy = random.randint(0, WIDTH-width), random.randint(0, HEIGHT-height)
                    source = torch.from_numpy(image[:3, half*HEIGHT:(half+1)*HEIGHT].copy())[None]
                    resized = F.interpolate(source, (height,width), mode='bilinear', align_corners=False)[0].numpy()
                    image[:3, half*HEIGHT:(half+1)*HEIGHT] = 1
                    image[:3, half*HEIGHT+oy:half*HEIGHT+oy+height, ox:ox+width] = resized
                    def transformed(value):
                        value = value.copy()
                        value[...,0] = value[...,0]*width/WIDTH + ox/WIDTH
                        value[...,1] = ((value[...,1]*2-half)*height+oy)/(HEIGHT*2)+half/2
                        return value
                    if half == 1:
                        points, gold_points = transformed(points), transformed(gold_points)
                        geometry = transformed(original_geometry.reshape(MAX_EVENTS,2,4,2)).reshape(MAX_EVENTS,2,8)
                    else:
                        previous_geometry = transformed(original_geometry.reshape(MAX_EVENTS,2,4,2)).reshape(MAX_EVENTS,2,8)
                for i,event in enumerate(row['events']):
                    for effect,previous in enumerate(event.get('anchor_previous',[False,False])):
                        if previous:
                            geometry[i,effect,4:] = previous_geometry[i,effect,4:]
            # The deployed locator is imperfect. Supervised visual event
            # pointers recover the printed note from its approximate query.
            if random.random() < .3:
                points[~padding] = gold_points[~padding]
            points[~padding] += np.random.normal(0, [3/960, 2/512], (int((~padding).sum()), 2))
            # Mild photometric variation preserves every musical coordinate.
            if random.random() < .3:
                image[:3] = np.clip(image[:3] * random.uniform(.75, 1.05) + random.uniform(-.03, .06), -1, 1)
            if random.random() < .15:
                # Text-erasure is an explicit unknown control, never a false
                # negative example of the original readable score.
                effect = random.randrange(2)
                erased = set()
                for i in range(len(row['events'])):
                    if present[i, effect]:
                        box = tuple(geometry[i, effect, 4:])
                        if box not in erased:
                            erased.add(box)
                            x0, y0, x1, y1 = np.round(np.array(box) * [WIDTH, HEIGHT * 2, WIDTH, HEIGHT * 2]).astype(int)
                            image[:3, max(0, y0 - 2):min(HEIGHT * 2, y1 + 3), max(0, x0 - 2):min(WIDTH, x1 + 3)] = 1
                        present[i, effect] = 0
                        if labels[i] in (effect + 1, 3):
                            labels[i] = 4
                            effect_labels[i, effect] = -1.
        return tuple(torch.from_numpy(a) for a in (
            image, points, padding, labels, geometry, present, gold_points, effect_labels))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    logits, labels, errors, oracle = [], [], [], []
    for cpu_batch in loader:
        images, points, padding, label, target, present, gold_points, _ = [value.cuda() for value in cpu_batch]
        prediction, geometry, existence, *_ = model(images, points, padding)
        valid = ~padding
        logits.append(prediction[valid].cpu().numpy())
        labels.append(label[valid].cpu().numpy())
        oracle_coverage = model.coverage_from_geometry(images, gold_points, target).sigmoid() * present
        pm, let = oracle_coverage[..., 0], oracle_coverage[..., 1]
        oracle.append(torch.stack(((1-pm)*(1-let), pm*(1-let), let*(1-pm), pm*let, torch.zeros_like(pm)), -1)[valid].cpu().numpy())
        error = (geometry - target).abs() * images.new_tensor([WIDTH, HEIGHT * 2] * 4)
        errors.extend(error[(present.bool() & valid[..., None])].mean(-1).cpu().tolist())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    report = metrics(logits, labels)
    report['span_geometry_mae_pixels'] = float(np.mean(errors))
    report['macro_effect_f1'] = sum(value['f1'] for value in report['per_effect'].values()) / 2
    report['gold_span_membership'] = metrics(np.concatenate(oracle), labels)
    return report, logits, labels


def effect_classification_loss(logits, labels, fields, valid):
    """A hidden let flag must not erase supervision for a visible PM flag."""
    known = fields >= 0
    joint = valid & (known.all(-1) | ~known.any(-1))
    loss = F.nll_loss(logits[joint], labels[joint], reduction='sum') if joint.any() else logits.sum() * 0
    partial = valid[..., None] & known & ~known.all(-1)[..., None]
    if partial.any():
        positive = torch.stack((logits[..., [1, 3]].logsumexp(-1),
                                logits[..., [2, 3]].logsumexp(-1)), -1)
        absent = torch.stack((logits[..., [0, 2]].logsumexp(-1),
                              logits[..., [0, 1]].logsumexp(-1)), -1)
        loss = loss - torch.where(fields > 0, positive, absent)[partial].sum()
    return loss / valid.sum().clamp_min(1)


def run(
    data,
    output,
    steps=2400,
    max_seconds=900,
    initial=None,
    learning_rate=0.0005,
    short_label_weight=3.0,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Technique span training requires a CUDA device")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261008)
    random.seed(20261008)
    np.random.seed(20261008)
    torch.set_num_threads(2)
    train_data, val_data = (
        SpanDataset([root / "manifest_train.jsonl" for root in data], True),
        SpanDataset([root / "manifest_validation.jsonl" for root in data]),
    )
    weights = [
        short_label_weight
        if any(
            (
                e["label"] in (kind, "both")
                and e.get("span_styles", [None, None])[i] == "label_only"
                for e in row["events"]
                for i, kind in enumerate(("pm", "let"))
            )
        )
        else 3.0
        if row.get("intervention")
        else 1.0
        for row in train_data.rows
    ]
    weights = [
        max(weight, float(row.get("sampling_weight", 1.0)))
        for weight, row in zip(weights, train_data.rows, strict=True)
    ]
    if any((not np.isfinite(weight) or weight <= 0 for weight in weights)):
        raise ValueError("Technique sampling weights must be finite and positive")
    train = DataLoader(
        train_data,
        batch_size=6,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
        num_workers=2,
        persistent_workers=True,
    )
    validation = DataLoader(
        val_data, batch_size=6, num_workers=2, persistent_workers=True
    )
    model = (
        TechniqueSpanRelationNetwork.from_state_dict(load_file(str(initial)))
        if initial
        else TechniqueSpanRelationNetwork()
    ).cuda()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    provenance = dict(
        data=[str(root.resolve()) for root in data],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        parameters=sum((p.numel() for p in model.parameters())),
        image=[4, HEIGHT * 2, WIDTH],
        query="class-free actual event pixels; previous/current full systems",
        supervision=[
            "event class",
            "effect-specific visible span endpoints",
            "label rectangle",
            "span existence",
            "event-to-span membership",
        ],
        source_note_owners="unchanged",
        train_systems=len(train_data),
        validation_systems=len(val_data),
        test_opened=False,
        initial=str(initial),
        learning_rate=learning_rate,
        short_label_weight=short_label_weight,
        trainable_parameters=sum(
            (p.numel() for p in model.parameters() if p.requires_grad)
        ),
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    started, step, best, history = (time.monotonic(), 0, -1.0, [])
    while step < steps and time.monotonic() - started < max_seconds:
        for cpu_batch in train:
            model.train()
            step += 1
            images, points, padding, labels, target, present, gold_points, fields = [
                value.cuda() for value in cpu_batch
            ]
            logits, geometry, existence, coverage, pointers, event_pointers, refined = (
                model(images, points, padding)
            )
            valid = ~padding
            observed = present.bool() & valid[..., None]
            positive = fields.clamp_min(0)
            known = observed & (fields >= 0)
            class_loss = effect_classification_loss(logits, labels, fields, valid)
            geometry_loss = (
                F.smooth_l1_loss(geometry[observed], target[observed], beta=0.02)
                if observed.any()
                else geometry.sum() * 0
            )
            if observed.any():
                anchor = geometry[..., 4:][observed]
                gold_anchor = target[..., 4:][observed]
                size = (gold_anchor[:, 2:] - gold_anchor[:, :2]).clamp_min(1 / WIDTH)
                label_box_loss = F.smooth_l1_loss(
                    (anchor - gold_anchor) / size.repeat(1, 2),
                    torch.zeros_like(anchor),
                    beta=0.1,
                )
            else:
                label_box_loss = geometry.sum() * 0
            existence_loss = F.binary_cross_entropy_with_logits(
                existence[valid], present[valid]
            )
            coverage_loss = (
                F.binary_cross_entropy_with_logits(coverage[known], positive[known])
                if known.any()
                else coverage.sum() * 0
            )
            teacher = model.coverage_from_geometry(images, gold_points, target)
            teacher_loss = (
                F.binary_cross_entropy_with_logits(teacher[known], positive[known])
                if known.any()
                else teacher.sum() * 0
            )
            height, width = (HEIGHT * 2 // 8, WIDTH // 8)
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, height, device=images.device),
                torch.linspace(0, 1, width, device=images.device),
                indexing="ij",
            )
            grid = torch.stack((xx, yy), -1).reshape(1, 1, 1, 1, height * width, 2)
            target_points = torch.stack(
                (
                    target[..., :2],
                    target[..., 2:4],
                    (target[..., 4:6] + target[..., 6:]) / 2,
                ),
                -2,
            )
            distance = (
                (
                    (grid - target_points[..., None, :])
                    * images.new_tensor([width, height])
                )
                .square()
                .sum(-1)
            )
            heatmap = (-distance / 2).softmax(-1)
            pointer_loss = (
                -(heatmap * pointers.log_softmax(-1)).sum(-1)[observed].mean()
                if observed.any()
                else pointers.sum() * 0
            )
            event_distance = (
                (
                    (grid.reshape(1, 1, height * width, 2) - gold_points[..., None, :])
                    * images.new_tensor([width, height])
                )
                .square()
                .sum(-1)
            )
            event_heatmap = (-event_distance / 2).softmax(-1)
            event_pointer_loss = (
                -(event_heatmap * event_pointers.log_softmax(-1)).sum(-1)[valid].mean()
            )
            event_location_loss = F.smooth_l1_loss(
                refined[valid], gold_points[valid], beta=0.02
            )
            loss = (
                class_loss
                + 35 * geometry_loss
                + existence_loss
                + coverage_loss
                + pointer_loss
                + event_pointer_loss
                + 35 * event_location_loss
                + max(0.0, 1 - step / 1000) * teacher_loss
                + 3 * label_box_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            step=step,
                            loss=float(loss.detach()),
                            classification=float(class_loss.detach()),
                            geometry=float(geometry_loss.detach()),
                            seconds=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
            if (
                step % 300 == 0
                or step >= steps
                or time.monotonic() - started >= max_seconds
            ):
                report, logits_np, labels_np = evaluate(model, validation)
                report.update(step=step, seconds=time.monotonic() - started)
                history.append(report)
                print(json.dumps(report), flush=True)
                (output / "validation-history.json").write_text(
                    json.dumps(history, indent=2)
                )
                score = (
                    report["macro_effect_f1"]
                    - 0.0002 * report["span_geometry_mae_pixels"]
                )
                if score > best:
                    best = score
                    save_file(
                        {
                            k: v.detach().cpu().contiguous()
                            for k, v in model.state_dict().items()
                        },
                        str(output / "model.safetensors"),
                    )
                    (output / "selected-validation.json").write_text(
                        json.dumps(report, indent=2)
                    )
                    np.savez(
                        output / "validation-logits.npz",
                        logits=logits_np,
                        labels=labels_np,
                    )
            if step >= steps or time.monotonic() - started >= max_seconds:
                break
    print(
        json.dumps(
            dict(finished=True, seconds=time.monotonic() - started, best_score=best)
        ),
        flush=True,
    )



class SpanInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, events, padding):
        return self.model(images, events, padding)[:4]


def export(candidate, data):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = candidate / "model.safetensors"
    weights = load_file(str(checkpoint))
    network = TechniqueSpanRelationNetwork.from_state_dict(weights).eval()
    model = SpanInference(network).eval()
    dataset = SpanDataset([root / "manifest_validation.jsonl" for root in data])
    output = candidate / "event_span_relation.onnx"
    metadata = dict(
        version=1,
        width=960,
        height=512,
        system_height=256,
        max_events=32,
        resize="contain_lanczos",
        labels=LABELS,
        effects=["pm", "let"],
        pair_order=["previous", "current"],
        normalization="RGB/127.5-1; channel4=0",
        event_coordinates="normalized pair canvas; current system in lower half",
        geometry=[
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "label_left",
            "label_top",
            "label_right",
            "label_bottom",
        ],
        absent_previous="white RGB; zero fourth channel",
        scope="visible event relation; no source note ownership",
        decision_model="independent_presence_coverage_v1",
        calibrated=False,
        native_validated=False,
    )
    inputs = ("images", "events", "padding")
    outputs = (
        "log_probabilities",
        "span_geometry",
        "presence_logits",
        "coverage_logits",
    )
    sample = tuple(value[None] for value in dataset[0][:3])
    torch.onnx.export(
        model,
        sample,
        str(output),
        input_names=list(inputs),
        output_names=list(outputs),
        opset_version=17,
        dynamo=False,
    )
    exported = onnx.load(output)
    onnx.helper.set_model_props(
        exported, {"guitarocr.event_span_relation": json.dumps(metadata)}
    )
    onnx.save(exported, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    errors, timings = [], []
    for index in (0, len(dataset) // 3, len(dataset) // 2, len(dataset) - 1):
        sample = tuple(value[None] for value in dataset[index][:3])
        with torch.inference_mode():
            expected = model(*sample)
        start = time.perf_counter()
        actual = session.run(
            None,
            {name: value.numpy() for name, value in zip(inputs, sample, strict=True)},
        )
        timings.append((time.perf_counter() - start) * 1000)
        for reference, value in zip(expected, actual, strict=True):
            # Log probabilities of near-zero classes are sensitive; compare
            # deployed probabilities separately from the geometric outputs.
            reference = reference.numpy()
            if reference.shape[-1] == len(LABELS):
                reference, value = np.exp(reference), np.exp(value)
            np.testing.assert_allclose(reference, value, rtol=2e-3, atol=2e-4)
            errors.append(float(np.max(np.abs(reference - value))))
    report = dict(
        path=str(output),
        bytes=output.stat().st_size,
        parameters=sum(p.numel() for p in model.parameters()),
        maximum_output_error=max(errors),
        cpu_threads=2,
        milliseconds_per_system_pair=timings,
        metadata=metadata,
        test_opened=False,
    )
    (candidate / "export.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--initial', type=Path)
    train.add_argument('--steps', type=int, default=2400)
    train.add_argument('--max-seconds', type=float, default=900)
    train.add_argument('--learning-rate', type=float, default=0.0005)
    train.add_argument('--short-label-weight', type=float, default=3.0)
    conversion = commands.add_parser('export')
    conversion.add_argument('--candidate', type=Path, required=True)
    for command in (train, conversion):
        command.add_argument('--data', type=Path, nargs='+', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = (run if command == 'train' else export)(**args)
    if result is not None:
        print(json.dumps(result, indent=2))
