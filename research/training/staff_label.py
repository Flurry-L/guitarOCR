"""Train a complete-staff name detector without using held-out document images."""

import argparse
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from research.models.auxiliary import StaffLabelNetwork
from research.common.staff_label import WIDTH, HEIGHT, STRIDE, ANGLES, staff_label_image

@lru_cache(maxsize=256)
def source_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def read_sources(root, split):
    context = Path(root) / f"{split}-labels.jsonl"
    if context.exists():
        return [json.loads(line) for line in context.read_text().splitlines()]
    rows = [
        json.loads(line)
        for line in (Path(root) / f"{split}.jsonl").read_text().splitlines()
    ]
    records = []
    for row in rows:
        prov = row["provenance"]
        if prov["task"] != "staff":
            continue
        geometry = json.loads(Path(prov["geometry"]).read_text())
        bounds = geometry["staff_crop"]["bounds"]
        labels = geometry["instrument_labels"]
        labelled = prov["labelled"]
        if bool(labels) != labelled:
            raise ValueError("Label geometry disagrees with visibility annotation")
        boxes = [
            [
                r["bbox"][0] - bounds[0],
                r["bbox"][1] - bounds[1],
                r["bbox"][2],
                r["bbox"][3],
            ]
            for r in labels
        ]
        if len(boxes) > 1:
            raise ValueError("Expected one complete instrument label per staff sample")
        records.append(
            {
                "image": row["images"][0],
                "family": prov["family"],
                "box": boxes[0] if boxes else None,
                "angle": 270,
                "name": json.loads(row["messages"][-1]["content"])["name"],
                "source": prov,
            }
        )
    return records


class StaffLabelData(Dataset):
    def __init__(self, records, count, *, training):
        self.records = records
        self.count = count
        self.training = training
        self.fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        self.names = sorted({r['name'] for r in records if r.get('name')})

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = random.Random(902171 + index + (0 if self.training else 1000000))
        record = self.records[index % len(self.records)]
        image = source_image(record["image"]).copy()
        box = list(record["box"]) if record["box"] else None
        angle = record["angle"]
        if self.training and box is not None and rng.random() < 0.5:
            x, y, w, h = box
            ImageDraw.Draw(image).rectangle(
                (x - 2, y - 2, x + w + 2, y + h + 2), fill="white"
            )
            font = ImageFont.truetype(rng.choice(self.fonts), rng.randint(10, 26))
            name = rng.choice(self.names)
            bx, by, br, bb = font.getbbox(name)
            label = Image.new("RGB", (br - bx + 6, bb - by + 6), "white")
            ImageDraw.Draw(label).text((3 - bx, 3 - by), name, font=font, fill="black")
            label = label.rotate(-angle, expand=True)
            x = max(
                0, min(image.width - label.width, round(x + w / 2 - label.width / 2))
            )
            y = max(
                0, min(image.height - label.height, round(y + h / 2 - label.height / 2))
            )
            image.paste(label, (x, y))
            box = [x, y, label.width, label.height]
        # Source geometry is training annotation only. Relocate complete names
        # into varied margins so an edge or fixed strip cannot determine role.
        if self.training and rng.random() < 0.6:
            pad_x, pad_y = rng.randint(25, 120), rng.randint(35, 110)
            canvas = Image.new(
                "RGB", (image.width + 2 * pad_x, image.height + 2 * pad_y), "white"
            )
            patch = None
            if box is not None:
                x, y, w, h = box
                bounds = (
                    max(0, math.floor(x) - 2),
                    max(0, math.floor(y) - 2),
                    min(image.width, math.ceil(x + w) + 2),
                    min(image.height, math.ceil(y + h) + 2),
                )
                patch = image.crop(bounds).rotate(angle, expand=True, fillcolor="white")
                ImageDraw.Draw(image).rectangle(bounds, fill="white")
            canvas.paste(image, (pad_x, pad_y))
            if patch is not None:
                rotation = rng.choice(ANGLES)
                patch = patch.rotate(rotation, expand=True, fillcolor="white")
                # Rendering augmentation only; inference does not see side IDs.
                side = rng.randrange(4)
                if side < 2:
                    factor = min(
                        1, (pad_y - 6) / patch.height, (canvas.width - 6) / patch.width
                    )
                else:
                    factor = min(
                        1, (pad_x - 6) / patch.width, (canvas.height - 6) / patch.height
                    )
                patch = patch.resize(
                    (
                        max(1, round(patch.width * factor)),
                        max(1, round(patch.height * factor)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                if side < 2:
                    x = rng.randint(3, max(3, canvas.width - patch.width - 3))
                    y = 3 if side == 0 else canvas.height - patch.height - 3
                else:
                    x = 3 if side == 2 else canvas.width - patch.width - 3
                    y = rng.randint(3, max(3, canvas.height - patch.height - 3))
                canvas.paste(patch, (x, y))
                box = [x, y, patch.width, patch.height]
                angle = (-rotation) % 360
            image = canvas
        if self.training and rng.random() < 0.5:
            draw = ImageDraw.Draw(image)
            for _ in range(rng.randint(1, 4)):
                size = rng.randint(9, 22)
                font = ImageFont.truetype(rng.choice(self.fonts), size)
                word = rng.choice(
                    [
                        "Intro",
                        "Chorus",
                        "Bridge",
                        "Verse",
                        "Allegro",
                        "120",
                        "mf",
                        "D.S. al Coda",
                        "sim.",
                        "P.M.",
                        "let ring",
                        "A7",
                        "Cmaj7",
                    ]
                )
                x, y = (
                    rng.randint(5, max(5, image.width - 140)),
                    rng.randint(2, max(2, image.height - size - 2)),
                )
                if box and not (
                    x + 140 < box[0]
                    or x > box[0] + box[2]
                    or y + size < box[1]
                    or y > box[1] + box[3]
                ):
                    continue
                draw.text((x, y), word, font=font, fill="black")
        if self.training and rng.random() < 0.15:
            image = image.rotate(180)
            if box:
                x, y, w, h = box
                box = [image.width - x - w, image.height - y - h, w, h]
                angle = (angle - 180) % 360
        pixels, transform = staff_label_image(image)
        heatmap = np.zeros((HEIGHT // STRIDE, WIDTH // STRIDE), np.float32)
        normalized = np.zeros(4, np.float32)
        center = np.zeros(2, np.int64)
        if box:
            sx, sy, left, top = transform
            x, y, w, h = box
            x = x * sx + left
            y = y * sy + top
            w *= sx
            h *= sy
            cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
            ix, iy = (
                min(heatmap.shape[1] - 1, max(0, int(cx))),
                min(heatmap.shape[0] - 1, max(0, int(cy))),
            )
            yy, xx = np.ogrid[: heatmap.shape[0], : heatmap.shape[1]]
            heatmap = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / 2).astype(np.float32)
            normalized = np.array(
                [
                    (ix * STRIDE - x) / WIDTH,
                    (iy * STRIDE - y) / HEIGHT,
                    (x + w - ix * STRIDE) / WIDTH,
                    (y + h - iy * STRIDE) / HEIGHT,
                ],
                np.float32,
            )
            normalized = np.maximum(normalized, 0)
            center[:] = [iy, ix]
        return (
            pixels,
            heatmap,
            normalized,
            center,
            np.int64(ANGLES.index(angle)),
            np.float32(box is not None),
        )


def focal(logits, target):
    probability = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = target.eq(1).float()
    negative = target.lt(1).float() * (1 - target).pow(4)
    loss = -(
        positive * (1 - probability).pow(2) * probability.log()
        + negative * probability.pow(2) * (1 - probability).log()
    ).sum()
    return loss / positive.sum().clamp_min(1)


def evaluate(model, loader):
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, heatmap, boxes, centers, angles, present in loader:
            scores, bounds, directions = model(images.cuda())
            b = scores.shape[0]
            flat = scores[:, 0].sigmoid().flatten(1)
            confidence, index = flat.max(1)
            yy, xx = index // (WIDTH // STRIDE), index % (WIDTH // STRIDE)
            ar = torch.arange(b, device="cuda")
            pred_box = bounds[ar, :, yy, xx].cpu().numpy()
            pred_angles = directions[ar, :, yy, xx].argmax(-1).cpu().numpy()
            for i in range(b):
                py, px = int(yy[i]), int(xx[i])
                gy, gx = centers[i].tolist()
                pb = pred_box[i]
                gb = boxes[i].numpy()
                pred = [
                    px * STRIDE - pb[0] * WIDTH,
                    py * STRIDE - pb[1] * HEIGHT,
                    px * STRIDE + pb[2] * WIDTH,
                    py * STRIDE + pb[3] * HEIGHT,
                ]
                truth = [
                    gx * STRIDE - gb[0] * WIDTH,
                    gy * STRIDE - gb[1] * HEIGHT,
                    gx * STRIDE + gb[2] * WIDTH,
                    gy * STRIDE + gb[3] * HEIGHT,
                ]
                intersection = max(
                    0, min(pred[2], truth[2]) - max(pred[0], truth[0])
                ) * max(0, min(pred[3], truth[3]) - max(pred[1], truth[1]))
                pa = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
                ga = max(0, truth[2] - truth[0]) * max(0, truth[3] - truth[1])
                iou = intersection / max(1e-9, pa + ga - intersection)
                rows.append(
                    {
                        "score": float(confidence[i]),
                        "present": bool(present[i]),
                        "iou": float(iou) if present[i] else None,
                        "coverage": float(intersection / max(ga, 1e-9))
                        if present[i]
                        else None,
                        "orientation_correct": bool(pred_angles[i] == int(angles[i]))
                        if present[i]
                        else None,
                        "bbox": list(map(float, pred)),
                        "target_bbox": list(map(float, truth)) if present[i] else None,
                    }
                )
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    def metric(threshold):
        accepted = [r for r in rows if r["score"] >= threshold]
        good = sum(
            r["present"] and r["iou"] >= 0.5 and r["orientation_correct"]
            for r in accepted
        )
        return {
            "threshold": threshold,
            "accepted": len(accepted),
            "correct": good,
            "present": sum(r["present"] for r in rows),
            "false_label_on_absent": sum(not r["present"] for r in accepted),
            "coverage_90_correct_orientation": sum(
                r["present"] and r["coverage"] >= 0.9 and r["orientation_correct"]
                for r in accepted
            ),
            "coverage_99_correct_orientation": sum(
                r["present"]
                and r["coverage"] >= 0.99
                and r["iou"] >= 0.5
                and r["orientation_correct"]
                for r in accepted
            ),
        }

    choices = [metric(t) for t in thresholds]
    eligible = [m for m in choices if m["false_label_on_absent"] == 0]
    selection = max(
        eligible or choices,
        key=lambda m: (
            m["coverage_99_correct_orientation"],
            m["correct"],
            -m["accepted"],
            -m["threshold"],
        ),
    )
    return {"selected": selection, "thresholds": choices, "rows": rows}


def export(directory):
    import onnx
    from research.inference import cpu_session

    torch.set_num_threads(2)
    validation = json.loads((directory / "validation.json").read_text())
    model = StaffLabelNetwork().eval()
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    )
    images = torch.rand(
        2, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(190273)
    ) * 2 - 1
    images[0].fill_(1)
    target = directory / "staff_label.onnx"
    names = ["label_logits", "bounds", "orientation_logits"]
    torch.onnx.export(
        model, images[:1], target, input_names=["images"], output_names=names,
        dynamic_axes={key: {0: "batch"} for key in ["images", *names]},
        opset_version=17, dynamo=False,
    )
    graph = onnx.load(target)
    # Slice export leaves symbolic spatial dimensions despite the fixed input.
    for output, channels in zip(graph.graph.output, [1, 4, 4], strict=True):
        for dimension, size in zip(
            output.type.tensor_type.shape.dim[1:],
            [channels, HEIGHT // STRIDE, WIDTH // STRIDE], strict=True,
        ):
            dimension.dim_value = size
    config = dict(
        version=1, input_size=[WIDTH, HEIGHT], stride=STRIDE,
        resize="contain_lanczos_center_white", rounding="positive_round_ties_even",
        normalization="rgb_minus1_plus1", angles_ccw=list(ANGLES),
        box_encoding="normalized_ltrb_from_integer_grid_origin",
        label_threshold=validation["selected"]["threshold"],
        scope="one_visible_name_in_complete_staff",
        score_selection="first_row_major_maximum_logit",
        crop_rounding="floor_left_top_ceil_right_bottom", source_checkpoint=validation["step"],
    )
    onnx.helper.set_model_props(graph, {"guitarocr.staff_label": json.dumps(config)})
    onnx.save(graph, target)
    with torch.inference_mode():
        expected = [value.numpy() for value in model(images)]
    actual = cpu_session(target).run(names, {"images": images.numpy()})
    errors = []
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-4, atol=1e-4)
        errors.append(float(np.abs(value - reference).max()))
    print(json.dumps(dict(
        output=str(target), threshold=config["label_threshold"],
        torch_onnx_max_errors=dict(zip(names, errors, strict=True)),
    )))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--export", type=Path, metavar="MODEL_DIR")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--context-data", type=Path)
    p.add_argument("--initialize", type=Path)
    args = p.parse_args()
    if args.export:
        export(args.export)
        return
    if args.data is None or args.output is None:
        p.error("Training requires --data and --output")
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(190273)
    sources = {s: read_sources(args.data, s) for s in ["train", "validation"]}
    if args.context_data:
        for split in sources:
            sources[split].extend(read_sources(args.context_data, split))
    assert not {r["family"] for r in sources["train"]} & {
        r["family"] for r in sources["validation"]
    }
    train = DataLoader(
        StaffLabelData(sources["train"], 8000, training=True),
        batch_size=24,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val = DataLoader(
        StaffLabelData(
            sources["validation"], len(sources["validation"]), training=False
        ),
        batch_size=24,
        num_workers=2,
    )
    model = StaffLabelNetwork().cuda()
    if args.initialize:
        model.load_state_dict(
            torch.load(args.initialize, map_location="cpu", weights_only=True)
        )
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    start = time.monotonic()
    step = 0
    best = -1
    history = []
    while step < args.steps:
        model.train()
        for images, heatmap, boxes, centers, angles, present in train:
            images, heatmap, boxes, centers, angles, present = [
                v.cuda() for v in [images, heatmap, boxes, centers, angles, present]
            ]
            optim.zero_grad(set_to_none=True)
            score, bounds, direction = model(images)
            loss_heat = focal(score[:, 0], heatmap)
            mask = present.bool()
            idx = torch.arange(images.shape[0], device="cuda")[mask]
            yy, xx = centers[mask, 0], centers[mask, 1]
            if idx.numel():
                predictions, targets, rotations, rotation_targets = [], [], [], []
                expected_center = boxes[mask]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        shifted = expected_center + expected_center.new_tensor(
                            [
                                dx * STRIDE / WIDTH,
                                dy * STRIDE / HEIGHT,
                                -dx * STRIDE / WIDTH,
                                -dy * STRIDE / HEIGHT,
                            ]
                        )
                        valid = (
                            (shifted >= 0).all(1)
                            & (yy + dy >= 0)
                            & (yy + dy < HEIGHT // STRIDE)
                            & (xx + dx >= 0)
                            & (xx + dx < WIDTH // STRIDE)
                        )
                        if valid.any():
                            predictions.append(
                                bounds[idx[valid], :, yy[valid] + dy, xx[valid] + dx]
                            )
                            targets.append(shifted[valid])
                            rotations.append(
                                direction[idx[valid], :, yy[valid] + dy, xx[valid] + dx]
                            )
                            rotation_targets.append(angles[mask][valid])
                predicted = torch.cat(predictions)
                expected = torch.cat(targets)
                size = (
                    torch.stack(
                        (
                            expected[:, 0] + expected[:, 2],
                            expected[:, 1] + expected[:, 3],
                        ),
                        dim=1,
                    )
                    .repeat(1, 2)
                    .clamp_min(0.002)
                )
                error = (predicted - expected) / size
                # Missing an edge removes letters. Source-box containment is
                # trained with asymmetric loss, without a runtime crop rule.
                loss_box = torch.where(error < 0, -4 * error, error).mean()
                # The selected heatmap maximum may move by one grid cell.
                # Train orientation on the same valid cells as box bounds.
                loss_direction = F.cross_entropy(
                    torch.cat(rotations), torch.cat(rotation_targets)
                )
            else:
                loss_box = score.sum() * 0
                loss_direction = score.sum() * 0
            loss = loss_heat + 3 * loss_box + loss_direction
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            step += 1
            if step % 50 == 0:
                print(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "heat": float(loss_heat.detach()),
                        "box": float(loss_box.detach()),
                        "direction": float(loss_direction.detach()),
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
            if step % 150 == 0 or step == args.steps:
                report = evaluate(model, val)
                selection = report["selected"]
                history.append({"step": step, **selection})
                print({"validation_step": step, **selection}, flush=True)
                score_value = (
                    selection["coverage_99_correct_orientation"]
                    + selection["correct"] / (1 + len(sources["validation"]))
                    - selection["false_label_on_absent"]
                )
                if score_value > best:
                    best = score_value
                    torch.save(model.cpu().state_dict(), args.output / "model.pt")
                    model.cuda()
                    (args.output / "validation.json").write_text(
                        json.dumps({"step": step, **report}, indent=2)
                    )
                model.train()
            if step >= args.steps:
                break
    (args.output / "training.json").write_text(
        json.dumps(
            {
                "steps": step,
                "parameters": sum(p.numel() for p in model.parameters()),
                "seconds": time.monotonic() - start,
                "history": history,
                "source_counts": {s: len(v) for s, v in sources.items()},
                "data": str(args.data),
                "context_data": str(args.context_data) if args.context_data else None,
                "initialization": str(args.initialize) if args.initialize else None,
                "source_family_intersection": 0,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
