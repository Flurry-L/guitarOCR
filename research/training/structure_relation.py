"""Prepare data, train, evaluate and export system and part relation networks."""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 768, 192
MODES = {'unknown': 0, 'notation': 1, 'tab': 2, 'both': 3}


def relation_inputs(image, centers, modes=None):
    """Read the complete horizontal interval between adjacent row centers.

    Centers come from layout boxes at inference; no staff-line, connector,
    intensity threshold, left-margin search, or expected system count is used.
    """
    image = image.convert('L')
    centers = np.asarray(centers, dtype=np.float32)
    modes = modes or ['unknown'] * len(centers)
    pairs, geometry = [], []
    for index, (top, bottom) in enumerate(zip(centers, centers[1:])):
        strip = image.crop((0, max(0, int(top)), image.width, min(image.height, max(int(top)+1, int(bottom)))))
        scale = min(WIDTH / strip.width, HEIGHT / strip.height)
        size = (max(1, round(strip.width * scale)), max(1, round(strip.height * scale)))
        resized = strip.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new('L', (WIDTH, HEIGHT), 255)
        canvas.paste(resized, ((WIDTH - size[0]) // 2, (HEIGHT - size[1]) // 2))
        pairs.append(np.asarray(canvas, dtype=np.float32)[None] / 255.)
        features = [float(top/image.height), float(bottom/image.height), float((bottom-top)/image.width), float(image.width/image.height)]
        for mode in modes[index:index+2]:
            features.extend(float(MODES.get(mode, 0) == value) for value in range(4))
        geometry.append(features)
    return np.asarray(pairs, dtype=np.float32).reshape(-1, 1, HEIGHT, WIDTH), np.asarray(geometry, dtype=np.float32).reshape(-1, 12)


def sample(job):
    row, output = job
    image = Image.open(row["image"]).convert("RGB")
    centers = row["centers"]
    if len(centers) != len(row["expected"]["rows"]):
        return {
            "error": "row_count",
            "image": row["image"],
            "expected": len(row["expected"]["rows"]),
            "actual": len(centers),
        }
    values, geometry = relation_inputs(image, centers, row.get("row_modes"))
    labels = np.asarray(
        [
            float(a[0] == b[0])
            for a, b in zip(row["expected"]["rows"], row["expected"]["rows"][1:])
        ],
        dtype=np.float32,
    )
    target = Path(output) / f"{sha256(row['image'].encode()).hexdigest()[:20]}.npz"
    np.savez_compressed(
        target, pixels=(values * 255).astype(np.uint8), geometry=geometry, labels=labels
    )
    return {
        **row,
        "cache": str(target.resolve()),
        "centers": centers,
        "pairs": len(labels),
        "image_sha256": sha256(Path(row["image"]).read_bytes()).hexdigest(),
    }


def build_crops(args):
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    errors = []
    for split in ("train", "validation", "test"):
        sources = getattr(args, split)
        if not sources:
            continue
        unique = {}
        for source in sources:
            for raw in map(json.loads, Path(source).read_text().splitlines()):
                expected = raw["expected"]
                path = Path(raw["image"])
                path = str((source.parent / path).resolve())
                if raw["split"] != split or not raw.get("family"):
                    raise ValueError(f"Missing or inconsistent source assignment: {source}")
                unique[path] = {
                    "image": path,
                    "expected": expected,
                    "row_modes": raw["row_modes"],
                    "centers": raw["centers"],
                    "source": str(source),
                    "split": split,
                    "family": raw["family"],
                    "reference_rows": True,
                }
        cache = args.output / split
        cache.mkdir(exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(
                pool.map(
                    sample, [(r, str(cache)) for r in unique.values()], chunksize=16
                )
            )
        errors.extend(r for r in rows if "error" in r)
        rows = [r for r in rows if "error" not in r]
        all_rows[split] = rows
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        print(split, len(rows), sum(r["pairs"] for r in rows), flush=True)
    families, hashes = {}, {}
    for split, rows in all_rows.items():
        for row in rows:
            if (families.setdefault(row["family"], split) != split
                    or hashes.setdefault(row["image_sha256"], split) != split):
                raise ValueError("Source family or image crosses dataset splits")
    (args.output / "summary.json").write_text(json.dumps({
        "counts": {split: {"pages": len(rows), "pairs": sum(row["pairs"] for row in rows)}
                   for split, rows in all_rows.items()},
        "errors": errors,
    }, indent=2))


def export(args):
    import onnx
    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader

    from research.models.structure_relation import (
        PartRelationNet,
        RelationNet,
        StructureRelationNetwork,
    )

    rows, data = load(args.validation)
    if any(row.get('split') != 'validation' for row in rows):
        raise ValueError('Export validation requires validation-only rows')
    system, part = RelationNet(), PartRelationNet()
    for model, checkpoint in [(system, args.system_checkpoint), (part, args.part_checkpoint)]:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
    model = StructureRelationNetwork(system, part).eval()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ['same_system_logits', 'same_part_logits']
    torch.onnx.export(model, (torch.ones(2, 1, HEIGHT, WIDTH), torch.zeros(2, 12)), str(args.output),
                      input_names=['pixels', 'geometry'], output_names=names,
                      dynamic_axes={name: {0: 'pairs'} for name in ['pixels', 'geometry', *names]},
                      opset_version=17, dynamo=False)
    contract = {'width': WIDTH, 'height': HEIGHT, 'channels': 1, 'geometry_features': 12,
                'grayscale': 'ITU-R-601', 'resize': 'contain_lanczos_white_center',
                'crop': 'full_width_between_adjacent_row_centers',
                'modes': ['unknown', 'notation', 'tab', 'both'], 'same_part': True,
                'temperature': 1., 'part_temperature': 1.,
                'review_confidence': .95, 'part_review_confidence': .95,
                'operating_point': 'review threshold; calibration under novel rendering styles is not established',
                'part_scope': 'adjacent rows within the same system; no instrument identity prediction'}
    graph = onnx.load(str(args.output))
    onnx.helper.set_model_props(graph, {'guitarocr.structure_relation': json.dumps(contract)})
    onnx.checker.check_model(graph)
    onnx.save(graph, str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    loader = DataLoader(data, batch_size=32)
    maximum = 0.
    with torch.inference_mode():
        for pixels, geometry, _ in loader:
            pixels = pixels.float()/255
            expected = model(pixels, geometry)
            actual = session.run(names, {'pixels': pixels.numpy(), 'geometry': geometry.numpy()})
            for reference, observed in zip(expected, actual, strict=True):
                maximum = max(maximum, float(np.abs(reference.numpy()-observed).max()))
                np.testing.assert_allclose(observed, reference.numpy(), rtol=2e-4, atol=1e-4)
    report = {**evaluate(model, loader, rows, 'cpu', 'part'),
              'pytorch_onnx_max_logit_error': maximum, 'contract': contract}
    args.output.with_suffix('.validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'predictions'}), flush=True)


def load(paths):
    import torch
    from torch.utils.data import TensorDataset

    rows, pixels, geometry, labels = [], [], [], []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            triples = row['expected']['rows']
            truth = np.asarray([[a[0] == b[0], a[1] == b[1]]
                                for a, b in zip(triples, triples[1:])], dtype=np.float32).reshape(-1, 2)
            with np.load(row['cache'], allow_pickle=False) as cache:
                if not len(truth) == row['pairs'] == len(cache['pixels']) == len(cache['geometry']):
                    raise ValueError(f"Row-pair annotations do not match pixels: {row['image']}")
                pixels.append(cache['pixels'])
                geometry.append(cache['geometry'])
            rows.append(row)
            labels.append(truth)
    if not rows or not sum(map(len, labels)):
        raise ValueError('Relation training/evaluation requires annotated row pairs')
    return rows, TensorDataset(*(torch.from_numpy(np.concatenate(values))
                                for values in (pixels, geometry, labels)))


def model_for_task(task):
    from research.models.structure_relation import PartRelationNet, RelationNet

    return PartRelationNet() if task == 'part' else RelationNet()


def evaluate(model, loader, rows, device, task):
    import torch

    probabilities, labels = [], []
    model.eval()
    with torch.inference_mode():
        for pixels, geometry, truth in loader:
            output = model(pixels.to(device).float()/255, geometry.to(device))
            logits = torch.stack(output, 1) if task == 'part' else output[:, None]
            probabilities.extend(logits.sigmoid().cpu().tolist())
            labels.extend(truth.tolist())
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    system_correct = (probabilities[:, 0] >= .5) == labels[:, 0]
    grouping_correct = system_correct.copy()
    if task == 'part':
        grouping_correct &= (labels[:, 0] == 0) | ((probabilities[:, 1] >= .5) == labels[:, 1])
    pages, offset = [], 0
    for row in rows:
        end = offset + row['pairs']
        pages.append({'image': row['image'], 'family': row['family'],
                      'system_correct': bool(system_correct[offset:end].all()),
                      'grouping_correct': bool(grouping_correct[offset:end].all()),
                      'probabilities': probabilities[offset:end].tolist()})
        offset = end
    return {'pages': len(rows), 'pairs': len(labels),
            'system_exact': sum(row['system_correct'] for row in pages),
            'grouping_exact': sum(row['grouping_correct'] for row in pages),
            'pair_correct': int(grouping_correct.sum()),
            'false_join': int(((probabilities[:, 0] >= .5) & (labels[:, 0] == 0)).sum()),
            'false_split': int(((probabilities[:, 0] < .5) & (labels[:, 0] == 1)).sum()),
            'predictions': pages}


def evaluate_checkpoint(args):
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(4)
    model = model_for_task(args.task)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict'])
    reports = {}
    for path in args.manifest:
        rows, data = load([path])
        reports[str(path)] = evaluate(model, DataLoader(data, batch_size=args.batch_size), rows, 'cpu', args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    print(json.dumps({key: {name: value for name, value in report.items() if name != 'predictions'}
                      for key, report in reports.items()}))



def train(args):
    import torch
    from torch.utils.data import DataLoader

    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError('epochs, batch-size and learning-rate must be positive')
    torch.set_num_threads(4)
    train_rows, train = load([root/'train.jsonl' for root in args.data])
    validation_rows, validation = load([root/'validation.jsonl' for root in args.data])
    for split, rows in [('train', train_rows), ('validation', validation_rows)]:
        if any(row.get('split') != split or not row.get('family') for row in rows):
            raise ValueError('Every relation page must identify its source family and split')
    if {row['family'] for row in train_rows} & {row['family'] for row in validation_rows}:
        raise ValueError('Training and validation source families overlap')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261003)
    np.random.seed(20261003)
    random.seed(20261003)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_for_task(args.task).to(device)
    if args.initial:
        model.load_state_dict(torch.load(args.initial, map_location='cpu', weights_only=True)['state_dict'])
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history, best, started = [], (-1, -1), time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for pixels, geometry, labels in loader:
            pixels, geometry, labels = pixels.to(device).float()/255, geometry.to(device), labels.to(device)
            pixels = (1-(1-pixels)*(torch.rand(len(pixels), 1, 1, 1, device=device)*.3+.85)).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            output = model(pixels, geometry)
            system, part = output if args.task == 'part' else (output, None)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(system, labels[:, 0])
            within_system = labels[:, 0] > 0
            if part is not None and within_system.any():
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    part[within_system], labels[within_system, 1])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        schedule.step()
        metrics = evaluate(model, validation_loader, validation_rows, device, args.task)
        report = {'epoch': epoch+1, 'loss': float(np.mean(losses)),
                  'seconds': time.monotonic()-started,
                  **{key: value for key, value in metrics.items() if key != 'predictions'}}
        history.append(report)
        score = (metrics['system_exact'], metrics['grouping_exact'])
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'task': args.task,
                        'epoch': epoch+1, 'metrics': report}, args.output/'best.pt')
            (args.output/'validation.json').write_text(json.dumps(metrics, indent=2))
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(report), flush=True)
    (args.output/'training.json').write_text(json.dumps({
        'arguments': vars(args), 'training_pages': len(train_rows),
        'validation_pages': len(validation_rows), 'test_opened': False,
        'device': str(device)}, default=str, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parser_prepare = commands.add_parser('prepare')
    parser_prepare.add_argument('--output', type=Path, required=True)
    for split in ('train', 'validation', 'test'):
        parser_prepare.add_argument(f'--{split}', type=Path, nargs='+', required=split != 'test')
    parser_prepare.add_argument('--workers', type=int, default=6)
    parser_train = commands.add_parser('train')
    parser_train.add_argument('--data', type=Path, nargs='+', required=True)
    parser_train.add_argument('--output', type=Path, required=True)
    parser_train.add_argument('--task', choices=['system', 'part'], required=True)
    parser_train.add_argument('--initial', type=Path)
    parser_train.add_argument('--epochs', type=int, default=20)
    parser_train.add_argument('--batch-size', type=int, default=128)
    parser_train.add_argument('--learning-rate', type=float, default=.0003)
    parser_evaluate = commands.add_parser('evaluate')
    parser_evaluate.add_argument('--manifest', type=Path, nargs='+', required=True)
    parser_evaluate.add_argument('--checkpoint', type=Path, required=True)
    parser_evaluate.add_argument('--task', choices=['system', 'part'], required=True)
    parser_evaluate.add_argument('--output', type=Path, required=True)
    parser_evaluate.add_argument('--batch-size', type=int, default=128)
    parser_export = commands.add_parser('export')
    parser_export.add_argument('--system-checkpoint', type=Path, required=True)
    parser_export.add_argument('--part-checkpoint', type=Path, required=True)
    parser_export.add_argument('--validation', type=Path, nargs='+', required=True)
    parser_export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        build_crops(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate_checkpoint(args)
    elif args.command == 'export':
        export(args)


if __name__ == "__main__":
    main()
