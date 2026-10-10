"""Train, export and evaluate staff evidence with one shared image transform."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import torch
from PIL import Image, ImageEnhance, ImageOps
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from research.inference import cpu_session
from research.models.auxiliary import StaffNetwork

COUNTS = (0, 4, 5, 6, 7, 8)


MODES = ("notation", "tab", "both")


def staff_array(image, augment=False):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if augment:
        image = image.rotate(
            float(np.random.uniform(-0.6, 0.6)),
            Image.Resampling.BICUBIC,
            expand=True,
            fillcolor="white",
        )
        x, y = (
            int(image.width * np.random.uniform(0, 0.04)),
            int(image.height * np.random.uniform(0, 0.08)),
        )
        image = ImageOps.expand(image, (x, y, x, y), "white")
        image = ImageEnhance.Contrast(image).enhance(
            float(np.random.uniform(0.65, 1.15))
        )
        image = ImageEnhance.Brightness(image).enhance(
            float(np.random.uniform(0.85, 1.1))
        )
    scale = min(768 / image.width, 256 / image.height)
    width, height = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (768, 256), "white")
    x, y = (768 - width) // 2, (256 - height) // 2
    if augment:
        x, y = np.random.randint(768 - width + 1), np.random.randint(256 - height + 1)
    canvas.paste(image, (int(x), int(y)))
    values = np.asarray(canvas, dtype=np.float32) / 255
    return (values.transpose(2, 0, 1) - 0.5) / 0.5


class StaffReader:
    def __init__(self, path):
        self.session = cpu_session(path)
        self.config = json.loads(
            self.session.get_modelmeta().custom_metadata_map["guitarocr.staff_config"]
        )
        c = self.config
        if (
            c["version"] != 2
            or (c["width"], c["height"]) != (768, 256)
            or c["resize"] != "contain_lanczos_white_center"
            or c["normalization"] != "minus_one_to_one"
            or c["counts"] != [0, 4, 5, 6, 7, 8]
            or c["modes"] != ["notation", "tab", "both"]
            or c.get("count_authority") != "joint_visual"
        ):
            raise ValueError("Unsupported staff evidence contract")
        if (
            len(c["temperatures"]) != 2
            or len(c["thresholds"]) != 2
            or any(not np.isfinite(v) or v <= 0 for v in c["temperatures"])
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in c["thresholds"])
        ):
            raise ValueError("Invalid staff evidence calibration")

    def predict(self, image):
        logits = self.session.run(["logits"], {"images": staff_array(image)[None]})[0][
            0
        ]
        if logits.shape != (9,) or not np.isfinite(logits).all():
            raise ValueError("Invalid staff model output")
        fields = {}
        for i, (name, sl, labels) in enumerate(
            [
                ("string_count", slice(0, 6), self.config["counts"]),
                ("mode", slice(6, 9), self.config["modes"]),
            ]
        ):
            z = logits[sl].astype(np.float64) / self.config["temperatures"][i]
            p = np.exp(z - z.max())
            p /= p.sum()
            index = int(p.argmax())
            fields[name] = {
                "value": labels[index] or None,
                "confidence": float(p[index]),
                "accepted": bool(p[index] >= self.config["thresholds"][i]),
                "probabilities": p.tolist(),
            }
        fields["count_authority"] = "joint_visual"
        return {"source": "staff_network_v2", **fields}


def evaluate_export(model, manifests, output):
    reader = StaffReader(model)
    predictions = []
    groups = defaultdict(lambda: dict(samples=0, accepted=0, correct=0, wrong=0))
    for manifest in manifests:
        text = manifest.read_text()
        rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines()]
        for row in rows:
            sample = row.get('sample', row)
            try:
                target = row.get('target') or json.loads(sample['messages'][-1]['content'])
            except ValueError:
                continue
            if 'string_count' not in target:
                continue
            image_path = sample.get('image') or sample['images'][0]
            with Image.open(image_path) as image:
                evidence = reader.predict(image)
                count, mode = evidence['string_count'], evidence['mode']
                accepted = count['accepted'] and mode['accepted'] and ((count['value'] is None) == (mode['value'] == 'notation'))
                correct = count['value'] == target['string_count']
                group = f"{sample.get('provenance', {}).get('mode', 'unknown')}:{target['string_count']}"
            for key in ['all', group]:
                groups[key]['samples'] += 1
                groups[key]['accepted'] += int(accepted)
                groups[key]['correct'] += int(accepted and correct)
                groups[key]['wrong'] += int(accepted and not correct)
            predictions.append(dict(image=image_path, target=target,
                provenance=sample.get('provenance', {}), prediction=evidence,
                jointly_accepted=accepted, correct=correct))
    report = dict(**groups.pop('all'), by_mode_count=dict(groups), predictions=predictions)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'predictions'}, indent=2))


def records(path):
    path = Path(path).resolve()
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines()]
    result = []
    for row in rows:
        try:
            target = row.get('target') or json.loads(row['messages'][-1]['content'])
        except ValueError:
            continue
        if 'string_count' not in target:
            continue
        provenance = row.get('provenance', {})
        image = Path(row.get('image') or row['images'][0])
        if not image.is_absolute():
            image = path.parent / image
        result.append(dict(image=str(image),
            count=COUNTS.index(target['string_count'] or 0), mode=MODES.index(provenance['mode']),
            family=provenance['family'], domain=str(path)))
    return result


class CountData(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['image']) as image:
            tensor = staff_array(image, self.augment)
        return tensor, np.asarray([row['count'], row['mode']], dtype=np.int64)


def evaluate(model, loader):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for images, target in loader:
            logits.extend(model(images.cuda()).cpu().tolist())
            targets.extend(target.tolist())
    z, targets = np.asarray(logits), np.asarray(targets)
    probabilities = []
    for section in [slice(0, 6), slice(6, 9)]:
        values = z[:, section]
        p = np.exp(values-values.max(1, keepdims=True))
        probabilities.append(p/p.sum(1, keepdims=True))
    predictions = np.stack([p.argmax(1) for p in probabilities], axis=1)
    confidence = np.stack([p.max(1) for p in probabilities], axis=1)
    correct = (predictions == targets).all(1)
    consistent = (predictions[:, 0] == 0) == (predictions[:, 1] == 0)
    choices = []
    for count in [.9, .95, .98, .99, .995, .998, .999, .9995, .9999, 1.0]:
        for mode in [.8, .9, .95, .98, .99]:
            accepted = (confidence[:, 0] >= count) & (confidence[:, 1] >= mode) & consistent
            choices.append(dict(thresholds=[count, mode], accepted=int(accepted.sum()),
                correct=int((accepted & correct).sum()), wrong=int((accepted & ~correct).sum())))
    selected = max([c for c in choices if not c['wrong']], key=lambda r: (r['correct'], -sum(r['thresholds'])))
    return dict(samples=len(targets), argmax_joint_correct=int(correct.sum()), selected=selected,
        choices=choices, logits=logits, targets=targets.tolist())


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(962703)
    np.random.seed(962703)
    rows = {split: [row for path in getattr(args, split) for row in records(path)] for split in ['train', 'validation']}
    families = {split: {row['family'] for row in values} for split, values in rows.items()}
    if families['train'] & families['validation']:
        raise ValueError('Source-family overlap between training and validation')
    groups = Counter((row['domain'], row['count']) for row in rows['train'])
    weights = [1/groups[row['domain'], row['count']] for row in rows['train']]
    train = DataLoader(CountData(rows['train'], True), batch_size=args.batch_size, num_workers=6,
        sampler=WeightedRandomSampler(weights, args.steps*args.batch_size, replacement=True), persistent_workers=True, pin_memory=True)
    validation = DataLoader(CountData(rows['validation']), batch_size=48, num_workers=4, persistent_workers=True)
    model = StaffNetwork().cuda()
    model.load_state_dict(load_file(str(args.initialize)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.00001)
    provenance = dict(seed=962703, device=torch.cuda.get_device_name(),
        steps=args.steps, batch_size=args.batch_size,
        samples={split: len(values) for split, values in rows.items()}, family_overlap=0,
        families={split: len(values) for split, values in families.items()},
        manifests=[str(path) for path in args.train+args.validation],
        initialization=str(args.initialize))
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    best, history, started = -1, [], time.monotonic()
    for step, (images, target) in enumerate(train, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images.cuda(non_blocking=True))
        target = target.cuda(non_blocking=True)
        loss = torch.nn.functional.cross_entropy(logits[:, :6], target[:, 0]) + .5*torch.nn.functional.cross_entropy(logits[:, 6:], target[:, 1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(dict(step=step, loss=float(loss.detach()), seconds=time.monotonic()-started), flush=True)
        if step % 300 == 0 or step == args.steps:
            report = evaluate(model, validation)
            entry = dict(step=step, selected=report['selected'], argmax_joint_correct=report['argmax_joint_correct'])
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if report['selected']['correct'] > best:
                best = report['selected']['correct']
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(args.output/'model.safetensors'))
                (args.output/'validation.json').write_text(json.dumps(dict(step=step, **report), indent=2)+'\n')
            (args.output/'training.json').write_text(json.dumps(history, indent=2)+'\n')
    model = StaffNetwork().eval()
    model.load_state_dict(load_file(str(args.output/'model.safetensors')))
    validation = json.loads((args.output/'validation.json').read_text())
    config = dict(version=2, width=768, height=256, resize='contain_lanczos_white_center',
        normalization='minus_one_to_one', counts=COUNTS, modes=MODES, temperatures=[1, 1],
        thresholds=validation['selected']['thresholds'], count_authority='joint_visual',
        parameters=sum(p.numel() for p in model.parameters()), supported_string_counts=list(COUNTS[1:]),
        scope='complete engraved staff row; no pixel-rule fallback')
    target = args.output/'staff_evidence.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, 256, 768), target, input_names=['images'], output_names=['logits'],
        dynamic_axes={'images': {0:'batch'}, 'logits': {0:'batch'}}, opset_version=17, dynamo=False)
    graph = onnx.load(target)
    onnx.helper.set_model_props(graph, {'guitarocr.staff_config':json.dumps(config)})
    onnx.save(graph, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parser_train = commands.add_parser('train')
    parser_train.add_argument('--train', type=Path, nargs='+', required=True)
    parser_train.add_argument('--validation', type=Path, nargs='+', required=True)
    parser_train.add_argument('--initialize', type=Path, required=True)
    parser_train.add_argument('--output', type=Path, required=True)
    parser_train.add_argument('--steps', type=int, default=1800)
    parser_train.add_argument('--batch-size', type=int, default=48)
    parser_evaluate = commands.add_parser('evaluate')
    parser_evaluate.add_argument('--model', type=Path, required=True)
    parser_evaluate.add_argument('--manifest', type=Path, nargs='+', required=True)
    parser_evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate_export(args.model, args.manifest, args.output)


if __name__ == "__main__":
    main()
