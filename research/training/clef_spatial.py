"""Train, export and independently evaluate the spatial clef model."""

import argparse
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset

from research.common.clef_spatial import CONTRACT, LABELS, image_tensor
from research.inference import cpu_session
from research.models.auxiliary import ClefSpatialNetwork
from research.training import fit_classifier


class ClefSpatialReader:
    def __init__(self, path):
        self.session = cpu_session(path)
        self.config = json.loads(self.session.get_modelmeta().custom_metadata_map['guitarocr.clef_spatial'])
        if any(self.config.get(k) != v for k, v in CONTRACT.items()):
            raise ValueError('Unsupported spatial clef model contract')
        if not .5 <= self.config.get('threshold', -1) <= 1:
            raise ValueError('Invalid spatial clef threshold')

    def predict(self, image, query_bbox):
        values = image_tensor(image, query_bbox)[None]
        logits = self.session.run(['logits'], {'images': values})[0]
        if logits.shape != (1, len(self.config['labels'])) or not np.isfinite(logits).all():
            raise ValueError('Invalid spatial clef output')
        z = logits[0].astype(np.float64)
        probability = np.exp(z-z.max())
        probability /= probability.sum()
        index = int(probability.argmax())
        confidence = float(probability[index])
        accepted = confidence >= self.config['threshold']
        value = self.config['labels'][index] if accepted else dict(clef=None, clef_octave=None)
        return dict(source='clef_spatial_network', **value, accepted=accepted, confidence=confidence,
                    threshold=self.config['threshold'], probabilities=probability.tolist(),
                    status='recognized' if accepted else 'uncertain')


def export(output, data):
    import onnx

    torch.set_num_threads(2)
    validation = json.loads((output / 'validation.json').read_text())
    model = ClefSpatialNetwork().eval()
    model.load_state_dict(torch.load(output / 'model.pt', map_location='cpu', weights_only=True))
    target = output / 'clef_spatial.onnx'
    torch.onnx.export(model, torch.zeros(1, 2, 256, 256), target, input_names=['images'], output_names=['logits'],
                      dynamic_axes={key: {0: 'batch'} for key in ['images', 'logits']}, opset_version=17, dynamo=False)
    graph = onnx.load(target)
    config = dict(CONTRACT, threshold=validation['selected']['threshold'], source_checkpoint=validation['step'])
    onnx.helper.set_model_props(graph, {'guitarocr.clef_spatial': json.dumps(config)})
    onnx.save(graph, target)
    reader = ClefSpatialReader(target)
    records = [json.loads(line) for line in (data / 'validation.jsonl').read_text().splitlines()]
    errors = []
    for row in records[:14]:
        with Image.open(row['images'][0]) as image:
            tensor = image_tensor(image, row['query_bbox'])[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).numpy()
        actual = reader.session.run(['logits'], {'images': tensor})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-4)
        errors.append(float(np.abs(actual-expected).max()))
    print(json.dumps(dict(selected=validation['selected'],
                          torch_onnx_max_logit_error=max(errors)), indent=2))


def evaluate_export(model, manifest, output):
    reader = ClefSpatialReader(model / 'clef_spatial.onnx')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    predictions = []
    for row in rows:
        try:
            target = json.loads(row['messages'][-1]['content'])
        except ValueError:
            continue
        if 'clef' not in target:
            continue
        with Image.open(row['images'][0]) as image:
            evidence = reader.predict(image, row['query_bbox'])
        guess = LABELS[int(np.argmax(evidence['probabilities']))]
        predictions.append(dict(image=row['images'][0], query_bbox=row['query_bbox'],
                                family=row.get('provenance', {}).get('family'), target=target,
                                prediction=evidence, raw_argmax=guess))
    report = dict(samples=len(predictions), accepted=sum(p['prediction']['accepted'] for p in predictions),
        correct=sum(p['prediction']['accepted'] and all(p['prediction'][k] == p['target'][k] for k in ['clef','clef_octave']) for p in predictions),
        wrong=sum(p['prediction']['accepted'] and any(p['prediction'][k] != p['target'][k] for k in ['clef','clef_octave']) for p in predictions),
        raw_joint=sum(p['raw_argmax']==p['target'] for p in predictions),
        predictions=predictions)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'}, indent=2))


@lru_cache(maxsize=8192)
def pixels(path):
    with Image.open(path) as image:
        return image.convert('L')


class ClefData(Dataset):
    def __init__(self, rows, count=None):
        self.rows, self.count = rows, count
        domains = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            target = json.loads(row['messages'][-1]['content'])
            label = LABELS.index(target)
            domains[label][row['provenance']['renderer']].append(index)
        self.sampling_groups = {label: list(groups.values()) for label, groups in domains.items()}

    def __len__(self):
        return self.count or len(self.rows)

    def __getitem__(self, index):
        rng = random.Random(552203 + index)
        if self.count:
            # Give each renderer the same chance within a clef class. Rare
            # printed forms must not be drowned out by one synthetic font set.
            label = rng.choice(list(self.sampling_groups))
            index = rng.choice(rng.choice(self.sampling_groups[label]))
        row = self.rows[index]
        image = pixels(row['images'][0]).copy()
        query = list(row['query_bbox'])
        if self.count and rng.random() < .8:
            margins = tuple(rng.randrange(max(2, round(min(image.size) * .3))) for _ in range(4))
            image = ImageOps.expand(image, margins, fill=255)
            query[0] += margins[0]
            query[1] += margins[1]
            scale = rng.uniform(.8, 1.2)
            width, height = image.size
            image = image.resize((max(1, round(image.width * scale)),
                                  max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
            query = [query[0] * image.width / width, query[1] * image.height / height,
                     query[2] * image.width / width, query[3] * image.height / height]
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(.65, 1.2))
            if rng.random() < .25:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.2, .7)))
            if rng.random() < .3:
                size = image.size
                scale = min(1.0, rng.uniform(32, 120) / image.height)
                image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                                     Image.Resampling.LANCZOS).resize(size, Image.Resampling.BILINEAR)
            # Detector bounds vary independently from the printed clef. The
            # complete context stays available when the query becomes tight.
            dx, dy = query[2] * rng.uniform(-.1, .1), query[3] * rng.uniform(-.1, .1)
            query = [query[0] + dx, query[1] + dy, query[2] - dx, query[3] - dy]
        tensor = image_tensor(image, query)
        if self.count and rng.random() < .8:
            # Keep every staff line while varying the query's canvas position.
            # Real page edges make the same clef appear away from the centre.
            size = rng.randrange(160, 257)
            left, top = rng.randrange(257 - size), rng.randrange(257 - size)
            shifted = np.stack([np.ones((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.float32)])
            for channel, resample in ((0, Image.Resampling.LANCZOS), (1, Image.Resampling.NEAREST)):
                resized = Image.fromarray(tensor[channel]).resize((size, size), resample)
                shifted[channel, top:top + size, left:left + size] = np.asarray(resized)
            tensor = shifted
        return tensor, np.int64(LABELS.index(json.loads(row['messages'][-1]['content'])))


def evaluate(model, loader):
    model.eval()
    probabilities, labels = [], []
    with torch.inference_mode():
        for images, target in loader:
            probabilities.extend(model(images.cuda()).softmax(-1).cpu().tolist())
            labels.extend(target.tolist())
    p, target = np.asarray(probabilities), np.asarray(labels)
    correct = p.argmax(1) == target
    choices = []
    for threshold in [.5, .6, .7, .8, .9, .95, .97, .98, .99, .995, .999, .9999, 1.0]:
        accepted = p.max(1) >= threshold
        choices.append(dict(threshold=threshold, accepted=int(accepted.sum()),
                            correct=int((accepted & correct).sum()), wrong=int((accepted & ~correct).sum())))
    selected = max([row for row in choices if row['wrong'] <= .01 * row['accepted']],
                   key=lambda r: (r['correct'], -r['threshold']))
    return dict(samples=len(labels), argmax_correct=int(correct.sum()), selected=selected,
                thresholds=choices, probabilities=probabilities, targets=labels)


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(654721)
    records = {split: [json.loads(line) for line in (args.data / f'{split}.jsonl').read_text().splitlines()]
               for split in ['train', 'validation']}
    families = {split: {r['provenance']['family'] for r in rows} for split, rows in records.items()}
    assert not families['train'] & families['validation']
    train = DataLoader(ClefData(records['train'], args.steps * 48), batch_size=48, shuffle=True,
                       num_workers=6, persistent_workers=True, pin_memory=True)
    validation = DataLoader(ClefData(records['validation']),
                            batch_size=48, num_workers=4, persistent_workers=True)
    model = ClefSpatialNetwork().cuda()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    report = fit_classifier(
        model,
        train,
        optimizer,
        steps=args.steps,
        output=args.output,
        evaluate=lambda current: evaluate(current, validation),
        scheduler=scheduler,
        clip=5,
    )
    (args.output / "training.json").write_text(
        json.dumps(
            dict(
                **report,
                seed=654721,
                initialize=str(args.init) if args.init else None,
                data=str(args.data),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parser_train = commands.add_parser('train')
    parser_train.add_argument('--data', type=Path, required=True)
    parser_train.add_argument('--output', type=Path, required=True)
    parser_train.add_argument('--steps', type=int, default=2400)
    parser_train.add_argument('--init', type=Path)
    parser_train.add_argument('--learning-rate', type=float, default=.0005)
    parser_export = commands.add_parser('export')
    parser_export.add_argument('--model', type=Path, required=True)
    parser_export.add_argument('--data', type=Path, required=True)
    parser_evaluate = commands.add_parser('evaluate')
    parser_evaluate.add_argument('--model', type=Path, required=True)
    parser_evaluate.add_argument('--manifest', type=Path, required=True)
    parser_evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'export':
        export(args.model, args.data)
    elif args.command == 'evaluate':
        evaluate_export(args.model, args.manifest, args.output)


if __name__ == "__main__":
    main()
