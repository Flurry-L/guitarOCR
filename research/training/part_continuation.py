"""Train, calibrate, export and evaluate the part continuation model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.common.part_continuation import CONTRACT, MAX_PARTS, continuation_inputs, match_gallery
from research.data.part_continuation import ContinuationData
from research.inference import cpu_session
from research.models.auxiliary import PartContinuationNetwork
from research.training import fit_classifier

MIN_PRECISION = .99
THRESHOLDS = [.5, .6, .7, .8, .85, .9, .95, .97, .98, .99, .995, .999, 1.0]
ASSIGNMENT_MARGINS = [0., .25, .5, 1., 2.]


class PartContinuationReader:
    def __init__(self, path):
        self.session = cpu_session(path)
        self.config = json.loads(self.session.get_modelmeta().custom_metadata_map['guitarocr.part_continuation'])
        if any(self.config.get(k) != v for k, v in CONTRACT.items()):
            raise ValueError('Unsupported part continuation model contract')
        if not .5 <= self.config.get('threshold', -1) <= 1:
            raise ValueError('Invalid part continuation threshold')
        if self.config.get('assignment_margin', -1) < 0:
            raise ValueError('Invalid part assignment margin')

    def predict(self, previous, current, previous_index, current_index,
                previous_identity, previous_mode, current_mode):
        base = dict(source='part_continuation_network', decision='unknown', accepted=False,
                    confidence=None, threshold=self.config['threshold'], probabilities=None)
        if not previous or not current or max(len(previous), len(current)) > MAX_PARTS:
            return dict(base, status='unsupported_context')
        rows, contexts, features = continuation_inputs(previous, current, previous_index, current_index,
            previous_identity, previous_mode, current_mode)
        logits = self.session.run(['logits'], dict(rows=rows[None], contexts=contexts[None], features=features[None]))[0]
        if logits.shape != (1, 3) or not np.isfinite(logits).all():
            raise ValueError('Invalid part continuation output')
        z = logits[0].astype(np.float64)
        probability = np.exp(z-z.max())
        probability /= probability.sum()
        index = int(probability.argmax())
        confidence = float(probability[index])
        decision = self.config['labels'][index] if confidence >= self.config['threshold'] else 'unknown'
        return dict(base, decision=decision, accepted=decision == 'same', confidence=confidence,
                    probabilities=probability.tolist(), status='recognized' if decision != 'unknown' else 'uncertain')


def predict(model, records, count, *, visible_labels=False):
    dataset = ContinuationData(records, count, False, visible_labels=visible_labels)
    contexts, queued, destinations = [], [], []

    def flush():
        if not queued:
            return
        arrays = [np.stack([inputs[i] for inputs in queued]) for i in range(3)]
        if hasattr(model, 'run'):
            logits = model.run(['logits'], dict(zip(['rows', 'contexts', 'features'], arrays)))[0].astype(np.float64)
            values = np.exp(logits-logits.max(1, keepdims=True))
            values /= values.sum(1, keepdims=True)
        else:
            device = next(model.parameters()).device
            batch = tuple(torch.from_numpy(value).to(device) for value in arrays)
            with torch.inference_mode():
                values = model(*batch).softmax(-1).cpu().numpy()
        for (sample, previous, current), value in zip(destinations, values, strict=True):
            contexts[sample]['probabilities'][current][previous] = value.tolist()
        queued.clear()
        destinations.clear()

    for index in range(count):
        example = dataset.example(index, include_correspondence=True)
        truth = example['correspondence']
        previous, current = example['previous'], example['current']
        contexts.append(dict(index=index, **truth,
            probabilities=[[None for _ in previous] for _ in current]))
        for c in range(len(current)):
            for p in range(len(previous)):
                queued.append(continuation_inputs(previous, current, p, c,
                    truth['previous_identities'][p], truth['previous_modes'][p], truth['current_modes'][c]))
                destinations.append((index, p, c))
                if len(queued) == 64:
                    flush()
    flush()
    return contexts


def assess(contexts, threshold, margin=0.):
    report = Counter({key: 0 for key in ['current_parts', 'matchable', 'pair_same_accepted', 'pair_false_same',
        'competing_rejected', 'accepted', 'correct', 'wrong', 'unknown_identity_accepted',
        'unknown_identity_correct', 'unknown_identity_wrong', 'singleton_accepted', 'singleton_correct', 'singleton_wrong',
        'source_matchable', 'ambiguous', 'matchable_rejected']})
    groups = {}
    wrong, decisions = [], []
    for context in contexts:
        previous, current = context['previous_parts'], context['current_parts']
        duplicate = Counter(current)
        for c, row in enumerate(context['probabilities']):
            report['current_parts'] += 1
            source_matchable = current[c] in previous and duplicate[current[c]] == 1
            report['source_matchable'] += int(source_matchable)
            report['matchable'] += int(source_matchable and context['resolvable'][c])
            report['ambiguous'] += int(not context['resolvable'][c])
            candidates = [p for p, probability in enumerate(row)
                          if int(np.argmax(probability)) == 1 and probability[1] >= threshold]
            for p in candidates:
                report['pair_same_accepted'] += 1
                if previous[p] != current[c] or duplicate[current[c]] > 1 or not context['resolvable'][c]:
                    report['pair_false_same'] += 1
        accepted = {}
        assignments = match_gallery(context['probabilities'], threshold, margin)
        for assignment in assignments:
            c, p = assignment['current'], assignment['previous']
            if not assignment['accepted']:
                report['competing_rejected'] += 1
                continue
            report['accepted'] += 1
            correct = previous[p] == current[c] and duplicate[current[c]] == 1 and context['resolvable'][c]
            accepted[c] = correct
            report['correct' if correct else 'wrong'] += 1
            if context['previous_identities'][p] is None:
                report['unknown_identity_accepted'] += 1
                report['unknown_identity_correct' if correct else 'unknown_identity_wrong'] += 1
            if len(previous) == len(current) == 1:
                report['singleton_accepted'] += 1
                report['singleton_correct' if correct else 'singleton_wrong'] += 1
            decision = dict(index=context['index'], previous=p, current=c, correct=correct,
                            domain=context['domain'], confidence=context['probabilities'][c][p][1],
                            assignment_margin=assignment['margin'])
            decisions.append(decision)
            if not correct:
                wrong.append(decision)
        for c, part in enumerate(current):
            matchable = part in previous and duplicate[part] == 1 and context['resolvable'][c]
            report['matchable_rejected'] += int(matchable and c not in accepted)
            tags = ['single' if len(previous) == len(current) == 1 else 'multiple',
                    'changed_gallery' if context['gallery_changed'] else 'ordered_gallery']
            if context['previous_instruments'].count(context['current_instruments'][c]) > 1:
                tags.append('same_instrument_candidates')
            if duplicate[part] > 1:
                tags.append('duplicate')
            if part not in previous:
                tags.append('new_part')
            if not context['resolvable'][c]:
                tags.append('ambiguous')
            for tag in tags:
                counts = groups.setdefault(tag, Counter({k: 0 for k in
                    ['parts', 'matchable', 'accepted', 'correct', 'wrong', 'matchable_rejected']}))
                counts.update(parts=1, matchable=int(matchable), accepted=int(c in accepted),
                              correct=int(accepted.get(c, False)), wrong=int(c in accepted and not accepted[c]),
                              matchable_rejected=int(matchable and c not in accepted))
    return dict(threshold=threshold, assignment_margin=margin, **dict(report), groups={k: dict(v) for k, v in groups.items()},
                wrong_cases=wrong, decisions=decisions)


def gallery_contexts(model, data, split, samples):
    records = [
        json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines()
    ]
    contexts = []
    for visible in (False, True):
        contexts.extend(
            dict(row, visible_labels=visible)
            for row in predict(model, records, samples, visible_labels=visible)
        )
    return contexts


def evaluate_export(args):
    reader = PartContinuationReader(args.model / "part_continuation.onnx")
    contexts = gallery_contexts(reader.session, args.data, args.split, args.samples)
    report = assess(contexts, reader.config["threshold"], reader.config['assignment_margin'])
    report['thresholds'] = [{key: value for key, value in assess(contexts, threshold, reader.config['assignment_margin']).items()
                             if key not in ['wrong_cases', 'decisions']}
                            for threshold in THRESHOLDS]
    output = args.model / f"evaluation-{args.split}.json"
    output.write_text(
        json.dumps(dict(split=args.split, samples=len(contexts), **report), indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["wrong_cases", "decisions"]},
            indent=2,
        )
    )


def select_threshold(reports):
    """Calibrate normal score continuation independently of stress-case mix.

    Changed/anonymous galleries are deliberately oversampled for diagnosis.
    Their prevalence must not set a rejection threshold for ordinary scores.
    Report their real error/coverage separately instead of hiding that limit.
    """
    eligible = [row for row in reports
                if row['groups']['ordered_gallery']['wrong'] <= (1 - MIN_PRECISION)
                * row['groups']['ordered_gallery']['accepted']]
    return max(eligible, key=lambda row: (row['groups']['ordered_gallery']['correct'],
               -row['groups']['ordered_gallery']['wrong'], -row['wrong'], row['correct'], -row['threshold']))


def export(output, data, samples=1200, device='cpu'):
    import onnx

    torch.set_num_threads(2)
    model = PartContinuationNetwork().eval()
    model.load_state_dict(
        torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    )
    model.to(device)
    contexts = gallery_contexts(model, data, "validation", samples)
    model.cpu()
    reports = [assess(contexts, threshold, margin) for threshold in THRESHOLDS for margin in ASSIGNMENT_MARGINS]
    chosen = select_threshold(reports)
    selection = {
        key: value
        for key, value in chosen.items()
        if key not in ["wrong_cases", "decisions"]
    }
    validation = json.loads((output / "validation.json").read_text())
    config = dict(
        CONTRACT,
        threshold=selection["threshold"],
        assignment_margin=selection['assignment_margin'],
        source_checkpoint=validation["step"],
        visible_label_continuation=True,
        target_precision=MIN_PRECISION,
        calibration_objective='ordered_gallery_coverage',
    )
    target = output / "part_continuation.onnx"
    dummy = (
        torch.zeros(1, 2, 192, 768),
        torch.zeros(1, 4, 256, 384),
        torch.zeros(1, 17),
    )
    torch.onnx.export(
        model,
        dummy,
        target,
        input_names=["rows", "contexts", "features"],
        output_names=["logits"],
        dynamic_axes={
            key: {0: "batch"} for key in ["rows", "contexts", "features", "logits"]
        },
        opset_version=17,
        dynamo=False,
    )
    exported = onnx.load(target)
    onnx.helper.set_model_props(
        exported, {"guitarocr.part_continuation": json.dumps(config)}
    )
    onnx.save(exported, target)
    records = [
        json.loads(line)
        for line in (data / "validation.jsonl").read_text().splitlines()
    ]
    batch = next(
        iter(
            DataLoader(
                ContinuationData(records, 14, False, visible_labels=True), batch_size=14
            )
        )
    )
    tensors = batch[:3]
    with torch.inference_mode():
        expected = model(*tensors).numpy()
    actual = cpu_session(target).run(
        ["logits"],
        {
            name: value.numpy()
            for name, value in zip(
                ["rows", "contexts", "features"], tensors, strict=True
            )
        },
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-4)
    report = dict(
        samples=len(contexts),
        selected=selection,
        thresholds=[{key: value for key, value in row.items()
                     if key not in ['wrong_cases', 'decisions']} for row in reports],
        torch_onnx_max_logit_error=float(np.abs(actual - expected).max()),
    )
    (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def evaluate(model, loader):
    model.eval()
    probabilities, labels, properties = [], [], []
    with torch.inference_mode():
        for rows, contexts, features, target, flags in loader:
            logits = model(rows.cuda(), contexts.cuda(), features.cuda())
            probabilities.extend(logits.softmax(-1).cpu().tolist())
            labels.extend(target.tolist())
            properties.extend(flags.tolist())
    p = np.asarray(probabilities)
    target = np.asarray(labels)
    flags = np.asarray(properties)
    classes = p.argmax(1)
    confidence = p.max(1)
    choices = []
    for threshold in THRESHOLDS:
        same = (classes == 1) & (confidence >= threshold)
        choices.append(dict(threshold=threshold, same_accepted=int(same.sum()),
            same_correct=int((same & (target == 1)).sum()), false_same=int((same & (target != 1)).sum()),
            false_same_instrument_negative=int((same & (flags[:, 0] == 1)).sum()),
            false_same_on_ambiguous=int((same & (target == 2)).sum())))
    eligible = [row for row in choices if row['false_same'] <= (1 - MIN_PRECISION) * row['same_accepted']]
    selection = max(eligible, key=lambda row: (row['same_correct'], -row['threshold']))
    confusion = [[int(((target == a) & (classes == b)).sum()) for b in range(3)] for a in range(3)]
    return dict(samples=len(labels), argmax_correct=int((classes == target).sum()),
        confusion=confusion, selected=selection, thresholds=choices,
        probabilities=probabilities, targets=labels, properties=properties)


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(920317)
    records = {split: [json.loads(line) for line in (args.data / f'{split}.jsonl').read_text().splitlines()]
               for split in ['train', 'validation']}
    families = {split: {f for r in rows for f in r['source_families']} for split, rows in records.items()}
    assert not families['train'] & families['validation']
    train = DataLoader(ContinuationData(records['train'], args.steps * args.batch_size, True,
                                      visible_labels=True),
        batch_size=args.batch_size, shuffle=True, num_workers=6, persistent_workers=True, pin_memory=True)
    validation = DataLoader(ContinuationData(records['validation'], 4000, False,
                                           visible_labels=True),
        batch_size=32, num_workers=4, persistent_workers=True, pin_memory=True)
    model = PartContinuationNetwork().cuda()
    if args.initialize:
        model.load_state_dict(torch.load(args.initialize, map_location='cpu', weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    report = fit_classifier(
        model,
        train,
        optimizer,
        steps=args.steps,
        output=args.output,
        evaluate=lambda current: evaluate(current, validation),
        score_field="same_correct",
        inputs=3,
        scheduler=scheduler,
        clip=5,
    )
    (args.output / "training.json").write_text(
        json.dumps(
            dict(
                **report,
                seed=920317,
                learning_rate=args.learning_rate,
                data=str(args.data),
                initialize=str(args.initialize) if args.initialize else None,
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
    parser_train.add_argument('--steps', type=int, default=1800)
    parser_train.add_argument('--batch-size', type=int, default=32)
    parser_train.add_argument('--initialize', type=Path)
    parser_train.add_argument('--learning-rate', type=float, default=.0005)
    parser_evaluate = commands.add_parser('evaluate')
    parser_evaluate.add_argument('--model', type=Path, required=True)
    parser_evaluate.add_argument('--data', type=Path, required=True)
    parser_evaluate.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser_evaluate.add_argument('--samples', type=int, default=1200)
    parser_export = commands.add_parser('export')
    parser_export.add_argument('--model', type=Path, required=True)
    parser_export.add_argument('--data', type=Path, required=True)
    parser_export.add_argument('--samples', type=int, default=1200)
    parser_export.add_argument('--device', default='cpu', help='Device for validation gallery calibration')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate_export(args)
    elif args.command == 'export':
        export(args.model, args.data, args.samples, args.device)


if __name__ == "__main__":
    main()
