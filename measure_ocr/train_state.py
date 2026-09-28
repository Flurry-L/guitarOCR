"""Train the printed-signature classifier on complete-source data splits."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

from measure_ocr.state_reader import DENOMINATORS, SignatureNetwork, signature_views
from shared.score_state import parse_signature


class Signatures(Dataset):
    def __init__(self, roots, split, augment=False):
        self.rows, self.augment = [], augment
        for root in roots:
            path = root / f'state_{split}.jsonl'
            for line in path.open():
                row = json.loads(line)
                state = parse_signature(row['messages'][1]['content'])
                n, d = map(int, state['time'].split('/')) if state['time'] else (0, None)
                label = [state['key'] + 7 if state['key'] is not None else 15, n, DENOMINATORS.index(d)]
                self.rows.append((row['images'][0], label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        return signature_views(path, self.augment), torch.tensor(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--initial', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260928)
    model = SignatureNetwork(pretrained=args.initial is None).cuda()
    if args.initial:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(args.initial)))
    training = Signatures(args.data, 'train', True)
    validation = Signatures(args.data, 'validation')
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, persistent_workers=True)
    counts = Counter(label[0] for _, label in training.rows)
    weights = torch.tensor([1 / math.sqrt(max(1, counts[i])) for i in range(16)], device='cuda')
    key_loss = nn.CrossEntropyLoss(weight=weights / weights.mean())
    other_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(training=len(training), validation=len(validation), keys=dict(counts))), flush=True)
    best, step, started = 0., 0, time.perf_counter()
    total_steps = args.epochs * len(loader)
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        for images, labels in loader:
            step += 1
            factor = min(1., step / 100) * (.05 + .95 * .5 * (1 + math.cos(math.pi * step / total_steps)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                key, numerator, denominator = model(images)
                loss = key_loss(key.float(), labels[:, 0]) + other_loss(numerator, labels[:, 1]) + other_loss(denominator, labels[:, 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item()
            if step % 100 == 0:
                print(json.dumps(dict(step=step, epoch=epoch, loss=loss_sum/100,
                                      seconds=time.perf_counter()-started)), flush=True)
                loss_sum = 0.
        model.eval()
        stats = Counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for images, labels in val_loader:
                labels = labels.cuda(non_blocking=True)
                outputs = model(images.cuda(non_blocking=True))
                predicted = torch.stack([v.argmax(-1) for v in outputs], -1)
                key_ok = predicted[:, 0] == labels[:, 0]
                time_ok = (predicted[:, 1:] == labels[:, 1:]).all(-1)
                stats['samples'] += len(labels)
                stats['key_correct'] += int(key_ok.sum())
                stats['time_correct'] += int(time_ok.sum())
                for name, mask, correct in [('key',labels[:, 0] != 15,key_ok),('time',labels[:, 1] != 0,time_ok)]:
                    stats[name+'_positive'] += int(mask.sum())
                    stats[name+'_positive_correct'] += int(correct[mask].sum())
                    stats[name+'_negative'] += int((~mask).sum())
                    stats[name+'_negative_correct'] += int(correct[~mask].sum())
        score = sum(stats[f'{name}_{kind}_correct']/max(1,stats[f'{name}_{kind}'])
                    for name in ('key','time') for kind in ('positive','negative')) / 4
        report = dict(epoch=epoch+1, score=score, **stats)
        print(json.dumps(report), flush=True)
        with (args.output/'metrics.jsonl').open('a') as log:
            log.write(json.dumps(report)+'\n')
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, args.output/f'epoch-{epoch+1}.safetensors')
        if score > best:
            best=score
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, args.output/'state.safetensors')
    print('Signature classifier training complete',flush=True)


if __name__ == '__main__':
    main()
