"""Evaluate printed signatures across complete held-out source splits."""

import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, action='append', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from research.data.signature_data import Signatures
    from research.inference.measures.state_reader import StateReader

    torch.set_num_threads(1)
    dataset = Signatures(args.data, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    model = StateReader(args.model, 'cuda').model
    stats, mistakes = Counter(), []
    offset = 0
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for images, labels in loader:
            outputs = model(images.cuda(non_blocking=True))
            predicted = torch.stack([v.argmax(-1) for v in outputs], -1).cpu()
            probability = torch.stack([v.float().softmax(-1).max(-1).values for v in outputs], -1).cpu()
            absent = (predicted[:, 1] == 0) | (predicted[:, 2] == 0)
            predicted[absent, 1:] = 0
            key_ok = predicted[:, 0] == labels[:, 0]
            time_ok = (predicted[:, 1:] == labels[:, 1:]).all(-1)
            stats['samples'] += len(labels)
            stats['key_correct'] += int(key_ok.sum())
            stats['time_correct'] += int(time_ok.sum())
            for name, mask, correct in [('key', labels[:, 0] != 15, key_ok), ('time', labels[:, 1] != 0, time_ok)]:
                for kind, selected in [('positive', mask), ('negative', ~mask)]:
                    stats[f'{name}_{kind}'] += int(selected.sum())
                    stats[f'{name}_{kind}_correct'] += int(correct[selected].sum())
            for i in torch.where(~key_ok | ~time_ok)[0].tolist():
                mistakes.append({'image': dataset.rows[offset + i][0], 'expected': labels[i].tolist(),
                                 'predicted': predicted[i].tolist(), 'confidence': probability[i].tolist()})
            offset += len(labels)
    result = dict(model=str(args.model), split=args.split, **stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    args.output.with_suffix('.errors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in mistakes))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
