"""Map-style printed-signature dataset shared by training and evaluation.

Reading labels needs no training framework. Torch is loaded only when the
DataLoader requests an image and label tensor, using the inference transform.
"""

import json

from research.inference.measures.state_reader import DENOMINATORS
from research.inference.measures.state_reader import signature_views
from scorelib.score_state import parse_signature


class Signatures:
    def __init__(self, roots, split, augment=False):
        self.rows, self.augment = [], augment
        for root in roots:
            path = root / f'state_{split}.jsonl'
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    state = parse_signature(row['messages'][1]['content'])
                    n, d = map(int, state['time'].split('/')) if state['time'] else (0, None)
                    label = [state['key'] + 7 if state['key'] is not None else 15, n, DENOMINATORS.index(d)]
                    self.rows.append((row['images'][0], label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        path, label = self.rows[index]
        return signature_views(path, self.augment), torch.tensor(label)

