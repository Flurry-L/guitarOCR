"""Parallel recognition of printed signatures, without text generation."""

from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn


DENOMINATORS = (None, 1, 2, 4, 8, 16, 32, 64)


def signature_views(path, augment=False):
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    gray = np.asarray(image.convert('L'))
    rows = np.flatnonzero((gray < 160).mean(axis=1) > .45)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(rows) else []
    lines = [float(g.mean()) for g in groups if len(g)]
    staff = None
    for i in range(max(0, len(lines) - 4)):
        gaps = np.diff(lines[i:i + 5])
        if gaps.min() >= 3 and gaps.max() < gaps.min() * 1.4:
            staff = lines[i], float(np.median(gaps))
            break
    if staff:
        top, gap = staff
        prefix = image.crop((0, max(0, int(top - gap * 5)),
                             min(image.width, int(gap * 28)),
                             min(image.height, int(top + gap * 10))))
    else:
        prefix = image.crop((0, 0, min(image.width, max(160, image.height)), image.height))
    views = []
    for view in (image, prefix):
        view = ImageOps.contain(view, (512, 192), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (512, 192), 'white')
        x, y = 0, (192 - view.height) // 2
        if augment:
            x = int(np.random.randint(0, min(12, 512 - view.width) + 1))
            y = int(np.clip(y + np.random.randint(-4, 5), 0, 192 - view.height))
        canvas.paste(view, (x, y))
        values = np.array(canvas, dtype=np.float32) / 255.
        if augment and np.random.random() < .3:
            values = np.clip(values * np.random.uniform(.65, 1.) + np.random.uniform(0., .08), 0, 1)
        views.append(values.transpose(2, 0, 1))
    return torch.from_numpy((np.stack(views) - .5) / .5)


class SignatureNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2], nn.AdaptiveAvgPool2d((1, 4)))
        self.head = nn.Sequential(nn.Linear(4096, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 57))

    def forward(self, images):
        features = self.features(images.flatten(0, 1)).reshape(images.shape[0], -1)
        logits = self.head(features)
        return logits[:, :16], logits[:, 16:49], logits[:, 49:]


class StateReader:
    def __init__(self, path: Path, device='cuda'):
        from safetensors.torch import load_file

        self.device = torch.device(device)
        self.model = SignatureNetwork().to(self.device).eval()
        self.model.load_state_dict(load_file(str(path), device=str(self.device)))

    def predict(self, records, batch_size=64, cancelled=None):
        from shared.tasks import Cancelled

        predictions = []
        for start in range(0, len(records), batch_size):
            if cancelled and cancelled():
                raise Cancelled('识别已停止')
            images = torch.stack([signature_views(r['image']) for r in records[start:start + batch_size]])
            with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                        enabled=self.device.type == 'cuda'):
                outputs = self.model(images.to(self.device))
            labels = [v.argmax(-1).cpu().tolist() for v in outputs]
            for key, numerator, denominator in zip(*labels, strict=True):
                value = {'key': key - 7 if key < 15 else None,
                         'time': f'{numerator}/{DENOMINATORS[denominator]}' if numerator and denominator else None}
                predictions.append(value)
        return predictions


@lru_cache(maxsize=2)
def cached_reader(path, device, modified):
    return StateReader(Path(path), device)
