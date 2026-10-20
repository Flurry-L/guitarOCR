"""Spatial clef context and query mask, shared by training and inference."""

import math
import numpy as np
from research.common.images import contain_lanczos

SIZE = (256, 256)
LABELS = [dict(clef=clef, clef_octave=octave) for clef in ['G2', 'F4'] for octave in [-24, -12, 0, 12, 24]]
LABELS += [dict(clef=clef, clef_octave=0) for clef in ['C3', 'C4', 'percussion', 'tab']]
CONTRACT = dict(version=1, image_size=list(SIZE), grayscale='pil_l',
                resize='contain_lanczos_white_center', normalization='zero_one', labels=LABELS,
                channels=2, context_padding_ratio=2)


def clef_context(image, query_bbox):
    x, y, width, height = query_bbox
    padding = CONTRACT['context_padding_ratio'] * max(width, height)
    left, top = max(0, math.floor(x - padding)), max(0, math.floor(y - padding))
    right, bottom = min(image.width, math.ceil(x + width + padding)), min(image.height, math.ceil(y + height + padding))
    return image.crop((left, top, right, bottom)), [x - left, y - top, width, height]


def image_tensor(image, query_bbox):
    image, query = clef_context(image, query_bbox)
    image = image.convert('L')
    scale = min(SIZE[0] / image.width, SIZE[1] / image.height)
    canvas, (_, _, left, top) = contain_lanczos(image, SIZE)
    x, y, width, height = query
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.float32)
    x0, y0 = max(0, math.floor(x * scale + left)), max(0, math.floor(y * scale + top))
    x1, y1 = min(SIZE[0], math.ceil((x + width) * scale + left)), min(SIZE[1], math.ceil((y + height) * scale + top))
    mask[y0:y1, x0:x1] = 1
    return np.stack([np.asarray(canvas, dtype=np.float32) / 255, mask])
