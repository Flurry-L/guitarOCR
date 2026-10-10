"""Image-only spatial clef contract, shared by training and inference."""

import numpy as np
from PIL import Image

SIZE = (256, 256)
LABELS = [dict(clef=clef, clef_octave=octave) for clef in ['G2', 'F4'] for octave in [-24, -12, 0, 12, 24]]
LABELS += [dict(clef=clef, clef_octave=0) for clef in ['C3', 'C4', 'percussion', 'tab']]
CONTRACT = dict(version=1, image_size=list(SIZE), grayscale='pil_l',
                resize='contain_lanczos_white_center', normalization='zero_one', labels=LABELS)


def image_tensor(image):
    image = image.convert('L')
    scale = min(SIZE[0] / image.width, SIZE[1] / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                           Image.Resampling.LANCZOS)
    canvas = Image.new('L', SIZE, 255)
    canvas.paste(resized, ((SIZE[0] - resized.width) // 2, (SIZE[1] - resized.height) // 2))
    return np.asarray(canvas, dtype=np.float32)[None] / 255
