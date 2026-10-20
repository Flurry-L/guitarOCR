"""Staff-label image geometry used by training and OCR detail crops."""

import numpy as np
from research.common.images import contain_lanczos

WIDTH, HEIGHT, STRIDE = 768, 256, 4
ANGLES = (0, 90, 180, 270)


def staff_label_image(image, size=(WIDTH, HEIGHT)):
    image = image.convert("RGB")
    canvas, (width, height, left, top) = contain_lanczos(image, size)
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    return pixels, (width / image.width, height / image.height, left, top)


