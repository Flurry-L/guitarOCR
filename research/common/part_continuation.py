"""Portable image contract for learned cross-page part correspondence."""

import numpy as np
from PIL import Image

ROW_SIZE = (768, 192)
CONTEXT_SIZE = (384, 256)
IDENTITIES = [None, 'guitar', 'bass', 'pitched', 'drums']
MODES = ['unknown', 'notation', 'tab', 'both']
LABELS = ['different', 'same', 'unknown']
MAX_PARTS = 5
FEATURES = [f'previous_identity_{value or "unknown"}' for value in IDENTITIES]
FEATURES += [f'{side}_mode_{mode}' for side in ['previous', 'current'] for mode in MODES]
FEATURES += ['previous_count_div8', 'current_count_div8',
             'previous_query_index_normalized', 'current_query_index_normalized']
CONTRACT = dict(version=1, row_size=list(ROW_SIZE), context_size=list(CONTEXT_SIZE),
    labels=LABELS, identities=IDENTITIES, modes=MODES, features=FEATURES,
    max_parts=MAX_PARTS, grayscale='pil_l', resize='contain_lanczos_white_center',
    normalization='zero_one', context_channels=['previous_pixels', 'previous_query_mask',
                                               'current_pixels', 'current_query_mask'])


def contain(image, size):
    image = image.convert('L')
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                           Image.Resampling.LANCZOS)
    canvas = Image.new('L', size, 255)
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def context_image(rows, query):
    """Show every current candidate row, with a separate query-position mask."""
    if not rows or len(rows) > MAX_PARTS or not 0 <= query < len(rows):
        raise ValueError('A continuation context needs candidate rows and a valid query')
    width, height = CONTEXT_SIZE
    panel = Image.new('L', (width, height), 255)
    mask = np.zeros((height, width), dtype=np.float32)
    for index, row in enumerate(rows):
        top, bottom = round(index * height / len(rows)), round((index + 1) * height / len(rows))
        cell = contain(Image.fromarray(row), (width, bottom - top))
        panel.paste(Image.fromarray(cell), (0, top))
        if index == query:
            mask[top:bottom] = 1
    return np.stack([np.asarray(panel, dtype=np.float32) / 255, mask])


def continuation_inputs(previous, current, previous_index, current_index,
                        previous_identity, previous_mode, current_mode):
    rows = np.stack([previous[previous_index], current[current_index]]).astype(np.float32) / 255
    contexts = np.concatenate([context_image(previous, previous_index), context_image(current, current_index)])
    features = [float(previous_identity == value) for value in IDENTITIES]
    for mode in (previous_mode, current_mode):
        features.extend(float(mode == value) for value in MODES)
    features.extend([len(previous) / 8, len(current) / 8,
                     previous_index / max(1, len(previous) - 1), current_index / max(1, len(current) - 1)])
    return rows, contexts, np.asarray(features, dtype=np.float32)
