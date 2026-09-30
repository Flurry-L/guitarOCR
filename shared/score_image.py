"""Identical staff-scale, aspect-preserving inputs for training and recognition."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


DEFAULT_POLICY = {'version': 1, 'notation_spacing': 12, 'tab_spacing': 18,
                  'max_pixels': 768 * 768, 'max_upscale': 2.5, 'patch_multiple': 28}


def load_policy(model):
    path = Path(model) / 'score_image_policy.json'
    return json.loads(path.read_text()) if path.is_file() else None


def staff_scale(image):
    """Find repeated long horizontal strokes; reject uncertain spacing estimates."""
    gray = np.asarray(image.convert('L'))
    if min(gray.shape) < 24:
        return None
    ink = (gray < 225).mean(axis=1)
    rows = np.flatnonzero(ink > max(.22, float(ink.max()) * .50))
    if len(rows) < 5:
        return None
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    centers = np.array([np.average(r, weights=ink[r]) for r in runs])
    strengths = np.array([ink[r].max() for r in runs])
    candidates = []
    for start in range(len(centers) - 4):
        for count in (5, 6, 7, 8):
            if start + count > len(centers):
                continue
            selected = centers[start:start + count]
            gaps = np.diff(selected)
            spacing = float(np.median(gaps))
            if not 4 <= spacing <= 48 or np.max(np.abs(gaps - spacing)) > max(1.3, spacing * .13):
                continue
            strength = float(np.mean(strengths[start:start + count]))
            candidates.append((count * strength, spacing, count))
    if not candidates:
        return None
    _, spacing, lines = max(candidates)
    return spacing, lines


def normalize_score_image(image, policy=None):
    policy = policy or DEFAULT_POLICY
    image = ImageOps.exif_transpose(image).convert('RGB')
    spacing = staff_scale(image)
    scale = 1.
    if spacing:
        gap, lines = spacing
        target = policy['tab_spacing'] if lines >= 6 else policy['notation_spacing']
        scale = min(policy['max_upscale'], target / gap)
    # Compact diagrams have both horizontal and vertical grids. A thick barre
    # can hide one row, so staff-line counting must not shrink their fine dots
    # or finger numerals. Give all compact symbol crops a stable minimum size.
    short_edge, long_edge = sorted(image.size)
    minimum = policy.get('small_crop_min_side', 0)
    if (minimum and long_edge <= policy.get('small_crop_max_dimension', 512)
            and long_edge / short_edge <= policy.get('small_crop_max_aspect', 2)):
        scale = max(scale, min(policy.get('small_crop_max_upscale', 3), minimum / short_edge))
    multiple = int(policy['patch_multiple'])
    budget = int(policy['max_pixels'])
    if image.width >= 600 and image.height >= 900:
        budget = int(policy.get('page_max_pixels', budget))
    # Reserve padding inside the budget so the processor never has to resize
    # this already normalized image again to meet its patch geometry.
    scale = min(scale, math.sqrt(budget / (image.width * image.height)))
    while True:
        width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        padded = (math.ceil(width / multiple) * multiple, math.ceil(height / multiple) * multiple)
        if padded[0] * padded[1] <= budget:
            break
        scale *= .98
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    # Small crops also need the same minimum area as the GLM image processor.
    while padded[0] * padded[1] < 112 * 112:
        padded = (padded[0] + multiple, padded[1])
    canvas = Image.new('RGB', padded, 'white')
    canvas.paste(image, ((padded[0] - width) // 2, (padded[1] - height) // 2))
    return canvas


def install_training_policy(model):
    policy = load_policy(model)
    if policy is None:
        return
    from llamafactory.data.mm_plugin import MMPluginMixin

    def preprocess(self, image, image_max_pixels=None, image_min_pixels=None, **kwargs):
        return normalize_score_image(image, policy)

    MMPluginMixin._preprocess_image = preprocess


def normalize_messages(messages, policy):
    if not policy:
        return messages
    result = []
    for message in messages:
        if isinstance(message.get('content'), str):
            result.append(message)
            continue
        content = []
        for item in message.get('content', []):
            if item.get('type') == 'image':
                with Image.open(item['url']) as source:
                    item = {'type': 'image', 'image': normalize_score_image(source, policy)}
            content.append(item)
        result.append({**message, 'content': content})
    return result
