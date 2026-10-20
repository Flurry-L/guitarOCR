"""Shared OCR image policy and declared recognition task adapters."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps
from research.common.staff_label import staff_label_image

def load_policy(model):
    path = Path(model) / "score_image_policy.json"
    return json.loads(path.read_text())


def normalize_score_image(image, policy):
    if policy.get("version") != 2:
        raise ValueError("Unsupported score image policy version")
    minimum = policy["min_pixels"]
    budget = policy.get("max_pixels")
    page_budget = policy.get("page_max_pixels", budget)
    multiple = policy.get("patch_multiple")
    upscale = policy.get("max_upscale")
    if (
        type(minimum) is not int
        or type(budget) is not int
        or type(page_budget) is not int
        or not 112 * 112 <= minimum <= budget <= 40_000_000
        or not 112 * 112 <= page_budget <= 40_000_000
        or type(multiple) is not int
        or not 1 <= multiple <= 256
        or min(budget, page_budget) < multiple * multiple
        or type(upscale) not in (int, float)
        or not math.isfinite(upscale)
        or not 0 < upscale <= 16
    ):
        raise ValueError("Invalid score image policy bounds")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image dimensions must be positive")
    # Pixel content must not decide image scale: line thickness, broken rules,
    # chord grids and notation style are recognition inputs, not resize hints.
    area = image.width * image.height
    scale = min(
        upscale, max(1.0, math.sqrt(minimum / area))
    )
    if image.width >= 600 and image.height >= 900:
        budget = page_budget
    # Reserve padding inside the budget so the processor never has to resize
    # this already normalized image again to meet its patch geometry.
    scale = min(scale, math.sqrt(budget / (image.width * image.height)))
    while True:
        width, height = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        padded = (
            math.ceil(width / multiple) * multiple,
            math.ceil(height / multiple) * multiple,
        )
        if padded[0] * padded[1] <= budget:
            break
        scale *= 0.98
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    # Small crops also need the same minimum area as the GLM image processor.
    while padded[0] * padded[1] < 112 * 112:
        padded = (padded[0] + multiple, padded[1])
    if padded[0] * padded[1] > budget:
        raise ValueError("Image policy cannot fit minimum processor area")
    canvas = Image.new("RGB", padded, "white")
    canvas.paste(image, ((padded[0] - width) // 2, (padded[1] - height) // 2))
    return canvas


@lru_cache(maxsize=4)
def _staff_label_reader(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    return session, json.loads(session.get_modelmeta().custom_metadata_map['guitarocr.staff_label'])


def staff_detail(image, label_model, policy):
    """The product's learned label crop, orientation and detail image budget."""
    import numpy as np

    session, config = _staff_label_reader(str(label_model))
    width, height = config['input_size']
    image = image.convert('RGB')
    pixels, (sx, sy, left, top) = staff_label_image(image, (width, height))
    scores, bounds, directions = session.run(
        ['label_logits', 'bounds', 'orientation_logits'], {'images': pixels[None]})
    index = int(scores.argmax())
    probability = float(1 / (1 + np.exp(-np.clip(scores.flatten()[index], -80, 80))))
    if probability < config['label_threshold']:
        return None
    stride = config['stride']
    y, x = divmod(index, width // stride)
    left_extent, top_extent, right_extent, bottom_extent = map(float, bounds.reshape(4, -1)[:, index])
    x0, y0 = max(0., (x * stride - left_extent * width - left) / sx), max(0., (y * stride - top_extent * height - top) / sy)
    x1, y1 = min(image.width, (x * stride + right_extent * width - left) / sx), min(image.height, (y * stride + bottom_extent * height - top) / sy)
    if x1 <= x0 or y1 <= y0:
        return None
    angle = config['angles_ccw'][int(directions.reshape(4, -1)[:, index].argmax())]
    crop = image.crop((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))).rotate(angle, expand=True)
    return normalize_score_image(crop, dict(policy, max_upscale=16.))


def install_training_policy(model):
    policy = load_policy(model)
    from llamafactory.data.mm_plugin import MMPluginMixin

    def preprocess(self, image, image_max_pixels=None, image_min_pixels=None, **kwargs):
        return normalize_score_image(image, policy)

    MMPluginMixin._preprocess_image = preprocess


INFORMATION_TASKS = frozenset({'staff', 'clef', 'tempo'})


@dataclass(frozen=True)
class TaskAdapter:
    path: Path
    tasks: frozenset[str]

    def applies(self, task):
        return validate_task(task) in self.tasks


def load_task_adapter(model_directory, runtime):
    root = Path(model_directory)
    capabilities = root / 'capabilities.json'
    declaration = json.loads(capabilities.read_text()).get('information_adapter') if capabilities.is_file() else None
    if declaration is None:
        return None
    if (not isinstance(declaration, dict)
            or not isinstance(declaration.get('tasks'), list)
            or not declaration['tasks']
            or any(not isinstance(t, str) for t in declaration['tasks'])
            or len(set(declaration['tasks'])) != len(declaration['tasks'])
            or not set(declaration['tasks']) <= INFORMATION_TASKS):
        raise ValueError('Unsupported information adapter tasks')
    relative = declaration.get(runtime)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f'Missing declared information adapter for {runtime}')
    path = (root / relative).resolve()
    required = [path] if runtime == 'gguf' else [path / 'adapter_config.json', path / 'adapter_model.safetensors']
    if not all(p.is_file() for p in required):
        raise FileNotFoundError(f'Declared information adapter is missing: {path}')
    return TaskAdapter(path, frozenset(declaration['tasks']))


def validate_task(task):
    if task is not None and task not in INFORMATION_TASKS:
        raise ValueError(f'Unsupported recognition adapter task: {task}')
    return task


def batch_tasks(task, count):
    tasks = task if isinstance(task, list) else [task] * count
    if len(tasks) != count:
        raise ValueError('Each prompt needs its own adapter task')
    return [validate_task(t) for t in tasks]
