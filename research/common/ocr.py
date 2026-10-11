"""Shared OCR image policy and declared recognition task adapters."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

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
