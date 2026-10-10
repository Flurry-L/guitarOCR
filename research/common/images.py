"""Deterministic RGB8 resampling shared by layout training and inference.

The cubic contract is the scalar fixed-point algorithm in
``backend/src/image_transforms.rs``. OpenCV's IPP and baseline SIMD kernels
have different final rounding, even when ``setUseOptimized(False)`` is set.
Do not substitute ``cv2.resize`` here: a one-byte tie can change model scores.

Kernel and coefficient provenance: OpenCV 4.13.0 modules/imgproc/src/resize.cpp;
see THIRD_PARTY_NOTICES.md for the license and source notice. This NumPy
implementation uses bounded row blocks and integer accumulation, without a
native extension or architecture-dependent vector rounding.
"""

from pathlib import Path

import numpy as np
from PIL import Image

def _cubic_taps(source: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / (target / source)
    position = ((np.arange(target, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    base = np.floor(position).astype(np.int64)
    t = position - base.astype(np.float32)
    p, q = t + np.float32(1), np.float32(1) - t
    a = np.float32(-0.75)
    left = ((a * p - np.float32(5) * a) * p + np.float32(8) * a) * p - np.float32(4) * a
    center = ((a + np.float32(2)) * t - (a + np.float32(3))) * t * t + np.float32(1)
    right = ((a + np.float32(2)) * q - (a + np.float32(3))) * q * q + np.float32(1)
    fourth = np.float32(1) - left - center - right
    weights = np.rint(np.stack((left, center, right, fourth), axis=1) * np.float32(2048)).astype(np.int32)
    pixels = np.clip(base[:, None] + np.arange(-1, 3), 0, source - 1)
    return pixels, weights


def resize_rgb_cubic(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize uint8 RGB using scalar half-up cubic, returning a fresh array.

    Half-pixel positions and cubic polynomials round to float32 before the
    ties-to-even 11-bit coefficient quantization. Borders replicate. Pixel
    products accumulate as integers; the final 22-bit shift rounds half up
    before clipping to [0,255]. No antialias filter is applied on downsampling.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Cubic input must be a uint8 RGB image")
    source_height, source_width = rgb.shape[:2]
    if min(source_height, source_width, width, height) <= 0:
        raise ValueError("Cubic image dimensions must be positive")
    if (source_height, source_width) == (height, width):
        return rgb.copy()
    xs, xw = _cubic_taps(source_width, width)
    ys, yw = _cubic_taps(source_height, height)
    output = np.empty((height, width, 3), dtype=np.uint8)
    # Bound temporary storage independently of page height. Each block only
    # computes the source rows referenced by its vertical footprints.
    for start in range(0, height, 32):
        stop = min(start + 32, height)
        rows, inverse = np.unique(ys[start:stop], return_inverse=True)
        inverse = inverse.reshape(stop - start, 4)
        horizontal = np.zeros((len(rows), width, 3), dtype=np.int32)
        for tap in range(4):
            horizontal += rgb[rows[:, None], xs[None, :, tap]] * xw[None, :, tap, None]
        vertical = np.zeros((stop - start, width, 3), dtype=np.int64)
        for tap in range(4):
            vertical += horizontal[inverse[:, tap]].astype(np.int64) * yw[start:stop, tap, None, None]
        output[start:stop] = np.clip((vertical + (1 << 21)) >> 22, 0, 255).astype(np.uint8)
    return output


MODEL_RENDER_DPI = 180


def measure_crop_bounds(
    page: Image.Image, bbox_pixels, dpi=MODEL_RENDER_DPI, *, horizontal_mm=1.5
):
    """Exact xyxy origin of the materialized measure crop (ties-to-even rounding)."""
    left, top, width, height = (float(v) for v in bbox_pixels)
    pad_x, pad_y = horizontal_mm * dpi / 25.4, 4.0 * dpi / 25.4
    x0, y0 = max(0, round(left - pad_x)), max(0, round(top - pad_y))
    x1 = min(page.width, round(left + width + pad_x))
    y1 = min(page.height, round(top + height + pad_y))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def crop_measure(page: Image.Image, bbox_pixels, dpi=MODEL_RENDER_DPI) -> Image.Image:
    """Crop a pixel-space xywh box with 1.5 mm horizontal and 4 mm vertical context."""
    return page.crop(measure_crop_bounds(page, bbox_pixels, dpi))


def crop_region(image: Image.Image, box: list[float], pad: int) -> Image.Image:
    """Crop an xyxy pixel region for information OCR, with pixel padding."""
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, int(left) - pad),
            max(0, int(top) - pad),
            min(image.width, int(right) + pad + 1),
            min(image.height, int(bottom) + pad + 1),
        )
    ).convert("RGB")


LETTERBOX = "letterbox_v1"
METADATA_KEY = "guitarocr.layout_preprocess"
SIZE_METADATA_KEY = "guitarocr.layout_input_size"
SIZE = 800


def validate_contract(value: str) -> str:
    if value != LETTERBOX:
        raise ValueError(f"Unsupported layout preprocessing contract: {value}")
    return value


def validate_size(size: int) -> int:
    if not 32 <= size <= 2048 or size % 32:
        raise ValueError(f"Unsupported layout input size: {size}")
    return size


def paddle_preprocessing(model_dir: Path) -> tuple[str, int]:
    """Read the exported training transforms; reject unrecognized geometry."""
    import yaml

    config = yaml.safe_load((Path(model_dir) / "inference.yml").read_text())
    operations = config["Preprocess"]
    if any(row["type"] not in {"Resize", "Pad", "NormalizeImage", "Permute"} for row in operations):
        raise ValueError("Unsupported layout preprocessing operation")
    normalization = [row for row in operations if row["type"] == "NormalizeImage"]
    if (len(normalization) != 1
            or normalization[0].get("mean") != [0, 0, 0]
            or normalization[0].get("std") != [1, 1, 1]
            or normalization[0].get("norm_type") != "none"
            or normalization[0].get("is_scale", True) is not True):
        raise ValueError("Unsupported layout normalization")
    geometry = [row for row in config["Preprocess"]
                if row["type"] in {"Resize", "Pad"}]
    dimensions = geometry[0].get("target_size") if geometry else None
    if not dimensions or len(dimensions) != 2 or dimensions[0] != dimensions[1]:
        raise ValueError("Layout input must have a square canvas")
    size = validate_size(int(dimensions[0]))
    if (len(geometry) == 3
            and [row["type"] for row in geometry] == ["Resize", "Pad", "Resize"]
            and geometry[0].get("keep_ratio") is True
            and geometry[2].get("keep_ratio") is False
            and all(row.get("target_size") == [size, size] for row in (geometry[0], geometry[2]))
            and geometry[1].get("size") == [size, size]
            and geometry[1].get("pad_mode") == -1
            and geometry[1].get("offsets") == [0, 0]
            and geometry[1].get("fill_value") == [255, 255, 255]):
        return LETTERBOX, size
    raise ValueError("Unsupported layout resize/padding transforms")


def prepare_image(rgb: np.ndarray, contract: str, size: int = SIZE) -> dict[str, np.ndarray]:
    """Bound allocation to the network canvas and retain page coordinates."""
    validate_contract(contract)
    validate_size(size)
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Layout image dimensions must be positive")
    ratio = size / max(height, width)
    resized_height = max(1, int(height * ratio + 0.5))
    resized_width = max(1, int(width * ratio + 0.5))
    resized = resize_rgb_cubic(rgb, resized_width, resized_height)
    if (resized_height, resized_width) != (size, size):
        canvas = np.full((size, size, 3), 255, dtype=resized.dtype)
        canvas[:resized_height, :resized_width] = resized
        resized = canvas
    return {
        "image": resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0,
        # DocLayoutV3 normalizes against the full canvas, including padding.
        "im_shape": np.array([[size, size]], dtype=np.float32),
        "scale_factor": np.array([[resized_height / height, resized_width / width]], dtype=np.float32),
    }


MODES = ("tab", "notation", "both")
MEASURE_LABELS = {f"measure_{mode}": mode for mode in MODES}
HEADER_REGION_LABELS = ("title_region", "subtitle_region", "credit_region", "tuning_region", "header_text_region")
TYPED_CATEGORIES = (*MEASURE_LABELS, "tempo_region", "clef_region", "annotation_region", *HEADER_REGION_LABELS)
