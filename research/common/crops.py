"""Physical context around each measure, shared by training and inference."""

from PIL import Image

MODEL_RENDER_DPI = 180


def crop_measure(page: Image.Image, bbox_pixels, dpi=MODEL_RENDER_DPI) -> Image.Image:
    """Crop a pixel-space xywh box with 1.5 mm horizontal and 4 mm vertical context."""
    left, top, width, height = (float(v) for v in bbox_pixels)
    pad_x, pad_y = 1.5 * dpi / 25.4, 4.0 * dpi / 25.4
    x0, y0 = max(0, round(left - pad_x)), max(0, round(top - pad_y))
    x1 = min(page.width, round(left + width + pad_x))
    y1 = min(page.height, round(top + height + pad_y))
    return page.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))


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
