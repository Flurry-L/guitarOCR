from __future__ import annotations

import hashlib
import re

from shared.pdf import open_pdf
from PIL import Image, ImageOps
from pathlib import Path

from layout.pdf_renderer import MODEL_RENDER_DPI, render_pdf_pages


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_FORMATS = ("PNG", "JPEG", "BMP", "TIFF")


def expand_inputs(
    inputs: list[Path], render_root: Path, temp_root: Path, force_pdf_render: bool
) -> list[dict]:
    values: list[Path] = []
    for supplied in inputs:
        if supplied.is_dir():
            values.extend(
                path
                for path in sorted(
                    supplied.iterdir(),
                    key=lambda p: [
                        int(t) if t.isdigit() else t.lower()
                        for t in re.split(r"(\d+)", p.name)
                    ],
                )
                if path.is_file()
                and (
                    path.suffix.lower() in IMAGE_SUFFIXES
                    or path.suffix.lower() == ".pdf"
                )
            )
        elif supplied.is_file() and (
            supplied.suffix.lower() in IMAGE_SUFFIXES
            or supplied.suffix.lower() == ".pdf"
        ):
            values.append(supplied)
        else:
            raise FileNotFoundError(
                f"Not a supported PDF, page image, or directory: {supplied}"
            )
    unique = list(dict.fromkeys(path.resolve() for path in values))
    pages: list[dict] = []
    for value in unique:
        if value.suffix.lower() != ".pdf":
            identity = hashlib.sha1(str(value).encode()).hexdigest()[:8]
            with Image.open(value, formats=IMAGE_FORMATS) as image:
                for frame in range(getattr(image, "n_frames", 1)):
                    image.seek(frame)
                    normalized = render_root / f"{identity}_{frame + 1:03d}.png"
                    normalized.parent.mkdir(parents=True, exist_ok=True)
                    ImageOps.exif_transpose(image).convert("RGB").save(normalized)
                    pages.append(
                        {
                            "image": normalized,
                            "source_pdf": None,
                            "pdf_page": None,
                            "vector": False,
                        }
                    )
            continue
        identity = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
        pdf_output = render_root / f"{value.stem}_{identity}"
        rendered = render_pdf_pages(
            value, pdf_output, temp_root, dpi=MODEL_RENDER_DPI, force=force_pdf_render
        )
        with open_pdf(value) as document:
            pages.extend(
                {
                    "image": path,
                    "source_pdf": value,
                    "pdf_page": index,
                    "vector": document[index - 1].has_vectors(),
                }
                for index, path in enumerate(rendered, start=1)
            )
    return pages
