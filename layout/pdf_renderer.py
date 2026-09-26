from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from shared.pdf import open_pdf
from PIL import Image

MODEL_RENDER_DPI = 180


def cache_matches(manifest_path: Path, pdf: Path, page_count: int, dpi: int) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = pdf.stat()
    expected = {
        "source_pdf": str(pdf.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "dpi": dpi,
        "page_count": page_count,
        "mode": "gray",
        "renderer": "pdfium",
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def validate_rendered_pages(paths: list[Path], page_count: int) -> list[dict]:
    if len(paths) != page_count:
        raise ValueError(f"Expected {page_count} rendered pages, found {len(paths)}")
    metadata: list[dict] = []
    for page_number, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            metadata.append(
                {
                    "page": page_number,
                    "file": path.name,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )
    return metadata


def render_pdf_pages(
    pdf: Path,
    output_dir: Path,
    temp_root: Path,
    dpi: int = MODEL_RENDER_DPI,
    force: bool = False,
) -> list[Path]:
    pdf = pdf.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Not a PDF file: {pdf}")
    if dpi != MODEL_RENDER_DPI:
        raise ValueError(
            f"Current recognition models require {MODEL_RENDER_DPI} DPI; received {dpi} DPI."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    with open_pdf(pdf) as document:
        page_count = len(document)
    manifest_path = output_dir / "render_manifest.json"
    cached_pages = sorted(output_dir.glob("page_*.png"))
    if (
        not force
        and cache_matches(manifest_path, pdf, page_count, dpi)
        and len(cached_pages) == page_count
    ):
        validate_rendered_pages(cached_pages, page_count)
        return cached_pages

    with tempfile.TemporaryDirectory(prefix="guitarocr_pdf_", dir=temp_root) as temporary:
        rendered = []
        with open_pdf(pdf) as document:
            for index, page in enumerate(document, start=1):
                destination = Path(temporary) / f"page_{index:03d}.png"
                page.render(dpi).save(
                    str(destination)
                )
                rendered.append(destination)
        metadata = validate_rendered_pages(rendered, page_count)
        for stale in output_dir.glob("page_*.png"):
            stale.unlink()
        final_pages: list[Path] = []
        for index, source in enumerate(rendered, start=1):
            destination = output_dir / f"page_{index:03d}.png"
            shutil.move(str(source), destination)
            final_pages.append(destination)

    stat = pdf.stat()
    manifest = {
        "schema_version": "1.0",
        "source_pdf": str(pdf),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "dpi": dpi,
        "mode": "gray",
        "page_count": page_count,
        "pages": [
            {**item, "file": f"page_{item['page']:03d}.png"} for item in metadata
        ],
        "renderer": "pdfium",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_rendered_pages(final_pages, page_count)
    return final_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Render a PDF to the {MODEL_RENDER_DPI}-DPI grayscale pages required by GuitarOCR."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--temp-root", type=Path,
        help="Temporary rendering directory (default: OUTPUT/tmp)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pages = render_pdf_pages(
        args.pdf, args.output, args.temp_root or args.output / "tmp", force=args.force
    )
    print(json.dumps({
        "pdf": str(args.pdf.resolve()),
        "dpi": MODEL_RENDER_DPI,
        "pages": len(pages),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
