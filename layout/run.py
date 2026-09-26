"""Locate measures and document regions; persist the input for subsequent stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layout.crops import prepare_document_crops
from shared.artifacts import write_result


def run(
    inputs: list[Path], output: Path, *, mode: str = "auto",
    force_pdf_render: bool = False, layout_model_dir: Path | None = None,
    layout_python: Path | None = None, layout_source: str = "auto",
    pages: list[dict] | None = None, allow_empty: bool = False,
    detector=None,
) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode, records = prepare_document_crops(
        inputs, output, mode, force_pdf_render,
        layout_model_dir, layout_python, layout_source, pages, allow_empty,
        detector,
    )
    regions = json.loads((output / "document_regions.json").read_text(encoding="utf-8"))
    return write_result(
        output, "layout", mode=mode,
        inputs=[str(path.resolve()) for path in inputs],
        info_source="image" if regions else "pdf",
        regions=regions, records=records,
        pages=json.loads((output / "pages.json").read_text(encoding="utf-8")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate and crop measures in PDF pages or images.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("auto", "tab", "notation", "both"), default="auto")
    parser.add_argument("--force-pdf-render", action="store_true")
    parser.add_argument("--layout-model-dir", type=Path)
    parser.add_argument("--layout-python", type=Path)
    parser.add_argument("--layout-source", choices=("auto", "image", "geometry"), default="auto")
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
