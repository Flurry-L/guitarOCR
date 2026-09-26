"""Read document metadata and resolve the information used by OCR and export."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.defaults import MODEL, INFO_ADAPTER

from document_info.image_ocr import recognize_document_info
from document_info.pdf_metadata import extract_pdf_vector_metadata
from shared.artifacts import read_result, write_json, write_result
from shared.tuning import DEFAULT_TUNING


def run(
    layout: Path, output: Path, *, model: Path = MODEL,
    adapter: Path = INFO_ADAPTER,
    device: str = "cuda", title: str | None = None, artist: str | None = None,
    tuning: list[int] | None = None, capo: int = 0, backend=None,
) -> Path:
    source = read_result(layout, "layout")
    predictions = []
    metadata = {}
    if source["info_source"] == "image":
        metadata, predictions = recognize_document_info(
            source["regions"], model.resolve(), adapter.resolve(), device, backend=backend
        )
    else:
        pdfs = {row.get("source_pdf") for row in source["records"]}
        if len(pdfs) == 1 and None not in pdfs:
            metadata = extract_pdf_vector_metadata(Path(next(iter(pdfs))))
    predictions_path = write_json(output.resolve() / "predictions.json", predictions)
    return write_result(
        output, "document_info", layout=str(layout.resolve()),
        document_metadata=metadata, predictions=str(predictions_path),
        title=title or metadata.get("title") or Path(source["inputs"][0]).stem,
        artist=artist if artist is not None else metadata.get("artist") or "",
        tuning_used=tuning or metadata.get("tuning_midi_high_to_low") or list(DEFAULT_TUNING),
        capo=capo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read metadata from a layout stage result.")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--title")
    parser.add_argument("--artist")
    parser.add_argument("--tuning", help="Comma-separated MIDI pitches from highest to lowest string")
    parser.add_argument("--capo", type=int, default=0)
    args = parser.parse_args()
    if args.tuning is not None:
        args.tuning = [int(value) for value in args.tuning.split(",") if value.strip()]
        if not args.tuning:
            parser.error("--tuning must contain at least one MIDI pitch")
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
