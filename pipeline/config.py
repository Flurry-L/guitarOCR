from __future__ import annotations

import argparse
from pathlib import Path

from shared.defaults import MODEL, MEASURE_ADAPTER, INFO_ADAPTER


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize regular guitar score PDF pages or images as score text and GP5."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("auto", "tab", "notation", "both"), default="auto"
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=MEASURE_ADAPTER,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-new-tokens-ceiling",
        type=int,
        default=2048,
        help=(
            "upper bound for adaptive structural retries; unterminated optional "
            "text retries retain --max-new-tokens"
        ),
    )
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume accepted measures in output/03_measure_ocr/recognition.jsonl",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Export GP5 even when OCR substituted rests",
    )
    parser.add_argument("--force-pdf-render", action="store_true")
    parser.add_argument("--layout-model-dir", type=Path)
    parser.add_argument("--layout-python", type=Path)
    parser.add_argument(
        "--info-adapter",
        type=Path,
        default=INFO_ADAPTER,
    )
    parser.add_argument(
        "--layout-source",
        choices=("auto", "image", "geometry"),
        default="auto",
        help="use the image layout model for rendered PDFs as well as images",
    )
    parser.add_argument("--title")
    parser.add_argument("--artist")
    parser.add_argument("--tuning")
    parser.add_argument("--capo", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_new_tokens_ceiling < args.max_new_tokens:
        parser.error("--max-new-tokens-ceiling must be >= --max-new-tokens")
    if args.max_new_tokens <= 0 or args.maximum_attempts < 1:
        parser.error("--max-new-tokens and --maximum-attempts must be positive")
    if args.tuning is not None:
        try:
            args.tuning = [
                int(value) for value in args.tuning.split(",") if value.strip()
            ]
        except ValueError:
            parser.error("--tuning must contain comma-separated MIDI pitches")
        if not args.tuning:
            parser.error("--tuning must contain at least one MIDI pitch")
    return args
