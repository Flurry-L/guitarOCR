"""Assemble layout, document information, measure OCR and GP5 export."""

from __future__ import annotations

import argparse
import json

from layout import run as layout_stage
from document_info import run as info_stage
from measure_ocr import run as measure_stage
from gp5_export import run as export_stage
from pipeline.config import parse_args
from pipeline.manifest import RunManifest
from shared.artifacts import read_result
from shared.glm_backend import BackendPool


def run(args: argparse.Namespace) -> dict:
    manifest = RunManifest(args.output, args.inputs)
    backends = BackendPool(args.model, args.device)
    with manifest.stage("layout", "01_layout") as output:
        layout = layout_stage.run(
            args.inputs, output, mode=args.mode, force_pdf_render=args.force_pdf_render,
            layout_model_dir=args.layout_model_dir, layout_python=args.layout_python,
            layout_source=args.layout_source,
        )
    with manifest.stage("document_info", "02_document_info") as output:
        info = info_stage.run(
            layout, output, model=args.model, adapter=args.info_adapter,
            device=args.device, title=args.title, artist=args.artist,
            tuning=args.tuning, capo=args.capo, backend=backends.adapter(args.info_adapter),
        )
    with manifest.stage("measure_ocr", "03_measure_ocr") as output:
        recognition = measure_stage.run(
            layout, info, output, model=args.model, adapter=args.adapter,
            device=args.device, max_new_tokens=args.max_new_tokens,
            max_new_tokens_ceiling=args.max_new_tokens_ceiling,
            maximum_attempts=args.maximum_attempts, resume=args.resume,
            backend=backends.adapter(args.adapter),
        )
    recognized = read_result(recognition, "measure_ocr")
    gp5 = None
    if not recognized.get("review_measures") or args.allow_unreviewed:
        with manifest.stage("gp5_export", "04_gp5_export") as output:
            exported = export_stage.run(recognition, output, allow_unreviewed=args.allow_unreviewed)
            gp5 = read_result(exported, "gp5_export")["gp5"]
    return manifest.complete(
        **{key: recognized[key] for key in (
            "mode", "measures", "m2", "score_text", "recognition_log", "document_metadata", "tuning_used", "review_measures",
        )},
        gp5=gp5,
    )


def main() -> None:
    result = run(parse_args())
    print(json.dumps(
        {key: result[key] for key in ("status", "mode", "measures", "review_measures", "score_text", "gp5")},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
