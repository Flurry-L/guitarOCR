"""Evaluate complete inputs, including automatic layout and predicted context."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from measure_ocr.metrics import MeasureSequenceMetrics
from pipeline.config import parse_args
from pipeline.run import run
from shared.artifacts import write_json


def evaluate(cases_file, output, device="cuda", layout_python=None):
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    results = []
    torch = None
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(device)
    for index, case in enumerate(cases, start=1):
        inputs = [(cases_file.parent / name).resolve() for name in case["inputs"]]
        expected_path = (cases_file.parent / case["expected_m2"]).resolve()
        expected = expected_path.read_text(encoding="utf-8").splitlines()
        arguments = [
            *[str(p) for p in inputs],
            "--output",
            str(output / f"case-{index:03d}"),
            "--device",
            device,
            "--mode",
            case.get("mode", "tab"),
        ]
        if layout_python:
            arguments += ["--layout-python", str(layout_python)]
        if case.get("layout_source"):
            arguments += ["--layout-source", case["layout_source"]]
        started = time.perf_counter()
        result = run(parse_args(arguments))
        predicted = Path(result["m2"]).read_text(encoding="utf-8").splitlines()
        aligned = len(expected) == len(predicted)
        metrics = MeasureSequenceMetrics()
        if aligned:
            for gold, guess in zip(expected, predicted, strict=True):
                metrics.update(
                    gold, guess, result["mode"], tuning=result["tuning_used"]
                )
        results.append(
            {
                "name": case["name"],
                "elapsed_seconds": time.perf_counter() - started,
                "inputs": [
                    {"name": p.name, "sha256": sha256(p.read_bytes()).hexdigest()}
                    for p in inputs
                ],
                "expected_sha256": sha256(expected_path.read_bytes()).hexdigest(),
                "expected_measures": len(expected),
                "detected_measures": len(predicted),
                "equal_measure_count": aligned,
                "alignment": "reading_order",
                "metrics": metrics.result() if aligned else None,
                "review_measures": result["review_measures"],
                "gp5_exported": bool(result["gp5"]),
            }
        )
    report = {
        "scope": "full_pipeline",
        "context_source": "predicted",
        "human_edits": 0,
        "system": platform.platform(),
        "python": platform.python_version(),
        "device": device,
        "versions": {
            name: version(name) for name in ("torch", "transformers", "peft", "PyMuPDF")
        },
        "model_manifest": json.loads(
            Path("weights/manifest.json").read_text(encoding="utf-8")
        ),
        "cases": results,
    }
    if torch:
        report.update(
            gpu=torch.cuda.get_device_name(device),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
        )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("examples/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout-python", type=Path)
    args = parser.parse_args()
    report = evaluate(args.cases, args.output, args.device, args.layout_python)
    print(json.dumps(report["cases"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
