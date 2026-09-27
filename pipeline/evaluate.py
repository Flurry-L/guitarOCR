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
from shared.artifacts import read_result, write_json
from shared.defaults import MODEL, MEASURE_ADAPTER, INFO_ADAPTER, LAYOUT_MODEL


def evaluate(
    cases_file,
    output,
    device="cuda",
    layout_python=None,
    *,
    model=MODEL,
    adapter=MEASURE_ADAPTER,
    info_adapter=INFO_ADAPTER,
    layout_model_dir=LAYOUT_MODEL,
):
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    results = []
    torch = None
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(device)
    for index, case in enumerate(cases, start=1):
        case_output = output / f"case-{index:03d}"
        inputs = [(cases_file.parent / name).resolve() for name in case["inputs"]]
        expected_path = (cases_file.parent / case["expected_m2"]).resolve()
        expected = expected_path.read_text(encoding="utf-8").splitlines()
        arguments = [
            *[str(p) for p in inputs],
            "--output",
            str(case_output),
            "--device",
            device,
            "--mode",
            case.get("mode", "auto"),
        ]
        for option, path in (
            ("model", model),
            ("adapter", adapter),
            ("info-adapter", info_adapter),
            ("layout-model-dir", layout_model_dir),
        ):
            arguments += ["--" + option, str(path)]
        for option in ("instrument", "tuning", "midi_program", "transpose"):
            if case.get(option) is not None:
                value = case[option]
                if isinstance(value, list):
                    value = ",".join(map(str, value))
                arguments += ["--" + option.replace("_", "-"), str(value)]
        if layout_python:
            arguments += ["--layout-python", str(layout_python)]
        if case.get("layout_source"):
            arguments += ["--layout-source", case["layout_source"]]
        started = time.perf_counter()
        error = None
        try:
            result = run(parse_args(arguments))
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            recognition = case_output / "03_measure_ocr" / "manifest.json"
            result = (
                read_result(recognition, "measure_ocr") if recognition.is_file() else {}
            )
            result["gp5"] = None
        predicted = (
            Path(result["m2"]).read_text(encoding="utf-8").splitlines()
            if result.get("m2")
            else []
        )
        layout = case_output / "01_layout" / "manifest.json"
        detected = (
            len(read_result(layout, "layout")["records"]) if layout.is_file() else 0
        )
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
                "instrument": result.get("instrument"),
                "manual_metadata": {
                    key: case[key]
                    for key in ("instrument", "tuning", "midi_program", "transpose")
                    if key in case
                },
                "elapsed_seconds": time.perf_counter() - started,
                "inputs": [
                    {"name": p.name, "sha256": sha256(p.read_bytes()).hexdigest()}
                    for p in inputs
                ],
                "expected_sha256": sha256(expected_path.read_bytes()).hexdigest(),
                "expected_measures": len(expected),
                "detected_measures": detected,
                "recognized_measures": len(predicted),
                "equal_measure_count": aligned,
                "alignment": "reading_order",
                "metrics": metrics.result() if aligned else None,
                "review_measures": result.get("review_measures", []),
                "gp5_exported": bool(result["gp5"]),
                "pipeline_error": error,
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
            name: version(name)
            for name in ("torch", "transformers", "peft", "pypdfium2", "pdfplumber")
        },
        "model_files": model_files(model, adapter, info_adapter, layout_model_dir),
        "cases": results,
    }
    if torch:
        report.update(
            gpu=torch.cuda.get_device_name(device),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
        )
    write_json(output / "report.json", report)
    return report


def model_files(model, adapter, info_adapter, layout_model_dir):
    result = {}
    for stage, root in (
        ("base", model),
        ("measures", adapter),
        ("information", info_adapter),
        ("layout", layout_model_dir),
    ):
        files = {}
        for path in sorted(Path(root).glob("*")):
            if not path.is_file() or path.suffix not in {
                ".safetensors",
                ".pdiparams",
                ".json",
                ".yml",
            }:
                continue
            digest = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    digest.update(chunk)
            files[path.name] = digest.hexdigest()
        result[stage] = {"path": str(Path(root).resolve()), "sha256": files}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="完整谱面及参考小节文本的案例清单")
    parser.add_argument("--output", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout-python", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=MEASURE_ADAPTER)
    parser.add_argument("--info-adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--layout-model-dir", type=Path, default=LAYOUT_MODEL)
    args = parser.parse_args()
    report = evaluate(
        args.cases,
        args.output,
        args.device,
        args.layout_python,
        model=args.model,
        adapter=args.adapter,
        info_adapter=args.info_adapter,
        layout_model_dir=args.layout_model_dir,
    )
    print(json.dumps(report["cases"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
