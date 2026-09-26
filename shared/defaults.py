"""Defaults shared by stage CLIs, the orchestrator and the local workbench."""

import os
from pathlib import Path

MODEL = Path("tools/models/GLM-OCR")
MEASURE_ADAPTER = Path("weights/measure_ocr")
INFO_ADAPTER = Path("weights/document_info")
LAYOUT_MODEL = Path("weights/layout")


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def paddle_python() -> Path:
    for root in (Path("tools/webui-paddle-venv"), Path("tools/paddlex-venv")):
        candidate = environment_python(root)
        if candidate.is_file():
            return candidate
    return environment_python(Path("tools/webui-paddle-venv"))
