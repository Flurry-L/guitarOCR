"""Defaults shared by stage CLIs, the orchestrator and the local workbench."""

import os
from pathlib import Path
from shared.paths import PROJECT_ROOT

MODEL = PROJECT_ROOT / Path("tools/models/GLM-OCR")
MEASURE_ADAPTER = PROJECT_ROOT / Path("weights/measure_ocr")
INFO_ADAPTER = PROJECT_ROOT / Path("weights/document_info")
LAYOUT_MODEL = PROJECT_ROOT / Path("weights/layout")


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def paddle_python() -> Path:
    for root in (PROJECT_ROOT / "tools/webui-paddle-venv", PROJECT_ROOT / "tools/paddlex-venv"):
        candidate = environment_python(root)
        if candidate.is_file():
            return candidate
    return environment_python(PROJECT_ROOT / "tools/webui-paddle-venv")
