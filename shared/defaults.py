"""Defaults shared by stage CLIs, the orchestrator and the local workbench."""

import os
from pathlib import Path

MODEL = Path("tools/models/GLM-OCR")
MEASURE_ADAPTER = Path("weights/glm_ocr_measure_sequence_v3_lora")
INFO_ADAPTER = Path("weights/glm_ocr_document_info_v3_headers_lora")
LAYOUT_MODEL = Path("weights/pp_doclayout_v3_score_joint_v3")


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def paddle_python() -> Path:
    for root in (Path("tools/webui-paddle-venv"), Path("tools/paddlex-venv")):
        candidate = environment_python(root)
        if candidate.is_file():
            return candidate
    return environment_python(Path("tools/webui-paddle-venv"))
