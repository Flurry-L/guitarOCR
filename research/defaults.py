"""Model and tool defaults for research entrypoints."""

import os
from pathlib import Path
from research.paths import PROJECT_ROOT

WEIGHTS_ROOT = Path(os.environ.get("GUITAROCR_WEIGHTS_ROOT", PROJECT_ROOT / "weights"))
SCORE_ADAPTER = WEIGHTS_ROOT / "score_ocr"
MODEL = SCORE_ADAPTER / 'merged'
MEASURE_ADAPTER = SCORE_ADAPTER
INFO_ADAPTER = SCORE_ADAPTER
LAYOUT_MODEL = WEIGHTS_ROOT / 'layout'


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def paddle_python() -> Path:
    return environment_python(PROJECT_ROOT / "tools/paddlex-venv")
