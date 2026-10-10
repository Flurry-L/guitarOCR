"""PaddleX subprocess entry point with the shared layout RGB8 transform.

This is also imported by multiprocessing's spawn mode, so installation is
outside the main guard. Import upstream under its own module name to preserve
this entry point when Paddle creates data-loader workers.
"""

import importlib.util
import os
from pathlib import Path
import sys

upstream = Path(os.environ["GUITAROCR_LAYOUT_UPSTREAM_TRAIN"])
sys.path.insert(0, str(upstream.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.training.layout import install_cubic_resize  # noqa: E402

install_cubic_resize()

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("guitarocr_paddle_train", upstream)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()
