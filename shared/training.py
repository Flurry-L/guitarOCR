"""Launch LLaMA-Factory with a stage's own training configuration."""

import argparse
from pathlib import Path
import shutil
import subprocess


def llamafactory_main(default_config: Path) -> None:
    parser = argparse.ArgumentParser(description="Train this OCR stage with LLaMA-Factory.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("overrides", nargs="*", help="LLaMA-Factory key=value overrides")
    args = parser.parse_args()
    executable = shutil.which("llamafactory-cli")
    if executable is None:
        parser.error("Install LLaMA-Factory in this environment before training")
    subprocess.run([executable, "train", str(args.config), *args.overrides], check=True)
