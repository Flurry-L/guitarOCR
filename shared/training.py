"""Launch LLaMA-Factory with a stage's own training configuration."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def llamafactory_main(default_config: Path, *, worker_module: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train this OCR stage with LLaMA-Factory.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("overrides", nargs="*", help="LLaMA-Factory key=value overrides")
    args = parser.parse_args()
    if worker_module:
        import torch

        workers = int(os.environ.get('NPROC_PER_NODE', max(1, torch.cuda.device_count())))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(workers), '--module', worker_module]
        subprocess.run([*command, str(args.config), *args.overrides], check=True)
        return
    executable = shutil.which("llamafactory-cli")
    if executable is None:
        parser.error("Install LLaMA-Factory in this environment before training")
    subprocess.run([executable, "train", str(args.config), *args.overrides], check=True)
