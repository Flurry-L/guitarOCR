"""Read OCR training options and launch LLaMA-Factory without loading models."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


OCR_TRAINING_DEFAULTS = {
    "vocab_trainable_from": None,
    "vocab_learning_rate": 5e-4,
    "vocab_freeze_original": True,
    "music_field_loss": False,
    "batch_token_budget": None,
    "maximum_batch_examples": 128,
    "share_context_images": False,
    "context_chunk_size": 1,
}


def load_training_config(path: Path, overrides=()) -> tuple[dict, dict]:
    """Separate framework arguments from OCR-only options before either use.

    Inactive features still consume their options: LLaMA-Factory rejects
    unknown keys even when the corresponding OCR hook is not installed.
    YAML stays an optional dependency until a training command is run.
    """
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must be a mapping: {path}")
    for override in overrides:
        key, separator, value = override.partition("=")
        if separator:
            config[key] = yaml.safe_load(value)
    options = {
        key: config.pop(key, default)
        for key, default in OCR_TRAINING_DEFAULTS.items()
    }
    return config, options


def llamafactory_main(default_config: Path, *, worker_module: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train this OCR stage with LLaMA-Factory.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("overrides", nargs="*", help="LLaMA-Factory key=value overrides")
    args = parser.parse_args()
    if worker_module:
        if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
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
