"""Read OCR training options and launch LLaMA-Factory without loading models."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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


def validate_score_dataset(config):
    """Keep score supervision and the held-out split explicit."""
    directory = Path(config["dataset_dir"]).resolve()
    dataset = json.loads((directory / "dataset.json").read_text())
    if (
        dataset.get("complete") is not True
        or dataset.get("supervision") != "visible_score"
        or dataset.get("pitch") != "written_before_transposition"
        or dataset.get("image_geometry") != "preserve_aspect_ratio"
    ):
        raise ValueError("Build the current score dataset before tokenization")
    if config.get("dataset") != "train" or config.get("eval_dataset") != "validation":
        raise ValueError(
            "Score training requires train and validation; test stays independent"
        )
    if Path(config.get("media_dir") or directory).resolve() != directory:
        raise ValueError("Score image paths resolve relative to dataset_dir")


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
    audit_required = config.pop("require_score_dataset", False)
    options = {
        key: config.pop(key, default)
        for key, default in OCR_TRAINING_DEFAULTS.items()
    }
    if audit_required:
        validate_score_dataset(config)
    return config, options


def tokenize(config):
    """Build the configured token cache without loading a model."""
    import torch
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    torch.set_num_threads(1)
    from research.common.ocr import install_training_policy
    install_training_policy(config['model_name_or_path'])
    # Tokenization loads no model and must work alongside a multi-GPU job.
    config.update(use_cpu=True, bf16=False, fp16=False)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    datasets = get_dataset(
        template, model_args, data_args, training_args,
        finetuning_args.stage, **tokenizer_module,
    )
    print({key: len(value) if value is not None else None for key, value in datasets.items()})


def main() -> None:
    default_config = Path(__file__).parents[1] / "configs/measures/train.yaml"
    parser = argparse.ArgumentParser(description="Train this OCR stage with LLaMA-Factory.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--tokenize-only", action="store_true", help="Build tokenized_path without training")
    parser.add_argument("overrides", nargs="*", help="LLaMA-Factory key=value overrides")
    args = parser.parse_args()
    config, _ = load_training_config(args.config, args.overrides)
    # Tokenization already runs in several processes. A full CPU thread
    # pool per worker can stall preprocessing through oversubscription.
    # Keep explicit operator settings and use one thread by default.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    if args.tokenize_only:
        tokenize(config)
        return
    if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    import torch

    workers = int(os.environ.get('NPROC_PER_NODE', max(1, torch.cuda.device_count())))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', str(workers), '--module', 'research.training.worker']
    subprocess.run([*command, str(args.config), *args.overrides], check=True)


if __name__ == '__main__':
    main()
