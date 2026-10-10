"""Prepare OCR batches on CPU or launch LLaMA-Factory on selected GPUs."""

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
    "batch_token_budget": None,
    "maximum_batch_examples": 128,
}


def validate_score_dataset(config):
    """Keep score supervision and the held-out split explicit."""
    directory = Path(config["dataset_dir"]).resolve()
    dataset = json.loads((directory / "dataset.json").read_text())
    if (
        dataset.get("complete") is not True
        or dataset.get("supervision") != "visible_observation"
        or dataset.get("measure_images") != 1
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
        if not separator or not key:
            raise ValueError(f"Expected a key=value override, got: {override}")
        config[key] = yaml.safe_load(value)
    audit_required = config.pop("require_score_dataset", False)
    options = {
        key: config.pop(key, default)
        for key, default in OCR_TRAINING_DEFAULTS.items()
    }
    if audit_required:
        validate_score_dataset(config)
    return config, options


def validate_training_output(config):
    """LLaMA-Factory otherwise silently resumes the last saved optimizer."""
    output = Path(config["output_dir"])
    if (config.get("do_train") and not config.get("resume_from_checkpoint")
            and not config.get("overwrite_output_dir")
            and output.is_dir() and any(output.iterdir())):
        raise ValueError(
            f"Training output is not empty: {output}. Use a new output_dir for "
            "fresh fine-tuning, or explicitly set resume_from_checkpoint for this run."
        )


def prepare_data(config, *, check=False):
    """Build tokens or exercise cached image/label batches without model weights."""
    import torch
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    torch.set_num_threads(1)
    from research.common.ocr import install_training_policy
    install_training_policy(config['model_name_or_path'])
    # Tokenization loads no model and must work alongside a multi-GPU job.
    if check and not Path(config["tokenized_path"], "dataset_dict.json").is_file():
        raise ValueError("Build tokenized_path with --tokenize-only before --check")
    cpu_config = dict(config, use_cpu=True, bf16=False, fp16=False)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(cpu_config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    datasets = get_dataset(
        template, model_args, data_args, training_args,
        finetuning_args.stage, **tokenizer_module,
    )
    report = {key: len(value) if value is not None else None for key, value in datasets.items()}
    if check:
        import pyarrow.compute as pc
        from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask

        collator = SFTDataCollatorWith4DAttentionMask(
            template=template, pad_to_multiple_of=8,
            attn_implementation="flash_attention_2" if model_args.flash_attn == "fa2" else "eager",
            compute_dtype=torch.bfloat16 if config.get("bf16") else torch.float32,
            **tokenizer_module,
        )
        report["batches"] = {}
        for split, dataset in datasets.items():
            if dataset is None:
                continue
            lengths = pc.list_value_length(dataset.data.column("input_ids")).to_pylist()
            indices = sorted(set((0, min(range(len(lengths)), key=lengths.__getitem__),
                                  max(range(len(lengths)), key=lengths.__getitem__))))
            batch = collator([dataset[index] for index in indices])
            if not (batch["labels"] != -100).any(dim=1).all():
                raise ValueError(f"Missing supervised tokens in {split}")
            report["batches"][split] = {
                "rows": indices,
                "input_ids": list(batch["input_ids"].shape),
                "pixel_values": list(batch["pixel_values"].shape),
            }
        report["cuda_initialized"] = torch.cuda.is_initialized()
    print(json.dumps(report, indent=2))


def main() -> None:
    default_config = Path(__file__).parents[1] / "configs/measures/train.yaml"
    parser = argparse.ArgumentParser(description="Train this OCR stage with LLaMA-Factory.")
    parser.add_argument("--config", type=Path, default=default_config)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tokenize-only", action="store_true", help="Build tokenized_path without training")
    mode.add_argument("--check", action="store_true", help="Check training imports and cached image/label batches on CPU")
    parser.add_argument("overrides", nargs="*", help="LLaMA-Factory key=value overrides")
    args = parser.parse_args()
    config, options = load_training_config(args.config, args.overrides)
    # Tokenization already runs in several processes. A full CPU thread
    # pool per worker can stall preprocessing through oversubscription.
    # Keep explicit operator settings and use one thread by default.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    if args.tokenize_only or args.check:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if args.check:
            from research.training.worker import configure_training
            validate_training_output(config)
            configure_training(config, options)
            if config.get("flash_attn") == "fa2":
                __import__("flash_attn")
        prepare_data(config, check=args.check)
        return
    validate_training_output(config)
    if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    import torch

    devices = torch.cuda.device_count()
    workers = int(os.environ.get('NPROC_PER_NODE', devices))
    if not 1 <= workers <= devices:
        parser.error("Select available GPUs with CUDA_VISIBLE_DEVICES; NPROC_PER_NODE must fit the visible GPU count")
    print(f"Training on {workers} GPUs: {config['model_name_or_path']} -> {config['output_dir']}", flush=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', str(workers), '--module', 'research.training.worker']
    subprocess.run([*command, str(args.config), *args.overrides], check=True)


if __name__ == '__main__':
    main()
