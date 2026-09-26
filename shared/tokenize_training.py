"""Build a LLaMA-Factory token cache without starting model training."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Dataset.map starts multiple processes; each image processor must not
    # create its own full CPU thread pool.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    import yaml
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    config["tokenized_path"] = str(args.output)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    datasets = get_dataset(
        template, model_args, data_args, training_args,
        finetuning_args.stage, **tokenizer_module,
    )
    print({key: len(value) if value is not None else None for key, value in datasets.items()})


if __name__ == "__main__":
    main()
