"""GLM SFT with fused loss on supervised tokens, without a dense prompt logit tensor."""

import math
import random
from functools import wraps
from pathlib import Path

from research.training.measures import load_training_config


class Epoch:
    value = 0

    def set_epoch(self, epoch):
        self.value = epoch


class TokenBatchSampler:
    batch_size = None
    drop_last = False

    def __init__(self, lengths, world, budget, maximum, seed):
        self.world, self.budget, self.maximum, self.seed = world, budget, maximum, seed
        self.sampler = Epoch()
        self.buckets = {}
        for index, length in enumerate(lengths):
            ceiling = max(128, math.ceil(length / 128) * 128)
            self.buckets.setdefault(ceiling, []).append(index)
        self.steps = sum(math.ceil(len(rows) / (self.local_size(ceiling) * world))
                         for ceiling, rows in self.buckets.items())

    def local_size(self, ceiling):
        return max(1, min(self.maximum, self.budget // ceiling))

    def __len__(self):
        return self.steps * self.world

    def __iter__(self):
        rng = random.Random(self.seed + self.sampler.value)
        steps = []
        for ceiling, values in self.buckets.items():
            rows = list(values)
            rng.shuffle(rows)
            size = self.local_size(ceiling) * self.world
            for offset in range(0, len(rows), size):
                batch = rows[offset:offset + size]
                # At most world-1 repeated examples per length bucket. Never
                # drop a rare long measure to equalize distributed ranks.
                padding = (-len(batch)) % self.world
                batch.extend(batch[i % len(batch)] for i in range(padding))
                steps.append(batch)
        rng.shuffle(steps)
        for batch in steps:
            size = len(batch) // self.world
            for rank in range(self.world):
                yield batch[rank * size:(rank + 1) * size]


def install_token_batching(budget, maximum=128):
    from torch.utils.data import DataLoader
    from transformers import Trainer

    original = Trainer._get_dataloader

    def dataloader(self, dataset, description, batch_size, sampler_fn=None, is_training=False, dataloader_key=None):
        if not is_training:
            return original(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
        import pyarrow.compute as pc
        if 'length' in dataset.column_names:
            lengths = list(dataset['length'])
        else:
            lengths = pc.list_value_length(dataset.data.column('input_ids'))
            if dataset._indices is not None:
                lengths = pc.take(lengths, dataset._indices.column(0))
            lengths = lengths.to_pylist()
        sampler = TokenBatchSampler(lengths, self.accelerator.num_processes, budget, maximum, self.args.seed)
        dataset = self._remove_unused_columns(dataset, description=description)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers,
                            pin_memory=self.args.dataloader_pin_memory,
                            persistent_workers=self.args.dataloader_persistent_workers,
                            prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers else None)
        if self.is_world_process_zero():
            print(f'Token batches: {len(lengths)} samples, {sampler.steps} steps/epoch, '
                  f'{budget} padded tokens/GPU, maximum {maximum} samples/GPU', flush=True)
        # These batches already contain equally sized shards for every rank.
        # Scope the flags to this loader: ordinary validation must pad its
        # final batch, or some ranks finish before the final metrics gather.
        configuration = self.accelerator.dataloader_config
        previous = configuration.even_batches, configuration.split_batches
        configuration.even_batches, configuration.split_batches = False, False
        try:
            return self.accelerator.prepare(loader)
        finally:
            configuration.even_batches, configuration.split_batches = previous

    Trainer._get_dataloader = dataloader


def stage_adapter_weights_on_cpu():
    """PEFT's generic 'cuda' restore otherwise loads every DDP rank on GPU 0."""
    from peft import PeftModel

    original = PeftModel.load_adapter

    @wraps(original)
    def load(self, model_id, adapter_name, is_trainable=False, torch_device=None, **kwargs):
        return original(self, model_id, adapter_name, is_trainable=is_trainable,
                        torch_device=torch_device or 'cpu', **kwargs)

    PeftModel.load_adapter = load


def exact_linear_targets():
    """LLaMA-Factory's substring expansion also matches GLM projector norms."""
    import llamafactory.model.adapter as adapters
    import torch

    original = adapters.patch_target_modules

    def patch(model, arguments, targets):
        names = original(model, arguments, targets)
        if model.config.model_type == 'glm_ocr':
            return [name for name in names if isinstance(model.get_submodule(name), torch.nn.Linear)]
        return names

    adapters.patch_target_modules = patch


def materialize_sampler_lengths():
    from transformers.trainer_pt_utils import LengthGroupedSampler

    original = LengthGroupedSampler.__init__

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # Dataset columns are lazy: sorting millions of random Arrow lookups
        # otherwise spends minutes on the CPU before the first GPU batch.
        if not isinstance(self.lengths, list):
            self.lengths = list(self.lengths)

    LengthGroupedSampler.__init__ = initialize


def enable_fused_loss():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        GlmOcrCausalLMOutputWithPast,
        GlmOcrForConditionalGeneration,
    )

    original = GlmOcrForConditionalGeneration.forward
    cross_entropy = LigerFusedLinearCrossEntropyLoss()

    @wraps(original)
    def forward(self, *args, **kwargs):
        labels = kwargs.get('labels')
        if labels is None:
            return original(self, *args, **kwargs)
        kwargs.pop('labels')
        kwargs.pop('logits_to_keep', None)
        kwargs.pop('num_items_in_batch', None)
        kwargs.pop('length', None)
        outputs = self.model(*args, **kwargs)
        selected = labels[:, 1:] != -100
        hidden = outputs.last_hidden_state[:, :-1][selected].contiguous()
        target = labels[:, 1:][selected].contiguous()
        loss = cross_entropy(self.lm_head.weight, hidden, target)
        return GlmOcrCausalLMOutputWithPast(
            loss=loss, logits=None, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    GlmOcrForConditionalGeneration.forward = forward


def train_music_vocabulary(first_new_token, learning_rate, freeze_original=True):
    """Learn new lexemes faster while preserving the pretrained vocabulary."""
    from transformers import Trainer

    original = Trainer.create_optimizer

    def create_optimizer(self):
        parameters = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and ('embed_tokens' in name or 'lm_head' in name) and parameter.ndim == 2:
                parameters[id(parameter)] = parameter
                def preserve_original(gradient):
                    gradient[:first_new_token] = 0
                    return gradient
                if freeze_original:
                    parameter.register_hook(preserve_original)
        result = original(self)
        groups = []
        for group in self.optimizer.param_groups:
            vocabulary = [p for p in group['params'] if id(p) in parameters]
            rest = [p for p in group['params'] if id(p) not in parameters]
            if rest:
                groups.append({**group, 'params': rest})
            if vocabulary:
                groups.append({**group, 'params': vocabulary, 'lr': learning_rate, 'weight_decay': 0.})
        self.optimizer.param_groups[:] = groups
        return result

    Trainer.create_optimizer = create_optimizer


def configure_training(config, options):
    """Install the same hooks for training and the CPU readiness check."""
    import os

    from research.common.ocr import install_training_policy
    install_training_policy(config['model_name_or_path'])
    if token_budget := options['batch_token_budget']:
        install_token_batching(int(token_budget), int(options['maximum_batch_examples']))
    needs_logits = any(config.get(key) for key in ('compute_accuracy', 'use_dft_loss', 'use_eaft_loss', 'use_asft_loss'))
    if not needs_logits and os.environ.get('GUITAROCR_DENSE_LOSS') != '1':
        enable_fused_loss()
    materialize_sampler_lengths()
    exact_linear_targets()
    stage_adapter_weights_on_cpu()
    first_new_token = options['vocab_trainable_from']
    if first_new_token is not None:
        train_music_vocabulary(int(first_new_token), float(options['vocab_learning_rate']),
                               bool(options['vocab_freeze_original']))


def main():
    import sys

    config, options = load_training_config(Path(sys.argv[1]), sys.argv[2:])
    configure_training(config, options)
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    class ResumeSchedule(TrainerCallback):
        last_evaluation_step = None
        last_evaluation_metrics = None

        def on_train_begin(self, args, state, control, **kwargs):
            # Transformers restores these intervals from trainer_state.json,
            # even when a continuation config deliberately changes them.
            state.compute_steps(args, state.max_steps)
            state.train_batch_size = args.train_batch_size

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.last_evaluation_step = state.global_step
            self.last_evaluation_metrics = metrics

        def on_save(self, args, state, control, **kwargs):
            from pathlib import Path

            if (Path(args.output_dir) / 'STOP_AFTER_CHECKPOINT').exists():
                control.should_training_stop = True

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                import json
                dataset = Path(config["dataset_dir"]) / "dataset.json"
                if dataset.is_file() and json.loads(dataset.read_text()).get("supervision") == "visible_observation":
                    (Path(args.output_dir) / "capabilities.json").write_text(json.dumps({
                        "measure_output": "observation", "measure_images": 1,
                    }, indent=2) + "\n")
            latest_is_best = (not args.load_best_model_at_end or
                              state.best_model_checkpoint and
                              state.best_model_checkpoint.endswith(f'checkpoint-{state.global_step}'))
            if (args.do_eval and latest_is_best
                    and self.last_evaluation_step == state.global_step and self.last_evaluation_metrics):
                # LLaMA-Factory otherwise repeats the complete validation set
                # immediately after a final-step evaluation of these weights.
                args.do_eval = False
                if state.is_world_process_zero:
                    import json
                    (Path(args.output_dir) / 'eval_results.json').write_text(json.dumps(self.last_evaluation_metrics, indent=2))

    run_exp(args=config, callbacks=[ResumeSchedule()])


if __name__ == '__main__':
    main()
