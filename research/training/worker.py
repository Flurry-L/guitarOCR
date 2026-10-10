"""GLM SFT with fused loss on supervised tokens, without a dense prompt logit tensor."""

import math
import random
from collections import defaultdict, deque
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from research.training.measures import load_training_config


def neighbour_chunks(images, maximum=8):
    """Keep every sample; follow actual context links, including repeated scores."""
    available = defaultdict(deque)
    for index, paths in enumerate(images):
        if paths:
            available[paths[0]].append(index)
    used, chunks = bytearray(len(images)), []
    for start in range(len(images)):
        if used[start]:
            continue
        index, chunk = start, []
        while True:
            chunk.append(index)
            used[index] = 1
            paths = images[index]
            if len(chunk) == maximum or len(paths) != 3 or paths[2] == paths[0]:
                break
            following = available.get(paths[2])
            while following and used[following[0]]:
                following.popleft()
            if not following:
                break
            index = following.popleft()
        chunks.append(chunk)
    return chunks


class Epoch:
    value = 0

    def set_epoch(self, epoch):
        self.value = epoch


class TokenBatchSampler:
    batch_size = None
    drop_last = False

    def __init__(self, lengths, world, budget, maximum, seed, chunks=None):
        self.world, self.budget, self.maximum, self.seed = world, budget, maximum, seed
        self.sampler = Epoch()
        self.buckets = {}
        for chunk in chunks if chunks is not None else ([i] for i in range(len(lengths))):
            ceiling = max(128, math.ceil(max(lengths[i] for i in chunk) / 128) * 128)
            self.buckets.setdefault((ceiling, len(chunk)), []).append(chunk)
        self.steps = sum(math.ceil(len(rows) / (self.local_size(ceiling, size) * world))
                         for (ceiling, size), rows in self.buckets.items())

    def local_size(self, ceiling, size=1):
        return max(1, min(self.maximum // size, self.budget // (ceiling * size)))

    def __len__(self):
        return self.steps * self.world

    def __iter__(self):
        rng = random.Random(self.seed + self.sampler.value)
        steps = []
        for (ceiling, chunk_size), values in self.buckets.items():
            rows = list(values)
            rng.shuffle(rows)
            size = self.local_size(ceiling, chunk_size) * self.world
            for offset in range(0, len(rows), size):
                batch = [index for chunk in rows[offset:offset + size] for index in chunk]
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


def install_token_batching(budget, maximum=128, context_chunk_size=1):
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
        chunks = None
        if context_chunk_size > 1:
            images = dataset.data.column('images')
            if dataset._indices is not None:
                images = pc.take(images, dataset._indices.column(0))
            chunks = neighbour_chunks(images.to_pylist(), context_chunk_size)
        sampler = TokenBatchSampler(lengths, self.accelerator.num_processes, budget, maximum, self.args.seed, chunks)
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


def install_shared_vision():
    import torch
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrModel

    original_collate = MultiModalDataCollatorForSeq2Seq.__call__
    original_forward = GlmOcrModel.forward
    original_features = GlmOcrModel.get_image_features
    current_images = ContextVar('shared_training_images', default=None)

    @wraps(original_collate)
    def collate(self, features):
        paths = [path for row in features for path in (row.get('images') or [])]
        first, inverse, lookup = [], [], {}
        if paths and all(isinstance(path, str) for path in paths):
            for index, path in enumerate(paths):
                if path not in lookup:
                    lookup[path] = len(first)
                    first.append(index)
                inverse.append(lookup[path])
        batch = original_collate(self, features)
        if inverse and len(first) < len(paths):
            batch['image_unique_indices'] = torch.tensor(first, dtype=torch.long)
            batch['image_inverse_indices'] = torch.tensor(inverse, dtype=torch.long)
        return batch

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        first = kwargs.pop('image_unique_indices', None)
        inverse = kwargs.pop('image_inverse_indices', None)
        context = current_images.set((first, inverse) if first is not None else None)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            current_images.reset(context)

    @wraps(original_features)
    def features(self, pixel_values, image_grid_thw=None, **kwargs):
        shared = current_images.get()
        if shared is None:
            return original_features(self, pixel_values, image_grid_thw, **kwargs)
        first, inverse = shared
        first, inverse = first.tolist(), inverse.tolist()
        if len(inverse) != len(image_grid_thw):
            raise ValueError('Shared image map does not match the image grid')
        chunks = torch.split(pixel_values, image_grid_thw.prod(-1).tolist())
        unique_pixels = torch.cat([chunks[i] for i in first])
        unique_grid = image_grid_thw[first]
        result = original_features(self, unique_pixels, unique_grid, **kwargs)
        # Repeated tensors stay attached to the same graph: all measure losses
        # contribute to the gradient of their shared image representation.
        unique_features = tuple(value.float() for value in result.pooler_output)
        # Accumulate contributions from repeated uses in FP32 before casting
        # their combined gradient back to the vision tower's compute dtype.
        result.pooler_output = tuple(unique_features[i] for i in inverse)
        return result

    MultiModalDataCollatorForSeq2Seq.__call__ = collate
    GlmOcrModel.forward = forward
    GlmOcrModel.get_image_features = features


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


def enable_fused_loss(field_weights=None):
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        GlmOcrCausalLMOutputWithPast,
        GlmOcrForConditionalGeneration,
    )

    original = GlmOcrForConditionalGeneration.forward
    cross_entropy = LigerFusedLinearCrossEntropyLoss()
    weighted = {}

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
        criterion = cross_entropy
        if field_weights:
            import torch
            key = (hidden.device, self.lm_head.weight.shape[0])
            if key not in weighted:
                weights = torch.ones(key[1], device=hidden.device, dtype=torch.float32)
                for token, weight in field_weights.items():
                    weights[int(token)] = float(weight)
                weighted[key] = LigerFusedLinearCrossEntropyLoss(ce_weight=weights)
            criterion = weighted[key]
        loss = criterion(self.lm_head.weight, hidden, target)
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


def main():
    import os
    import sys

    config, options = load_training_config(Path(sys.argv[1]), sys.argv[2:])
    from research.common.ocr import install_training_policy
    install_training_policy(config['model_name_or_path'])
    if options['share_context_images']:
        install_shared_vision()
    context_chunk_size = int(options['context_chunk_size'])
    if token_budget := options['batch_token_budget']:
        install_token_batching(int(token_budget), int(options['maximum_batch_examples']), context_chunk_size)
    needs_logits = any(config.get(key) for key in ('compute_accuracy', 'use_dft_loss', 'use_eaft_loss', 'use_asft_loss'))
    field_weights = None
    if options['music_field_loss']:
        import json
        path = Path(config['model_name_or_path']) / 'music_vocabulary.json'
        field_weights = json.loads(path.read_text()).get('loss_weights')
    if not needs_logits and os.environ.get('GUITAROCR_DENSE_LOSS') != '1':
        enable_fused_loss(field_weights)
    materialize_sampler_lengths()
    exact_linear_targets()
    stage_adapter_weights_on_cpu()
    first_new_token = options['vocab_trainable_from']
    if first_new_token is not None:
        train_music_vocabulary(int(first_new_token), float(options['vocab_learning_rate']),
                               bool(options['vocab_freeze_original']))
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
                    from pathlib import Path

                    (Path(args.output_dir) / 'eval_results.json').write_text(json.dumps(self.last_evaluation_metrics, indent=2))

    run_exp(args=config, callbacks=[ResumeSchedule()])


if __name__ == '__main__':
    main()
