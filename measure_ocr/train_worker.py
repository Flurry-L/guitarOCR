"""GLM SFT with fused loss on supervised tokens, without a dense prompt logit tensor."""

from functools import wraps


def exact_linear_targets():
    """LLaMA-Factory's substring expansion also matches GLM projector norms."""
    import torch
    import llamafactory.model.adapter as adapters

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
        GlmOcrForConditionalGeneration, GlmOcrCausalLMOutputWithPast,
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


def train_music_vocabulary(first_new_token, learning_rate):
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
    import yaml

    config = yaml.safe_load(open(sys.argv[1]))
    for override in sys.argv[2:]:
        key, sep, value = override.partition('=')
        if sep:
            config[key] = yaml.safe_load(value)
    needs_logits = any(config.get(key) for key in ('compute_accuracy', 'use_dft_loss', 'use_eaft_loss', 'use_asft_loss'))
    if not needs_logits and os.environ.get('GUITAROCR_DENSE_LOSS') != '1':
        enable_fused_loss()
    materialize_sampler_lengths()
    exact_linear_targets()
    first_new_token = config.pop('vocab_trainable_from', None)
    if first_new_token is not None:
        train_music_vocabulary(int(first_new_token), float(config.pop('vocab_learning_rate', 5e-4)))
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

        def on_train_end(self, args, state, control, **kwargs):
            if (args.do_eval and not args.load_best_model_at_end
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
