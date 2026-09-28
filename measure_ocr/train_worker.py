"""GLM SFT with fused loss on supervised tokens, without a dense prompt logit tensor."""

from functools import wraps


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
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == '__main__':
    main()
