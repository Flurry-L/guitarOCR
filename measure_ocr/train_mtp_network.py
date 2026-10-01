"""Native GLM-OCR draft layer and its checkpoint representation."""

from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrRMSNorm, GlmOcrTextDecoderLayer, GlmOcrTextRotaryEmbedding,
)


class MusicMTP(nn.Module):
    def __init__(self, teacher, model_path, initial_mtp=None):
        super().__init__()
        config = teacher.config.text_config
        self.enorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.block = GlmOcrTextDecoderLayer(config, 0)
        self.norm = GlmOcrRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary = GlmOcrTextRotaryEmbedding(config=config)
        self.prefix = f'model.language_model.layers.{config.num_hidden_layers}.'
        state = {}
        paths = [Path(initial_mtp)] if initial_mtp else Path(model_path).glob('*.safetensors')
        for path in paths:
            with safe_open(path, framework='pt') as source:
                for key in source.keys():
                    if not key.startswith(self.prefix):
                        continue
                    name = key.removeprefix(self.prefix)
                    if name in {'embed_tokens.weight', 'shared_head.head.weight'}:
                        continue
                    if name == 'shared_head.norm.weight':
                        name = 'norm.weight'
                    elif name not in {'enorm.weight', 'hnorm.weight', 'eh_proj.weight'}:
                        name = 'block.' + name
                    state[name] = source.get_tensor(key)
        self.load_state_dict(state, strict=True)

    def forward(self, embeddings, hidden, attention_mask, position_ids=None):
        # vLLM pairs hidden(P) with token(P+1), keeping the original P position.
        if position_ids is None:
            positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
            position_ids = positions[None].expand(3, -1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        embeddings = embeddings.masked_fill((position_ids[0] == 0).unsqueeze(-1), 0)
        fused = self.eh_proj(torch.cat((self.enorm(embeddings), self.hnorm(hidden)), dim=-1))
        # Image tokens use temporal/row/column coordinates, followed by a
        # compressed text offset. Plain arange disagrees with the deployed MTP.
        rope = self.rotary(fused, position_ids)
        result = self.block(fused, position_embeddings=rope, attention_mask=attention_mask, use_cache=False)
        return self.norm(result), self.hnorm(result)

    def save(self, path, teacher):
        values = {}
        for name, value in self.state_dict().items():
            if name.startswith('block.'):
                name = name.removeprefix('block.')
            elif name == 'norm.weight':
                name = 'shared_head.norm.weight'
            values[self.prefix + name] = value.detach().cpu().contiguous()
        values[self.prefix + 'embed_tokens.weight'] = teacher.model.language_model.embed_tokens.weight.detach().cpu()
        values[self.prefix + 'shared_head.head.weight'] = teacher.lm_head.weight.detach().cpu()
        save_file(values, path)

