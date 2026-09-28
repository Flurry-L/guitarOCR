"""Distill the native GLM-OCR next-token layer against a frozen music model."""

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor
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

    def forward(self, embeddings, hidden, attention_mask):
        # vLLM pairs hidden(P) with token(P+1), keeping the original P position.
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
        embeddings = embeddings.clone()
        embeddings[:, 0] = 0
        fused = self.eh_proj(torch.cat((self.enorm(embeddings), self.hnorm(hidden)), dim=-1))
        rope = self.rotary(fused, positions[None].expand(3, -1, -1))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Merged music model with the original MTP weights')
    parser.add_argument('--tokenized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-mtp', type=Path, help='Continue distillation from a previously trained MTP layer')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--eval-batch-size', type=int, default=48)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if world > 1:
        dist.init_process_group('nccl')
    teacher = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    teacher.requires_grad_(False)
    draft = MusicMTP(teacher, args.model, args.initial_mtp).cuda().float()
    wrapped = DistributedDataParallel(draft, device_ids=[device], broadcast_buffers=False) if world > 1 else draft
    processor = AutoProcessor.from_pretrained(args.model)
    from datasets import load_from_disk
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    template = get_template_and_fix_tokenizer(processor.tokenizer, DataArguments(template='glm_ocr'))
    # Same raster budget used by LLaMA-Factory's multimodal preprocessing.
    processor.image_max_pixels = 768 * 768
    processor.image_min_pixels = 32 * 32
    cache = load_from_disk(args.tokenized)
    collator = SFTDataCollatorWith4DAttentionMask(template=template, model=teacher,
                                               tokenizer=processor.tokenizer, processor=processor,
                                               pad_to_multiple_of=8, attn_implementation='flash_attention_2',
                                               compute_dtype=torch.bfloat16)
    sampler = DistributedSampler(cache['train'], num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(cache['train'], batch_size=args.batch_size, sampler=sampler, collate_fn=collator,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    val_sampler = DistributedSampler(cache['validation'], num_replicas=world, rank=rank, shuffle=False) if 'validation' in cache else None
    validation = DataLoader(cache['validation'], batch_size=args.eval_batch_size, sampler=val_sampler, collate_fn=collator,
                            num_workers=2, pin_memory=True) if val_sampler else None
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=.01, fused=True)
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    cross_entropy = LigerFusedLinearCrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    iterator, epoch = iter(loader), 0
    started, best = time.perf_counter(), float('inf')

    def forward(batch):
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch.pop('labels')
        batch.pop('length', None)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = teacher.model(**batch, use_cache=False).last_hidden_state
            embeddings = teacher.model.language_model.embed_tokens(batch['input_ids'][:, 1:])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, recurrence = wrapped(embeddings, hidden[:, :-1], batch['attention_mask'][:, :-1])
            mask = labels[:, 2:] != -100
            student = output[:, :-1][mask]
            target = hidden[:, 1:-1][mask]
            recurrence = recurrence[:, :-1][mask]
            with torch.no_grad():
                recurrent_target = draft.hnorm(target)
            alignment = (1 - F.cosine_similarity(recurrence.float(), recurrent_target.float())).mean()
            expected, correct, total = [], 0, student.shape[0]
            with torch.no_grad():
                for i in range(0, total, 512):
                    tokens = teacher.lm_head(target[i:i + 512]).argmax(-1)
                    expected.append(tokens)
                    correct += (teacher.lm_head(student[i:i + 512]).argmax(-1) == tokens).sum()
            loss = cross_entropy(teacher.lm_head.weight, student.contiguous(), torch.cat(expected)) + .1 * alignment
        return loss, correct, total

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        progress = (step - 1) / max(1, args.steps - 1)
        factor = min(1., step / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * factor
        losses, hits, tokens = 0., 0, 0
        for micro in range(args.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
            with sync:
                loss, correct, total = forward(batch)
                (loss / args.accumulation).backward()
            losses += loss.detach().item() / args.accumulation
            hits += int(correct)
            tokens += total
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.)
        optimizer.step()
        if rank == 0 and step % 20 == 0:
            print(json.dumps({'step': step, 'loss': losses, 'token_agreement': hits / max(1, tokens),
                              'seconds': time.perf_counter() - started}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            draft.eval()
            stats = torch.zeros(3, device='cuda', dtype=torch.float64)
            if validation is not None:
                with torch.no_grad():
                    for batch_index, batch in enumerate(validation, 1):
                        loss, correct, total = forward(batch)
                        stats += torch.tensor([float(loss) * total, int(correct), total], device='cuda')
                        if rank == 0 and batch_index % 100 == 0:
                            print(json.dumps({'step': step, 'validation_batches': batch_index,
                                              'validation_total_batches': len(validation)}), flush=True)
                if world > 1:
                    dist.all_reduce(stats)
                values = stats.tolist()
                score = values[0] / max(1, values[2])
            else:
                score, values = losses, [losses * tokens, hits, tokens]
            if rank == 0:
                print(json.dumps({'step': step, 'validation_loss': score,
                                  'validation_agreement': values[1] / max(1, values[2]),
                                  'validation_tokens': values[2]}), flush=True)
                draft.save(args.output / f'mtp-{step}.safetensors', teacher)
                if score < best:
                    draft.save(args.output / 'mtp.safetensors', teacher)
            best = min(best, score)
            draft.train()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
