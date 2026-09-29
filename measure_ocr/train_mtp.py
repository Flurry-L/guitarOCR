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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Merged music model with the original MTP weights')
    parser.add_argument('--tokenized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-mtp', type=Path, help='Continue distillation from a previously trained MTP layer')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--epochs', type=float, help='Cover this many complete data passes instead of --steps')
    parser.add_argument('--token-budget', type=int, default=0, help='Padded tokens per GPU; zero uses fixed batches')
    parser.add_argument('--maximum-batch-examples', type=int, default=128)
    parser.add_argument('--context-chunk-size', type=int, default=8)
    parser.add_argument('--share-context-images', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--eval-batch-size', type=int, default=48)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--early-stopping-patience', type=int, default=0,
                        help='Stop after this many full validations without meaningful improvement; zero disables')
    parser.add_argument('--minimum-relative-improvement', type=float, default=.01)
    parser.add_argument('--image-max-pixels', type=int, default=768 * 768,
                        help='Match the image budget used to build the tokenized training data')
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    if args.early_stopping_patience < 0 or not 0 <= args.minimum_relative_improvement < 1:
        parser.error('Early-stopping patience must be nonnegative and relative improvement must be in [0, 1)')
    from shared.score_image import install_training_policy
    install_training_policy(args.model)
    if args.share_context_images:
        from measure_ocr.shared_vision import install_shared_vision
        install_shared_vision()
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
    processor.image_max_pixels = args.image_max_pixels
    processor.image_min_pixels = 32 * 32
    cache = load_from_disk(args.tokenized)
    collator = SFTDataCollatorWith4DAttentionMask(template=template, model=teacher,
                                               tokenizer=processor.tokenizer, processor=processor,
                                               pad_to_multiple_of=8, attn_implementation='flash_attention_2',
                                               compute_dtype=torch.bfloat16)
    if args.token_budget:
        import pyarrow.compute as pc
        from measure_ocr.token_batching import TokenBatchSampler, RankBatchSampler, neighbour_chunks
        data = cache['train']
        lengths = list(data['length']) if 'length' in data.column_names else [len(ids) for ids in data['input_ids']]
        images = data.data.column('images')
        if data._indices is not None:
            images = pc.take(images, data._indices.column(0))
        chunks = neighbour_chunks(images.to_pylist(), args.context_chunk_size)
        batches = TokenBatchSampler(lengths, world, args.token_budget, args.maximum_batch_examples, args.seed, chunks)
        sampler = RankBatchSampler(batches, rank)
        loader = DataLoader(data, batch_sampler=sampler, collate_fn=collator,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        sampler = DistributedSampler(cache['train'], num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        loader = DataLoader(cache['train'], batch_size=args.batch_size, sampler=sampler, collate_fn=collator,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    if args.epochs is not None:
        args.steps = math.ceil(len(loader) * args.epochs / args.accumulation)
    if rank == 0:
        print(json.dumps({'training_examples': len(cache['train']), 'steps_per_epoch': len(loader),
                          'steps': args.steps, 'token_budget': args.token_budget}), flush=True)
    val_sampler = DistributedSampler(cache['validation'], num_replicas=world, rank=rank, shuffle=False) if 'validation' in cache else None
    validation = DataLoader(cache['validation'], batch_size=args.eval_batch_size, sampler=val_sampler, collate_fn=collator,
                            num_workers=2, pin_memory=True) if val_sampler else None
    if args.early_stopping_patience and validation is None:
        raise ValueError('Early stopping requires a validation split')
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=.01, fused=True)
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    cross_entropy = LigerFusedLinearCrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    iterator, epoch = iter(loader), 0
    started, best = time.perf_counter(), float('inf')
    progress_best, stalled = float('inf'), 0
    evaluations = []

    def forward(batch):
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch.pop('labels')
        batch.pop('length', None)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = teacher.model(**batch, use_cache=False).last_hidden_state
            embeddings = teacher.model.language_model.embed_tokens(batch['input_ids'][:, 1:])
            positions = batch.get('position_ids')
            if positions is None and batch.get('image_grid_thw') is not None:
                positions, _ = teacher.model.get_rope_index(
                    batch['input_ids'], image_grid_thw=batch['image_grid_thw'],
                    attention_mask=batch['attention_mask'], mm_token_type_ids=batch.get('mm_token_type_ids'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, recurrence = wrapped(embeddings, hidden[:, :-1], batch['attention_mask'][:, :-1],
                                         positions[..., :-1] if positions is not None else None)
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
            if not math.isfinite(score):
                raise ValueError(f'Non-finite MTP validation loss at step {step}: {score}')
            if score < progress_best * (1 - args.minimum_relative_improvement):
                progress_best, stalled = score, 0
            else:
                stalled += 1
            stop_early = bool(args.early_stopping_patience and stalled >= args.early_stopping_patience)
            evaluation = {'step': step, 'validation_loss': score,
                          'validation_agreement': values[1] / max(1, values[2]),
                          'validation_tokens': values[2]}
            evaluations.append(evaluation)
            if rank == 0:
                print(json.dumps(evaluation), flush=True)
                draft.save(args.output / f'mtp-{step}.safetensors', teacher)
                if score < best:
                    draft.save(args.output / 'mtp.safetensors', teacher)
                (args.output / 'training_results.json').write_text(json.dumps({
                    'planned_steps': args.steps, 'completed_steps': step,
                    'completed_epochs': step * args.accumulation / len(loader),
                    'best_validation_loss': min(best, score),
                    'early_stopping_patience': args.early_stopping_patience,
                    'minimum_relative_improvement': args.minimum_relative_improvement,
                    'stop_reason': 'validation_plateau' if stop_early else
                                   'step_limit' if step == args.steps else None,
                    'evaluations': evaluations,
                }, indent=2))
            best = min(best, score)
            draft.train()
            if stop_early:
                if rank == 0:
                    print(f'Stopping MTP at step {step}: full validation loss has plateaued', flush=True)
                break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
