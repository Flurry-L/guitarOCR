"""Batch by padded token budget while keeping every distributed rank in step."""

import math
import random
from collections import defaultdict, deque


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


class RankBatchSampler:
    """Use the same batches with native DDP, without Accelerate sharding."""

    def __init__(self, batches, rank):
        self.batches, self.rank = batches, rank

    def __len__(self):
        return self.batches.steps

    def set_epoch(self, epoch):
        self.batches.sampler.set_epoch(epoch)

    def __iter__(self):
        for index, batch in enumerate(self.batches):
            if index % self.batches.world == self.rank:
                yield batch


def evaluation_batches(lengths, budget, maximum, seed, rank=0, world=1, chunks=None):
    """Balance validation work without padding the dataset with duplicate rows.

    Evaluation does not synchronize each forward pass, so ranks may have
    different batch counts. Every example contributes exactly once.
    """
    batches = TokenBatchSampler(lengths, 1, budget, maximum, seed, chunks)
    ranked, loads = [[] for _ in range(world)], [0] * world
    costs = [(max(lengths[i] for i in batch) * len(batch), batch) for batch in batches]
    for cost, batch in sorted(costs, key=lambda item: item[0], reverse=True):
        target = min(range(world), key=lambda index: (loads[index], len(ranked[index]), index))
        ranked[target].append(batch)
        loads[target] += cost
    return ranked[rank]


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
