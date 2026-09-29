"""Add reversible music lexemes and initialize their embedding/output rows."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def structured_lexemes(base, training_data=None):
    """Cover the supported musical fields, with ordinary-token fallback."""
    pitches = [f'p{i}' for i in range(128)]
    fingerings = [f's{s}f{f}' for s in range(1, 13) for f in [*range(37), 'x']]
    onsets = [f'@{i}' for i in range(0, 15361, 40)]
    durations = [f':{d}{dots}{tuplet}:' for d in 'whqestf'
                 for dots in ['', '.', '..'] for tuplet in ['', '[3:2]', '[5:4]', '[7:4]', '[7:8]']]
    fields = pitches + fingerings + onsets + durations
    if training_data:
        fields += select_lexemes(base, training_data, minimum_count=20, limit=4096)
    return list(dict.fromkeys(fields))


def select_lexemes(base, training_data, minimum_count=50, limit=2048):
    """Rank whole music fields by saved tokens using training labels only."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    pattern = re.compile(r'@\d+|s\d+f(?:\d+|x)|p\d+|:(?:[whqestf]|d\d+)(?:\.{1,2})?(?:\[\d+:\d+\])?:')
    counts = Counter()
    with Path(training_data).open() as source:
        for line in source:
            row = json.loads(line)
            target = row.get('target') or row['messages'][-1]['content']
            counts.update(pattern.findall(target))
    ranked = [(count * (len(tokenizer.encode(text, add_special_tokens=False)) - 1), text)
              for text, count in counts.items() if count >= minimum_count]
    return [text for saved, text in sorted(ranked, reverse=True)[:limit] if saved > 0]


def retokenize_dataset(base, expanded, source, output, workers=16):
    """Replace target tokens while retaining multimodal prompt/image alignment."""
    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    old = AutoTokenizer.from_pretrained(base)
    new = AutoTokenizer.from_pretrained(expanded)

    def rewrite(batch):
        result = {k: [] for k in ('input_ids', 'labels', 'attention_mask', 'length')}
        for ids, labels in zip(batch['input_ids'], batch['labels'], strict=True):
            start = next(i for i, token in enumerate(labels) if token != -100)
            text = old.decode(labels[start:], skip_special_tokens=True)
            tokens = new.encode(text, add_special_tokens=False)
            if new.decode(tokens, skip_special_tokens=False) != text:
                raise ValueError('Vocabulary changed the score text')
            tokens.append(new.eos_token_id)
            values = ids[:start] + tokens
            result['input_ids'].append(values)
            result['labels'].append([-100] * start + tokens)
            result['attention_mask'].append([1] * len(values))
            result['length'].append(len(values))
        return result

    data = load_from_disk(str(source))
    DatasetDict({split: rows.map(rewrite, batched=True, batch_size=128, num_proc=workers,
                                desc='Music vocabulary ' + split) for split, rows in data.items()}).save_to_disk(str(output))


def prepare_vocabulary(base, adapter, output, lexemes):
    import torch
    from shared.checkpoint import load_checkpoint, save_checkpoint
    from transformers import AddedToken, AutoConfig, AutoProcessor
    from shared.export_glm import export_model

    output = Path(output)
    export_model(base, adapter, output)
    processor = AutoProcessor.from_pretrained(output)
    tokenizer = processor.tokenizer
    original = len(tokenizer)
    pieces = {text: tokenizer.encode(text, add_special_tokens=False) for text in lexemes}
    tokenizer.add_tokens([AddedToken(text, normalized=False, special=False) for text in lexemes])
    size = (len(tokenizer) + 63) // 64 * 64
    weights = load_checkpoint(output)
    for name in ('model.language_model.embed_tokens.weight', 'lm_head.weight'):
        old = weights[name]
        value = torch.empty((size, old.shape[1]), dtype=old.dtype)
        value[:original] = old[:original]
        value[original:] = old[:original].float().mean(0).to(old.dtype)
        for text, ids in pieces.items():
            index = tokenizer.convert_tokens_to_ids(text)
            if index < original:
                continue
            source = old[ids].float()
            average = source.mean(0)
            average *= source.norm(dim=1).mean() / average.norm().clamp_min(1e-8)
            value[index] = average.to(old.dtype)
        weights[name] = value
    config = AutoConfig.from_pretrained(output)
    prefix = f'model.language_model.layers.{config.text_config.num_hidden_layers}.'
    weights[prefix + 'embed_tokens.weight'] = weights['model.language_model.embed_tokens.weight'].clone()
    weights[prefix + 'shared_head.head.weight'] = weights['lm_head.weight'].clone()
    config.text_config.vocab_size = size
    config.save_pretrained(output)
    processor.save_pretrained(output)
    save_checkpoint(weights, output)
    metadata = {'original_vocabulary': original, 'vocabulary': len(tokenizer),
                'embedding_rows': size, 'lexemes': list(lexemes),
                'loss_weights': {str(tokenizer.convert_tokens_to_ids(text)): min(4., float(len(ids)))
                                 for text, ids in pieces.items() if tokenizer.convert_tokens_to_ids(text) >= original}}
    (output / 'music_vocabulary.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k not in {'lexemes', 'loss_weights'}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--lexemes', type=Path)
    selection.add_argument('--training-data', type=Path, help='Select frequent lexemes from training JSONL')
    parser.add_argument('--structured', action='store_true', help='Complete supported pitch/fingering/rhythm fields')
    parser.add_argument('--tokenized-input', type=Path)
    parser.add_argument('--tokenized-output', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if bool(args.tokenized_input) != bool(args.tokenized_output):
        parser.error('Both tokenized input and output are required for retokenization')
    lexemes = (structured_lexemes(args.base, args.training_data) if args.structured else
               json.loads(args.lexemes.read_text())['lexemes'] if args.lexemes else select_lexemes(args.base, args.training_data))
    prepare_vocabulary(args.base, args.adapter, args.output, lexemes)
    if args.tokenized_input:
        retokenize_dataset(args.base, args.output, args.tokenized_input, args.tokenized_output, args.workers)
