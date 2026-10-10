"""Add reversible music lexemes and initialize their embedding/output rows."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


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


def prepare_vocabulary(base, adapter, output, lexemes):
    import torch
    from transformers import AddedToken, AutoConfig, AutoProcessor

    from research.export.export_glm import (
        export_model,
        load_checkpoint,
        save_checkpoint,
    )

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
    args = parser.parse_args()
    lexemes = (structured_lexemes(args.base, args.training_data) if args.structured else
               json.loads(args.lexemes.read_text())['lexemes'] if args.lexemes else select_lexemes(args.base, args.training_data))
    prepare_vocabulary(args.base, args.adapter, args.output, lexemes)
