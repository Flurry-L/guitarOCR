"""Merge a task adapter while preserving GLM-OCR's native MTP weights."""

import argparse
import json
from pathlib import Path
import shutil


def export_model(base, adapter, output, mtp=None):
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from shared.checkpoint import save_checkpoint
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base, output = Path(base), Path(output)
    torch.set_num_threads(4)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).to(dtype=torch.bfloat16).merge_and_unload()
    state = model.state_dict()
    layer = model.config.text_config.num_hidden_layers
    prefix = f'model.language_model.layers.{layer}.'
    for path in sorted(base.glob('*.safetensors')):
        with safe_open(path, framework='pt') as source:
            for key in source.keys():
                if key.startswith(prefix):
                    state[key] = source.get_tensor(key)
    if mtp:
        with safe_open(mtp, framework='pt') as source:
            for key in source.keys():
                if not key.startswith(prefix):
                    raise ValueError(f'Unexpected MTP parameter: {key}')
                state[key] = source.get_tensor(key)
    if not any(k.startswith(prefix) for k in state):
        raise ValueError('No MTP layer found in the original model')
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(state, output)
    model.config.save_pretrained(output)
    model.generation_config.save_pretrained(output)
    AutoProcessor.from_pretrained(base).save_pretrained(output)
    if (base / 'music_vocabulary.json').is_file():
        shutil.copyfile(base / 'music_vocabulary.json', output / 'music_vocabulary.json')
    if (base / 'score_image_policy.json').is_file():
        shutil.copyfile(base / 'score_image_policy.json', output / 'score_image_policy.json')
    capability_root = Path(adapter) if adapter and (Path(adapter) / 'capabilities.json').is_file() else base
    if (capability_root / 'capabilities.json').exists():
        capabilities = json.loads((capability_root / 'capabilities.json').read_text())
        if capabilities.get('state_reader'):
            source = (capability_root / capabilities['state_reader']).resolve()
            destination = output / 'state_reader' / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            capabilities['state_reader'] = str(destination.relative_to(output))
            if source != destination.resolve():
                shutil.copyfile(source, destination)
        (output / 'capabilities.json').write_text(json.dumps(capabilities))
    print(json.dumps({'output': str(output), 'parameters': sum(v.numel() for v in state.values()),
                      'mtp_tensors': sum(k.startswith(prefix) for k in state)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('tools/models/GLM-OCR'))
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mtp', type=Path)
    args = parser.parse_args()
    export_model(args.base, args.adapter, args.output, args.mtp)
