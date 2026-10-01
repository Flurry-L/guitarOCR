"""Read and write standard Hugging Face safetensors checkpoints."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.common.model_files import checkpoint_files


def load_checkpoint(folder):
    from safetensors.torch import load_file

    state = {}
    for path in checkpoint_files(Path(folder)):
        if path.suffix == '.safetensors':
            state.update(load_file(path))
    return state


def save_checkpoint(state, folder, max_shard_size='1GB'):
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shards = split_torch_state_dict_into_shards(state, max_shard_size=max_shard_size)
    names = set(shards.filename_to_tensors)
    with TemporaryDirectory(prefix='.checkpoint-', dir=folder) as temporary:
        temporary = Path(temporary)
        for filename, keys in shards.filename_to_tensors.items():
            save_file({key: state[key].contiguous() for key in keys},
                      temporary / filename, metadata={'format': 'pt'})
        if shards.is_sharded:
            names.add('model.safetensors.index.json')
            (temporary / 'model.safetensors.index.json').write_text(json.dumps({
                'metadata': shards.metadata, 'weight_map': shards.tensor_to_filename,
            }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        for filename in sorted(names, key=lambda name: name.endswith('.json')):
            (temporary / filename).replace(folder / filename)
    previous = {folder / 'model.safetensors', folder / 'model.safetensors.index.json',
                *folder.glob('model-*-of-*.safetensors')}
    for path in previous:
        if path.name not in names:
            path.unlink(missing_ok=True)
    return sorted(folder / name for name in names)
