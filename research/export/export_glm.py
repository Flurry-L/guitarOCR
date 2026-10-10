"""Merge a task adapter while preserving GLM-OCR's native MTP weights."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from research.common.ocr import load_task_adapter
from research.common.signature_image import signature_config


def checkpoint_files(folder: Path) -> list[Path]:
    """Include the index and every shard, while retaining single-file support."""
    index = folder / 'model.safetensors.index.json'
    if not index.is_file():
        return [folder / 'model.safetensors']
    names = sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
    if not names or any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
        raise ValueError(f'Invalid checkpoint index: {index}')
    return [index, *(folder / name for name in names)]


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
    with tempfile.TemporaryDirectory(prefix='.checkpoint-', dir=folder) as temporary:
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


def _asset(root, relative, field):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Invalid model capability {field}")
    source = (root / relative).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Declared {field} is missing: {source}")
    return source

def plan_capability_assets(base, adapter, output):
    """Validate all dependencies before an expensive model merge or any copy."""
    base, output = Path(base), Path(output).resolve()
    root = (
        Path(adapter)
        if adapter and (Path(adapter) / "capabilities.json").is_file()
        else base
    )
    manifest = root / "capabilities.json"
    if not manifest.is_file():
        return None, []
    capabilities = json.loads(manifest.read_text())
    if not isinstance(capabilities, dict):
        raise ValueError("Model capabilities must be an object")
    copies = []
    for field, relative in list(capabilities.items()):
        if relative is None or not (
            field == "state_reader" or field.endswith(("_model", "_calibration"))
        ):
            continue
        source = _asset(root, relative, field)
        if field == "state_reader" and source.suffix.lower() != ".onnx":
            signature_config(source)
        destination = (
            output
            / ("state_reader" if field == "state_reader" else "auxiliary")
            / source.name
        )
        copies.append((source, destination))
        capabilities[field] = destination.relative_to(output).as_posix()
        if field == "state_reader":
            for name in ("config.json", "calibration.json"):
                companion = source.parent / name
                if companion.is_file():
                    copies.append((companion.resolve(), destination.parent / name))

    declaration = capabilities.get("information_adapter")
    if declaration is not None:
        # Reuse the runtime's task and file validation. Both runtimes must be
        # present; exporting only the PEFT half breaks the desktop backend.
        peft = load_task_adapter(root, "peft").path
        gguf = load_task_adapter(root, "gguf").path
        copies.extend(
            (path.resolve(), output / "information_adapter" / path.relative_to(peft))
            for path in sorted(peft.rglob("*"))
            if path.is_file()
        )
        copies.append((gguf, output.parent / "gguf" / "information-adapter.gguf"))
        capabilities["information_adapter"] = {
            **declaration,
            "peft": "information_adapter",
            "gguf": "../gguf/information-adapter.gguf",
        }

    destinations = {}
    for source, destination in copies:
        previous = destinations.get(destination)
        if previous is not None and previous != source:
            raise ValueError(
                f"Declared assets collide at {destination}: {previous}, {source}"
            )
        destinations[destination] = source
    return capabilities, [(source, target) for target, source in destinations.items()]

def copy_capability_assets(plan, output):
    """Publish rewritten capabilities only after their declared files exist."""
    capabilities, copies = plan
    if capabilities is None:
        return
    output = Path(output)
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == destination.resolve():
            continue
        # Never write through an existing inode: training snapshots can share
        # hardlinks with the source or a previously published model generation.
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".asset-"
        ) as staging:
            staged = Path(staging) / destination.name
            shutil.copyfile(source, staged)
            staged.replace(destination)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".capabilities.json.tmp"
    temporary.write_text(json.dumps(capabilities, indent=2) + "\n")
    temporary.replace(output / "capabilities.json")


def export_model(base, adapter, output):
    capability_plan = plan_capability_assets(base, adapter, output)
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base, output = Path(base), Path(output)
    torch.set_num_threads(4)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).to(dtype=torch.bfloat16)
        model = model.merge_and_unload()
    state = model.state_dict()
    layer = model.config.text_config.num_hidden_layers
    prefix = f"model.language_model.layers.{layer}."
    for path in sorted(base.glob("*.safetensors")):
        with safe_open(path, framework="pt") as source:
            for key in source.keys():
                if key.startswith(prefix):
                    state[key] = source.get_tensor(key)
    if not any(k.startswith(prefix) for k in state):
        raise ValueError("No MTP layer found in the original model")
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(state, output)
    model.config.save_pretrained(output)
    model.generation_config.save_pretrained(output)
    AutoProcessor.from_pretrained(base).save_pretrained(output)
    if (base / "music_vocabulary.json").is_file():
        shutil.copyfile(
            base / "music_vocabulary.json", output / "music_vocabulary.json"
        )
    if (base / "score_image_policy.json").is_file():
        shutil.copyfile(
            base / "score_image_policy.json", output / "score_image_policy.json"
        )
    copy_capability_assets(capability_plan, output)
    (output / "merge_provenance.json").write_text(
        json.dumps(
            {
                "base": str(base.resolve()),
                "adapter": str(Path(adapter).resolve()) if adapter else None,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "parameters": sum(v.numel() for v in state.values()),
                "mtp_tensors": sum(k.startswith(prefix) for k in state),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("tools/models/GLM-OCR"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_model(args.base, args.adapter, args.output)
