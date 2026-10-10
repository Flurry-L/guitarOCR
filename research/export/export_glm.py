"""Merge a task adapter while preserving GLM-OCR's native MTP weights."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from research.common.ocr import load_task_adapter


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
    roots = [base] + ([Path(adapter)] if adapter else [])
    roots = [root for root in roots if (root / "capabilities.json").is_file()]
    if not roots:
        return None, []
    capabilities, declared_at = {}, {}
    for root in roots:
        values = json.loads((root / "capabilities.json").read_text())
        if not isinstance(values, dict):
            raise ValueError("Model capabilities must be an object")
        # A task adapter overrides its own declarations while retaining the
        # base checkpoint's inference settings and auxiliary readers.
        capabilities.update(values)
        declared_at.update({field: root for field in values})
    # This is product routing, not an asset of the measure checkpoint being
    # merged. The information model is maintained and packaged separately.
    capabilities.pop("information_model", None)
    copies = []
    for field, relative in list(capabilities.items()):
        if relative is None or not field.endswith(("_model", "_calibration")):
            continue
        source = _asset(declared_at[field], relative, field)
        destination = output / "auxiliary" / source.name
        copies.append((source, destination))
        capabilities[field] = destination.relative_to(output).as_posix()

    declaration = capabilities.get("information_adapter")
    if declaration is not None:
        # Reuse the runtime's task and file validation. Both runtimes must be
        # present; exporting only the PEFT half breaks the desktop backend.
        root = declared_at["information_adapter"]
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
    image_policy = Path(base) / "score_image_policy.json"
    if not image_policy.is_file():
        raise FileNotFoundError(f"Declare the OCR image policy before exporting: {image_policy}")
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
    shutil.copyfile(image_policy, output / "score_image_policy.json")
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_model(args.base, args.adapter, args.output)
