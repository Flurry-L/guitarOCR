"""Export a language-only information LoRA for per-request task selection."""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

from research.common.ocr import INFORMATION_TASKS


def install_permutation_support():
    from conversion.glm import Glm4Model

    original = Glm4Model.normal_to_neox

    def normal_to_neox(weights, *args, **kwargs):
        if callable(getattr(weights, "get_lora_A_B", None)):
            a, b = weights.get_lora_A_B()
            return type(weights)(a, original(b, *args, **kwargs))
        return original(weights, *args, **kwargs)

    Glm4Model.normal_to_neox = staticmethod(normal_to_neox)


def convert_worker():
    llama_cpp = Path(sys.argv.pop(1)).resolve()
    sys.path.insert(0, str(llama_cpp))
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    install_permutation_support()
    converter = llama_cpp / "convert_lora_to_gguf.py"
    sys.argv[0] = str(converter)
    runpy.run_path(str(converter), run_name="__main__")


def export(base, adapter, output, llama_cpp, tasks=None):
    from safetensors.torch import load_file, save_file

    tasks = list(tasks) if tasks is not None else sorted(INFORMATION_TASKS)
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= INFORMATION_TASKS:
        raise ValueError('Declare a nonempty unique subset of supported information tasks')

    peft = output / "merged/information_adapter"
    peft.mkdir(parents=True, exist_ok=True)
    gguf = output / "gguf/information-adapter.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    source = adapter / "adapter_model.safetensors"
    tensors = load_file(source, device="cpu")
    language, excluded = {}, []
    for name, value in tensors.items():
        if ".visual." in name:
            excluded.append(name)
        elif ".language_model." in name and ".lora_" in name:
            language[name] = value.contiguous()
        else:
            raise ValueError(f"Unexpected adapter tensor: {name}")
    if not language:
        raise ValueError("No language LoRA tensors found")
    save_file(language, peft / "adapter_model.safetensors")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("modules_to_save"):
        raise ValueError("Task adapter must contain only LoRA deltas")
    config["target_modules"] = [
        name for name in config["target_modules"] if not name.startswith("merger.")
    ]
    config["base_model_name_or_path"] = str(base.resolve())
    (peft / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.export.export_information_adapter",
            "--convert",
            str(llama_cpp),
            "--base",
            str(base),
            "--outtype",
            "f16",
            "--outfile",
            str(gguf),
            str(peft),
        ],
        check=True,
    )
    report = {
        "base": str(base.resolve()),
        "source_adapter": str(adapter.resolve()),
        "conversion_fix": "GLM rotary row permutation applies to LoRA B only: P(BA)=(PB)A",
        "language_parameters": sum(t.numel() for t in language.values()),
        "excluded_visual_tensors": excluded,
        "tasks": tasks,
        "artifacts": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
            }
            for path in [peft / "adapter_model.safetensors", peft / "adapter_config.json", gguf]
        },
    }
    (output / "information-adapter-export.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    if sys.argv[1:2] == ["--convert"]:
        sys.argv.pop(1)
        convert_worker()
        raise SystemExit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--tasks", nargs='+', choices=sorted(INFORMATION_TASKS))
    export(**vars(parser.parse_args()))
