"""Build self-hosted WASM assets from the installed GuitarOCR weights.

The upstream ONNX graphs supply operators only. Every learned initializer is
replaced from the local base model plus its LoRA, including vision weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from uuid import uuid4

GRAPH_REPO = "onnx-community/GLM-OCR-ONNX"
GRAPH_REVISION = "aea46198f09e3aa2b63422dd234f1cc66afffe52"
TRANSFORMERS_VERSION = "4.3.0"
ORT_VERSION = "1.31.0-dev.20260914-8d85527a0"


def mapped_weight(name, part):
    transpose = ".MatMul." in name
    clean = name.replace(".MatMul.", ".").replace(".Add.", ".")
    split = None
    if part == "embed_tokens":
        return "model.language_model.embed_tokens.weight", False, None
    if part == "decoder_model_merged":
        if name == "model.inv_freq":
            return None
        clean = clean.replace("model.layers.16.final_norm_layernorm", "model.norm")
        if clean.startswith("model."):
            clean = clean.replace("model.", "model.language_model.", 1).replace(
                ".attn.", ".self_attn."
            )
        if ".mlp.gate_proj." in clean or ".mlp.up_proj." in clean:
            split = 0 if ".gate_proj." in clean else 1
            clean = clean.replace(".gate_proj.", ".gate_up_proj.").replace(
                ".up_proj.", ".gate_up_proj."
            )
    else:
        if name == "model.vision.inv_freq":
            return None
        clean = clean.replace(
            "model.layers.post.post_layernorm_layernorm", "model.visual.post_layernorm"
        )
        clean = clean.replace(
            "model.embeddings.patch_embed", "model.visual.patch_embed.proj"
        )
        clean = clean.replace("model.layers.", "model.visual.blocks.")
        clean = clean.replace(".norm1_layernorm.", ".norm1.").replace(
            ".norm2_layernorm.", ".norm2."
        )
        clean = clean.replace("model.downsample.", "model.visual.downsample.").replace(
            "model.merger.", "model.visual.merger."
        )
    return clean, transpose, split


def fill_graph(template, part, base, adapter, scale):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(template / "onnx" / f"{part}.onnx", load_external_data=False)
    used = set()
    for item in graph.graph.initializer:
        if not item.name.startswith(("model.", "lm_head.")):
            if item.external_data:
                raise ValueError(f"Unexpected external constant: {item.name}")
            continue
        mapping = mapped_weight(item.name, part)
        if mapping is None:
            continue
        key, transpose, split = mapping
        if key not in base.keys():
            raise ValueError(f"Unmapped learned initializer: {item.name} -> {key}")
        tensor = base.get_tensor(key).float()
        prefix = "base_model.model." + key.removesuffix(".weight")
        a_key, b_key = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
        if a_key in adapter:
            tensor = tensor + scale * (adapter[b_key].float() @ adapter[a_key].float())
            used.update([a_key, b_key])
        if split is not None:
            tensor = tensor.chunk(2, dim=0)[split]
        if "patch_embed.weight" in item.name:
            tensor = tensor.flatten(1).T
        elif transpose:
            tensor = tensor.T
        array = tensor.contiguous().numpy().astype(np.float32)
        if list(array.shape) != list(item.dims):
            raise ValueError(
                f"Shape mismatch for {item.name}: {array.shape} != {item.dims}"
            )
        item.CopyFrom(numpy_helper.from_array(array, item.name))
    return graph, used


def npm_assets(output, cache):
    """Pin browser runtimes and verify npm integrity; users fetch only our origin."""
    import base64

    for package, version, files in [
        (
            "@huggingface/transformers",
            TRANSFORMERS_VERSION,
            ["dist/transformers.min.js", "LICENSE"],
        ),
        (
            "onnxruntime-web",
            ORT_VERSION,
            [
                "dist/ort.wasm.min.mjs",
                "dist/ort-wasm-simd-threaded.wasm",
                "dist/ort-wasm-simd-threaded.mjs",
                "dist/ort-wasm-simd-threaded.jsep.wasm",
                "dist/ort-wasm-simd-threaded.jsep.mjs",
                "dist/ort-wasm-simd-threaded.asyncify.wasm",
                "dist/ort-wasm-simd-threaded.asyncify.mjs",
            ],
        ),
    ]:
        with urllib.request.urlopen(
            f"https://registry.npmjs.org/{package}/{version}", timeout=60
        ) as r:
            metadata = json.load(r)
        archive = cache / (package.split("/")[-1] + ".tgz")
        payload = (
            archive.read_bytes()
            if archive.exists()
            else urllib.request.urlopen(metadata["dist"]["tarball"], timeout=120).read()
        )
        digest = base64.b64encode(hashlib.sha512(payload).digest()).decode()
        if metadata["dist"]["integrity"] != "sha512-" + digest:
            raise ValueError("npm runtime integrity check failed")
        archive.write_bytes(payload)
        with tarfile.open(archive) as tar:
            for name in files:
                contents = tar.extractfile("package/" + name).read()
                dest = (
                    (package.split("/")[-1] + "-LICENSE")
                    if name == "LICENSE"
                    else Path(name).name
                )
                (output / dest).write_bytes(contents)
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/microsoft/onnxruntime/8d85527a0/LICENSE",
        timeout=60,
    ) as response:
        (output / "onnxruntime-web-LICENSE").write_bytes(response.read())


def build(config, cache):
    import onnx
    import torch
    from huggingface_hub import snapshot_download
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        DefaultWeightOnlyQuantConfig,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file

    torch.set_num_threads(8)
    cache.mkdir(parents=True, exist_ok=True)
    template = Path(
        snapshot_download(
            GRAPH_REPO,
            revision=GRAPH_REVISION,
            local_dir=cache / "template",
            allow_patterns=[
                "*.json",
                "*.jinja",
                "onnx/decoder_model_merged.onnx",
                "onnx/vision_encoder.onnx",
                "onnx/embed_tokens.onnx",
            ],
        )
    )
    output = config.browser_models / (".building-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    npm_assets(runtime, cache)
    for kind, path in [
        ("information", config.info_adapter),
        ("measures", config.measure_adapter),
    ]:
        destination = output / kind
        (destination / "onnx").mkdir(parents=True, exist_ok=True)
        for pattern in ("*.json", "*.jinja"):
            for file in template.glob(pattern):
                shutil.copy2(file, destination / file.name)
        preprocessing = json.loads(
            (destination / "preprocessor_config.json").read_text()
        )
        # Python defaults to normalization; Transformers.js requires it explicitly.
        preprocessing["do_normalize"] = True
        (destination / "preprocessor_config.json").write_text(json.dumps(preprocessing))
        adapter = load_file(str(Path(path) / "adapter_model.safetensors"))
        settings = json.loads((Path(path) / "adapter_config.json").read_text())
        scale = settings["lora_alpha"] / settings["r"]
        used = set()
        with safe_open(
            str(Path(config.model) / "model.safetensors"), framework="pt"
        ) as base:
            for part in ("embed_tokens", "vision_encoder", "decoder_model_merged"):
                print(f"Building {kind}/{part}", flush=True)
                graph, keys = fill_graph(template, part, base, adapter, scale)
                used.update(keys)
                with tempfile.TemporaryDirectory(dir=cache) as tmp:
                    raw = Path(tmp) / "model.onnx"
                    onnx.save_model(
                        graph,
                        raw,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location="weights.data",
                        size_threshold=1024,
                    )
                    del graph
                    result = destination / "onnx" / f"{part}_quantized.onnx"
                    result.with_suffix(".onnx.data").unlink(missing_ok=True)
                    # Keep activations in float32. Dynamic activation quantization
                    # substantially changes this fine-tuned vision encoder.
                    quantizer = MatMulNBitsQuantizer(
                        str(raw),
                        algo_config=DefaultWeightOnlyQuantConfig(
                            block_size=32, is_symmetric=True, bits=8, accuracy_level=0
                        ),
                    )
                    quantizer.process()
                    quantizer.model.save_model_to_file(
                        str(result), use_external_data_format=True
                    )
                    del quantizer
        if used != set(adapter):
            raise ValueError(f"Unapplied adapter tensors: {set(adapter) - used}")
        cfg = json.loads((destination / "config.json").read_text())
        cfg["transformers.js_config"] = {"use_external_data_format": True}
        # ORT's quantizer names external weights .onnx.data; Transformers.js expects _data.
        for part in ("embed_tokens", "vision_encoder", "decoder_model_merged"):
            path = destination / "onnx" / f"{part}_quantized.onnx"
            graph = onnx.load(path, load_external_data=False)
            old = path.with_suffix(".onnx.data")
            new_name = path.name + "_data"
            if old.exists():
                old.replace(path.parent / new_name)
                for tensor in graph.graph.initializer:
                    for entry in tensor.external_data:
                        if entry.key == "location":
                            entry.value = new_name
                onnx.save(graph, path)
        (destination / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    layout = output / "layout.onnx"
    subprocess.run(
        [
            str(Path(config.layout_python).parent / "paddle2onnx"),
            "--model_dir",
            config.layout_model,
            "--model_filename",
            "inference.json",
            "--params_filename",
            "inference.pdiparams",
            "--save_file",
            str(layout),
            "--opset_version",
            "17",
        ],
        check=True,
    )
    publish(output, config.browser_models)


def publish(output, root):
    def digest(path):
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024**2), b""):
                checksum.update(block)
        return checksum.hexdigest()

    files = {
        p.relative_to(output).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in output.rglob("*")
        if p.is_file()
    }
    revision = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    manifest = {
        "revision": revision,
        "graph_revision": GRAPH_REVISION,
        "runtime": TRANSFORMERS_VERSION,
        "files": files,
        "bytes": sum(x["bytes"] for x in files.values()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    destination = root / revision
    if destination.exists():
        shutil.rmtree(output)
    else:
        output.rename(destination)
    from shared.artifacts import write_json

    write_json(root / "manifest.json", manifest)
    print(f"Browser assets ready: {destination}", flush=True)


def main():
    from server.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("tools/browser-build"))
    args = parser.parse_args()
    build(Config.load(args.config), args.cache.resolve())


if __name__ == "__main__":
    main()
