"""Diagnose a local installation without loading recognition models."""

from __future__ import annotations

import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from shared.model_files import verify_files
from shared.defaults import MODEL, paddle_python


def inspect(
    root: Path, model: Path, layout_python: Path, device="cpu", hashes=False, core=False
):
    checks = []

    def check(name, errors, fix=""):
        checks.append({"name": name, "ok": not errors, "errors": errors, "fix": fix})

    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "PIL": "Pillow",
        "pypdfium2": "pypdfium2",
        "pdfplumber": "pdfplumber",
        "guitarpro": "PyGuitarPro",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "multipart": "python-multipart",
    }
    engine = os.environ.get('GUITAROCR_BACKEND', 'transformers')
    auxiliary = os.environ.get('GUITAROCR_AUX_MODELS')
    if not core and auxiliary:
        packages.update(onnxruntime='onnxruntime', cv2='opencv-python-headless')
    if not core and engine == 'vllm':
        packages.update(vllm='vllm', torch='torch')
    if not core and engine == 'transformers':
        packages.update(
            torch="torch",
            torchvision="torchvision",
            transformers="transformers",
            peft="peft",
            accelerate="accelerate",
        )
    versions = {}
    for module, distribution in packages.items():
        found = util.find_spec(module) is not None
        if found:
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                pass
        check(
            distribution,
            [] if found else [f"未安装 {distribution}"],
            "重新运行 install.bat 或 bash install.sh",
        )
    if not core:
        if auxiliary:
            from scripts.distribution import check_models
            check('所选模型', check_models(engine, Path(auxiliary).parent), '重新运行安装脚本')
            if engine == 'llamacpp':
                import shutil
                from scripts.llamacpp_runtime import probe
                executable = shutil.which(os.environ.get('GUITAROCR_LLAMA_SERVER', 'llama-server'))
                try:
                    if not executable:
                        raise ValueError('找不到 llama-server')
                    probe(executable)
                    check('llama-server 固定版本', [])
                except ValueError as error:
                    check('llama-server 固定版本', [str(error)], '安装官方预编译 runtime 或指定固定版本可执行文件')
            from shared.onnx_runtime import cpu_session
            for name in ('layout', 'signature'):
                try:
                    cpu_session(Path(auxiliary) / f'{name}.onnx')
                    check(name + ' ONNX', [])
                except Exception as error:
                    check(name + ' ONNX', [str(error)], '重新安装所选运行环境')
        else:
            manifest = json.loads((root / 'weights/manifest.json').read_text(encoding='utf-8'))
            for entry in manifest['models']:
                check(entry['stage'] + ' 权重', verify_files(root / entry['path'], entry['files'], hashes),
                      '重新运行安装脚本；Git 检出也可执行 git lfs pull')
        if util.find_spec("torch"):
            try:
                import torch

                if device.startswith("cuda"):
                    # Exercise a kernel; CUDA availability alone misses unsupported GPU architectures.
                    tensor = torch.ones((8, 8), device=device)
                    (tensor @ tensor).sum().item()
                    versions["gpu"] = torch.cuda.get_device_name(device)
                check("运算设备", [])
            except Exception as error:
                check(
                    "运算设备",
                    [str(error)],
                    "运行 install.bat --device cuda / bash install.sh --device cuda 安装 GPU 版；检查 NVIDIA 驱动，或改用 --device cpu",
                )
        if not auxiliary:
            try:
                command = [
                    str(layout_python),
                    "-c",
                    "import paddle; from paddlex import create_model; print(paddle.__version__)",
                ]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
                check(
                    "Paddle 版面环境",
                    [] if result.returncode == 0 else [result.stderr[-2000:]],
                    "重新运行安装脚本；Linux 若缺 libGL.so.1：sudo apt-get install libgl1 libglib2.0-0",
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                check(
                    "Paddle 版面环境",
                    [str(error)],
                    "重新运行 install.bat 或 bash install.sh",
                )
    return {
        "ok": all(c["ok"] for c in checks),
        "system": platform.platform(),
        "python": sys.version.split()[0],
        "device": device,
        "versions": versions,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layout-python", type=Path, default=paddle_python())
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hashes", action="store_true", help="Verify complete model SHA-256 hashes"
    )
    parser.add_argument(
        "--core", action="store_true", help="Check editor/export dependencies only"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = inspect(
        args.root, args.model, args.layout_python, args.device, args.hashes, args.core
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for check in result["checks"]:
            print(f"[{'OK' if check['ok'] else '缺失'}] {check['name']}")
            for error in check["errors"]:
                print("  " + error)
            if not check["ok"]:
                print("  处理方法：" + check["fix"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
