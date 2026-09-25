"""Diagnose a local installation without loading recognition models."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata, util
import json
from pathlib import Path
import platform
import subprocess
import sys

from shared.defaults import MODEL, paddle_python


def verify_files(folder: Path, files: list[dict], hashes: bool = True) -> list[str]:
    errors = []
    for item in files:
        path = folder / item["name"]
        if not path.is_file():
            errors.append(f"缺少文件：{path}")
            continue
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                errors.append(f"尚未下载 Git LFS 权重：{path}")
                continue
            if path.stat().st_size != item["bytes"]:
                errors.append(f"文件大小不符，可能下载未完成：{path}")
                continue
            if hashes:
                handle.seek(0)
                digest = sha256()
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != item["sha256"]:
                    errors.append(f"SHA-256 不符：{path}")
    return errors


def inspect(
    root: Path, model: Path, layout_python: Path, device="cpu", hashes=False, core=False
):
    checks = []

    def check(name, errors, fix=""):
        checks.append({"name": name, "ok": not errors, "errors": errors, "fix": fix})

    packages = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "pymupdf": "PyMuPDF",
        "guitarpro": "PyGuitarPro",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "multipart": "python-multipart",
    }
    if not core:
        packages.update(
            torch="torch",
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
        manifest = json.loads(
            (root / "weights/manifest.json").read_text(encoding="utf-8")
        )
        for entry in manifest["models"]:
            check(
                entry["stage"] + " 权重",
                verify_files(root / entry["path"], entry["files"], hashes),
                "重新运行安装脚本；Git 检出也可执行 git lfs pull",
            )
        check(
            "GLM-OCR 基座",
            verify_files(model, manifest["base_model"]["files"], hashes),
            "重新运行 install.bat 或 bash install.sh，自动继续下载",
        )
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
