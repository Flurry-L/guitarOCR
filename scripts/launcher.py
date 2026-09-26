"""First-run installer and launcher. Uses only Python's standard library."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.defaults import environment_python  # noqa: E402
from shared.environment import verify_files  # noqa: E402
from scripts.downloads import acquire_base_model  # noqa: E402

PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_MIRROR = "https://mirror.sjtu.edu.cn/pytorch-wheels"


def has_uv_config():
    """Let an explicitly configured index take precedence over our default."""
    settings = {"UV_CONFIG_FILE", "UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX",
                "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS", "UV_NO_INDEX", "UV_OFFLINE"}
    if settings.intersection(os.environ):
        return True
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    windows_config = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    return any(path.is_file() for path in (
        ROOT / "uv.toml", config_home / "uv/uv.toml",
        windows_config / "uv/uv.toml", Path("/etc/uv/uv.toml"),
    ))


def run(command, *, capture=False, env=None):
    print("\n> " + subprocess.list2cmdline([str(p) for p in command]), flush=True)
    result = subprocess.run(
        [str(p) for p in command],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return result.stdout if capture else None


def run_uv(uv, arguments, *, capture=False):
    environment = os.environ.copy()
    environment.setdefault("UV_HTTP_TIMEOUT", "30")
    environment.setdefault("UV_HTTP_RETRIES", "2")
    preferred = list(arguments)
    if preferred[:2] == ["pip", "install"]:
        if not has_uv_config():
            environment["UV_DEFAULT_INDEX"] = PYPI_MIRROR
            print("Python 包使用清华镜像。", flush=True)
        if "--torch-backend" in preferred:
            index = preferred.index("--torch-backend")
            backend = preferred[index + 1]
            # This mirror includes older copies of ordinary PyPI dependencies.
            # Exhaust it before looking for the pinned version on the PyPI mirror.
            preferred[index:index + 2] = [
                "--index", f"{TORCH_MIRROR}/{backend}/",
                "--index-strategy", "unsafe-first-match",
            ]
            # UV_TORCH_BACKEND overrides index selection even without the CLI flag.
            environment.pop("UV_TORCH_BACKEND", None)
            print("PyTorch 使用上海交大镜像。", flush=True)
    try:
        return run([uv, *preferred], capture=capture, env=environment)
    except subprocess.CalledProcessError:
        print("当前步骤执行失败，正在使用默认配置和官方源重试。", flush=True)
    # Keep cache, proxy and certificate settings; reset uv's resolver settings
    # only in the retry process, without changing the user's configuration.
    preserved = {
        "UV_CACHE_DIR",
        "UV_HTTP_TIMEOUT",
        "UV_HTTP_RETRIES",
        "UV_NATIVE_TLS",
        "UV_SYSTEM_CERTS",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_CACHE_DIR",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("UV_") or key in preserved
    }
    return run([uv, "--no-config", *arguments], capture=capture, env=environment)


def install_fingerprint():
    digest = sha256()
    for name in (
        "uv.lock",
        "pyproject.toml",
        "weights/manifest.json",
        "scripts/launcher.py",
        "scripts/downloads.py",
    ):
        digest.update((ROOT / name).read_bytes())
    digest.update(f"{platform.system()}:{platform.machine()}".encode())
    return digest.hexdigest()


def download(url, destination, expected=None):
    """Retry safely; never replace a good file with a partial download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            print(f"下载 {destination.name}（{attempt + 1}/3）", flush=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "GuitarOCR-installer"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected:
                errors = verify_files(
                    partial.parent, [{**expected, "name": partial.name}]
                )
                if errors:
                    raise ValueError("；".join(errors))
            partial.replace(destination)
            return
        except (OSError, ValueError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2)


def acquire_weights(manifest):
    missing = [
        (entry, item)
        for entry in manifest["models"]
        for item in entry["files"]
        if verify_files(ROOT / entry["path"], [item])
    ]
    if not missing:
        print("随项目发布的 LoRA 和版面权重校验通过。", flush=True)
        return
    # Release ZIPs normally include all files; their pinned provenance also
    # allows repair without requiring Git on the user's machine.
    release_path = ROOT / "release.json"
    if release_path.is_file():
        release = json.loads(release_path.read_text(encoding="utf-8"))
        repository = release.get("repository", "")
        commit = release.get("commit") or ""
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("安装包信息不完整，请重新下载 Release 中的 GuitarOCR ZIP 并完整解压。")
        for entry, item in missing:
            path = Path(entry["path"]) / item["name"]
            host = "media.githubusercontent.com/media" if path.suffix in {".safetensors", ".pdiparams"} else "raw.githubusercontent.com"
            download(f"https://{host}/{repository}/{commit}/{path.as_posix()}", ROOT / path, item)
        return
    # Git checkouts fetch the revision they actually checked out.
    try:
        remote = run(["git", "remote", "get-url", "origin"], capture=True).strip()
        commit = run(["git", "rev-parse", "HEAD"], capture=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(
            "当前目录缺少模型。请从 https://github.com/Flurry-L/guitarOCR/releases/latest 下载 GuitarOCR ZIP 并完整解压。"
        ) from None
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?",
        remote,
    )
    if not match or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("请在此 Git 检出执行 git lfs pull，以取得 weights/ 中的模型。")
    for entry, item in missing:
        path = Path(entry["path"]) / item["name"]
        attributes = run(
            ["git", "check-attr", "-z", "filter", "--", path.as_posix()], capture=True
        ).split("\0")
        host = (
            "media.githubusercontent.com/media"
            if attributes[2] == "lfs"
            else "raw.githubusercontent.com"
        )
        url = f"https://{host}/{match[1]}/{commit}/{path.as_posix()}"
        download(url, ROOT / path, item)


def choose_device(requested):
    if requested != "auto":
        return requested
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver = (
            int(result.stdout.strip().split(".")[0]) if result.returncode == 0 else 0
        )
        if driver >= 580:
            return "cuda"
        if driver:
            print(
                "检测到 NVIDIA 显卡，但驱动低于 580。此次使用 CPU；升级驱动后可重新安装 GPU 环境。"
            )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return "cpu"


def torch_reinstall_args(python, device):
    """A +cpu wheel also satisfies ==X.Y; force replacement when switching devices."""
    result = subprocess.run(
        [str(python), "-c", "import torch; print(torch.version.cuda or 'cpu')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        return []
    installed = result.stdout.strip()
    if (device == "cuda" and installed != "13.0") or (
        device == "cpu" and installed != "cpu"
    ):
        return ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
    return []


@contextmanager
def installation_lock(tools):
    """OS lock releases on crash, so interrupted setup can be rerun."""
    tools.mkdir(parents=True, exist_ok=True)
    with (tools / "install.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError("另一个安装窗口正在运行，请等待它完成") from None
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("另一个安装窗口正在运行，请等待它完成") from None
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def export_requirements(uv, python, device):
    # CI checks lock freshness; installation must not re-resolve it using local indexes.
    requirements = run_uv(
        uv,
        [
            "export",
            "--frozen",
            "--python",
            python,
            "--extra",
            "glm-ocr",
            "--extra",
            "webui",
            "--no-dev",
            "--no-hashes",
            "--no-emit-project",
            "--format",
            "requirements-txt",
        ],
        capture=True,
    )
    if device == "cpu":
        requirements = (
            "\n".join(
                line
                for line in requirements.splitlines()
                if not re.match(r"^(?:nvidia-|cuda-|triton[=;])", line)
            )
            + "\n"
        )
    backend = "cpu" if device == "cpu" else "cu130"
    # Keep a missing CPU/CUDA mirror wheel from resolving to another backend.
    return re.sub(
        r"^(torch|torchvision)==([0-9.]+)(?:\+[^\s;]+)?",
        rf"\1==\2+{backend}", requirements, flags=re.MULTILINE,
    )


def install(args, uv, tools):
    if platform.system() not in {
        "Windows",
        "Linux",
    } or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError(
            "一键完整安装目前支持 Windows / Linux x64；其他平台可按 docs/setup.md 安装编辑与导出环境。"
        )
    device = choose_device(args.device)
    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    print(
        f"开始安装：{'NVIDIA GPU' if device == 'cuda' else 'CPU（识别较慢）'}。首次需要下载数 GB，请保持窗口开启。",
        flush=True,
    )
    acquire_weights(manifest)
    app_python = environment_python(tools / "webui-venv")
    layout_python = environment_python(tools / "webui-paddle-venv")
    for folder in (app_python, layout_python):
        if not folder.is_file():
            run_uv(
                uv,
                [
                    "venv",
                    "--python",
                    "3.11",
                    "--allow-existing",
                    folder.parent.parent,
                ],
            )
    requirements = export_requirements(uv, app_python, device)
    requirements_path = tools / "webui-requirements.txt"
    requirements_path.write_text(requirements, encoding="utf-8")
    run_uv(
        uv,
        [
            "pip",
            "install",
            "--python",
            app_python,
            "--torch-backend",
            "cu130" if device == "cuda" else "cpu",
            *torch_reinstall_args(app_python, device),
            "-r",
            requirements_path,
        ],
    )
    run_uv(uv, ["pip", "install", "--python", app_python, "--no-deps", "-e", ROOT])
    # Paddle's CPU runtime keeps both platforms on the same supported package set.
    # GLM is the dominant workload and uses the chosen GPU independently.
    run_uv(
        uv,
        [
            "pip",
            "install",
            "--python",
            layout_python,
            "paddlepaddle==3.2.0",
            "paddlex[ocr,cv]==3.7.2",
            "numpy==1.26.4",
            "Pillow>=12.3,<13",
            "pypdfium2>=5.13,<6",
            "pdfplumber>=0.11.10,<1",
        ],
    )
    run_uv(uv, ["pip", "install", "--python", layout_python, "--no-deps", "-e", ROOT])
    base = manifest["base_model"]
    model = ROOT / base["path"]
    acquire_base_model(base, ROOT)
    run(
        [
            app_python,
            "-m",
            "shared.environment",
            "--hashes",
            "--device",
            device,
            "--layout-python",
            layout_python,
        ]
    )
    state = {
        "root": str(ROOT),
        "device": device,
        "model": str(model),
        "layout_python": str(layout_python),
        "lock": sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "installer": install_fingerprint(),
    }
    (tools / "install-state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print("\n安装完成。以后双击 start.bat 或运行 bash start.sh 即可。", flush=True)
    return state


def launch(args, tools, state):
    port = args.port
    with socket.socket() as sock:
        if os.name != "nt":
            # Match Uvicorn: closed connections must not block an immediate restart.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise ValueError(
                f"端口 {port} 已被占用。已有工作台可直接打开 http://127.0.0.1:{port}，或使用 --port 7861。"
            ) from None
    url = f"http://127.0.0.1:{port}"
    stopped = threading.Event()

    def open_browser():
        for _ in range(120):
            if stopped.wait(0.5):
                return
            try:
                with urllib.request.urlopen(url + "/api/config", timeout=1):
                    webbrowser.open(url)
                    return
            except (OSError, urllib.error.URLError):
                pass

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()
    print(f"\n工作台地址：{url}\n使用期间保留此窗口。按 Ctrl+C 停止。", flush=True)
    try:
        run(
            [
                environment_python(tools / "webui-venv"),
                "-m",
                "webapp.app",
                "--port",
                str(port),
                "--device",
                state["device"],
                "--model",
                state["model"],
                "--layout-python",
                state["layout_python"],
                "--output",
                args.output,
            ]
        )
    finally:
        stopped.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "start", "check"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output", type=Path, default=ROOT / "output/webui")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "tools")
    args = parser.parse_args()
    os.chdir(ROOT)
    tools = args.runtime_root.resolve()
    uv = os.environ.get("GUITAROCR_UV") or shutil.which("uv")
    if not uv:
        raise ValueError("找不到 uv，请使用根目录的 install / start 启动脚本")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    if not 1 <= args.port <= 65535:
        raise ValueError("端口范围为 1–65535")
    state_path = tools / "install-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else None
    )
    changed = state and (
        state.get("lock") != sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
        or state.get("root") != str(ROOT)
        or state.get("installer") != install_fingerprint()
        or not environment_python(tools / "webui-venv").is_file()
        or not environment_python(tools / "webui-paddle-venv").is_file()
    )
    if args.command == "install" or (
        args.command == "start" and (not state or changed)
    ):
        if args.device == "auto" and state:
            args.device = state["device"]
        with installation_lock(tools):
            state = install(args, uv, tools)
    if args.command == "check":
        if not state:
            raise ValueError("尚未完成安装，请运行 install.bat 或 bash install.sh")
        run(
            [
                environment_python(tools / "webui-venv"),
                "-m",
                "shared.environment",
                "--hashes",
                "--device",
                state["device"],
                "--layout-python",
                state["layout_python"],
            ]
        )
    elif args.command == "start":
        if args.device != "auto":
            state = {**state, "device": args.device}
        launch(args, tools, state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止。下次运行启动脚本可继续。")
        raise SystemExit(130)
    except Exception as error:
        print(
            f"\n未能完成：{error}\n请查看 output/logs/ 最新日志；处理后重跑同一脚本即可。",
            file=sys.stderr,
        )
        raise SystemExit(1)
