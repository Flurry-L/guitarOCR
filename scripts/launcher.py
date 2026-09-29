"""First-run installer and launcher. Uses only Python's standard library."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from shared.model_files import verify_files  # noqa: E402
from scripts.downloads import download_verified  # noqa: E402
from scripts.progress import progress  # noqa: E402

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


def runtime_manifest():
    return json.loads((ROOT / 'scripts/runtime-manifest.json').read_text(encoding='utf-8'))


def installation_current(state, tools, profile=None):
    if not state or not state.get('engine'):
        return False
    manifest = runtime_manifest()
    from scripts.distribution import catalog
    return (state.get('generation') == manifest['generation']
            and state.get('engine_generation') == manifest['engines'][state['engine']]['generation']
            and (profile is None or profile == state.get('profile'))
            and Path(state.get('python', '')).is_file()
            and Path(state.get('models', '')).is_dir()
            and Path(state['models']).name == catalog()['generation'])


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


def acquire_weights(manifest, root=None):
    root = Path(root or ROOT)
    missing = [
        (entry, item)
        for entry in manifest["models"]
        for item in entry["files"]
        if verify_files(root / entry["path"], [item], hashes=False)
    ]
    if root != ROOT:
        from scripts.distribution import reuse
        missing = [(entry, item) for entry, item in missing
                   if not reuse(ROOT / entry['path'] / item['name'],
                                root / entry['path'] / item['name'], item)]
    if not missing:
        print("OCR 和版面模型已就绪。", flush=True)
        return
    # Release packages pin their weight assets and source revision, so installs
    # and repairs do not require Git on the user's machine.
    release_path = ROOT / "release.json"
    if release_path.is_file():
        release = json.loads(release_path.read_text(encoding="utf-8"))
        repository = release.get("repository", "")
        commit = release.get("commit") or ""
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("安装包信息不完整，请重新下载 Release 中的 GuitarOCR ZIP 并完整解压。")
        tag = release.get('model_release', '')
        assets = release.get('model_assets', {})
        if tag and not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag):
            raise ValueError('安装包的模型版本无效，请重新下载安装包。')
        total = sum(item['bytes'] for _, item in missing)
        completed = 0
        for entry, item in missing:
            path = Path(entry["path"]) / item["name"]
            asset = assets.get(path.as_posix())
            progress('models', '正在下载识别模型', completed=completed, total=total, detail=path.as_posix())
            if tag and asset:
                if not re.fullmatch(r'[\w.\-]+', asset):
                    raise ValueError('安装包的模型文件名无效。')
                url = f'https://github.com/{repository}/releases/download/{tag}/{asset}'
                download_verified(url, root / path, item,
                                  on_progress=lambda received, size: progress(
                                      'models', '正在下载识别模型', completed=completed + received,
                                      total=total, detail=path.as_posix()))
                completed += item['bytes']
                continue
            host = "media.githubusercontent.com/media" if path.suffix in {".safetensors", ".pdiparams"} else "raw.githubusercontent.com"
            download(f"https://{host}/{repository}/{commit}/{path.as_posix()}", root / path, item)
            completed += item['bytes']
        progress('models', '识别模型已就绪', completed=total, total=total)
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
        download(url, root / path, item)


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


def export_requirements(uv, python, device, engine="transformers"):
    # CI checks lock freshness; installation must not re-resolve it using local indexes.
    requirements = run_uv(
        uv,
        [
            "export",
            "--frozen",
            "--python",
            python,
            *(["--extra", "glm-ocr"] if engine == "transformers" else []),
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


def resolve_profile(args):
    device = choose_device(args.device)
    engine = getattr(args, 'engine', 'auto')
    if engine == 'auto':
        engine = 'vllm' if device == 'cuda' and platform.system() == 'Linux' else 'transformers'
    if engine == 'vllm' and (device != 'cuda' or platform.system() != 'Linux'):
        raise ValueError('vLLM 需要 Linux 和 NVIDIA GPU；此平台请选择 transformers。')
    if engine != 'llamacpp' and (platform.system() not in {'Windows', 'Linux'}
                                or platform.machine().lower() not in {'amd64', 'x86_64'}):
        raise ValueError('此平台请连接服务器，或配置 llama.cpp 后使用 --engine llamacpp。')
    return engine, device, f'{engine}-{device}'


def install(args, uv, tools):
    from scripts.distribution import acquire, environment

    engine, device, profile = resolve_profile(args)
    llama = getattr(args, 'llama_server', None) or os.environ.get('GUITAROCR_LLAMA_SERVER')
    if engine == 'llamacpp':
        llama = shutil.which(str(llama or 'llama-server'))
        if not llama:
            raise ValueError('请用 --llama-server 指定 llama-server 可执行文件，详见 docs/setup.md。')
    root = acquire(engine)
    python = environment_python(tools / 'runtimes' / profile).absolute()
    manifest = runtime_manifest()
    progress('ocr', '正在准备所选运行环境', detail=f'{engine} / {device}；仅安装这一种 OCR 引擎。')
    if not python.is_file():
        run_uv(uv, ['venv', '--python', manifest['python'], python.parent.parent])
    requirements = ((ROOT / 'scripts/runtime-vllm.txt').read_text() if engine == 'vllm'
                    else export_requirements(uv, python, device, engine))
    requirements_path = tools / f'{profile}-requirements.txt'
    requirements_path.write_text(requirements, encoding='utf-8')
    packages = [*manifest['auxiliary'], *manifest['engines'][engine].get('packages', [])]
    backend_args = (['--torch-backend', 'cu130' if device == 'cuda' else 'cpu',
                     *torch_reinstall_args(python, device)] if engine == 'transformers' else [])
    run_uv(uv, ['pip', 'install', '--python', python, *backend_args, '-r', requirements_path, *packages])
    run_uv(uv, ['pip', 'install', '--python', python, '--no-deps', '-e', ROOT])
    state = dict(engine=engine, device=device, profile=profile, python=str(python), models=str(root),
                 model=str(root / 'weights/measure_ocr/merged'), layout_python=str(python),
                 generation=manifest['generation'], engine_generation=manifest['engines'][engine]['generation'],
                 llama_server=llama)
    progress('check', '正在检查识别环境')
    run([python, '-m', 'shared.environment', '--device', device], env=environment(state))
    path = tools / 'install-state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(path)
    print(f'安装完成：{profile}。模型和运行环境会在应用更新时复用。', flush=True)
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
    from scripts.distribution import environment
    try:
        run(
            [
                state["python"],
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
            ], env=environment(state)
        )
    finally:
        stopped.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'start', 'check', 'status', 'clean', 'run'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--engine', choices=('auto', 'transformers', 'vllm', 'llamacpp'), default='auto')
    parser.add_argument('--llama-server')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/webui')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--runtime-root', type=Path, default=ROOT / 'tools')
    # `run -- ...` uses exactly the installed environment for CLI/server commands.
    argv = sys.argv[1:]
    command = argv[argv.index('--') + 1:] if '--' in argv else []
    args = parser.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    os.chdir(ROOT)
    tools = args.runtime_root.absolute()
    uv = os.environ.get('GUITAROCR_UV') or shutil.which('uv')
    if not uv and args.command in ('install', 'start'):
        raise ValueError('找不到 uv，请使用根目录的 install / start 启动脚本')
    if not 1 <= args.port <= 65535:
        raise ValueError('端口范围为 1–65535')
    path = tools / 'install-state.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if args.command in ('status', 'clean'):
        return manage_runtimes(tools, state, clean=args.command == 'clean')
    if state:
        if args.device == 'auto':
            args.device = state['device']
        if args.engine == 'auto' and state.get('engine') and args.device == state['device']:
            args.engine = state['engine']
        if not args.llama_server:
            args.llama_server = state.get('llama_server')
    profile = resolve_profile(args)[2]
    if args.command == 'install' or (args.command == 'start' and not installation_current(state, tools, profile)):
        with installation_lock(tools):
            state = install(args, uv, tools)
    if not state or not state.get('engine'):
        raise ValueError('请先运行安装脚本，迁移到按后端管理的运行环境。')
    from scripts.distribution import environment
    if args.command == 'check':
        run([state['python'], '-m', 'shared.environment', '--device', state['device']], env=environment(state))
    elif args.command == 'start':
        with installation_lock(tools / 'in-use'):
            launch(args, tools, state)
    elif args.command == 'run':
        if not command:
            raise ValueError('用法：launcher.py run -- -m pipeline.run score.pdf --output output/score')
        with installation_lock(tools / 'in-use'):
            run([state['python'], *command], env=environment(state))


def manage_runtimes(tools, state, *, clean=False):
    active = Path(state['python']).parent.parent if state and state.get('python') else None
    candidates = [*sorted((tools / 'runtimes').glob('*')),
                  *(tools / name for name in ('webui-venv', 'webui-paddle-venv', 'vllm-venv', 'desktop-edit'))]
    if clean and not active:
        raise ValueError('请先完成新环境安装；当前环境未迁移，不能清理。')
    with installation_lock(tools / 'in-use'), installation_lock(tools):
        for folder in candidates:
            if not (folder / 'pyvenv.cfg').is_file() or folder.is_symlink():
                continue
            size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
            print(f'{"当前" if folder == active else "未使用"} {folder} {size / 1e9:.2f} GB')
            if clean and folder != active:
                shutil.rmtree(folder)
                print('  已移除旧运行环境')
    if state and state.get('models'):
        print('模型缓存：' + state['models'])
    if not clean:
        print('关闭本机服务后，可运行 clean 移除上述未使用环境。用户项目和模型不参与清理。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已停止；下次启动可继续下载。')
        raise SystemExit(130)
    except Exception as error:
        print(f'未能完成：{error}', file=sys.stderr)
        raise SystemExit(1)
