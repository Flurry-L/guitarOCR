"""Managed desktop runtime. Installs only the environment selected by the user."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import launcher  # noqa: E402
from shared.defaults import MODEL, environment_python  # noqa: E402
from scripts.progress import progress  # noqa: E402


def install_edit(uv, tools):
    state_path = tools / 'install-state.json'
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if launcher.installation_current(state, tools):
        return Path(state['python'])
    python = environment_python(tools / 'runtimes/edit')
    fingerprint = str(launcher.runtime_manifest()['generation'])
    stamp = tools / 'desktop-edit.json'
    if python.is_file() and stamp.exists() and stamp.read_text() == fingerprint:
        return python
    progress('editor', '正在安装校对与导出依赖')
    launcher.run_uv(uv, ['venv', '--python', '3.11', '--allow-existing', python.parent.parent])
    requirements = launcher.run_uv(uv, ['export', '--frozen', '--extra', 'webui', '--no-dev',
                                       '--no-hashes', '--no-emit-project', '--format', 'requirements-txt'], capture=True)
    path = tools / 'desktop-edit-requirements.txt'
    path.write_text(requirements, encoding='utf-8')
    launcher.run_uv(uv, ['pip', 'install', '--python', python, '-r', path])
    stamp.write_text(fingerprint)
    return python


def serve(args):
    import uvicorn
    from webapp.app import create_app
    from pipeline.workspace import Workspace

    state = json.loads((ROOT / 'tools/install-state.json').read_text()) if args.mode != 'edit' else {}
    workflow = Workspace(args.output, device=state.get('device', 'cpu'), model=Path(state.get('model', MODEL)),
                        layout_python=Path(state['layout_python']) if state else None)
    app = create_app(workflow, inference_enabled=args.mode != 'edit')
    # Bind once and pass the socket to Uvicorn; no find-free-port race.
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))

    def ready():
        while not server.started and not server.should_exit:
            time.sleep(.05)
        if server.started:
            print(f'GUITAROCR_READY http://127.0.0.1:{port}', flush=True)
    threading.Thread(target=ready, daemon=True).start()
    server.run(sockets=[sock])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('gpu', 'cpu', 'edit'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.serve:
        serve(args)
        return
    tools = ROOT / 'tools'
    uv = os.environ['GUITAROCR_UV']
    with launcher.installation_lock(tools):
        env = os.environ.copy()
        if args.mode != 'edit':
            device = 'cuda' if args.mode == 'gpu' else 'cpu'
            if device == 'cuda' and launcher.choose_device('auto') != 'cuda':
                raise ValueError('本机 GPU 识别需要 NVIDIA 显卡及 580 或更新驱动。')
            options = argparse.Namespace(device=device, engine='auto', llama_server=None)
            profile = launcher.resolve_profile(options)[2]
            path = tools / 'install-state.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            if not launcher.installation_current(state, tools, profile):
                state = launcher.install(options, uv, tools)
            python = Path(state['python'])
            from scripts.distribution import environment
            env = environment(state)
        else:
            python = install_edit(uv, tools)
    progress('check', '正在启动工作台')
    # The Tauri parent owns the process group / Windows Job, including this child.
    with launcher.installation_lock(tools / 'in-use'):
        result = subprocess.run([str(python), str(Path(__file__).resolve()), '--serve',
                             '--mode', args.mode, '--output', str(args.output)], env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'启动失败：{exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
