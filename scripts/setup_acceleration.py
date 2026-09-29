"""Install the managed vLLM runtime without adding a second Torch environment."""

import argparse
import os
import shutil

from scripts.launcher import install, installation_lock
from shared.paths import PROJECT_ROOT


def main():
    uv = os.environ.get('GUITAROCR_UV') or shutil.which('uv')
    if not uv:
        raise RuntimeError('Run bash install.sh --engine vllm --device cuda')
    tools = PROJECT_ROOT / 'tools'
    with installation_lock(tools):
        install(argparse.Namespace(engine='vllm', device='cuda', llama_server=None), uv, tools)


if __name__ == '__main__':
    main()
