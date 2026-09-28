"""Install the isolated CUDA inference runtime used by the published models."""

import os
import shutil
import subprocess

from shared.defaults import environment_python
from shared.paths import PROJECT_ROOT


def main():
    uv = os.environ.get('GUITAROCR_UV') or shutil.which('uv')
    if not uv:
        candidate = PROJECT_ROOT / 'tools/uv/uv'
        uv = str(candidate) if candidate.is_file() else None
    if not uv:
        raise RuntimeError('Install uv or run scripts/bootstrap.sh first')
    folder = PROJECT_ROOT / 'tools/vllm-venv'
    python = environment_python(folder)
    if not python.is_file():
        subprocess.run([uv, 'venv', '--python', '3.11', str(folder)], check=True)
    subprocess.run([uv, 'pip', 'install', '--python', str(python), 'vllm==0.30.0'], check=True)
    print(f'CUDA inference runtime ready: {folder}')


if __name__ == '__main__':
    main()
