"""Package a self-hosted native backend using staged desktop resources.

Models remain in the single shared download catalog. No Python environment,
training sources or weights are included in the server archive.
"""
import argparse
import json
from pathlib import Path
import platform
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def build(resources, output):
    native = resources / 'native'
    executable = native / ('guitarocr-backend.exe' if platform.system() == 'Windows' else 'guitarocr-backend')
    if not executable.is_file() or not (native / 'manifest.json').is_file():
        raise ValueError('Stage complete native resources first: npm run resources -- --manifest PATH --require-inference')
    manifest = json.loads((native / 'manifest.json').read_text())
    if not manifest.get('inferenceInventoryComplete'):
        raise ValueError('Server distribution requires complete native inference resources')
    output.mkdir(parents=True, exist_ok=True)
    name = f'GuitarOCR-0.1.0-server-{platform.system().lower()}-{platform.machine().lower()}'
    paths = [p for folder in ('native', 'ui', 'licenses') for p in (resources / folder).rglob('*') if p.is_file()]
    if any(p.suffix in {'.py', '.pyc', '.whl', '.gguf', '.safetensors'} for p in paths):
        raise ValueError('Native distribution contains unexpected Python or model files')
    if platform.system() == 'Windows':
        destination = output / (name + '.zip')
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
            for p in paths:
                archive.write(p, Path(name) / p.relative_to(resources))
    else:
        destination = output / (name + '.tar.gz')
        with tarfile.open(destination, 'w:gz') as archive:
            for p in paths:
                archive.add(p, Path(name) / p.relative_to(resources), recursive=False)
    print(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, default=ROOT / 'desktop/src-tauri/resources')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/releases')
    args = parser.parse_args()
    build(args.resources, args.output)
