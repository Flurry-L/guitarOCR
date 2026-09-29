"""Prepare lightweight Tauri resources; local recognition downloads its models."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
import os
import tomllib
import shutil
import sys
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.model_bundle import model_files, release_metadata, release_model_assets  # noqa: E402

DEST = ROOT / 'desktop/src-tauri/resources'
UV_VERSION = '0.12.17'
DIRECTORIES = ('layout', 'document_info', 'measure_ocr', 'gp5_export', 'pipeline', 'shared', 'webapp')
SUFFIXES = {'.py', '.json', '.yaml', '.html', '.css', '.js', '.mjs', '.woff2'}


def backend_files(root):
    """Select the Python workbench and its installer, excluding models and user data."""
    paths = []
    for directory in DIRECTORIES:
        paths.extend(p for p in (root / directory).rglob('*') if p.is_file()
                     and (p.suffix in SUFFIXES or p.name.endswith('LICENSE'))
                     and '__pycache__' not in p.parts and not p.is_symlink())
    paths.extend(root / p for p in ('pyproject.toml', 'uv.lock', 'README.md', 'THIRD_PARTY_NOTICES.md',
                                   'weights/manifest.json', 'webapp/static/vendor/README.md', 'scripts/launcher.py', 'scripts/downloads.py',
                                   'scripts/desktop_runtime.py', 'scripts/bootstrap-uv.toml',
                                   'scripts/model_bundle.py', 'scripts/progress.py', 'scripts/setup_acceleration.py'))
    paths.extend(p for p in (root / 'weights').rglob('*.json') if p.is_file() and not p.is_symlink())
    manifest = json.loads((root / 'weights/manifest.json').read_text(encoding='utf-8'))
    assets = release_model_assets(manifest)
    paths.extend(root / name for name in model_files(manifest) if name not in assets)
    paths.extend(p for p in (root / 'weights/licenses').rglob('*') if p.is_file())
    return sorted(set(paths))


def prepare():
    if DEST.exists():
        shutil.rmtree(DEST)
    # Cargo dependencies are fetched by the build workflow before this step.
    cargo_home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo'))
    registry = cargo_home / 'registry/src'
    licenses = DEST / 'licenses'
    licenses.mkdir(parents=True)
    packages = tomllib.loads((ROOT / 'desktop/src-tauri/Cargo.lock').read_text())['package']
    inventory = []
    for package in packages:
        if not package.get('source', '').startswith('registry+'):
            continue
        name = f"{package['name']}-{package['version']}"
        sources = list(registry.glob(f'*/{name}'))
        if not sources:
            raise ValueError(f'Missing {name}; run cargo fetch --locked before preparing resources')
        source = sources[0]
        metadata = tomllib.loads((source / 'Cargo.toml').read_text())['package']
        inventory.append({'package': name, 'license': metadata.get('license'),
                          'source': f"https://crates.io/crates/{package['name']}/{package['version']}"})
        for path in source.rglob('*'):
            if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE', 'COPYRIGHT')):
                dest = licenses / name / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    (licenses / 'components.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    backend = DEST / 'backend'
    backend.mkdir(parents=True)
    for p in backend_files(ROOT):
        dest = backend / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    (backend / 'release.json').write_text(json.dumps(release_metadata(ROOT)), encoding='utf-8')
    machine = platform.machine().lower()
    tag = {('Windows', 'amd64'): 'win_amd64', ('Windows', 'x86_64'): 'win_amd64',
           ('Darwin', 'arm64'): 'macosx_11_0_arm64', ('Darwin', 'x86_64'): 'macosx_10_12_x86_64',
           ('Linux', 'x86_64'): 'manylinux_2_17_x86_64.manylinux2014_x86_64'}[(platform.system(), machine)]
    with urllib.request.urlopen(f'https://pypi.org/pypi/uv/{UV_VERSION}/json', timeout=60) as response:
        metadata = json.load(response)
    item = next(p for p in metadata['urls'] if p['filename'].endswith(f'-{tag}.whl'))
    with urllib.request.urlopen(item['url'], timeout=120) as response:
        data = response.read()
    if sha256(data).hexdigest() != item['digests']['sha256']:
        raise ValueError('uv wheel checksum mismatch')
    uv_dir = DEST / 'uv'
    uv_dir.mkdir()
    with ZipFile(io.BytesIO(data)) as archive:
        binary = next(n for n in archive.namelist() if n.endswith(('/uv', '/uv.exe')))
        executable = uv_dir / Path(binary).name
        executable.write_bytes(archive.read(binary))
        executable.chmod(0o755)
        for name in archive.namelist():
            if '/licenses/' in name and not name.endswith('/'):
                (uv_dir / Path(name).name).write_bytes(archive.read(name))
    (uv_dir / 'source.json').write_text(json.dumps({'version': UV_VERSION, 'url': item['url'], 'sha256': item['digests']['sha256']}, indent=2), encoding='utf-8')
    print(f'Prepared {platform.system()} {machine}; uv and workbench included, models downloaded on demand.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
