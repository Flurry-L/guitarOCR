"""Select release files from the training and distribution manifests."""

import json
from pathlib import Path


def model_entries(manifest):
    base = manifest.get('base_model')
    return [*manifest['models'], *([base] if base and base.get('required_for_inference', True) else [])]


def model_files(manifest):
    return {f"{model['path']}/{item['name']}": item
            for model in model_entries(manifest) for item in model['files']}


def release_model_assets(manifest):
    """Keep each original weight shard below the release hosting size limit."""
    return {name: name.replace('/', '--') for name in model_files(manifest)
            if Path(name).suffix == '.safetensors' and '/state_reader/' not in name}



def release_metadata(root):
    import subprocess
    import tomllib

    version = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    manifest = json.loads((root / 'weights/manifest.json').read_text(encoding='utf-8'))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return dict(version=version, repository='Flurry-L/guitarOCR', commit=commit,
                model_release=json.loads((root / 'weights/distribution.json').read_text())['release'],
                model_assets=release_model_assets(manifest),
                bundled_models=False)
