"""Download only the selected engine's models into a reusable model cache."""

import json
import os
from pathlib import Path
import shutil

from scripts.downloads import download_verified
from scripts.progress import progress
from shared.model_files import verify_files

ROOT = Path(__file__).resolve().parent.parent


def catalog():
    return json.loads((ROOT / 'weights/distribution.json').read_text(encoding='utf-8'))


def model_root():
    base = Path(os.environ.get('GUITAROCR_MODEL_CACHE', ROOT / 'tools/model-cache')).expanduser().absolute()
    return base / catalog()['generation']


def selected_manifest(engine):
    manifest = json.loads((ROOT / 'weights/manifest.json').read_text(encoding='utf-8'))
    models = []
    for entry in manifest['models']:
        if entry['stage'] != 'score_ocr':
            continue
        files = [item for item in entry['files'] if 'state_reader/' not in item['name']
                 and (engine != 'llamacpp' or not item['name'].endswith('.safetensors'))]
        models.append({**entry, 'files': files})
    return {**manifest, 'models': models}


def acquire(engine):
    from scripts.launcher import acquire_weights

    root = model_root()
    acquire_weights(selected_manifest(engine), root=root)
    release = catalog()
    selected = ['auxiliary', *(['llamacpp'] if engine == 'llamacpp' else [])]
    items = [item for group in selected for item in release['files'][group]]
    total = sum(item['bytes'] for item in items)
    completed = 0
    for item in items:
        path = root / item['path']
        missing = verify_files(path.parent, [{**item, 'name': path.name}])
        if missing:
            local = ROOT / ('weights/' + item['path'] if item['path'].startswith('auxiliary/') else item['path'])
            missing = not reuse(local, path, item)
        if missing:
            url = f"https://github.com/{release['repository']}/releases/download/{release['release']}/{item['asset']}"
            download_verified(url, path, item, on_progress=lambda received, _: progress(
                '正在下载所选模型', completed=completed + received, total=total, detail=path.name))
        completed += item['bytes']
    progress('所选模型已就绪', completed=total, total=total)
    return root


def environment(state):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', GUITAROCR_BACKEND=state['engine'],
               GUITAROCR_WEIGHTS_ROOT=str(Path(state['models']) / 'weights'),
               GUITAROCR_AUX_MODELS=str(Path(state['models']) / 'auxiliary'),
               GUITAROCR_MODEL_CACHE=str(Path(state['models']).parent))
    if state['engine'] == 'vllm':
        env['GUITAROCR_VLLM_PYTHON'] = state['python']
    if state.get('llama_server'):
        env['GUITAROCR_LLAMA_SERVER'] = state['llama_server']
    env['GUITAROCR_LLAMA_GPU_LAYERS'] = '99' if state['device'] in {'cuda', 'metal'} else '0'
    return env


def check_models(engine, root):
    errors = []
    for entry in selected_manifest(engine)['models']:
        errors += verify_files(Path(root) / entry['path'], entry['files'])
    release = catalog()
    for group in ('auxiliary', *(['llamacpp'] if engine == 'llamacpp' else [])):
        for item in release['files'][group]:
            path = Path(root) / item['path']
            errors += verify_files(path.parent, [{**item, 'name': path.name}])
    return errors


def reuse(source, destination, item):
    if verify_files(source.parent, [{**item, 'name': source.name}]):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def download_size(engine):
    """Bytes missing from the cache, excluding reusable files in this checkout."""
    root = model_root()
    items = [(entry['path'] + '/' + item['name'], item)
             for entry in selected_manifest(engine)['models'] for item in entry['files']]
    for group in ('auxiliary', *(['llamacpp'] if engine == 'llamacpp' else [])):
        items.extend((item['path'], item) for item in catalog()['files'][group])
    total = 0
    for name, item in items:
        cached = root / name
        local = ROOT / ('weights/' + name if name.startswith('auxiliary/') else name)
        expected = {**item, 'name': cached.name}
        if verify_files(cached.parent, [expected]) and verify_files(local.parent, [expected]):
            total += item['bytes']
    return total
