"""Build and restore the exact inference files listed in weights/manifest.json."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

from shared.model_files import verify_files
from scripts.downloads import acquire_base_model
from scripts.progress import progress


def model_entries(manifest):
    return [*manifest['models'], manifest['base_model']]


def model_files(manifest):
    return {f"{model['path']}/{item['name']}": item
            for model in model_entries(manifest) for item in model['files']}


def build_bundle(root, destination, *, fetch_models=False):
    manifest_bytes = (root / 'weights/manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = root / 'tools/model-bundles' / f'{sha256(manifest_bytes).hexdigest()}.tar.xz'
    # Only a completed build is renamed into this cache. The manifest identifies
    # its contents, so subsequent platform packages can reuse it directly.
    if cache.is_file():
        shutil.copyfile(cache, destination)
        return destination
    # The downloader verifies the base itself. Check it here only when the
    # caller supplied local files instead of using the downloader.
    if fetch_models:
        acquire_base_model(manifest['base_model'], root)
    entries = manifest['models'] if fetch_models else model_entries(manifest)
    errors = [error for model in entries
              for error in verify_files(root / model['path'], model['files'])]
    if errors:
        raise ValueError('Cannot package incomplete models: ' + '; '.join(errors))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.part')
    try:
        with tarfile.open(temporary, 'w:xz', preset=6) as archive:
            for name in model_files(manifest):
                print(f'Packaging {name}', flush=True)
                # Dereference local model caches; the package must be standalone.
                path = (root / name).resolve()
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                with path.open('rb') as source:
                    archive.addfile(info, source)
        temporary.replace(cache)
        shutil.copyfile(cache, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def restore_bundle(archive_path, root, manifest):
    """Install a bundled model version once; later launches reuse its files."""
    files = model_files(manifest)
    version = sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    stamp = root / 'tools/model-bundle.version'
    current = stamp.is_file() and stamp.read_text(encoding='utf-8') == version
    missing = {name for name in files
               if not current or not (root / name).is_file()}
    if not missing:
        progress('models', '包内模型已就绪', completed=1, total=1)
        return
    total = sum(item['bytes'] for item in files.values())
    completed = 0
    progress('models', '正在展开包内模型', completed=0, total=total)
    seen = set()
    with tarfile.open(archive_path, 'r|xz') as archive:
        for member in archive:
            name = member.name
            if name not in files or name in seen or not member.isfile() or member.size != files[name]['bytes']:
                raise ValueError('安装包模型内容不完整，请重新下载安装包。')
            seen.add(name)
            if name not in missing:
                completed += member.size
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.part')
            try:
                with archive.extractfile(member) as source, temporary.open('wb') as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                        completed += len(block)
                        progress('models', '正在展开包内模型', completed=completed, total=total,
                                 detail=Path(name).name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    if seen != set(files):
        raise ValueError('安装包缺少模型文件，请重新下载安装包。')
    stamp.parent.mkdir(parents=True, exist_ok=True)
    temporary_stamp = stamp.with_suffix('.part')
    temporary_stamp.write_text(version, encoding='utf-8')
    temporary_stamp.replace(stamp)
    progress('models', '包内模型已就绪', completed=total, total=total)
