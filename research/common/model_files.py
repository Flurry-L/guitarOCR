"""Verify model artifacts locally without importing inference dependencies."""

from hashlib import sha256
import json
from pathlib import Path


def checkpoint_files(folder: Path) -> list[Path]:
    """Include the index and every shard, while retaining single-file support."""
    index = folder / 'model.safetensors.index.json'
    if not index.is_file():
        return [folder / 'model.safetensors']
    names = sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
    if not names or any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
        raise ValueError(f'Invalid checkpoint index: {index}')
    return [index, *(folder / name for name in names)]


def remove_obsolete_checkpoints(folder: Path, files: list[dict]) -> None:
    """Avoid loading a stale single file after installing a sharded release."""
    expected = {folder / item['name'] for item in files}
    indexes = [path for path in expected if path.name == 'model.safetensors.index.json']
    if not indexes:
        return
    errors = verify_files(folder, files, hashes=False)
    if errors:
        raise ValueError('; '.join(errors))
    for index in indexes:
        previous = [index.parent / 'model.safetensors', *index.parent.glob('model-*-of-*.safetensors')]
        for path in previous:
            if path not in expected:
                path.unlink(missing_ok=True)


def verify_files(folder: Path, files: list[dict], hashes: bool = True) -> list[str]:
    errors = []
    for item in files:
        path = folder / item["name"]
        if not path.is_file():
            errors.append(f"缺少文件：{path}")
            continue
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                errors.append(f"尚未下载 Git LFS 权重：{path}")
                continue
            if path.stat().st_size != item["bytes"]:
                errors.append(f"文件大小不符，可能下载未完成：{path}")
                continue
            if hashes and item.get('sha256'):
                handle.seek(0)
                digest = sha256()
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != item["sha256"]:
                    errors.append(f"SHA-256 不符：{path}")
    return errors
