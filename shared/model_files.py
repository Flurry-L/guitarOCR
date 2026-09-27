"""Verify model artifacts locally without importing inference dependencies."""

from hashlib import sha256
from pathlib import Path


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
            if hashes:
                handle.seek(0)
                digest = sha256()
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != item["sha256"]:
                    errors.append(f"SHA-256 不符：{path}")
    return errors
