"""Package an already built and tested pinned runtime, including its shared libraries.

Maintainer tool only: no compilation or system libraries are installed on users'
computers. Supply a minimal staging directory with llama-server, runtime shared
libraries, upstream LICENSE and any dependency notices. Publish the returned ZIP,
then review/copy its entry into scripts/llamacpp-runtime.json after native testing.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.llamacpp_runtime import COMMIT, probe, validate_entry  # noqa: E402


def package(source, staging, output, key, url):
    if not re.fullmatch(r'(linux|windows|macos)-(x64|arm64)-(cpu|cuda|metal)', key):
        raise ValueError('Invalid runtime target')
    if output.resolve().is_relative_to(staging.resolve()):
        raise ValueError('Output must be outside the staging directory')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != COMMIT:
        raise ValueError(f'Build source must be pinned to {COMMIT}')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip():
        raise ValueError('Build source has tracked modifications; use the exact pinned revision')
    if (staging / 'LICENSE').read_bytes() != (source / 'LICENSE').read_bytes():
        raise ValueError('Include the exact upstream MIT LICENSE')
    executable = 'llama-server.exe' if key.startswith('windows-') else 'llama-server'
    probe(staging / executable)
    files = []
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'llamacpp-{COMMIT[:12]}-{key}.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
        for path in sorted(staging.rglob('*')):
            if path.is_symlink():
                raise ValueError('Stage actual library files instead of symbolic links')
            if not path.is_file():
                continue
            name = path.relative_to(staging).as_posix()
            contents = path.read_bytes()
            files.append(dict(name=name, bytes=len(contents), sha256=sha256(contents).hexdigest()))
            bundle.write(path, name)
    entry = dict(url=url, bytes=archive.stat().st_size, sha256=sha256(archive.read_bytes()).hexdigest(),
                 executable=executable, files=files)
    validate_entry(entry)
    manifest = output / f'{key}.json'
    manifest.write_text(json.dumps({key: entry}, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--staging', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    print(package(args.source, args.staging, args.output, args.target, args.url))
