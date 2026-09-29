"""Build a launcher ZIP and separately downloadable, original model shards."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.model_bundle import build_bundle, model_files, release_metadata  # noqa: E402
from shared.model_files import verify_files  # noqa: E402

DIRECTORIES = {
    "datagen",
    "layout",
    "document_info",
    "measure_ocr",
    "gp5_export",
    "pipeline",
    "shared",
    "webapp",
    "server",
    "scripts",
    "docs",
    "weights",
}
ROOT_FILES = {
    "README.md",
    "使用说明.txt",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
    ".gitattributes",
    ".gitignore",
    "install.bat",
    "start.bat",
    "install.sh",
    "start.sh",
}
EXCLUDE = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv", "runtime"}


def build(output, *, bundle_models=False):
    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    release = release_metadata(ROOT)
    files_by_path = model_files(manifest)
    errors = [error for name, item in files_by_path.items()
              for error in verify_files((ROOT / name).parent, [{**item, 'name': Path(name).name}])]
    if errors:
        raise ValueError('Cannot package incomplete models: ' + '; '.join(errors))
    bundle = build_bundle(ROOT, output / 'models.tar.xz') if bundle_models else None
    if bundle_models:
        release['bundled_models'] = True
    else:
        assets = output / 'models'
        assets.mkdir(exist_ok=True)
        for name, asset in release['model_assets'].items():
            source, target = ROOT / name, assets / asset
            if source.stat().st_size >= 2**31:
                raise ValueError(f'{name} exceeds the GitHub asset limit; shard the model first')
            target.unlink(missing_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    prefix = f"GuitarOCR-{version}"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{prefix}.zip"
    root_files = {ROOT / name for name in ROOT_FILES}
    root_files.update(ROOT.glob("LICENSE*"))
    root_files.update(ROOT.glob("NOTICE*"))
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    files = [path for path in root_files if path.is_file() and not path.is_symlink()]
    for directory in sorted(DIRECTORIES):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.relative_to(ROOT).as_posix() not in tracked:
                continue
            if any(part in EXCLUDE for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix in {".safetensors", ".pdiparams", ".pdparams", ".pt"}:
                continue
            if path.name == ".env" or (
                path.name.startswith(".env.") and path.name != ".env.example"
            ):
                continue
            if path.suffix.lower() in {
                ".pyc",
                ".pdb",
                ".lib",
                ".exp",
                ".log",
                ".pem",
                ".key",
            }:
                continue
            files.append(path)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        contents = (json.dumps(release, indent=2) + "\n").encode()
        archive.writestr(f"{prefix}/release.json", contents)
        for path in sorted(set(files)):
            relative = path.relative_to(ROOT).as_posix()
            contents = path.read_bytes()
            if path.suffix == ".bat" or path.name == "使用说明.txt":
                contents = contents.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            archive.writestr(f"{prefix}/{relative}", contents)
        # The model archive is already compressed; stream it without keeping
        # several gigabytes in memory or compressing it a second time.
        if bundle:
            entry = ZipInfo(f'{prefix}/models.tar.xz')
            entry.compress_type = ZIP_STORED
            with bundle.open('rb') as source, archive.open(entry, 'w', force_zip64=True) as dest:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
    print(f"{destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/releases")
    parser.add_argument('--bundle-models', action='store_true', help='Build a large offline-model ZIP instead of release model assets')
    args = parser.parse_args()
    build(args.output, bundle_models=args.bundle_models)
