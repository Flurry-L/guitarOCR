"""Build a novice-friendly source ZIP with actual model weights and checksums."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tomllib
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.environment import verify_files  # noqa: E402

DIRECTORIES = {
    "datagen",
    "layout",
    "document_info",
    "measure_ocr",
    "gp5_export",
    "pipeline",
    "shared",
    "webapp",
    "scripts",
    "docs",
    "examples",
    "tests",
    "weights",
    ".github",
}
ROOT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
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


def build(output):
    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    errors = [
        error
        for model in manifest["models"]
        for error in verify_files(ROOT / model["path"], model["files"])
    ]
    if errors:
        raise ValueError("Cannot package incomplete weights: " + "; ".join(errors))
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    prefix = f"GuitarOCR-{version}"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{prefix}.zip"
    root_files = {ROOT / name for name in ROOT_FILES}
    root_files.update(ROOT.glob("LICENSE*"))
    root_files.update(ROOT.glob("NOTICE*"))
    files = [path for path in root_files if path.is_file() and not path.is_symlink()]
    for directory in sorted(DIRECTORIES):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in EXCLUDE for part in path.relative_to(ROOT).parts):
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
    checksums = []
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            contents = path.read_bytes()
            if path.suffix == ".bat":
                contents = contents.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            archive.writestr(f"{prefix}/{relative}", contents)
            checksums.append(f"{sha256(contents).hexdigest()}  {relative}")
        archive.writestr(f"{prefix}/SHA256SUMS", "\n".join(checksums) + "\n")
    digest = sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    print(f"{destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/releases")
    build(parser.parse_args().output)
