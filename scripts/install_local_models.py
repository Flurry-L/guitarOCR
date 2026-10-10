"""Install the current local model inventory for the native application."""

import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def install(cache):
    catalog = json.loads((ROOT / "weights/distribution.json").read_text())
    if catalog.get("availability") != "local_unpublished":
        raise ValueError("This command installs the local development model inventory")
    destination = Path(cache).resolve() / catalog["generation"]
    files = []
    for group in catalog["files"].values():
        for item in group:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid inventory path: {relative}")
            source = ROOT / relative if relative.parts[0] == "weights" else ROOT / "weights" / relative
            with source.open("rb") as stream:
                prefix = stream.read(80)
            if not prefix or prefix.startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Download the actual model file first: {source}")
            files.append((source, destination / relative))
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and os.path.samefile(source, target):
            continue
        temporary = target.with_name(target.name + ".installing")
        temporary.unlink(missing_ok=True)
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {"directory": str(destination), "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tools/model-cache")
    print(json.dumps(install(parser.parse_args().cache), indent=2))
