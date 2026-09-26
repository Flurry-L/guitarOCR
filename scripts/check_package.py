"""Verify that the built wheel includes the runnable stages and static assets."""

import argparse
from pathlib import Path
from zipfile import ZipFile


def check(path):
    if path.is_dir():
        wheels = list(path.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one wheel in the build directory")
        path = wheels[0]
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {
            "webapp/static/index.html",
            "webapp/static/app.js",
            "webapp/static/api.js",
            "webapp/static/boxes.js",
            "webapp/static/pages.js",
            "webapp/static/measure-editor.js",
            "webapp/static/dom.js",
            "webapp/static/state.js",
            "webapp/static/style.css",
            "measure_ocr/configs/train.yaml",
            "measure_ocr/configs/release_gate.json",
            "datagen/native-source/build.ps1",
            "datagen/native-source/build_linux.py",
            "datagen/native-source/dllmain.cpp",
            "datagen/native-source/score_dump.cpp",
            "datagen/native-source/toolchain/GPCore.def",
        }
        missing = required - names
        if missing:
            raise ValueError(f"Wheel is missing: {sorted(missing)}")
    print(f"Wheel assets OK: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    check(parser.parse_args().wheel)
