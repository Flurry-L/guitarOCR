"""Cross-compile the GP8 exporter using clang-cl, xwin and Qt's MSVC SDK."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True, help="xwin splat output")
    parser.add_argument("--qt", type=Path, required=True, help="Qt 5 MSVC x64 SDK")
    parser.add_argument("--clang", default="clang-cl-18")
    parser.add_argument("--linker", required=True, type=Path)
    parser.add_argument("--lib", default="/usr/lib/llvm-18/bin/llvm-lib")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    sdk, qt, output = args.sdk.resolve(), args.qt.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gp8-build-") as directory:
        build = Path(directory)
        libraries = []
        for name in ("GPCore", "AMUtils", "AMPainting", "AMProfProxy"):
            library = build / f"{name}.lib"
            subprocess.run(
                [
                    args.lib,
                    "/machine:x64",
                    f"/def:{source / 'toolchain' / (name + '.def')}",
                    f"/out:{library}",
                ],
                check=True,
            )
            libraries.append(str(library))
        includes = [
            sdk / "crt/include",
            sdk / "sdk/include/ucrt",
            sdk / "sdk/include/shared",
            sdk / "sdk/include/um",
            sdk / "sdk/include/winrt",
        ]
        objects = []
        for name in ("dllmain", "score_dump"):
            obj = build / f"{name}.obj"
            command = [
                args.clang,
                "--target=x86_64-pc-windows-msvc",
                "-fms-compatibility-version=19.40",
                "/nologo",
                "/O2",
                "/EHsc",
                "/std:c++17",
                "/MD",
                "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                "/c",
                str(source / f"{name}.cpp"),
                f"/Fo{obj}",
            ]
            for include in includes:
                command.extend(["/imsvc", str(include)])
            for include in (
                qt / "include",
                qt / "include/QtCore",
                qt / "include/QtGui",
                qt / "include/QtWidgets",
                qt / "mkspecs/win32-msvc",
                source / "toolchain",
            ):
                command.append(f"/I{include}")
            subprocess.run(command, check=True)
            objects.append(str(obj))
        for name, preload in (
            ("gpomr_native_export", False),
            ("gpomr_amprof_preload", True),
        ):
            command = [
                str(args.linker.absolute()),
                "/dll",
                "/machine:x64",
                f"/out:{build / (name + '.dll')}",
                *objects,
            ]
            for path in (
                sdk / "crt/lib/x86_64",
                sdk / "sdk/lib/um/x86_64",
                sdk / "sdk/lib/ucrt/x86_64",
                qt / "lib",
            ):
                command.append(f"/libpath:{path}")
            command.extend(
                [
                    "Qt5Core.lib",
                    "Qt5Gui.lib",
                    "Qt5Widgets.lib",
                    "kernel32.lib",
                    "user32.lib",
                    *libraries[:3],
                ]
            )
            if preload:
                command.extend(
                    [f"/def:{source / 'toolchain/AMProfProxy.def'}", libraries[3]]
                )
            subprocess.run(command, check=True)
            shutil.copy2(build / f"{name}.dll", output / f"{name}.dll")
    record = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "compiler": subprocess.check_output(
            [args.clang, "--version"], text=True
        ).splitlines()[0],
        "qt_root": str(qt),
        "sdk_root": str(sdk),
        "source_sha256": {
            str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "outputs": [
            {"name": p.name, "sha256": sha256(p.read_bytes()).hexdigest()}
            for p in sorted(output.glob("*.dll"))
        ],
        "runtime_validation": "not_run",
    }
    (output / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
