"""Isolated parallel Guitar Pro workers for Linux/Wine batch callers."""
from __future__ import annotations
import concurrent.futures
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

_PRELOAD_DLL = Path(__file__).parent / "bin/gpomr_amprof_preload.dll"

def run_workers(*, runtime: Path, prefix_template: Path, wine_python: Path,
                worker_root: Path, workers: int, script: Path, arguments,
                display_base: int = 100) -> None:
    """Run a Windows Python worker in isolated GP/Wine sessions.

    arguments(index, worker_directory) returns Windows arguments for the caller's
    worker script. Dataset identities and output formats stay with that caller.
    """
    runtime, prefix_template, wine_python, worker_root = (
        Path(p).expanduser().resolve() for p in (runtime, prefix_template, wine_python, worker_root)
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    if not (prefix_template / "system.reg").is_file():
        raise FileNotFoundError(prefix_template / "system.reg")
    if not wine_python.is_file():
        raise FileNotFoundError(wine_python)
    for path in (runtime, prefix_template):
        _require_separate(worker_root, path)
    worker_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    pty_launcher = _required_executable("script", environment)
    xvfb_command = (_required_executable("Xvfb", environment),)
    worker_dirs = tuple(
        worker_root / f"worker-{index:03d}" for index in range(workers)
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        tuple(
            executor.map(
                _prepare_worker,
                ((directory, runtime, prefix_template) for directory in worker_dirs),
            )
        )

    processes: list[subprocess.Popen[bytes]] = []
    displays: list[tuple[subprocess.Popen[bytes], BinaryIO, int]] = []
    wine_sessions: list[tuple[dict[str, str], tuple[str, ...]]] = []
    claimed_displays: set[int] = set()
    try:
        for index, worker_dir in enumerate(worker_dirs):
            worker_environment = dict(environment)
            display = _available_display(display_base, claimed_displays)
            claimed_displays.add(display)
            worker_environment.update(
                {
                    "DISPLAY": f":{display}",
                    "GPOMR_NATIVE_PRELOADED": "1",
                    "WINEPREFIX": str(worker_dir / "wine-prefix"),
                    "WINEDEBUG": "-all",
                    "LIBGL_ALWAYS_SOFTWARE": "1",
                }
            )
            wine = (_required_executable("wine", worker_environment),)
            wineserver = (_required_executable("wineserver", worker_environment),)
            wine_sessions.append((worker_environment, wineserver))
            subprocess.run(
                [*wineserver, "-k"],
                env=worker_environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            log = (worker_dir / "export.log").open("wb")
            xvfb = subprocess.Popen(
                [
                    *xvfb_command,
                    worker_environment["DISPLAY"],
                    "-screen",
                    "0",
                    "1280x1024x24",
                    "-nolisten",
                    "tcp",
                ],
                env=worker_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            displays.append((xvfb, log, display))
            try:
                xvfb.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise RuntimeError(f"Xvfb worker {index} exited during startup")

            worker_arguments = list(arguments(index, worker_dir))
            command = [
                *wine,
                _wine_path(wine_python),
                _wine_path(script),
                *worker_arguments,
            ]
            process = subprocess.Popen(
                [pty_launcher, "-q", "-e", "-c", shlex.join(command), "/dev/null"],
                env=worker_environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)

        exit_codes = [process.wait() for process in processes]
        failed = [str(index) for index, code in enumerate(exit_codes) if code != 0]
        if failed:
            raise RuntimeError("Wine workers failed: " + ", ".join(failed))
    finally:
        for worker_environment, wineserver in wine_sessions:
            subprocess.run(
                [*wineserver, "-k"],
                env=worker_environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for xvfb, log, display in displays:
            _terminate_process_group(xvfb)
            _remove_display_files(display)
            log.close()


def _prepare_worker(arguments: tuple[Path, Path, Path]) -> None:
    worker, runtime, prefix = arguments
    worker.mkdir(parents=True, exist_ok=True)
    _copy_tree_once(runtime, worker / "runtime", "GuitarPro.exe")
    _install_preload(worker / "runtime")
    _copy_tree_once(prefix, worker / "wine-prefix", "system.reg")


def _copy_tree_once(source: Path, destination: Path, required: str) -> None:
    if (destination / required).is_file():
        return
    if destination.exists():
        shutil.rmtree(destination)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        subprocess.run(
            ["cp", "-a", "--reflink=auto", f"{source}{os.sep}.", str(temporary)],
            check=True,
        )
        if not (temporary / required).is_file():
            raise FileNotFoundError(temporary / required)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _install_preload(runtime: Path) -> None:
    _require_pe(_PRELOAD_DLL)
    official = runtime / "AMProf.dll"
    original = runtime / "AMProf_original.dll"
    if not original.is_file():
        if not official.is_file():
            raise FileNotFoundError(official)
        official.replace(original)
    shutil.copy2(_PRELOAD_DLL, official)


def _require_pe(path: Path) -> None:
    with path.open("rb") as dll:
        header = dll.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError(f"invalid preload DLL: {path}")
        dll.seek(int.from_bytes(header[60:64], "little"))
        if dll.read(4) != b"PE\0\0":
            raise ValueError(f"invalid preload DLL: {path}")


def _required_executable(name: str, environment: dict[str, str]) -> str:
    value = shutil.which(name, path=environment.get("PATH"))
    if value is None:
        raise FileNotFoundError(f"required executable is missing: {name}")
    return value


def _wine_path(path: Path) -> str:
    source = path.resolve()
    if not source.is_absolute():
        raise ValueError(f"Wine path must be absolute: {path}")
    return "Z:" + str(source).replace("/", "\\")


def _available_display(minimum: int, claimed: set[int]) -> int:
    for number in range(minimum, minimum + 10_000):
        if number in claimed:
            continue
        if (Path("/tmp/.X11-unix") / f"X{number}").exists():
            continue
        if Path(f"/tmp/.X{number}-lock").exists():
            continue
        return number
    raise RuntimeError("no free X display")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _remove_display_files(display: int) -> None:
    (Path("/tmp/.X11-unix") / f"X{display}").unlink(missing_ok=True)
    Path(f"/tmp/.X{display}-lock").unlink(missing_ok=True)


def _require_separate(first: Path, second: Path) -> None:
    if first == second or first.is_relative_to(second) or second.is_relative_to(first):
        raise ValueError(f"paths must be separate: {first}, {second}")


wine_path = _wine_path
