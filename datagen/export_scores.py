from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


if __package__:
    from .native.guitarpro_pdf import (GuitarProPdfExporter, PdfExportJob, SourceTrack, load_render_layout)
else:
    from native.guitarpro_pdf import (GuitarProPdfExporter, PdfExportJob, SourceTrack, load_render_layout)


_PRELOAD_DLL = (
    Path(__file__).resolve().parent
    / "native-bin"
    / "gpomr_amprof_preload.dll"
)
_MANIFEST_FIELDS = frozenset(
    {"document_id", "source_family_id", "split", "instrument_kind", "source"}
)
_SOURCE_SUFFIXES = frozenset({".gp", ".gp3", ".gp4", ".gp5", ".gtp"})
_SAFE_DOCUMENT = re.compile(r"[a-z0-9][a-z0-9.-]*")
_SAFE_TRACK = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    document_id: str
    source_family_id: str
    split: str
    instrument_kind: str
    source: str
    display_mode: str = "tab"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a family-safe Guitar Pro corpus with isolated native "
            "renderer workers."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--ready-timeout", type=_positive_int, default=120)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--wine-prefix-template", type=Path)
    parser.add_argument("--wine-python", type=Path)
    parser.add_argument("--display-base", type=_positive_int, default=90)
    parser.add_argument("--worker-index", type=_nonnegative_int, help=argparse.SUPPRESS)
    parser.add_argument("--failure-log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--temp-root", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker_index is not None:
        return _run_worker(args)
    return _launch(args)


def _launch(args: argparse.Namespace) -> int:
    manifest = args.manifest.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    entries = _load_manifest(manifest)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not runtime.is_dir():
        raise FileNotFoundError(runtime)
    for entry in entries:
        _manifest_source(source_root, entry)
    _require_separate(source_root, output_dir)
    _require_separate(runtime, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_text(
        output_dir / "document-manifest.jsonl",
        _manifest_jsonl(entries),
    )
    if os.name == "nt":
        if args.workers != 1:
            raise ValueError("parallel export requires isolated Wine prefixes")
        worker_values = vars(args).copy()
        worker_values.update(
            worker_index=0,
            failure_log=output_dir / ".worker-failures" / "worker-000.jsonl",
            temp_root=output_dir / ".in-progress" / "worker-000",
        )
        worker_args = argparse.Namespace(**worker_values)
        _run_worker(worker_args)
        _publish_failures(
            output_dir,
            (Path(worker_args.failure_log),),
        )
        return 0

    _launch_wine_workers(
        args,
        manifest=manifest,
        source_root=source_root,
        runtime=runtime,
        output_dir=output_dir,
    )
    return 0


def _launch_wine_workers(
    args: argparse.Namespace,
    *,
    manifest: Path,
    source_root: Path,
    runtime: Path,
    output_dir: Path,
) -> None:
    if args.wine_prefix_template is None or args.wine_python is None:
        raise ValueError(
            "Linux export requires --wine-prefix-template and --wine-python"
        )
    prefix_template = args.wine_prefix_template.expanduser().resolve()
    wine_python = args.wine_python.expanduser().resolve()
    if not (prefix_template / "system.reg").is_file():
        raise FileNotFoundError(prefix_template / "system.reg")
    if not wine_python.is_file():
        raise FileNotFoundError(wine_python)
    worker_root = (
        args.worker_root.expanduser().resolve()
        if args.worker_root is not None
        else output_dir.parent / f".{output_dir.name}-workers"
    )
    for path in (source_root, runtime, output_dir, prefix_template):
        _require_separate(worker_root, path)
    worker_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    pty_launcher = _required_executable("script", environment)
    xvfb_command = (_required_executable("Xvfb", environment),)
    worker_dirs = tuple(
        worker_root / f"worker-{index:03d}" for index in range(args.workers)
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        tuple(
            executor.map(
                _prepare_worker,
                ((directory, runtime, prefix_template) for directory in worker_dirs),
            )
        )

    processes: list[subprocess.Popen[bytes]] = []
    displays: list[tuple[subprocess.Popen[bytes], BinaryIO, int]] = []
    wine_sessions: list[tuple[dict[str, str], tuple[str, ...]]] = []
    failure_logs: list[Path] = []
    claimed_displays: set[int] = set()
    try:
        for index, worker_dir in enumerate(worker_dirs):
            worker_environment = dict(environment)
            display = _available_display(args.display_base, claimed_displays)
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

            failure_log = worker_dir / "failures.jsonl"
            failure_logs.append(failure_log)
            arguments = [
                "--manifest",
                _wine_path(manifest),
                "--source-root",
                _wine_path(source_root),
                "--runtime",
                _wine_path(worker_dir / "runtime"),
                "--output-dir",
                _wine_path(output_dir),
                "--workers",
                str(args.workers),
                "--ready-timeout",
                str(args.ready_timeout),
                "--worker-index",
                str(index),
                "--failure-log",
                _wine_path(failure_log),
                "--temp-root",
                _wine_path(
                    output_dir / ".in-progress" / f"worker-{index:03d}",
                ),
            ]
            command = [
                *wine,
                _wine_path(wine_python),
                _wine_path(Path(__file__)),
                *arguments,
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
        _publish_failures(output_dir, tuple(failure_logs))
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


def _run_worker(args: argparse.Namespace) -> int:
    if args.worker_index is None or args.worker_index >= args.workers:
        raise ValueError("worker index must be smaller than worker count")
    if args.failure_log is None or args.temp_root is None:
        raise ValueError("worker paths are required")
    entries = _load_manifest(args.manifest)
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    temp_root = args.temp_root.expanduser().resolve()
    failure_log = args.failure_log.expanduser().resolve()
    if not temp_root.is_relative_to(output_dir / ".in-progress"):
        raise ValueError("worker temporary data must remain under output-dir")
    (output_dir / "documents").mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    exporter: GuitarProPdfExporter | None = None

    def open_exporter() -> GuitarProPdfExporter:
        nonlocal exporter
        if exporter is None:
            exporter = GuitarProPdfExporter(
                args.runtime,
                ready_timeout=args.ready_timeout,
            )
            exporter.__enter__()
        return exporter

    def close_exporter() -> None:
        nonlocal exporter
        current, exporter = exporter, None
        if current is not None:
            current.__exit__(None, None, None)

    def release_or_restart() -> None:
        if exporter is None:
            return
        try:
            exporter.release_source()
        except BaseException:
            close_exporter()

    with failure_log.open("w", encoding="utf-8", newline="\n") as failures:
        try:
            for entry in entries[args.worker_index :: args.workers]:
                final = output_dir / "documents" / entry.document_id
                staging: Path | None = None
                try:
                    source = _manifest_source(source_root, entry)
                    if final.exists():
                        _validate_document_source(final, source)
                        _validate_document(final)
                        _validate_document_geometry(final)
                        _validate_document_mode(final, entry.display_mode)
                        continue
                    staging = Path(
                        tempfile.mkdtemp(
                            prefix=f"{entry.document_id}-",
                            dir=temp_root,
                        )
                    )
                    copied_source = staging / f"source{source.suffix.lower()}"
                    shutil.copy2(source, copied_source)
                    metadata = source.with_name(source.name + ".metadata.json")
                    if metadata.is_file():
                        shutil.copy2(metadata, copied_source.with_name(copied_source.name + ".metadata.json"))
                    current_exporter = open_exporter()
                    tracks = tuple(
                        track
                        for track in current_exporter.list_tracks(copied_source)
                        if track.instrument_kind == entry.instrument_kind
                    )
                    if not tracks:
                        raise ValueError(
                            f"source contains no {entry.instrument_kind} track"
                        )
                    _export_document(
                        current_exporter,
                        staging,
                        copied_source,
                        tracks,
                        entry.display_mode,
                    )
                    _validate_document(staging)
                    _validate_document_geometry(staging)
                    _validate_document_mode(staging, entry.display_mode)
                    staging.replace(final)
                except Exception as exception:
                    failures.write(
                        json.dumps(
                            {
                                "document_id": entry.document_id,
                                "source": entry.source,
                                "error_type": type(exception).__name__,
                                "error": str(exception),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    failures.flush()
                finally:
                    release_or_restart()
                    if staging is not None and staging.exists():
                        shutil.rmtree(staging)
        finally:
            close_exporter()
    return 0


def _export_document(
    exporter: GuitarProPdfExporter,
    destination: Path,
    source: Path,
    tracks: tuple[SourceTrack, ...],
    display_mode: str = "tab",
) -> None:
    for track in tracks:
        track_dir = destination / "tracks" / _track_name(track)
        track_dir.mkdir(parents=True)
        exporter.export(
            PdfExportJob(
                source=source,
                pdf=track_dir / "score.pdf",
                layout=track_dir / "layout.json",
                official_score=track_dir / "official-score.json",
                track_index=track.source_track_index,
                display_mode=display_mode,
            )
        )


def _validate_document_mode(document: Path, display_mode: str) -> None:
    for path in (document / "tracks").glob("*/layout.json"):
        layout = json.loads(path.read_text(encoding="utf-8"))
        # The original exporter produced only TAB and omitted this field.
        actual = layout.get("display_mode", "tab")
        if actual != display_mode:
            raise ValueError(
                f"Native export mode mismatch: {actual} != {display_mode}: {path}; "
                "use a new output directory when changing display mode"
            )


def _validate_document_source(document: Path, source: Path) -> None:
    """Never reuse a native render after its GP input or metadata has changed."""
    archived = document / f"source{source.suffix.lower()}"
    for current, saved in (
        (source, archived),
        (source.with_name(source.name + ".metadata.json"),
         archived.with_name(archived.name + ".metadata.json")),
    ):
        if current.is_file() != saved.is_file() or (
            current.is_file() and current.read_bytes() != saved.read_bytes()
        ):
            raise ValueError(
                f"Native export input changed: {current}; use a new output directory"
            )


def _validate_document(document: Path) -> None:
    sources = [
        path
        for path in document.iterdir()
        if path.is_file()
        and path.stem == "source"
        and path.suffix.casefold() in _SOURCE_SUFFIXES
    ]
    if len(sources) != 1:
        raise ValueError(f"document has no unique GP5 source: {document}")
    tracks_dir = document / "tracks"
    tracks = tuple(path for path in tracks_dir.iterdir() if path.is_dir())
    if not tracks:
        raise ValueError(f"document contains no tracks: {document}")
    for track in tracks:
        pdf = track / "score.pdf"
        layout = track / "layout.json"
        official = track / "official-score.json"
        for path in (pdf, layout, official):
            if not path.is_file():
                raise FileNotFoundError(path)
        if pdf.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"invalid score PDF: {pdf}")
        load_render_layout(layout)
        json.loads(official.read_text(encoding="utf-8"))


def _validate_document_geometry(document: Path) -> None:
    if __package__:
        from .native.score_native_note_geometry import group_glyphs_by_owner
    else:
        from native.score_native_note_geometry import group_glyphs_by_owner

    for track in sorted((document / "tracks").iterdir()):
        if track.is_dir():
            layout = json.loads((track / "layout.json").read_text(encoding="utf-8"))
            if layout.get("display_mode", "tab") != "tab":
                # Fret-glyph ownership is a TAB-only annotation contract. Page,
                # system and measure coverage are validated for every mode.
                continue
            score = json.loads(
                (track / "official-score.json").read_text(encoding="utf-8")
            )
            group_glyphs_by_owner(score, layout)


def _load_manifest(path: Path) -> tuple[ManifestEntry, ...]:
    source = path.expanduser().resolve()
    entries: list[ManifestEntry] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict) or not (_MANIFEST_FIELDS <= set(value) <= _MANIFEST_FIELDS | {"display_mode"}):
            raise ValueError(f"invalid manifest fields at line {line_number}")
        entry = ManifestEntry(**value)
        _validate_manifest_entry(entry, line_number)
        entries.append(entry)
    if not entries:
        raise ValueError("manifest contains no documents")
    if len({entry.document_id for entry in entries}) != len(entries):
        raise ValueError("manifest repeats a document ID")
    if len({entry.source for entry in entries}) != len(entries):
        raise ValueError("manifest repeats a source path")
    family_splits: dict[str, str] = {}
    for entry in entries:
        previous = family_splits.setdefault(entry.source_family_id, entry.split)
        if previous != entry.split:
            raise ValueError(f"source family {entry.source_family_id} crosses splits")
    return tuple(entries)


def _validate_manifest_entry(entry: ManifestEntry, line_number: int) -> None:
    if entry.display_mode not in {"tab", "notation", "both"}:
        raise ValueError(f"invalid display mode at line {line_number}")
    if _SAFE_DOCUMENT.fullmatch(entry.document_id) is None:
        raise ValueError(f"unsafe document ID at line {line_number}")
    if (
        not isinstance(entry.source_family_id, str)
        or not entry.source_family_id.strip()
    ):
        raise ValueError(f"invalid source family at line {line_number}")
    if entry.split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid split at line {line_number}")
    if entry.instrument_kind not in {"guitar", "bass"}:
        raise ValueError(f"invalid instrument at line {line_number}")
    relative = Path(entry.source)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path at line {line_number}")
    if (
        relative.as_posix() != entry.source
        or relative.suffix.casefold() not in _SOURCE_SUFFIXES
    ):
        raise ValueError(f"invalid source path at line {line_number}")


def _manifest_jsonl(entries: tuple[ManifestEntry, ...]) -> str:
    return "".join(
        json.dumps(
            {
                "document_id": entry.document_id,
                "source_family_id": entry.source_family_id,
                "split": entry.split,
                "instrument_kind": entry.instrument_kind,
                "source": entry.source,
                **({"display_mode": entry.display_mode} if entry.display_mode != "tab" else {}),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for entry in entries
    )


def _manifest_source(root: Path, entry: ManifestEntry) -> Path:
    source = (root / Path(entry.source)).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise FileNotFoundError(source)
    return source


def _publish_failures(output: Path, worker_logs: tuple[Path, ...]) -> None:
    failures: list[dict[str, str]] = []
    for worker_log in worker_logs:
        if not worker_log.is_file():
            raise FileNotFoundError(worker_log)
        for line in worker_log.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid worker failure: {worker_log}")
            failures.append(value)
    failures.sort(key=lambda value: value["document_id"])
    failure_ids = [value["document_id"] for value in failures]
    if len(set(failure_ids)) != len(failure_ids):
        raise ValueError("workers reported the same failed document more than once")
    expected = {
        entry.document_id
        for entry in _load_manifest(output / "document-manifest.jsonl")
    }
    documents = output / "documents"
    published = (
        {path.name for path in documents.iterdir() if path.is_dir()}
        if documents.is_dir()
        else set()
    )
    overlap = published.intersection(failure_ids)
    missing = expected.difference(published, failure_ids)
    unexpected = published.union(failure_ids).difference(expected)
    if overlap or missing or unexpected:
        raise ValueError(
            "corpus export is incomplete: "
            f"overlap={sorted(overlap)[:10]}, "
            f"missing={sorted(missing)[:10]}, "
            f"unexpected={sorted(unexpected)[:10]}"
        )
    path = output / "failures.jsonl"
    if not failures:
        path.unlink(missing_ok=True)
        return
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in failures
        ),
        encoding="utf-8",
        newline="\n",
    )


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


def _write_or_validate_text(path: Path, expected: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing manifest differs: {path}")
        return
    path.write_text(expected, encoding="utf-8", newline="\n")


def _require_separate(first: Path, second: Path) -> None:
    if first == second or first.is_relative_to(second) or second.is_relative_to(first):
        raise ValueError(f"paths must be separate: {first}, {second}")


def _track_name(track: SourceTrack) -> str:
    name = _SAFE_TRACK.sub("-", track.name.strip()).strip("-._") or "unnamed"
    return f"track-{track.source_track_index:03d}-{name}"


if __name__ == "__main__":
    raise SystemExit(main())
