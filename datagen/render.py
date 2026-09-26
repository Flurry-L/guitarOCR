from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from datagen.files import _write_jsonl
from datagen.inventory import source_catalog


def render_modes(
    output: Path,
    modes: list[str],
    runtime: Path,
    wine_prefix_template: Path,
    wine_python: Path,
    workers: int,
) -> None:
    labels = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "labels").glob("*.json"))
    ]
    if not labels:
        raise ValueError("No selected source labels found; run --phase select first")
    catalog = source_catalog(output)
    from datagen.gp_sources import parse_song

    def instrument_kind(mode: str, source_id: str) -> str:
        prepared = output / "prepared" / mode / f"{source_id}.gp5"
        song, _ = parse_song(prepared)
        if len(song.tracks) != 1:
            raise ValueError(f"Expected one prepared track: {prepared}")
        return "bass" if 32 <= int(song.tracks[0].channel.instrument) <= 39 else "guitar"

    entries = [
        {
            "document_id": f"{mode}-{label['source_id']}",
            "source_family_id": catalog[label["source_id"]]["family"],
            "split": {"validation": "dev"}.get(catalog[label["source_id"]]["split"], catalog[label["source_id"]]["split"]),
            "display_mode": mode,
            "instrument_kind": instrument_kind(mode, label["source_id"]),
            "source": f"{mode}/{label['source_id']}.gp5",
        }
        for label in labels for mode in modes
    ]
    manifest = output / "native-export-manifest.jsonl"
    _write_jsonl(manifest, entries)
    export_root = output / "native-export"
    subprocess.run(
        [
            sys.executable, str(Path(__file__).with_name("export_scores.py")),
            "--manifest", str(manifest),
            "--source-root", str((output / "prepared").resolve()),
            "--runtime", str(runtime.resolve()),
            "--output-dir", str(export_root.resolve()),
            "--wine-prefix-template", str(wine_prefix_template.resolve()),
            "--wine-python", str(wine_python.resolve()),
            "--workers", str(workers),
        ],
        check=True,
    )
    failures = export_root / "failures.jsonl"
    if failures.is_file():
        raise RuntimeError(f"Native export reported failures: {failures}")
    for entry in entries:
        document = export_root / "documents" / entry["document_id"] / "tracks"
        tracks = sorted(document.glob("*/layout.json"))
        if len(tracks) != 1:
            raise RuntimeError(f"Expected one exported track in {document}; found {len(tracks)}")
        layout = tracks[0]
        pdf = layout.with_name("score.pdf")
        mode, source_id = entry["source"].split("/")[0], Path(entry["source"]).stem
        native_layout = json.loads(layout.read_text(encoding="utf-8"))
        if native_layout.get("schema") != "gpomr.render-layout":
            raise ValueError(f"Unexpected native layout schema: {layout}")
        if bool(native_layout["tab_only"]) != (mode == "tab"):
            raise ValueError(f"Native layout mode differs from {mode}: {layout}")
        for source, destination in (
            (layout, output / "layout" / mode / f"{source_id}.layout.json"),
            (pdf, output / "pdf" / mode / f"{source_id}.pdf"),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
