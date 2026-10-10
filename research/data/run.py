from __future__ import annotations

import argparse
import heapq
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scorelib.instruments import track_instrument

from research.common.images import MODES
from research.data.catalog import source_catalog, write_json, write_jsonl
from research.data.gp_sources import analyze_source, prepare_single_track_gp5

SUPPORTED_SUFFIXES = {".gp3", ".gp4", ".gp5", ".gtp"}

def _candidate_pool(
    corpus: Path, seed: int, pool_per_format: int
) -> tuple[dict[str, list[Path]], Counter[str]]:
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    inventory: Counter[str] = Counter()
    for path in corpus.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        inventory[path.suffix.lower()] += 1
        relative = path.relative_to(corpus).as_posix()
        key = random.Random(f"{seed}:{relative}").getrandbits(64)
        extension = path.suffix.lower()
        heap = heaps[extension]
        item = (-key, str(path))
        if len(heap) < pool_per_format:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    result = {}
    for extension, heap in heaps.items():
        result[extension] = [
            Path(path) for _negative_key, path in sorted(heap, key=lambda item: -item[0])
        ]
    return result, inventory

def _interleave(values: dict[str, list[Path]]) -> list[Path]:
    ordered = []
    extensions = sorted(values)
    maximum = max((len(values[key]) for key in extensions), default=0)
    for index in range(maximum):
        for extension in extensions:
            if index < len(values[extension]):
                ordered.append(values[extension][index])
    return ordered

def _eligible(payload: dict[str, Any], minimum_measures: int, maximum_measures: int) -> bool:
    statistics = payload["statistics"]
    strings = int(payload["track"]["string_count"])
    return (
        minimum_measures <= int(statistics["measure_count"]) <= maximum_measures
        and int(statistics["note_count"]) >= 16
        and 4 <= strings <= 8
    )

def _diverse_selection(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected = []
    tag_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    while remaining and len(selected) < count:
        best_index = 0
        best_score = float("-inf")
        for index, payload in enumerate(remaining):
            tags = payload["statistics"].get("tags", [])
            rarity = sum(1.0 / (1.0 + tag_counts[tag]) for tag in tags)
            format_bonus = 2.0 / (1.0 + format_counts[payload["source_format"]])
            multi_voice = min(2.0, payload["statistics"]["multi_voice_measure_count"] / 8.0)
            score = rarity + format_bonus + multi_voice
            if score > best_score:
                best_score = score
                best_index = index
        payload = remaining.pop(best_index)
        selected.append(payload)
        tag_counts.update(payload["statistics"].get("tags", []))
        format_counts[payload["source_format"]] += 1
    return selected

def select_sources(
    corpus: Path,
    output: Path,
    *,
    source_count: int,
    seed: int,
    minimum_measures: int,
    maximum_measures: int,
    modes: list[str],
    source_catalog: Path | None = None,
) -> list[dict[str, Any]]:
    assignments = {}
    if source_catalog is not None:
        source_catalog = source_catalog.resolve()
        families = {}
        for row in json.loads(source_catalog.read_text())['sources']:
            path = (source_catalog.parent / row['source_file']).resolve()
            if row['split'] not in {'train', 'validation', 'test'} or not row['family']:
                raise ValueError(f'Invalid source assignment: {row}')
            if families.setdefault(row['family'], row['split']) != row['split']:
                raise ValueError(f"Source family crosses splits: {row['family']}")
            if path in assignments:
                raise ValueError(f'Duplicate source path: {path}')
            assignments[path] = row
    labels_root = output / "labels"
    prepared_root = output / "prepared"
    pool, inventory = _candidate_pool(corpus, seed, max(200, source_count * 8))
    candidates = _interleave(pool)
    valid: list[dict[str, Any]] = []
    failures = []
    seen_hashes = set()
    target_pool = max(source_count, source_count * 3)
    for path in candidates:
        if len(valid) >= target_pool:
            break
        try:
            if source_catalog is not None and path.resolve() not in assignments:
                raise ValueError(f'Source is not declared in the catalog: {path}')
            payload = analyze_source(path)
            if source_catalog is not None:
                assignment = assignments[path.resolve()]
                payload.update({key: assignment[key] for key in ('source_id', 'family', 'split')})
            if payload["sha256"] in seen_hashes or not _eligible(
                payload, minimum_measures, maximum_measures
            ):
                continue
            seen_hashes.add(payload["sha256"])
            valid.append(payload)
        except Exception as error:  # pragma: no cover - corpus dependent
            if len(failures) < 500:
                failures.append({"path": str(path), "error": repr(error)})
    ranked = _diverse_selection(valid, len(valid))
    prepared = []
    for candidate in ranked:
        if len(prepared) >= source_count:
            break
        source_path = Path(candidate["source_path"])
        source_id = candidate["source_id"]
        payload = None
        try:
            for mode in modes:
                gp5_path = prepared_root / mode / f"{source_id}.gp5"
                mode_payload = prepare_single_track_gp5(source_path, gp5_path, mode)
                if payload is None:
                    payload = mode_payload
        except Exception as error:  # pragma: no cover - corpus dependent
            for mode in modes:
                partial = prepared_root / mode / f"{source_id}.gp5"
                if partial.is_file():
                    partial.unlink()
            if len(failures) < 500:
                failures.append({
                    "path": str(source_path),
                    "stage": "prepare_gp5",
                    "error": repr(error),
                })
            continue
        assert payload is not None
        payload['source_id'] = source_id
        if source_catalog is not None:
            payload.update({key: candidate[key] for key in ('family', 'split')})
        label_path = labels_root / f"{source_id}.json"
        write_json(label_path, payload)
        prepared.append(payload)
        print(
            f"[{len(prepared)}/{source_count}] {source_id} {payload['source_format']} "
            f"measures={payload['statistics']['measure_count']} "
            f"tags={','.join(payload['statistics']['tags'])}",
            flush=True,
        )
    selected_ids = {payload["source_id"] for payload in prepared}
    for stale in labels_root.glob("*.json"):
        if stale.stem not in selected_ids:
            stale.unlink()
    for mode in modes:
        for stale in (prepared_root / mode).glob("*.gp5"):
            if stale.stem not in selected_ids:
                stale.unlink()
    if len(prepared) < source_count:
        raise RuntimeError(
            f"Only prepared {len(prepared)} valid sources out of requested {source_count}"
        )
    write_json(output / "selection_failures.json", failures)
    selected_formats = Counter(payload["source_format"] for payload in prepared)
    selected_tags = Counter(
        tag for payload in prepared for tag in payload["statistics"].get("tags", [])
    )
    write_json(
        output / "selection_summary.json",
        {
            "corpus": str(corpus.resolve()),
            "corpus_files": sum(inventory.values()),
            "corpus_formats": {
                extension.lstrip("."): count
                for extension, count in sorted(inventory.items())
            },
            "requested_sources": source_count,
            "selected_sources": len(prepared),
            "selected_formats": dict(sorted(selected_formats.items())),
            "selected_technique_tags": dict(sorted(selected_tags.items())),
            "parse_failures_recorded": len(failures),
            "seed": seed,
        },
    )
    if source_catalog is not None:
        write_json(output / 'source_catalog.json', {'sources': [
            {key: value[key] for key in ('source_id', 'family', 'split', 'source_path')}
            for value in prepared
        ]})
    return prepared


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
    from research.data.gp_sources import parse_song

    def instrument_kind(mode: str, source_id: str) -> str:
        prepared = output / "prepared" / mode / f"{source_id}.gp5"
        song, _ = parse_song(prepared)
        if len(song.tracks) != 1:
            raise ValueError(f"Expected one prepared track: {prepared}")
        return track_instrument(song.tracks[0])

    entries = [
        {
            "document_id": f"{mode}-{label['source_id']}",
            "source_family_id": catalog[label["source_id"]]["family"],
            "split": {"validation": "dev"}.get(catalog[label["source_id"]]["split"], catalog[label["source_id"]]["split"]),
            "display_mode": mode,
            "instrument_kind": instrument_kind(mode, label["source_id"]),
            "source": f"{mode}/{label['source_id']}.gp5",
        }
        for label in labels for mode in modes if mode in label.get("modes", modes)
    ]
    manifest = output / "native-export-manifest.jsonl"
    write_jsonl(manifest, entries)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-disjoint official GP8 measure-image to event-sequence data."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Original source files; required for select and all phases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-catalog", type=Path, help="Preserved source_file, family and split assignments")
    parser.add_argument("--source-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--minimum-measures", type=int, default=8)
    parser.add_argument("--maximum-measures", type=int, default=128)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--runtime", type=Path, help="Guitar Pro 8 runtime directory")
    parser.add_argument("--wine-prefix-template", type=Path)
    parser.add_argument("--wine-python", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "select",
            "render",
            "crop",
            "datasets",
        ),
        default="all",
    )
    args = parser.parse_args()
    if args.phase in {"all", "select"} and args.corpus is None:
        parser.error("source selection requires --corpus")
    if args.phase in {"all", "render"}:
        if (
            args.runtime is None
            or args.wine_prefix_template is None
            or args.wine_python is None
        ):
            parser.error(
                "rendering requires --runtime, --wine-prefix-template and --wine-python"
            )
    modes = args.mode or list(MODES)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"all", "select"}:


        select_sources(
            args.corpus.resolve(),
            output,
            source_count=args.source_count,
            seed=args.seed,
            minimum_measures=args.minimum_measures,
            maximum_measures=args.maximum_measures,
            modes=modes,
            source_catalog=args.source_catalog,
        )
    if args.phase in {"all", "render"}:

        render_modes(
            output,
            modes,
            args.runtime,
            args.wine_prefix_template,
            args.wine_python,
            args.workers,
        )
    if args.phase in {"all", "crop"}:
        from research.data.build_measure_data import crop_and_manifest

        summary = crop_and_manifest(output, modes, args.dpi, args.seed, args.workers)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.phase in {"all", "datasets"}:
        from research.data.build_info_data import build_dataset as build_info
        from research.data.build_layout_data import build_dataset as build_layout
        from research.data.inventory import build_inventory

        inventory = output / "inventory"
        print(
            json.dumps(
                build_inventory(output, inventory, dpi=args.dpi, seed=args.seed, modes=modes, workers=args.workers), ensure_ascii=False
            )
        )
        print(
            json.dumps(
                {
                    'layout': build_layout([inventory], output / "datasets" / 'layout', include_test=True),
                    "information": build_info(
                        inventory, output / "datasets" / 'document_info', include_test=True
                    ),
                },
                ensure_ascii=False,
            )
        )

if __name__ == "__main__":
    main()
