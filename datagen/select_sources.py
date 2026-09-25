from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from datagen.files import _write_json, _write_jsonl


import heapq
from datagen.gp_sources import analyze_source, prepare_single_track_gp5

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
        key = int.from_bytes(sha256(f"{seed}:{relative}".encode("utf-8")).digest()[:8], "big")
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
            stable = int(payload["sha256"][:8], 16) / 0xFFFFFFFF
            score = rarity + format_bonus + multi_voice + stable * 0.01
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
) -> list[dict[str, Any]]:
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
            payload = analyze_source(path)
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
        label_path = labels_root / f"{source_id}.json"
        payload["label_json"] = str(label_path.resolve())
        _write_json(label_path, payload)
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
    _write_json(output / "selection_failures.json", failures)
    selected_formats = Counter(payload["source_format"] for payload in prepared)
    selected_tags = Counter(
        tag for payload in prepared for tag in payload["statistics"].get("tags", [])
    )
    _write_json(
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
    _write_jsonl(
        output / "sources.jsonl",
        (
            {
                "source_id": payload["source_id"],
                "sha256": payload["sha256"],
                "source_path": payload["source_path"],
                "source_format": payload["source_format"],
                "label_json": payload["label_json"],
                "statistics": payload["statistics"],
            }
            for payload in prepared
        ),
    )
    return prepared


def relabel_selected_sources(
    output: Path, modes: list[str], *, prepare: bool = True
) -> list[dict[str, Any]]:
    """Rebuild labels/prepared GP5 from an existing stable source selection."""
    source_manifest = output / "sources.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing prior source selection: {source_manifest}")
    previous = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    labels_root = output / "labels"
    prepared_root = output / "prepared"
    rebuilt = []
    failures = []
    for index, record in enumerate(previous, start=1):
        source_path = Path(record["source_path"])
        try:
            payload = analyze_source(source_path)
            if prepare:
                for mode in modes:
                    prepare_single_track_gp5(
                        source_path,
                        prepared_root / mode / f"{payload['source_id']}.gp5",
                        mode,
                    )
            label_path = labels_root / f"{payload['source_id']}.json"
            payload["label_json"] = str(label_path.resolve())
            _write_json(label_path, payload)
            rebuilt.append(payload)
            print(
                f"[relabel {index}/{len(previous)}] {payload['source_id']}",
                flush=True,
            )
        except Exception as error:  # pragma: no cover - corpus dependent
            failures.append({"path": str(source_path), "error": repr(error)})
    if failures or len(rebuilt) != len(previous):
        _write_json(output / "relabel_failures.json", failures)
        raise RuntimeError(
            f"Relabel rebuilt {len(rebuilt)}/{len(previous)} sources; "
            f"see {output / 'relabel_failures.json'}"
        )
    _write_jsonl(output / "sources.jsonl", rebuilt)
    _refresh_selection_summary(output, rebuilt)
    _write_json(output / "relabel_failures.json", [])
    return rebuilt


def _refresh_selection_summary(output: Path, payloads: list[dict[str, Any]]) -> None:
    """Keep schema-derived coverage counts in sync after relabeling."""
    summary_path = output / "selection_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    formats = Counter(str(payload["source_format"]) for payload in payloads)
    tags = Counter(
        tag
        for payload in payloads
        for tag in payload["statistics"].get("tags", [])
    )
    summary.update({
        "selected_sources": len(payloads),
        "selected_formats": dict(sorted(formats.items())),
        "selected_technique_tags": dict(sorted(tags.items())),
    })
    _write_json(summary_path, summary)
