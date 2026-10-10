"""One source-family assignment shared by all layouts and training tasks."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
from typing import Any, Iterable


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def native_document(label_path, mode):
    path = Path(label_path)
    root = path.parent.parent
    return root / 'native-export' / 'documents' / f'{mode}-{path.stem}'


SPLITS = ("train", "validation", "test")


def source_catalog(root: Path, seed: int = 20260715) -> dict[str, dict]:
    path = root / "source_catalog.json"
    labels = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((root / "labels").glob("*.json"))
    ]
    if not labels:
        raise ValueError(
            f"No source labels in {root / 'labels'}; run research.data.run --phase select first"
        )
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))["sources"]
        catalog = {row["source_id"]: row for row in rows}
        if len(catalog) != len(rows) or set(catalog) != {
            r["source_id"] for r in labels
        }:
            raise ValueError(
                "source_catalog.json must contain each selected source exactly once"
            )
    else:
        groups = {}
        for label in labels:
            family = str(
                label.get("family") or label.get("sha256") or label["source_id"]
            )
            if family not in groups:
                groups[family] = deepcopy(label)
                groups[family]["source_id"] = family
            else:
                tags = groups[family]["statistics"].get("tags", [])
                groups[family]["statistics"]["tags"] = sorted(
                    set(tags) | set(label["statistics"].get("tags", []))
                )
        family_splits, _ = stratified_source_splits(list(groups.values()), seed)
        catalog = {}
        for label in labels:
            sid = label["source_id"]
            family = str(label.get("family") or label.get("sha256") or sid)
            catalog[sid] = {
                "source_id": sid,
                "family": family,
                "split": family_splits[family],
                "source_path": label["source_path"],
            }
    families = {}
    for row in catalog.values():
        if row["split"] not in SPLITS or not row["family"]:
            raise ValueError(f"Invalid source assignment: {row}")
        previous = families.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Source family crosses dataset splits: {row['family']}")
    write_json(
        path, {"schema_version": "1.0", "seed": seed, "sources": list(catalog.values())}
    )
    return catalog


def stratified_source_splits(
    labels: list[dict[str, Any]], seed: int
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Create deterministic source-disjoint splits with long-tail coverage."""

    split_names = ("train", "validation", "test")
    ratios = {"train": 0.80, "validation": 0.10, "test": 0.10}
    total = len(labels)
    capacities = {
        "train": round(total * ratios["train"]),
        "validation": round(total * ratios["validation"]),
    }
    capacities["test"] = total - capacities["train"] - capacities["validation"]

    semantic_tags: dict[str, set[str]] = {}
    all_tags: dict[str, set[str]] = {}
    label_by_id = {label["source_id"]: label for label in labels}
    for label in labels:
        source_id = label["source_id"]
        semantics = set(label["statistics"].get("tags", []))
        if int(label["statistics"].get("multi_voice_measure_count", 0)):
            semantics.add("multi_voice")
        semantic_tags[source_id] = semantics
        all_tags[source_id] = {
            *semantics,
            f"format:{label['source_format']}",
            f"strings:{label['track']['string_count']}",
        }

    frequencies = Counter(tag for tags in all_tags.values() for tag in tags)
    semantic_universe = set().union(*semantic_tags.values())
    desired = {
        split: {
            tag: max(
                1
                if split != "train" and tag in semantic_universe and count >= 3
                else 0,
                round(count * ratios[split]),
            )
            for tag, count in frequencies.items()
        }
        for split in split_names
    }
    assignments: dict[str, str] = {}
    current = {split: Counter() for split in split_names}
    assigned_counts = Counter()

    def stable(source_id: str, split: str) -> float:
        return random.Random(f"{seed}:{split}:{source_id}").random()

    def assign(source_id: str, split: str) -> None:
        assignments[source_id] = split
        assigned_counts[split] += 1
        current[split].update(all_tags[source_id])

    # Give both held-out splits at least one source for every semantic tag
    # represented by three or more independent songs.
    semantic_frequency = Counter(tag for tags in semantic_tags.values() for tag in tags)
    for split in ("validation", "test"):
        for tag, count in sorted(
            semantic_frequency.items(), key=lambda item: (item[1], item[0])
        ):
            if (
                count < 3
                or current[split][tag]
                or assigned_counts[split] >= capacities[split]
            ):
                continue
            candidates = [
                source_id
                for source_id, tags in semantic_tags.items()
                if source_id not in assignments and tag in tags
            ]
            if not candidates:
                continue
            source_id = max(
                candidates,
                key=lambda value: (
                    sum(
                        1.0 / semantic_frequency[candidate_tag]
                        for candidate_tag in semantic_tags[value]
                        if not current[split][candidate_tag]
                    ),
                    stable(value, split),
                ),
            )
            assign(source_id, split)

    remaining = sorted(
        (source_id for source_id in label_by_id if source_id not in assignments),
        key=lambda source_id: (
            -sum(1.0 / frequencies[tag] for tag in all_tags[source_id]),
            stable(source_id, "order"),
        ),
    )
    for source_id in remaining:
        choices = [
            split for split in split_names if assigned_counts[split] < capacities[split]
        ]
        split = max(
            choices,
            key=lambda value: (
                sum(
                    max(0, desired[value][tag] - current[value][tag])
                    / max(1, desired[value][tag])
                    for tag in all_tags[source_id]
                ),
                (capacities[value] - assigned_counts[value])
                / max(1, capacities[value]),
                stable(source_id, value),
            ),
        )
        assign(source_id, split)

    coverage = {
        split: dict(
            sorted(
                Counter(
                    tag
                    for source_id, assigned_split in assignments.items()
                    if assigned_split == split
                    for tag in semantic_tags[source_id]
                ).items()
            )
        )
        for split in split_names
    }
    return assignments, coverage
