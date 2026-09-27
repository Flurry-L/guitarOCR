"""Deterministic measure selection for training repeats and fixed evaluation sets."""

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from shared.layout_labels import MODES

# These classes are visually or structurally important but are much rarer than
# ordinary fretted notes.  The main training file still contains every measure;
# a second, bounded file repeats representative training-only hard cases once.
# This avoids allowing common quarter/eighth-note measures to dominate LoRA.
HARDCASE_TAGS = {
    "accent",
    "bend",
    "chord",
    "dead",
    "fade",
    "ghost",
    "grace",
    "hammer",
    "harm",
    "heavy",
    "multi_voice",
    "pick_down",
    "pick_up",
    "pm",
    "rasg",
    "sia",
    "sib",
    "sl",
    "slap",
    "sod",
    "sou",
    "ss",
    "stacc",
    "stroke_down",
    "stroke_up",
    "tap",
    "trem",
    "trill",
}


def measure_semantic_tags(measure: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    nonempty_voices = 0
    for voice in measure.get("voices", []):
        events = voice.get("events", [])
        if events:
            nonempty_voices += 1
        for event in events:
            status = str(event.get("status", "normal"))
            if status in {"rest", "empty"}:
                tags.add(status)
            duration = event.get("duration") or {}
            if duration.get("dotted") or duration.get("double_dotted"):
                tags.add("dotted")
            if (
                int(duration.get("tuplet_enters", 1) or 1) != 1
                or int(duration.get("tuplet_times", 1) or 1) != 1
            ):
                tags.add("tuplet")
            notes = event.get("notes") or []
            if len(notes) > 1:
                tags.add("chord")
            for effect in event.get("effects", []):
                tags.add(str(effect).split(":", 1)[0])
            for note in notes:
                for effect in note.get("effects", []):
                    tags.add(str(effect).split(":", 1)[0])
    if nonempty_voices > 1:
        tags.add("multi_voice")
    return sorted(tags)


def balanced_hardcase_rows(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Return a deterministic, bounded union of rare-class examples.

    Each mode/tag bucket contributes at most 1.5% of that mode's ordinary
    training rows (and at least 256 when available).  Rare buckets smaller than
    the cap are therefore preserved in full, while common classes such as palm
    mute cannot overwhelm the rest of the dataset.
    """
    train_rows = [row for row in rows if row["split"] == "train"]
    per_mode = Counter(row["mode"] for row in train_rows)
    selected: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, int]] = defaultdict(dict)
    for mode in MODES:
        cap = max(256, (int(per_mode[mode]) * 15 + 999) // 1000)
        mode_rows = [row for row in train_rows if row["mode"] == mode]
        for tag in sorted(HARDCASE_TAGS):
            candidates = [
                row for row in mode_rows if tag in row.get("semantic_tags", [])
            ]
            candidates.sort(
                key=lambda row: sha256(
                    f"{seed}:hard:{tag}:{row['id']}".encode("utf-8")
                ).digest()
            )
            chosen = candidates[:cap]
            coverage[mode][tag] = len(chosen)
            for row in chosen:
                selected[row["id"]] = row
    hardcases = sorted(
        selected.values(),
        key=lambda row: (row["source_id"], row["mode"], row["measure_index"]),
    )
    return hardcases, {
        mode: dict(sorted(values.items())) for mode, values in sorted(coverage.items())
    }


def load_measure_rows(
    manifest: Path, maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]

    def stable(value):
        return sha256(f"{seed}:{value}".encode("utf-8")).digest()

    if not maximum:
        return sorted(rows, key=lambda row: stable(row["id"]))

    modes = sorted({row["mode"] for row in rows})
    quotas = {mode: maximum // len(modes) for mode in modes}
    for mode in modes[: maximum % len(modes)]:
        quotas[mode] += 1
    selected = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_selected: dict[str, dict[str, Any]] = {}

        # Reserve a small, source-diverse slice for every represented semantic
        # class before filling the quota. A purely uniform sample can contain
        # almost no bends, harmonics or multi-voice measures and therefore
        # cannot support the per-technique release gate.
        tags = sorted(
            {str(tag) for row in mode_rows for tag in row.get("semantic_tags", [])}
        )
        coverage_per_tag = min(20, max(4, quotas[mode] // 50))
        for tag in tags:
            candidates = sorted(
                (row for row in mode_rows if tag in row.get("semantic_tags", [])),
                key=lambda row: stable(f"technique:{mode}:{tag}:{row['id']}"),
            )
            tag_sources: set[str] = set()
            for row in candidates:
                if (
                    len(mode_selected) >= quotas[mode]
                    or len(tag_sources) >= coverage_per_tag
                ):
                    break
                if row["id"] in mode_selected or row["source_id"] in tag_sources:
                    continue
                mode_selected[row["id"]] = row
                tag_sources.add(row["source_id"])

        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in mode_rows:
            if row["id"] not in mode_selected:
                by_source.setdefault(row["source_id"], []).append(row)
        for values in by_source.values():
            values.sort(key=lambda row: stable(row["id"]))
        source_ids = sorted(by_source, key=lambda value: stable(f"{mode}:{value}"))
        while len(mode_selected) < quotas[mode]:
            progressed = False
            for source_id in source_ids:
                if not by_source[source_id]:
                    continue
                row = by_source[source_id].pop()
                mode_selected[row["id"]] = row
                progressed = True
                if len(mode_selected) >= quotas[mode]:
                    break
            if not progressed:
                break
        selected.extend(mode_selected.values())
    return sorted(selected, key=lambda row: stable(row["id"]))
