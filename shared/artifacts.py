"""JSON files exchanged between independently runnable pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from shared.schema import STAGE_SCHEMA_VERSION, Stage, StageResult


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def write_result(output: Path, stage: Stage, **values: Any) -> Path:
    return write_json(
        output.resolve() / "manifest.json",
        {"schema_version": STAGE_SCHEMA_VERSION, "stage": stage, **values},
    )


def read_result(path: Path, stage: Stage) -> StageResult:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != STAGE_SCHEMA_VERSION
        or value.get("stage") != stage
    ):
        raise ValueError(f"Expected a {stage} stage manifest (schema 1.0): {path}")
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
