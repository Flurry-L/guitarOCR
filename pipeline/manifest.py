"""Record the status and artifact locations of a document pipeline run."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from shared.artifacts import write_json


class RunManifest:
    def __init__(self, output: Path, inputs: list[Path]) -> None:
        self.output = output.resolve()
        self.path = self.output / "manifest.json"
        self.value: dict[str, Any] = {
            "schema_version": "3.0", "status": "running",
            "inputs": [str(path.resolve()) for path in inputs], "stages": {},
        }
        write_json(self.path, self.value)

    @contextmanager
    def stage(self, name: str, directory: str) -> Iterator[Path]:
        output = self.output / directory
        entry = {"status": "running", "manifest": str(output / "manifest.json")}
        self.value["stages"][name] = entry
        write_json(self.path, self.value)
        try:
            yield output
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.value["status"] = "failed"
            raise
        else:
            entry["status"] = "complete"
        finally:
            write_json(self.path, self.value)

    def complete(self, **summary: Any) -> dict[str, Any]:
        self.value.update(summary)
        self.value["status"] = "needs_review" if summary.get("review_measures") else "complete"
        write_json(self.path, self.value)
        return self.value
