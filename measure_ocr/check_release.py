from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    def require(scope: str, values: dict[str, Any], thresholds: dict[str, Any]) -> None:
        for name, minimum in thresholds.items():
            actual = float(values.get(name, 0.0))
            if actual < float(minimum):
                failures.append({
                    "scope": scope,
                    "metric": name,
                    "actual": actual,
                    "minimum": float(minimum),
                })

    require("overall", metrics.get("overall", {}), gate.get("overall", {}))
    by_mode = metrics.get("by_mode", {})
    required_modes = list(gate.get("required_modes", []))
    for mode in required_modes:
        if mode not in by_mode:
            failures.append({
                "scope": mode,
                "metric": "mode_present",
                "actual": 0.0,
                "minimum": 1.0,
            })
    mode_overrides = gate.get("mode_overrides", {})
    for mode, values in sorted(by_mode.items()):
        require(mode, values, gate.get("each_mode", {}))
        require(mode, values, mode_overrides.get(mode, {}))

    support = int(gate.get("technique_min_expected", 20))
    minimum_f1 = float(gate.get("technique_min_f1", 0.9))
    for mode, values in sorted(by_mode.items()):
        technique_values = values.get("technique_by_class", {})
        for technique in gate.get("required_techniques", []):
            result = technique_values.get(technique)
            expected = int(result.get("expected", 0)) if result else 0
            if expected < support:
                failures.append({
                    "scope": mode,
                    "metric": f"technique:{technique}:expected_support",
                    "actual": expected,
                    "minimum": support,
                })
        for technique, result in sorted(technique_values.items()):
            if int(result.get("expected", 0)) < support:
                continue
            actual = float(result.get("f1", 0.0))
            if actual < minimum_f1:
                failures.append({
                    "scope": mode,
                    "metric": f"technique:{technique}:f1",
                    "actual": actual,
                    "minimum": minimum_f1,
                    "expected": int(result.get("expected", 0)),
                })
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the non-negotiable M2 release thresholds to OCR metrics."
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path(__file__).with_name("configs") / "release_gate.json",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = check(metrics, gate)
    value = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(value, encoding="utf-8")
    print(value, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
