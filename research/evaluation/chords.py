"""Evaluate a diagram reader on held-out data or labelled image crops."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

from research.data.chord_evidence import ChordEvidenceData, diagram_from_labels
from research.inference.chord_evidence import ChordEvidenceReader


def record(counters, evidence, kind, expected):
    counters["samples"] += 1
    counters["kind_correct"] += evidence["kind"] == kind
    if kind != "diagram":
        counters["negative"] += 1
        counters["false_diagram"] += evidence["kind"] == "diagram"
        return evidence["kind"] != kind

    diagram = evidence["diagram"]
    geometry = bool(diagram) and all(
        diagram[key] == expected[key] for key in ("base_fret", "frets", "barres")
    )
    full = diagram == expected
    accepted_geometry = bool(diagram) and evidence["accepted"]["geometry"]
    accepted_full = accepted_geometry and evidence["accepted"]["full"]
    counters["positive"] += 1
    for key, value in {
        "raw_geometry_correct": geometry,
        "raw_full_correct": full,
        "geometry_accepted": accepted_geometry,
        "geometry_correct": accepted_geometry and geometry,
        "full_accepted": accepted_full,
        "full_correct": accepted_full and full,
    }.items():
        counters[key] += bool(value)
    return not full


def summarize(counters):
    metrics = dict(counters)
    for name, numerator, denominator in (
        ("kind_accuracy", "kind_correct", "samples"),
        ("raw_geometry_accuracy", "raw_geometry_correct", "positive"),
        ("raw_full_accuracy", "raw_full_correct", "positive"),
        ("accepted_geometry_precision", "geometry_correct", "geometry_accepted"),
        ("accepted_full_precision", "full_correct", "full_accepted"),
    ):
        metrics[name] = (
            counters[numerator] / counters[denominator]
            if counters[denominator]
            else None
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        help="Score data directory; defaults to database/score when --cases is absent",
    )
    parser.add_argument("--synthetic", type=int, default=2000)
    parser.add_argument(
        "--font", action="append", help="Fonts for the synthetic test images"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON array of image, kind and expected diagram; image paths are relative to this file",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output report JSON")
    args = parser.parse_args()
    reader = ChordEvidenceReader(args.model / "chord-evidence.onnx")
    groups = {name: Counter() for name in ("all", "native", "synthetic", "external")}
    failures = []
    source = args.data or (Path("database/score") if args.cases is None else None)
    if source is not None:
        data = ChordEvidenceData(source, "test", args.synthetic, fonts=args.font)
        offset = 0
        for images, labels in DataLoader(data, batch_size=32, num_workers=4):
            for index, evidence in enumerate(reader.predict_tensors(images.numpy())):
                kind = ("diagram", "text", "other")[int(labels["kind"][index])]
                expected = (
                    diagram_from_labels(labels, index) if kind == "diagram" else None
                )
                group = "native" if offset + index < len(data.rows) else "synthetic"
                wrong = record(groups[group], evidence, kind, expected)
                record(groups["all"], evidence, kind, expected)
                if wrong:
                    failures.append(
                        {
                            "group": group,
                            "index": offset + index,
                            "kind": kind,
                            "expected": expected,
                            "prediction": evidence,
                        }
                    )
            offset += len(images)
    if args.cases:
        for case in json.loads(args.cases.read_text()):
            path = args.cases.parent / case["image"]
            with Image.open(path) as image:
                evidence = reader.predict([image.convert("RGB")])[0]
            kind = case["kind"]
            expected = case.get("expected")
            wrong = record(groups["external"], evidence, kind, expected)
            record(groups["all"], evidence, kind, expected)
            if wrong:
                failures.append(
                    {
                        "image": str(path),
                        "kind": kind,
                        "expected": expected,
                        "prediction": evidence,
                    }
                )
    report = {
        "model": str(args.model),
        "data": str(source) if source else None,
        "cases": str(args.cases) if args.cases else None,
        "synthetic": args.synthetic if source else 0,
        "fonts": args.font,
        "metrics": {
            name: summarize(counts)
            for name, counts in groups.items()
            if counts["samples"]
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
