"""Calibrate and export diagram readers and their agreement using validation data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from research.data.chord_evidence import ChordEvidenceData, diagram_from_labels
from research.inference.chord_evidence import (
    CONTRACT,
    OUTPUT_NAMES,
    PREPROCESSING,
    ChordEvidenceReader,
    decode,
)
from research.models.chord_evidence import OUTPUTS, ChordEvidenceNetwork


def threshold(rows, minimum=40, precision=0.995):
    # Pick the broadest complete confidence prefix reaching the requested
    # validation precision. A value >1 means disabled, including low support.
    ordered = sorted(rows, reverse=True)
    best = (1.001, 0, 0)
    correct = 0
    for i, (confidence, ok) in enumerate(ordered, 1):
        correct += int(ok)
        complete_group = i == len(ordered) or ordered[i][0] != confidence
        if (
            complete_group
            and confidence >= 0.5
            and i >= minimum
            and correct / i >= precision
        ):
            best = (max(0.5, float(confidence)), i, correct)
    return {
        "threshold": best[0],
        "accepted": best[1],
        "correct": best[2],
        "samples": len(rows),
    }


def field_calibration(kind_rows, diagram_rows):
    """Calibrate a dependent field on the inputs its upstream gates accept.

    Frets, fingers and barres are indexed by the predicted string count. An
    already rejected count must not disable these heads through confident
    predictions for irrelevant slots. Incorrect counts that pass the count
    gate remain errors in every dependent field and in the final joint gate.
    """
    metrics = {"kind": threshold(kind_rows)}
    classified = [
        (confidence, correct)
        for confidence, correct in diagram_rows
        if correct["kind"] and confidence["kind"] >= metrics["kind"]["threshold"]
    ]
    metrics["strings"] = threshold(
        [
            (confidence["strings"], correct["strings"])
            for confidence, correct in classified
        ]
    )
    upstream = [
        (confidence, correct)
        for confidence, correct in classified
        if confidence["strings"] >= metrics["strings"]["threshold"]
    ]
    for field in ("base", "frets", "fingers", "barres"):
        metrics[field] = threshold(
            [
                (confidence[field], correct[field] and correct["strings"])
                for confidence, correct in upstream
            ]
        )
    return metrics


def branch_calibration(contract):
    return {
        key: contract[key] for key in ("temperature", "thresholds", "joint_thresholds")
    }


def calibrate_reader(model_dir, data):
    checkpoint = model_dir / "chord-evidence.safetensors"
    torch.set_num_threads(1)
    model = ChordEvidenceNetwork.from_weights(load_file(str(checkpoint)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(data, batch_size=192, num_workers=8)
    contract = {
        "schema": CONTRACT,
        "reader": "chord_evidence",
        "preprocessing": PREPROCESSING,
        "input_size": 256,
        "temperature": {name: 1.0 for name in OUTPUTS},
        "thresholds": {name: 1.001 for name in OUTPUTS},
        "kind_labels": ["diagram", "text", "other"],
        "calibration_split": "validation_only_shape_and_document_disjoint",
        "max_strings": 12,
        "max_relative_frets": 7,
        "source": "learned_visual_evidence",
    }
    kind_rows = []
    joint_rows = []
    with torch.inference_mode():
        for images, labels in loader:
            out = [v.float().cpu().numpy() for v in model(images.to(device))]
            for i in range(len(images)):
                evidence = decode([v[i] for v in out], contract)
                kind = int(labels["kind"][i])
                count = int(labels["strings"][i])
                predicted = [int(out[n][i].argmax()) for n in range(3)]
                kind_rows.append((evidence["confidence"]["kind"], predicted[0] == kind))
                if kind != 0:
                    continue
                ok = {
                    "strings": predicted[1] == count,
                    "base": predicted[2] == int(labels["base"][i]),
                    "frets": bool(
                        (
                            out[3][i, :count].argmax(-1)
                            == labels["frets"][i, :count].numpy()
                        ).all()
                    ),
                    "fingers": bool(
                        (
                            out[4][i, :count].argmax(-1)
                            == labels["fingers"][i, :count].numpy()
                        ).all()
                    ),
                    "barres": bool(
                        (
                            (out[5][i, :, : count - 1] >= 0)
                            == labels["barres"][i, :, : count - 1].numpy().astype(bool)
                        ).all()
                    ),
                }
                joint_rows.append(
                    (evidence["confidence"], {"kind": predicted[0] == kind, **ok})
                )
    calibration = field_calibration(kind_rows, joint_rows)
    contract["calibration_method"] = "conditional_kind_strings_v1"
    contract["thresholds"] = {name: r["threshold"] for name, r in calibration.items()}
    joint = {}
    for name, keys in [
        ("geometry", ["kind", "strings", "base", "frets", "barres"]),
        ("full", list(OUTPUTS)),
    ]:
        rows = [
            (min(confidence[key] for key in keys), all(correct[key] for key in keys))
            for confidence, correct in joint_rows
            if all(confidence[key] >= contract["thresholds"][key] for key in keys)
        ]
        joint[name] = threshold(rows)
    contract["joint_thresholds"] = {name: r["threshold"] for name, r in joint.items()}
    calibration["joint"] = joint
    import onnx

    output = model_dir / "chord-evidence.onnx"
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 256, 256),
        str(output),
        input_names=["images"],
        output_names=list(OUTPUTS),
        dynamic_axes={name: {0: "batch"} for name in ("images", *OUTPUTS)},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(
        graph, {"guitarocr.chord_evidence": json.dumps(contract, separators=(",", ":"))}
    )
    onnx.save(graph, str(output))
    (model_dir / "calibration.json").write_text(
        json.dumps({"contract": contract, "metrics": calibration}, indent=2)
    )
    print(json.dumps(calibration, indent=2))


def calibrate_consensus(model_dir, data, output):
    reader = ChordEvidenceReader(model_dir / "chord-evidence.onnx")
    if not reader.consensus or "agreement_calibration" in reader.contract:
        raise ValueError(
            "Calibrate the exported pair before adding agreement thresholds"
        )
    rows, failures, offset = [], [], 0
    for images, labels in DataLoader(data, batch_size=32, num_workers=4):
        for i, evidence in enumerate(reader.predict_tensors(images.numpy())):
            if int(labels["kind"][i]) != 0:
                continue
            gold = diagram_from_labels(labels, i)
            diagram = evidence["diagram"]
            correct = {key: bool(diagram) and diagram[key] == gold[key] for key in gold}
            geometry = all(value for key, value in correct.items() if key != "fingers")
            confidence = {
                key: min(
                    evidence["confidence"][key], evidence["secondary_confidence"][key]
                )
                for key in (*OUTPUT_NAMES, "geometry", "full")
            }
            count = int(labels["strings"][i])
            relative = labels["frets"][i, :count].tolist()
            fields = {
                "kind": evidence["kind"] == "diagram",
                "strings": evidence["strings"] == count,
                "base": evidence["base_fret"] == gold["base_fret"],
                "frets": evidence["relative_frets"]
                == ["x" if value == 0 else value - 1 for value in relative],
                "fingers": correct["fingers"],
                "barres": evidence["relative_barres"]
                == [
                    [fret - gold["base_fret"] + 1, a, b]
                    for fret, a, b in gold["barres"]
                ],
            }
            rows.append(
                {
                    "confidence": confidence,
                    "accepted": evidence["accepted"],
                    "geometry": geometry,
                    "full": all(correct.values()),
                    "fields": fields,
                }
            )
            if (
                evidence["accepted"]["geometry"]
                and not geometry
                or evidence["accepted"]["full"]
                and not all(correct.values())
            ):
                failures.append(
                    {
                        "index": offset + i,
                        "wrong_fields": [
                            key for key, value in correct.items() if not value
                        ],
                        "expected": gold,
                        "evidence": evidence,
                    }
                )
        offset += len(images)
    metrics, thresholds, field_thresholds = {}, {}, {}
    for key in OUTPUT_NAMES:
        scope = "full" if key == "fingers" else "geometry"
        candidates = [
            (row["confidence"][key], row["fields"][key])
            for row in rows
            if row["accepted"][scope]
        ]
        result = threshold(candidates, precision=1.0)
        value = (
            0.0
            if len(candidates) >= 40 and all(correct for _, correct in candidates)
            else result["threshold"]
        )
        field_thresholds[key] = (
            float(np.nextafter(np.float32(value), np.float32(np.inf)))
            if value
            else 0.0
        )
        metrics[key] = {**result, "export_threshold": field_thresholds[key]}
    for key in ("geometry", "full"):
        keys = [
            field for field in OUTPUT_NAMES if key == "full" or field != "fingers"
        ]
        selected = [
            row
            for row in rows
            if row["accepted"][key]
            and all(
                row["confidence"][field] >= field_thresholds[field]
                for field in keys
            )
        ]
        metrics[key] = {
            "export_accepted": len(selected),
            "export_correct": sum(row[key] for row in selected),
        }
        thresholds[key] = 0.0
    report = {
        "calibration_split": "validation_only_shape_and_document_disjoint",
        "minimum_precision": 1.0,
        "branch_calibration": [
            branch_calibration(reader.contract),
            branch_calibration(reader.contract["secondary_contract"]),
        ],
        "thresholds": thresholds,
        "field_thresholds": field_thresholds,
        "method": "fields",
        "metrics": metrics,
        "precalibration_failures": failures,
    }
    output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "thresholds": thresholds,
                "field_thresholds": field_thresholds,
                "metrics": metrics,
            },
            indent=2,
        )
    )


def export_consensus(args):
    import onnx

    from research.models.chord_evidence import ChordEvidenceConsensus

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    models = []
    contracts = []
    for directory in [args.primary, args.secondary]:
        checkpoint = directory / "chord-evidence.safetensors"
        model = ChordEvidenceNetwork.from_weights(load_file(str(checkpoint))).eval()
        models.append(model)
        contract = json.loads((directory / "calibration.json").read_text())["contract"]
        contracts.append(contract)
    agreement = None
    if args.agreement_calibration:
        agreement = json.loads(args.agreement_calibration.read_text())
        if (
            agreement["calibration_split"]
            != "validation_only_shape_and_document_disjoint"
        ):
            raise ValueError("Agreement calibration requires validation data")
        if agreement["branch_calibration"] != [
            branch_calibration(contract) for contract in contracts
        ]:
            raise ValueError("Reader thresholds changed after agreement calibration")
    model = ChordEvidenceConsensus(
        *models,
        contracts,
        agreement["thresholds"] if agreement else None,
        agreement.get("field_thresholds") if agreement else None,
    ).eval()
    names = [*OUTPUTS, *("secondary_" + name for name in OUTPUTS), "agreement"]
    path = args.output / "chord-evidence.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 256, 256),
        str(path),
        input_names=["images"],
        output_names=names,
        dynamic_axes={"images": {0: "batch"}, **{name: {0: "batch"} for name in names}},
        opset_version=17,
        dynamo=False,
    )
    contract = {
        **contracts[0],
        "reader": "chord_evidence",
        "consensus": "calibrated_exact_diagram_v1",
        "secondary_contract": contracts[1],
        "agreement_outputs": ["geometry", "full"],
    }
    if agreement:
        contract["agreement_calibration"] = agreement
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(
        graph, {"guitarocr.chord_evidence": json.dumps(contract, separators=(",", ":"))}
    )
    onnx.save(graph, str(path))
    (args.output / "calibration.json").write_text(
        json.dumps(
            {
                "contract": contract,
                "source": "unchanged per-branch validation calibration",
            },
            indent=2,
        )
    )
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_size": 256,
        "outputs": names,
        "preprocessing": contract["preprocessing"],
        "primary": str(args.primary),
        "secondary": str(args.secondary),
        "onnx_bytes": path.stat().st_size,
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    for name in ("reader", "consensus"):
        task = tasks.add_parser(name)
        task.add_argument("--model", type=Path, required=True)
        task.add_argument("--data", type=Path, default=Path("database/score"))
        task.add_argument("--synthetic", type=int, default=2000)
        task.add_argument("--font", action="append")
        if name == "consensus":
            task.add_argument("--output", type=Path, required=True)
    exporter = tasks.add_parser("export")
    exporter.add_argument("--primary", type=Path, required=True)
    exporter.add_argument("--secondary", type=Path, required=True)
    exporter.add_argument("--agreement-calibration", type=Path,
                          help="Validation calibration of agreeing predictions")
    exporter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task == "export":
        export_consensus(args)
        return
    data = ChordEvidenceData(args.data, "validation", args.synthetic, fonts=args.font)
    if args.task == "reader":
        calibrate_reader(args.model, data)
    else:
        calibrate_consensus(args.model, data, args.output)


if __name__ == "__main__":
    main()
