from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from PIL import Image

from layout.postprocess import order_measure_boxes, refine_measure_boxes
from shared.layout_labels import is_measure, measure_mode, mode_vote


def _type_matches(boxes: list[list[float]], measures: list[dict], mode: str) -> dict:
    """One-to-one IoU 0.50 matches, ignoring class until a box is matched."""
    import numpy as np

    truth = np.asarray(boxes, dtype=float).reshape(-1, 4)
    truth[:, 2:] += truth[:, :2]
    available = np.ones(len(truth), dtype=bool)
    confusion = {}
    for measure in sorted(measures, key=lambda row: row["score"], reverse=True):
        if not available.any():
            break
        prediction = np.asarray(measure["coordinate"], dtype=float)
        extent = np.maximum(0, np.minimum(truth[:, 2:], prediction[2:]) - np.maximum(truth[:, :2], prediction[:2]))
        intersection = extent.prod(axis=1)
        union = (truth[:, 2:] - truth[:, :2]).prod(axis=1) + (prediction[2:] - prediction[:2]).prod() - intersection
        iou = np.where(available, intersection / np.maximum(union, 1e-9), 0)
        index = int(iou.argmax())
        if iou[index] >= 0.5:
            available[index] = False
            predicted = measure_mode(measure.get("label", "")) or measure.get("mode") or "unknown"
            confusion[predicted] = confusion.get(predicted, 0) + 1
    return {"matched_measures": sum(confusion.values()), "correct_type_measures": confusion.get(mode, 0), "measure_type_confusion": confusion}


def _coco_metrics(truth, predictions, image_ids: list[int]) -> dict:
    from pycocotools.cocoeval import COCOeval

    evaluator = COCOeval(truth, predictions, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    categories = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, category_index, 0, 2]
        recall = evaluator.eval["recall"][0, category_index, 0, 2]
        valid_precision = precision[precision >= 0]
        valid_recall = recall[recall >= 0]
        categories[truth.cats[category_id]["name"]] = {
            "ap_50_95": float(valid_precision.mean()) if len(valid_precision) else None,
            "recall_50": float(valid_recall.max()) if len(valid_recall) else None,
            "ground_truth_boxes": len(
                truth.getAnnIds(imgIds=image_ids, catIds=[category_id])
            ),
        }
    return {
        "ap_50_95": float(evaluator.stats[0]),
        "ap_50": float(evaluator.stats[1]),
        "categories": categories,
        "total_pages": len(image_ids),
    }


def _count_metrics(rows: list[dict]) -> dict:
    sources = defaultdict(list)
    for row in rows:
        sources[(row["source_id"], row["mode"])].append(row)
    confusion = {mode: {} for mode in ("tab", "notation", "both")}
    for row in rows:
        predicted = row.get("predicted_mode") or "unknown"
        cell = confusion[row["mode"]]
        cell[predicted] = cell.get(predicted, 0) + 1
    correct = sum(row.get("predicted_mode") == row["mode"] for row in rows)
    matched = sum(row["matched_measures"] for row in rows)
    correct_types = sum(row["correct_type_measures"] for row in rows)
    return {
        "matched_measure_type": {
            "iou": 0.5,
            "matched_measures": matched,
            "correct_measures": correct_types,
            "accuracy": correct_types / matched if matched else None,
        },
        "mode_classification": {
            "correct_pages": correct,
            "total_pages": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "unknown_pages": sum(row.get("predicted_mode") is None for row in rows),
            "confusion": confusion,
            "correct_sources": sum(
                all(row.get("predicted_mode") == row["mode"] for row in group)
                for group in sources.values()
            ),
        },
        "exact_measure_count_pages": sum(
            row["predicted"] == row["expected"] for row in rows
        ),
        "raw_exact_measure_count_pages": sum(
            row["raw"] == row["expected"] for row in rows
        ),
        "changed_count_pages": sum(row["predicted"] != row["raw"] for row in rows),
        "exact_measure_count_sources": sum(
            all(row["predicted"] == row["expected"] for row in group)
            for group in sources.values()
        ),
        "total_sources": len(sources),
    }


def evaluate(
    model_dir: Path,
    dataset_dir: Path,
    output: Path,
    postprocess: bool = False,
    *,
    split: str = "val",
    modes: list[str] | None = None,
    device: str = "gpu:0",
    threshold: float = 0.25,
) -> dict:
    from paddlex import create_model
    from pycocotools.coco import COCO

    annotation_file = dataset_dir / "annotations" / f"instance_{split}.json"
    truth = COCO(str(annotation_file))
    typed_truth = truth if any(measure_mode(c["name"]) for c in truth.cats.values()) else None
    original_categories = {row["id"]: row["name"] for row in truth.cats.values()}
    # Compare localization with older two-class models independently of type
    # correctness. The separate typed AP also penalizes a wrong notation label.
    collapsed = COCO()
    collapsed.dataset = {
        **truth.dataset,
        "categories": [
            {"id": 1, "name": "measure", "supercategory": "score"},
            {"id": 2, "name": "tempo_region", "supercategory": "score"},
        ],
        "annotations": [
            {**row, "category_id": 1 if is_measure(original_categories[row["category_id"]]) else 2}
            for row in truth.dataset["annotations"]
        ],
    }
    collapsed.createIndex()
    truth = collapsed
    image_ids = sorted(
        image_id
        for image_id, row in truth.imgs.items()
        if not modes or row.get("mode", "tab") in modes
    )
    if not image_ids:
        raise ValueError("No pages match the requested split and modes")
    model = create_model(
        model_name="PP-DocLayoutV3", model_dir=str(model_dir), device=device
    )
    detections, typed_detections, counts = [], [], []
    by_mode = defaultdict(list)
    category_ids = {cat["name"]: cat_id for cat_id, cat in truth.cats.items()}
    typed_ids = {cat["name"]: cat_id for cat_id, cat in typed_truth.cats.items()} if typed_truth else {}
    for index, image_id in enumerate(image_ids, 1):
        metadata = truth.imgs[image_id]
        mode = metadata.get("mode", "tab")
        by_mode[mode].append(image_id)
        image_path = dataset_dir / "images" / metadata["file_name"]
        result = next(
            iter(
                model.predict(
                    str(image_path),
                    batch_size=1,
                    threshold=0.0,
                    layout_shape_mode="rect",
                    filter_overlap_boxes=False,
                )
            )
        )
        boxes = result.json["res"]["boxes"]
        raw_measures = order_measure_boxes(boxes, threshold)
        measures = raw_measures
        if postprocess:
            with Image.open(image_path) as opened:
                measures = refine_measure_boxes(opened, raw_measures)
            boxes = [
                box
                for box in boxes
                if box["label"] == "tempo_region" and box["score"] >= threshold
            ]
            boxes.extend(measures)
        counts.append(
            {
                "image_id": image_id,
                "mode": mode,
                "source_id": metadata.get(
                    "source_id", metadata.get("family", str(image_id))
                ),
                "predicted_mode": mode_vote(measures)["mode"],
                "mode_vote_fraction": mode_vote(measures)["mode_vote_fraction"],
                **_type_matches(
                    [row["bbox"] for row in truth.loadAnns(truth.getAnnIds(imgIds=[image_id], catIds=[category_ids["measure"]]))],
                    measures, mode,
                ),
                "expected": len(
                    truth.getAnnIds(imgIds=[image_id], catIds=[category_ids["measure"]])
                ),
                "raw": len(raw_measures),
                "predicted": len(measures),
            }
        )
        for box in boxes:
            left, top, right, bottom = (float(value) for value in box["coordinate"])
            if right <= left or bottom <= top:
                continue
            detection = {
                    "image_id": image_id,
                    "category_id": category_ids["measure" if is_measure(box["label"]) else box["label"]],
                    "bbox": [left, top, right - left, bottom - top],
                    "score": float(box["score"]),
                }
            detections.append(detection)
            if typed_truth:
                if box["label"] not in typed_ids:
                    raise ValueError("Typed annotation evaluation requires a four-class model")
                typed_detections.append({**detection, "category_id": typed_ids[box["label"]]})
        if index % 50 == 0 or index == len(image_ids):
            print(f"Evaluated {index}/{len(image_ids)} pages", flush=True)
    if detections:
        predictions = truth.loadRes(detections)
    else:
        predictions = COCO()
        predictions.dataset = {**truth.dataset, "annotations": []}
        predictions.createIndex()
    typed_predictions = None
    if typed_truth:
        if typed_detections:
            typed_predictions = typed_truth.loadRes(typed_detections)
        else:
            typed_predictions = COCO()
            typed_predictions.dataset = {**typed_truth.dataset, "annotations": []}
            typed_predictions.createIndex()
    metrics = {
        **_coco_metrics(truth, predictions, image_ids),
        **_count_metrics(counts),
        "detections": len(detections),
        **({"typed_detection": _coco_metrics(typed_truth, typed_predictions, image_ids)} if typed_truth else {}),
        "by_mode": {
            mode: {
                **_coco_metrics(truth, predictions, ids),
                **_count_metrics([r for r in counts if r["mode"] == mode]),
                **({"typed_detection": _coco_metrics(typed_truth, typed_predictions, ids)} if typed_truth else {}),
            }
            for mode, ids in sorted(by_mode.items())
        },
        "split": split,
        "postprocess": postprocess,
        "localization_categories": ["measure", "tempo_region"],
        "model_provides_notation_type": bool(typed_detections) or any(row["predicted_mode"] for row in counts),
        "threshold": threshold,
        "model_dir": str(model_dir.resolve()),
        "model_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(model_dir.glob("inference.*"))
        },
        "annotations_sha256": sha256(annotation_file.read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("paddlex", "pycocotools")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".pages.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )
    output.with_suffix(".detections.json").write_text(
        json.dumps(typed_detections if typed_truth else detections) + "\n", encoding="utf-8"
    )
    if typed_truth:
        output.with_suffix(".localization-detections.json").write_text(json.dumps(detections) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate score layout detection by category and notation mode"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("database/gp8_measure_sequence_v2/datasets/layout"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--mode", action="append", dest="modes", choices=("tab", "notation", "both")
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
