from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from layout.postprocess import order_measure_boxes, refine_measure_boxes


def evaluate(model_dir: Path, dataset_dir: Path, output: Path, postprocess: bool = False) -> dict:
    from paddlex import create_model
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO(str(dataset_dir / "annotations" / "instance_val.json"))
    model = create_model(model_name="PP-DocLayoutV3", model_dir=str(model_dir), device="gpu:0")
    detections = []
    count_correct = 0
    raw_count_correct = 0
    changed_count_pages = 0
    changed_examples = []
    for image_id in sorted(truth.imgs):
        image = truth.imgs[image_id]
        image_path = dataset_dir / "images" / image["file_name"]
        result = next(iter(model.predict(
            str(image_path), batch_size=1, threshold=0.0,
            layout_shape_mode="rect", filter_overlap_boxes=False,
        )))
        boxes = result.json["res"]["boxes"]
        if postprocess:
            raw_measures = order_measure_boxes(boxes, 0.25)
            with Image.open(image_path) as opened:
                measures = refine_measure_boxes(opened, raw_measures)
            expected_count = len(truth.getAnnIds(imgIds=[image_id], catIds=[1]))
            raw_count_correct += len(raw_measures) == expected_count
            count_correct += len(measures) == expected_count
            if len(measures) != len(raw_measures):
                changed_count_pages += 1
                if len(changed_examples) < 12:
                    changed_examples.append({
                        "image_id": image_id,
                        "raw": len(raw_measures),
                        "refined": len(measures),
                        "expected": expected_count,
                    })
            boxes = [box for box in boxes if box["label"] == "tempo_region" and box["score"] >= 0.25]
            boxes.extend(measures)
        for box in boxes:
            left, top, right, bottom = (float(value) for value in box["coordinate"])
            if right <= left or bottom <= top:
                continue
            detections.append({
                "image_id": image_id,
                "category_id": int(box["cls_id"]) + 1,
                "bbox": [left, top, right - left, bottom - top],
                "score": float(box["score"]),
            })
    if not detections:
        raise ValueError("The layout model produced no detections")
    predictions = truth.loadRes(detections)
    evaluator = COCOeval(truth, predictions, "bbox")
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
            "ground_truth_boxes": len(truth.getAnnIds(catIds=[category_id])),
        }
    metrics = {
        "ap_50_95": float(evaluator.stats[0]),
        "ap_50": float(evaluator.stats[1]),
        "categories": categories,
        "detections": len(detections),
    }
    if postprocess:
        metrics["exact_measure_count_pages"] = count_correct
        metrics["raw_exact_measure_count_pages"] = raw_count_correct
        metrics["changed_count_pages"] = changed_count_pages
        metrics["changed_examples"] = changed_examples
        metrics["total_pages"] = len(truth.imgs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate score layout detection by category")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("database/gp8_measure_sequence_v2/datasets/layout"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_dir, args.dataset_dir, args.output, args.postprocess), indent=2))


if __name__ == "__main__":
    main()
