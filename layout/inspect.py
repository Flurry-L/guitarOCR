from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from shared.defaults import LAYOUT_MODEL

from PIL import Image, ImageDraw

from layout.postprocess import order_measure_boxes, refine_measure_boxes


def inspect(pages: Path, model_dir: Path, output: Path, threshold: float) -> dict:
    from paddlex import create_model

    model = create_model(model_name="PP-DocLayoutV3", model_dir=str(model_dir), device="gpu:0")
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for page_path in sorted(pages.glob("page_*.png")):
        result = next(iter(model.predict(
            str(page_path), batch_size=1, threshold=threshold,
            layout_shape_mode="rect", filter_overlap_boxes=False,
        )))
        boxes = result.json["res"]["boxes"]
        measures = order_measure_boxes(boxes, minimum_score=threshold)
        with Image.open(page_path) as opened:
            overlay = opened.convert("RGB")
        measures = refine_measure_boxes(overlay, measures)
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            left, top, right, bottom = box["coordinate"]
            if box["label"] == "tempo_region":
                draw.rectangle((left, top, right, bottom), outline="#a2640b", width=3)
        for index, box in enumerate(measures, start=1):
            left, top, right, bottom = box["coordinate"]
            draw.rectangle((left, top, right, bottom), outline="#d33232", width=3)
            draw.text((left + 3, top + 3), str(index), fill="#d33232")
        overlay.save(output / page_path.name)
        records.append({
            "page": page_path.name,
            "counts": dict(Counter(box["label"] for box in boxes)),
            "ordered_measure_count": len(measures),
            "ordered_measures": measures,
            "boxes": boxes,
        })
    manifest = {"threshold": threshold, "pages": records}
    (output / "detections.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"pages": len(records), "ordered_measures": sum(
        record["ordered_measure_count"] for record in records
    ), "counts": dict(Counter(
        box["label"] for record in records for box in record["boxes"]
    ))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PP-DocLayoutV3 page detections")
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=LAYOUT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(inspect(args.pages, args.model_dir, args.output, args.threshold), indent=2))


if __name__ == "__main__":
    main()
