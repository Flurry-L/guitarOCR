from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from layout.postprocess import order_measure_boxes, refine_measure_boxes, deduplicate_pitch_boxes
from shared.layout_labels import mode_vote, PITCH_REGION_LABELS


def detect_pages(
    pages: list[str], model_dir: Path, threshold: float, include_tempo: bool = False,
    *, model=None, batch_size: int = 4,
) -> list[list[dict] | dict]:
    if model is None:
        import paddle
        from paddlex import create_model

        device = "gpu:0" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() else "cpu"
        model = create_model(model_name="PP-DocLayoutV3", model_dir=str(model_dir), device=device)
    results = []
    predictions = model.predict(
            pages, batch_size=batch_size, threshold=threshold,
            layout_shape_mode="rect", filter_overlap_boxes=False,
        ) if pages else []
    for page, prediction in zip(pages, predictions, strict=True):
        boxes = deduplicate_pitch_boxes(prediction.json["res"]["boxes"])
        ordered = order_measure_boxes(boxes, threshold)
        with Image.open(page) as image:
            measures = refine_measure_boxes(image, ordered)
        if include_tempo:
            tempos = [
                box for box in boxes
                if box.get("label") == "tempo_region"
                and float(box.get("score", 0)) >= threshold
            ]
            pitches = [box for box in boxes if box.get("label") in PITCH_REGION_LABELS
                       and float(box.get("score", 0)) >= threshold]
            results.append({"measures": measures, "tempo_regions": tempos,
                            "pitch_regions": pitches, **mode_vote(measures)})
        else:
            results.append(measures)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate ordered score measures on page images")
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--include-tempo", action="store_true")
    args = parser.parse_args()
    pages = json.loads(args.pages.read_text(encoding="utf-8"))
    results = detect_pages(pages, args.model_dir, args.threshold, args.include_tempo)
    args.output.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
