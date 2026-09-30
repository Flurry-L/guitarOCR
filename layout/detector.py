from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from layout.postprocess import order_measure_boxes, refine_measure_boxes, deduplicate_pitch_boxes
from shared.layout_labels import mode_vote, PITCH_REGION_LABELS


def refine_small_regions(image, boxes, model, threshold=.25, batch_size=4):
    """Read small annotations at a second scale on unusually large pages.

    Full-page detection owns measure geometry. Four overlapping detail views
    supplement only clefs, text and diagrams, without splitting music bars.
    """
    width, height = image.size
    if width <= 2400 and height <= 3400:
        return boxes
    import numpy as np

    tile_width, tile_height = math.ceil(width * .55), math.ceil(height * .55)
    origins = [(x, y) for y in (0, height - tile_height) for x in (0, width - tile_width)]
    tiles = [np.asarray(image.crop((x, y, x + tile_width, y + tile_height)).convert('RGB'))
             [:, :, ::-1].copy() for x, y in origins]
    kinds = {'clef_region', 'annotation_region', 'transposition_region', 'tempo_region'}
    candidates = [box for box in boxes if box.get('label') in kinds]
    predictions = model.predict(tiles, batch_size=batch_size, threshold=threshold,
                                layout_shape_mode='rect', filter_overlap_boxes=False)
    for (x, y), prediction in zip(origins, predictions, strict=True):
        for box in prediction.json['res']['boxes']:
            if box.get('label') not in kinds or float(box['score']) < threshold:
                continue
            a, b, c, d = box['coordinate']
            # A partial marker at an internal tile edge cannot establish a
            # pitch instruction. Its complete view is in the overlapping tile.
            if ((x and a < 3) or (y and b < 3)
                    or (x + tile_width < width and c > tile_width - 3)
                    or (y + tile_height < height and d > tile_height - 3)):
                continue
            candidates.append({**box, 'coordinate': [float(a + x), float(b + y),
                                                     float(c + x), float(d + y)],
                               'geometry_source': 'detail_crop'})
    kept = []
    for box in sorted(candidates, key=lambda box: float(box['score']), reverse=True):
        x, y, right, bottom = box['coordinate']
        area = (right - x) * (bottom - y)
        duplicate = False
        for other in kept:
            if other['label'] != box['label']:
                continue
            a, b, c, d = other['coordinate']
            intersection = max(0, min(right, c) - max(x, a)) * max(0, min(bottom, d) - max(y, b))
            if intersection / max(1, area + (c - a) * (d - b) - intersection) > .4:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return [box for box in boxes if box.get('label') not in kinds] + kept


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
                boxes = refine_small_regions(image, boxes, model, threshold, batch_size)
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
