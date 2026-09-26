from __future__ import annotations

from bisect import bisect_right
from statistics import median

import numpy as np
from PIL import Image
from shared.layout_labels import is_measure, measure_mode


def _coordinates(box: dict) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in box["coordinate"])


def _same_row(box: dict, row: list[dict]) -> bool:
    left, top, right, bottom = _coordinates(box)
    center = (top + bottom) / 2
    row_centers = []
    row_heights = []
    for member in row:
        _, row_top, _, row_bottom = _coordinates(member)
        row_centers.append((row_top + row_bottom) / 2)
        row_heights.append(row_bottom - row_top)
    return abs(center - median(row_centers)) <= 0.35 * min(bottom - top, median(row_heights))


def _select_nonoverlapping(row: list[dict]) -> list[dict]:
    ordered = sorted(row, key=lambda box: (_coordinates(box)[2], _coordinates(box)[0]))
    right_edges = [_coordinates(box)[2] for box in ordered]
    scores = [0.0]
    choices: list[list[int]] = [[]]
    for index, box in enumerate(ordered):
        left = _coordinates(box)[0]
        previous = bisect_right(right_edges, left + 6, hi=index)
        confidence = float(box["score"])
        selected_score = scores[previous] + confidence * confidence + 0.15
        if selected_score > scores[-1]:
            scores.append(selected_score)
            choices.append([*choices[previous], index])
        else:
            scores.append(scores[-1])
            choices.append(choices[-1])
    selected = sorted((ordered[index] for index in choices[-1]), key=lambda box: _coordinates(box)[0])
    for first, second in zip(selected, selected[1:]):
        first_box = list(_coordinates(first))
        second_box = list(_coordinates(second))
        if first_box[2] > second_box[0]:
            boundary = (first_box[2] + second_box[0]) / 2
            first_box[2] = boundary
            second_box[0] = boundary
            first["coordinate"] = first_box
            second["coordinate"] = second_box
    return selected


def order_measure_boxes(boxes: list[dict], minimum_score: float = 0.2) -> list[dict]:
    candidates = [
        {**box, "coordinate": list(_coordinates(box)),
         **({"mode": measure_mode(box["label"])} if measure_mode(box.get("label", "")) else {})}
        for box in boxes
        if is_measure(box.get("label", ""))
        and float(box.get("score", 0)) >= minimum_score
        and _coordinates(box)[2] > _coordinates(box)[0]
        and _coordinates(box)[3] > _coordinates(box)[1]
    ]
    candidates.sort(key=lambda box: ((_coordinates(box)[1] + _coordinates(box)[3]) / 2, _coordinates(box)[0]))
    rows: list[list[dict]] = []
    for box in candidates:
        matching = next((row for row in rows if _same_row(box, row)), None)
        if matching is None:
            rows.append([box])
        else:
            matching.append(box)
    rows.sort(key=lambda row: median((_coordinates(box)[1] + _coordinates(box)[3]) / 2 for box in row))
    result = []
    for system_index, row in enumerate(rows):
        for measure_index, box in enumerate(_select_nonoverlapping(row)):
            left, top, right, bottom = _coordinates(box)
            result.append({
                **box,
                "system_index": system_index,
                "system_measure_index": measure_index,
                "bbox": [left, top, right - left, bottom - top],
            })
    return result


def refine_measure_boxes(image: Image.Image, boxes: list[dict]) -> list[dict]:
    pixels = np.asarray(image.convert("L"))

    def has_barline(x: float, top: float, bottom: float) -> bool:
        height = bottom - top
        if height < 35:
            return False
        y0 = max(0, int(top + 0.04 * height))
        y1 = min(pixels.shape[0], int(bottom - 0.04 * height))
        x0 = max(0, int(x) - 18)
        x1 = min(pixels.shape[1], int(x) + 19)
        if y1 <= y0 or x1 <= x0:
            return False
        ink = pixels[y0:y1, x0:x1] < 150
        longest = 0
        for column in ink.T:
            edges = np.diff(np.r_[False, column, False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            if len(starts):
                longest = max(longest, int(np.max(ends - starts)))
        return longest >= max(35, min(60, 0.34 * height))

    rows: dict[int, list[dict]] = {}
    for box in boxes:
        rows.setdefault(int(box["system_index"]), []).append(box)
    row_sizes = [len(rows[index]) for index in sorted(rows)]
    typical_count = round(median(row_sizes[1:-1] if len(row_sizes) > 2 else row_sizes)) if row_sizes else 0
    result = []
    for system_index in sorted(rows):
        row = sorted(rows[system_index], key=lambda box: _coordinates(box)[0])
        if not 3 <= len(row) < typical_count:
            result.extend(row)
            continue
        widths = [_coordinates(box)[2] - _coordinates(box)[0] for box in row]
        usual_width = median(widths)
        merged: list[dict] = []
        for box in row:
            current = {**box, "coordinate": list(_coordinates(box))}
            if merged:
                previous = merged[-1]
                if previous.get("mode") != current.get("mode"):
                    merged.append(current)
                    continue
                left, top, right, bottom = _coordinates(previous)
                next_left, next_top, next_right, next_bottom = _coordinates(current)
                gap = next_left - right
                common_top = max(top, next_top)
                common_bottom = min(bottom, next_bottom)
                if common_bottom > common_top:
                    boundary = (right + next_left) / 2
                    if 0.35 * usual_width <= gap <= 1.5 * usual_width:
                        middle = (right + next_left) / 2
                        staff_ink = np.count_nonzero(
                            pixels[int(common_top + 0.2 * (common_bottom - common_top)):
                                   int(common_bottom - 0.2 * (common_bottom - common_top)),
                                   min(pixels.shape[1] - 1, int(middle))] < 150
                        )
                        if (staff_ink >= 3 and has_barline(right, common_top, common_bottom)
                                and has_barline(next_left, common_top, common_bottom)):
                            merged.append({
                                **current,
                                "score": min(float(previous["score"]), float(current["score"])),
                                "coordinate": [right, common_top, next_left, common_bottom],
                                "geometry_source": "barline_gap_recovery",
                            })
                        elif (not has_barline(right, common_top, common_bottom)
                              and not has_barline(next_left, common_top, common_bottom)):
                            previous["coordinate"] = [left, min(top, next_top), next_right, max(bottom, next_bottom)]
                            previous["score"] = max(float(previous["score"]), float(current["score"]))
                            previous["geometry_source"] = "barline_merge"
                            continue
                    elif gap <= 0.35 * usual_width and not has_barline(boundary, common_top, common_bottom):
                        previous["coordinate"] = [left, min(top, next_top), next_right, max(bottom, next_bottom)]
                        previous["score"] = max(float(previous["score"]), float(current["score"]))
                        previous["geometry_source"] = "barline_merge"
                        continue
            merged.append(current)
        for measure_index, box in enumerate(merged):
            left, top, right, bottom = _coordinates(box)
            result.append({
                **box,
                "system_measure_index": measure_index,
                "bbox": [left, top, right - left, bottom - top],
            })
    return result
