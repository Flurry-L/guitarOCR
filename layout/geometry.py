from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image

from layout.tab_geometry import detect_tab_boundaries_for_lines, detect_tab_geometry
from layout.score_geometry import detect_score_tab_geometry


def _notation_mode(layout: str) -> str:
    values = {"tab_only": "tab", "score_only": "notation", "score_tab": "both"}
    if layout not in values:
        raise ValueError(f"Unsupported or undetected page layout: {layout}")
    return values[layout]


def _clean_notation_boundaries(boundaries: list[float]) -> list[float]:
    """Merge a score note stem that crosses all five staff lines.

    A real barline normally ends at the outer staff lines, but some short
    stems do as well. Such a stem splits one normal-width measure into one
    very narrow and one residual interval. Repeats and genuinely short bars
    are retained unless the two adjacent intervals recombine to the page's
    normal measure width.
    """

    values = list(boundaries)
    changed = True
    while changed and len(values) >= 4:
        changed = False
        widths = np.diff(values)
        median = float(np.median(widths))
        if median <= 0:
            break
        candidates = []
        for index in range(1, len(values) - 1):
            left = float(values[index] - values[index - 1])
            right = float(values[index + 1] - values[index])
            combined = left + right
            if min(left, right) >= 0.50 * median:
                continue
            if not 0.70 * median <= combined <= 1.45 * median:
                continue
            candidates.append((min(left, right) / median, index))
        if candidates:
            _ratio, index = min(candidates)
            del values[index]
            changed = True
    return values


def _measure_boxes(page: Image.Image, mode: str) -> list[dict[str, Any]]:
    boxes = []
    if mode == "both":
        systems = detect_score_tab_geometry(page)
        for system_index, system in enumerate(systems):
            y0 = float(system["score_line_y"][0]) - 4.0 * float(system["score_spacing"])
            y1 = float(system["tab_string_y"][-1]) + 3.0 * float(system["tab_spacing"])
            for measure_index in range(len(system["boundaries"]) - 1):
                left = float(system["boundaries"][measure_index])
                right = float(system["boundaries"][measure_index + 1])
                boxes.append({
                    "system_index": system_index,
                    "system_measure_index": measure_index,
                    "bbox": [left, y0, right - left, y1 - y0],
                })
        return boxes

    staffs = detect_tab_geometry(
        page, minimum_string_spacing=6.0 if mode == "notation" else None
    )
    for system_index, staff in enumerate(staffs):
        lines = [float(value) for value in staff["string_y"]]
        spacing = float(staff["spacing"])
        if mode == "tab" and not 4 <= len(lines) <= 8:
            continue
        if mode == "notation" and len(lines) != 5:
            continue
        y0 = lines[0] - (4.0 if mode == "tab" else 5.0) * spacing
        y1 = lines[-1] + (3.0 if mode == "tab" else 4.0) * spacing
        boundaries = [float(value) for value in staff["boundaries"]]
        if mode == "notation":
            boundaries = _clean_notation_boundaries(boundaries)
        for measure_index in range(len(boundaries) - 1):
            left, right = boundaries[measure_index:measure_index + 2]
            boxes.append({
                "system_index": system_index,
                "system_measure_index": measure_index,
                "bbox": [left, y0, right - left, y1 - y0],
            })
    return boxes


def _hybrid_tab_pdf_boxes(
    page: Image.Image,
    pixel_boxes: list[dict[str, Any]],
    vector_systems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add whole TAB systems missed by pixels, retaining raster barlines."""

    groups: dict[int, list[dict[str, Any]]] = {}
    for box in pixel_boxes:
        groups.setdefault(int(box["system_index"]), []).append(box)
    combined: list[dict[str, Any]] = []
    pixel_centers = []
    for boxes in groups.values():
        boxes.sort(key=lambda item: int(item["system_measure_index"]))
        top = min(float(item["bbox"][1]) for item in boxes)
        bottom = max(float(item["bbox"][1] + item["bbox"][3]) for item in boxes)
        pixel_centers.append((0.5 * (top + bottom), boxes))
        combined.append({"center": 0.5 * (top + bottom), "boxes": boxes})

    matched_pixel: set[int] = set()
    for vector in vector_systems:
        string_y = [float(value) for value in vector["string_y"]]
        spacing = float(vector["spacing"])
        # Pixel crop centers sit half a string spacing above the mean string
        # position because the crop reserves four spaces above and three
        # below.  Match in that same coordinate system.
        center = float(np.mean(string_y)) - 0.5 * spacing
        candidates = [
            (abs(pixel_center - center), index)
            for index, (pixel_center, _boxes) in enumerate(pixel_centers)
            if index not in matched_pixel
        ]
        matched_index: int | None = None
        if candidates:
            distance, pixel_index = min(candidates)
            if distance <= max(20.0, 1.75 * spacing):
                matched_index = pixel_index
                matched_pixel.add(pixel_index)
                for box in pixel_centers[pixel_index][1]:
                    box["geometry_source"] = "pixel_staff+pdf_vector_system_check"

        # The vector rows already prove that this is a real TAB system, so a
        # dense run of fret glyphs must not fail the raster density guard.
        boundaries = detect_tab_boundaries_for_lines(
            page, string_y, minimum_horizontal_density=0.0
        )
        if len(boundaries) < 2:
            continue
        top = string_y[0] - 4.0 * spacing
        bottom = string_y[-1] + 3.0 * spacing
        boxes = []
        for measure_index, (left, right) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            boxes.append({
                "system_index": -1,
                "system_measure_index": measure_index,
                "bbox": [left, top, right - left, bottom - top],
                "geometry_source": vector["geometry_source"],
            })
        if matched_index is not None:
            pixel_group = pixel_centers[matched_index][1]
            # Exact vector-derived string rows can reveal a barline that was
            # missed when the pixel chain snapped to a neighbouring fragment.
            # Keep the established pixel result unless the seeded pass finds
            # strictly more complete measure intervals.
            if len(boxes) > len(pixel_group):
                for group in combined:
                    if group["boxes"] is pixel_group:
                        group["boxes"] = boxes
                        group["center"] = center
                        break
            continue
        combined.append({"center": center, "boxes": boxes})

    result = []
    for system_index, group in enumerate(sorted(combined, key=lambda item: item["center"])):
        for box in group["boxes"]:
            box["system_index"] = system_index
            box.setdefault("geometry_source", "pixel_staff_fallback")
            result.append(box)
    return result
