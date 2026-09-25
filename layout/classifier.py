from __future__ import annotations

from collections import Counter

from PIL import Image

from layout.tab_geometry import detect_tab_geometry
from layout.score_geometry import detect_score_tab_geometry


LAYOUTS = ("score_tab", "tab_only", "score_only")


def _classify_fixed_scale(page: Image.Image) -> dict:
    paired = detect_score_tab_geometry(page)
    if paired:
        return {
            "layout": "score_tab", "confidence": 0.99,
            "systems": len(paired), "method": "paired_five_and_tab_staffs",
        }
    staffs = detect_tab_geometry(page, minimum_string_spacing=6.0)
    if not staffs:
        return {"layout": "unknown", "confidence": 0.0, "systems": 0, "method": "no_staffs"}
    counts = Counter(int(staff["string_count"]) for staff in staffs)
    score_like = sum(
        1 for staff in staffs
        if int(staff["string_count"]) == 5 and float(staff["spacing"]) < 15.0
    )
    tab_like = sum(
        1 for staff in staffs
        if 4 <= int(staff["string_count"]) <= 8
        and (int(staff["string_count"]) != 5 or float(staff["spacing"]) >= 15.0)
    )
    if tab_like > score_like:
        layout = "tab_only"
        support = tab_like
    else:
        layout = "score_only"
        support = score_like
    return {
        "layout": layout,
        "confidence": support / max(1, len(staffs)),
        "systems": len(staffs),
        "staff_line_counts": dict(sorted(counts.items())),
        "method": "unpaired_staff_spacing_and_line_count",
    }


def classify_notation_layout(page: Image.Image) -> dict:
    """Classify TuxGuitar print layouts, normalizing common PDF raster scales."""
    candidates: list[tuple[float, Image.Image]] = [(1.0, page)]
    longest = max(page.size)
    if longest >= 600:
        scale = 2000.0 / longest
        if abs(scale - 1.0) >= 0.08:
            resized = page.resize(
                (max(1, round(page.width * scale)), max(1, round(page.height * scale))),
                Image.Resampling.BILINEAR,
            )
            candidates.append((scale, resized))

    results = []
    for scale, candidate in candidates:
        result = _classify_fixed_scale(candidate)
        result["analysis_scale"] = scale
        results.append(result)
    # Paired score+TAB geometry is stronger evidence than an unpaired staff
    # guess. Otherwise choose the result with the greatest supported fraction.
    return max(
        results,
        key=lambda item: (
            item["layout"] == "score_tab",
            float(item["confidence"]),
            int(item["systems"]),
        ),
    )
