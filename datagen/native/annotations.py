"""Visible labels and measure bounds from Guitar Pro's native export."""

import unicodedata
from collections import defaultdict
from typing import Any


def measure_boxes(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if layout.get("schema") == "gpomr.render-layout":
        boxes = {}
        for system in layout["systems"]:
            for measure in system["measure_boxes"]:
                index = int(measure["measure_index"])
                if index in boxes:
                    raise ValueError(f"Duplicate native measure index: {index}")
                boxes[index] = {
                    "page": int(system["page"]),
                    "bbox_mm": measure["bbox_mm"],
                    "staff_types": [],
                }
        return boxes
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in layout.get("records", []):
        if record.get("kind") != "bar":
            continue
        index = int(record.get("first_bar_index", -1))
        if index >= 0:
            grouped[index].append(record)
    boxes = {}
    for index, records in grouped.items():
        pages = {int(record["page"]) for record in records}
        if len(pages) != 1:
            continue
        coordinates = [
            record.get("page_bbox_mm") or record["bbox_mm"] for record in records
        ]
        left = min(float(value[0]) for value in coordinates)
        top = min(float(value[1]) for value in coordinates)
        right = max(float(value[0]) + float(value[2]) for value in coordinates)
        bottom = max(float(value[1]) + float(value[3]) for value in coordinates)
        boxes[index] = {
            "page": pages.pop(),
            "bbox_mm": [left, top, right - left, bottom - top],
            "staff_types": sorted(
                {int(record.get("staff_type", -1)) for record in records}
            ),
        }
    return boxes


def printed_text(value: str) -> str:
    # PDF font mappings may encode Chinese glyphs as compatibility radicals.
    # These supplemental radical glyphs used by GP8's CJK font have no NFKC
    # decomposition. Normalize only the visibility check, never the OCR label.
    value = value.translate(str.maketrans({"⻓": "长", "⻘": "青", "⻛": "风"}))
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def header_target(score: dict) -> dict:
    document = score["document"]
    metadata = document["metadata"]
    properties = metadata["properties"]
    header = metadata["first_page_header"]
    staff = score["tracks"][0]["staves"][0]
    return {
        "title": (properties["title"] or "").strip() or None
        if header["title"]["visible"]
        else None,
        "artist": (properties["artist"] or "").strip() or None
        if header["artist"]["visible"]
        else None,
        "tuning_name": (staff["tuning_displayed_label"] or "").strip() or None
        if staff["tuning_label_visible"]
        else None,
    }


def tempo_target(score: dict, indication: dict) -> dict | None:
    document = score["document"]
    if indication["source"] == "initial":
        tempo = document["tempo"]
    else:
        matches = [
            row
            for row in document["tempo_automations"]
            if row["bar_index"] == indication["master_measure_index"]
            and abs(float(row["position"]) - float(indication["position"])) < 1e-5
        ]
        if len(matches) != 1:
            return None
        tempo = matches[0]
    if not tempo["visible"] or tempo["unit_name"] != "Quarter":
        return None
    return {"tempo_quarter": int(tempo["value"])}
