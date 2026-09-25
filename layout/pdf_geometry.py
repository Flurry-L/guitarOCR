from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _horizontal_rows(page: Any) -> dict[float, list[tuple[float, float]]]:
    drawings = page.get_drawings()
    if not drawings:
        return {}

    def horizontal_count(drawing: dict[str, Any]) -> int:
        count = 0
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            if abs(end.x - start.x) > 20 and abs(end.y - start.y) < 0.15:
                count += 1
        return count

    # GP8 prints all staff lines in one path. Restricting extraction to that
    # dominant path prevents beams, text underlines and volta brackets from
    # becoming staff candidates.
    drawing = max(drawings, key=horizontal_count)
    rows: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for item in drawing.get("items", []):
        if item[0] != "l":
            continue
        start, end = item[1], item[2]
        # TAB string rules are split around every fret glyph.  A busy line can
        # therefore consist entirely of short fragments even though the union
        # of those fragments spans the full system width.  Keep the fragments
        # here and apply the full-row span filter below.
        if abs(end.x - start.x) <= 1.0 or abs(end.y - start.y) >= 0.15:
            continue
        y = round((float(start.y) + float(end.y)) / 2.0, 2)
        rows[y].append((min(float(start.x), float(end.x)), max(float(start.x), float(end.x))))
    return {
        y: segments for y, segments in rows.items()
        # A final system can contain only one narrow measure (about 27 mm in
        # GP8).  Parallel-row clustering below is the real staff guard, so do
        # not require every candidate row to span a normal full system.
        if max(segment[1] for segment in segments) - min(segment[0] for segment in segments) > 40
    }


def _staff_clusters(
    rows: dict[float, list[tuple[float, float]]], *, allow_missing_rows: bool = False
) -> list[list[float]]:
    clusters: list[list[float]] = []
    for y in sorted(rows):
        maximum_gap = 20.0 if allow_missing_rows else 10.0
        if not clusters or y - clusters[-1][-1] > maximum_gap:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    if not allow_missing_rows:
        return [cluster for cluster in clusters if 4 <= len(cluster) <= 8]
    result = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        spacing = _cluster_spacing(cluster)
        inferred_count = round((cluster[-1] - cluster[0]) / spacing) + 1
        if 4 <= inferred_count <= 8:
            result.append(cluster)
    return result


def _cluster_spacing(cluster: list[float]) -> float:
    return float(median(b - a for a, b in zip(cluster, cluster[1:])))


def _boundaries(
    cluster: list[float], rows: dict[float, list[tuple[float, float]]]
) -> list[float]:
    endpoints: Counter[float] = Counter()
    for y in cluster:
        for left, right in rows[y]:
            endpoints[round(left, 2)] += 1
            endpoints[round(right, 2)] += 1
    threshold = max(3, len(cluster) - 1)
    candidates = sorted(value for value, count in endpoints.items() if count >= threshold)
    merged: list[list[float]] = []
    for value in candidates:
        if not merged or value - merged[-1][-1] > 0.4:
            merged.append([value])
        else:
            merged[-1].append(value)
    values = [sum(group) / len(group) for group in merged]
    return values if len(values) >= 2 else []


def _page_boxes(page: Any, mode: str, dpi: int) -> list[dict[str, Any]]:
    rows = _horizontal_rows(page)
    clusters = _staff_clusters(rows, allow_missing_rows=mode == "tab")
    scale = dpi / 72.0
    staff_records: list[dict[str, Any]] = []
    for cluster in clusters:
        spacing = _cluster_spacing(cluster)
        line_count = (
            round((cluster[-1] - cluster[0]) / spacing) + 1
            if mode == "tab" else len(cluster)
        )
        lines = [cluster[0] + index * spacing for index in range(line_count)]
        kind = "score" if line_count == 5 and spacing < 5.5 else "tab"
        staff_records.append({
            "kind": kind,
            "lines": lines,
            "spacing": spacing,
            "boundaries": _boundaries(cluster, rows),
        })

    systems: list[dict[str, Any]] = []
    if mode == "notation":
        systems = [record for record in staff_records if record["kind"] == "score"]
    elif mode == "tab":
        systems = [record for record in staff_records if record["kind"] == "tab"]
    elif mode == "both":
        for index, score in enumerate(staff_records):
            if score["kind"] != "score":
                continue
            next_score = next(
                (record for record in staff_records[index + 1:] if record["kind"] == "score"),
                None,
            )
            following = [
                record for record in staff_records[index + 1:]
                if next_score is None or record["lines"][0] < next_score["lines"][0]
            ]
            tab = next((record for record in following if record["kind"] == "tab"), None)
            systems.append({"score": score, "tab": tab})
    else:
        raise ValueError(f"Unsupported vector measure mode: {mode}")

    boxes = []
    for system_index, system in enumerate(systems):
        if mode == "both":
            score = system["score"]
            tab = system["tab"]
            # Score staff paths are split only at real barlines. TAB strings
            # are additionally interrupted around fret glyphs, so their path
            # endpoints are not a reliable boundary source.
            boundaries = score["boundaries"] or (tab["boundaries"] if tab else [])
            top = score["lines"][0] - 5.0 * score["spacing"]
            bottom = (
                tab["lines"][-1] + 3.0 * tab["spacing"]
                if tab is not None
                else score["lines"][-1] + 22.0 * score["spacing"]
            )
        else:
            boundaries = system["boundaries"]
            if mode == "notation":
                top = system["lines"][0] - 5.0 * system["spacing"]
                bottom = system["lines"][-1] + 4.0 * system["spacing"]
            else:
                top = system["lines"][0] - 4.0 * system["spacing"]
                bottom = system["lines"][-1] + 3.0 * system["spacing"]
        for measure_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            # A repeat/double bar is emitted as two parallel boundaries about
            # six millimetres apart.  Treat that narrow slot as decoration,
            # not as an empty measure.  The smallest real measure in the GP8
            # validation corpus is more than 17 mm wide.
            if right - left <= 20:
                continue
            boxes.append({
                "system_index": system_index,
                "system_measure_index": measure_index,
                "bbox": [
                    left * scale,
                    max(0.0, top * scale),
                    (right - left) * scale,
                    max(1.0, (bottom - top) * scale),
                ],
                "geometry_source": "pdf_vector_staff_segments",
            })
    return boxes


def extract_pdf_vector_measure_boxes(
    pdf: str | Path, mode: str, *, dpi: int = 180
) -> dict[int, list[dict[str, Any]]]:
    import pymupdf

    result = {}
    with pymupdf.open(Path(pdf)) as document:
        for page_number, page in enumerate(document, start=1):
            result[page_number] = _page_boxes(page, mode, dpi)
    return result


def extract_pdf_vector_tab_systems(
    pdf: str | Path, *, dpi: int = 180
) -> dict[int, list[dict[str, Any]]]:
    """Return TAB string locations without trusting split string endpoints.

    Fret glyphs interrupt GP8's vector string rules.  Their y positions are
    still a reliable way to recover a complete system that the raster chain
    detector missed, while the raster vertical-stroke test remains the more
    reliable source of bar boundaries.
    """

    import pymupdf

    result: dict[int, list[dict[str, Any]]] = {}
    with pymupdf.open(Path(pdf)) as document:
        for page_number, page in enumerate(document, start=1):
            result[page_number] = _tab_systems_for_page(page, dpi)
    return result


def _tab_systems_for_page(page: Any, dpi: int) -> list[dict[str, Any]]:
    scale = dpi / 72.0
    rows = _horizontal_rows(page)
    clusters = _staff_clusters(rows, allow_missing_rows=True)
    systems = []
    for cluster in clusters:
        spacing = _cluster_spacing(cluster)
        line_count = round((cluster[-1] - cluster[0]) / spacing) + 1
        kind = "score" if line_count == 5 and spacing < 5.5 else "tab"
        if kind != "tab" or not 4 <= line_count <= 8:
            continue
        endpoints = _boundaries(cluster, rows)
        left = min(value[0] for y in cluster for value in rows[y])
        right = max(value[1] for y in cluster for value in rows[y])
        systems.append({
            "system_index": len(systems),
            "string_y": [
                (cluster[0] + index * spacing) * scale
                for index in range(line_count)
            ],
            "spacing": spacing * scale,
            "left": left * scale,
            "right": right * scale,
            "boundary_candidates": [value * scale for value in endpoints],
            "geometry_source": "pdf_vector_tab_rows+raster_barlines",
        })
    return systems


def _longest_consecutive_measure_numbers(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove fret/tuplet numbers that happen to sit above a TAB row."""

    if not candidates:
        return []
    lengths = [1] * len(candidates)
    previous = [-1] * len(candidates)
    for index, candidate in enumerate(candidates):
        for earlier in range(index):
            if candidates[earlier]["value"] + 1 != candidate["value"]:
                continue
            if lengths[earlier] + 1 > lengths[index]:
                lengths[index] = lengths[earlier] + 1
                previous[index] = earlier
    cursor = max(range(len(candidates)), key=lambda value: lengths[value])
    result = []
    while cursor >= 0:
        result.append(candidates[cursor])
        cursor = previous[cursor]
    return list(reversed(result))


def extract_pdf_vector_tab_measure_boxes(
    pdf: str | Path, *, dpi: int = 180
) -> dict[int, list[dict[str, Any]]]:
    """Recover GP8 TAB measures from vector rows and printed bar numbers.

    GP8's TAB-only style commonly omits internal vertical barlines. Chord
    stems can cross every string, so treating vertical raster strokes as
    barlines is intrinsically ambiguous. The PDF text layer prints a small,
    consecutive measure number at every true boundary and is the authoritative
    source whenever it is available. Raster geometry remains the fallback for
    flattened PDFs and ordinary images.
    """

    import pymupdf

    scale = dpi / 72.0
    result: dict[int, list[dict[str, Any]]] = {}
    with pymupdf.open(Path(pdf)) as document:
        for page_number, page in enumerate(document, start=1):
            systems = _tab_systems_for_page(page, dpi)
            words = page.get_text("words")
            candidates: list[dict[str, Any]] = []
            for system in systems:
                top_points = float(system["string_y"][0]) / scale
                spacing_points = float(system["spacing"]) / scale
                values = []
                for word in words:
                    x0, y0, _x1, y1, text, *_rest = word
                    token = str(text).strip()
                    if not token.isdigit():
                        continue
                    if y0 < top_points - 2.2 * spacing_points:
                        continue
                    if y1 > top_points + 0.20 * spacing_points:
                        continue
                    if y1 - y0 > 1.65 * spacing_points:
                        continue
                    values.append({
                        "value": int(token),
                        "system_index": int(system["system_index"]),
                        "x": float(x0) * scale,
                    })
                candidates.extend(sorted(values, key=lambda value: value["x"]))

            chosen = _longest_consecutive_measure_numbers(candidates)
            selected_systems = {value["system_index"] for value in chosen}
            # A partial text extraction is more dangerous than a complete
            # raster fallback because it silently drops whole systems.
            if systems and selected_systems != set(range(len(systems))):
                result[page_number] = []
                continue

            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for value in chosen:
                grouped[value["system_index"]].append(value)
            boxes = []
            for system in systems:
                system_index = int(system["system_index"])
                numbers = sorted(grouped[system_index], key=lambda value: value["x"])
                if not numbers:
                    continue
                boundaries = [float(system["left"])]
                endpoints = [float(value) for value in system["boundary_candidates"]]
                for number in numbers[1:]:
                    x = float(number["x"])
                    if endpoints:
                        nearest = min(endpoints, key=lambda value: abs(value - x))
                        if abs(nearest - x) <= 0.9 * float(system["spacing"]):
                            x = nearest
                    boundaries.append(x)
                boundaries.append(float(system["right"]))
                top = float(system["string_y"][0]) - 4.0 * float(system["spacing"])
                bottom = float(system["string_y"][-1]) + 3.0 * float(system["spacing"])
                for measure_index, (left, right) in enumerate(
                    zip(boundaries, boundaries[1:])
                ):
                    if right <= left:
                        continue
                    boxes.append({
                        "system_index": system_index,
                        "system_measure_index": measure_index,
                        "bbox": [left, top, right - left, bottom - top],
                        "measure_number": int(numbers[measure_index]["value"]),
                        "geometry_source": "pdf_vector_tab_rows+measure_numbers",
                    })
            result[page_number] = boxes
    return result
