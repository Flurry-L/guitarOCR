from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image


def group_runs(values: np.ndarray, maximum_gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw_value in values:
        value = int(raw_value)
        if not groups or value > groups[-1][-1] + maximum_gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def detect_tab_boundaries_for_lines(
    image: Image.Image,
    string_y: list[float],
    *,
    preserve_cross_staff_barlines: bool = False,
    minimum_horizontal_density: float = 0.62,
) -> list[float]:
    """Detect bar boundaries when the TAB string rows are already known.

    This is used by the GP8 PDF hybrid geometry path.  The PDF vector layer
    reliably supplies string y positions even when dense digits make the
    raster page-level chain detector discard the entire system.  Boundaries
    still come from raster vertical strokes because fret glyphs split the
    vector string rules into misleading endpoints.
    """

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    black = gray < 160
    rows = [int(round(value)) for value in string_y]
    return _detect_tab_boundaries(
        black, rows, preserve_cross_staff_barlines, minimum_horizontal_density
    )


def _detect_tab_boundaries(
    black: np.ndarray,
    rows: list[int],
    preserve_cross_staff_barlines: bool,
    minimum_horizontal_density: float = 0.62,
) -> list[float]:
    """Shared barline filtering for known string rows and full-page detection."""
    height, width = black.shape
    if len(rows) < 4:
        return []
    if rows[0] < 0 or rows[-1] >= height:
        return []
    spacing = float(np.median(np.diff(rows)))
    top_y, bottom_y = rows[0], rows[-1]
    vertical_counts = black[top_y : bottom_y + 1].sum(axis=0)
    bar_groups = group_runs(
        np.where(vertical_counts > (bottom_y - top_y + 1) * 0.90)[0]
    )
    if len(bar_groups) < 2:
        return []

    maximum_tail = max(2, round(spacing * 0.19))

    def continuous_tail(column: int) -> int:
        above = 0
        for y in range(top_y - 1, max(-1, round(top_y - 4 * spacing)), -1):
            if not black[y, column]:
                break
            above += 1
        below = 0
        for y in range(bottom_y + 1, min(height, round(bottom_y + 4 * spacing))):
            if not black[y, column]:
                break
            below += 1
        return max(above, below)

    filtered_groups = [bar_groups[0]]
    for group in bar_groups[1:-1]:
        shortest_tail = min(continuous_tail(column) for column in group)
        cross_staff_tail = (
            preserve_cross_staff_barlines
            and shortest_tail >= round(spacing * 3.5)
        )
        if shortest_tail <= maximum_tail or cross_staff_tail:
            filtered_groups.append(group)
    filtered_groups.append(bar_groups[-1])

    merged: list[list[int]] = []
    for group in filtered_groups:
        if merged and group[0] - merged[-1][-1] <= max(3.0, spacing * 0.75):
            merged[-1].extend(group)
        else:
            merged.append(group)
    if len(merged) < 2:
        return []

    raw_boundaries = [
        merged[0][0],
        *(group[0] for group in merged[1:-1]),
        merged[-1][-1],
    ]
    minimum_measure_width = max(80.0, spacing * 5.0)
    minimum_initial_width = max(minimum_measure_width, spacing * 7.0)
    boundaries = [raw_boundaries[0]]
    for raw_index, boundary in enumerate(raw_boundaries[1:-1], start=1):
        required_width = (
            minimum_initial_width if raw_index == 1 else minimum_measure_width
        )
        if boundary - boundaries[-1] >= required_width:
            boundaries.append(boundary)
    final_boundary = raw_boundaries[-1]
    if final_boundary - boundaries[-1] < minimum_measure_width and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(final_boundary)

    system_left = max(0, int(raw_boundaries[0]))
    system_right = min(width, int(final_boundary) + 1)
    horizontal_density = float(
        np.median(black[np.asarray(rows), system_left:system_right].mean(axis=1))
    )
    if horizontal_density < minimum_horizontal_density:
        return []
    return [float(value) for value in boundaries]


def detect_tab_geometry(
    image: Image.Image,
    *,
    preserve_cross_staff_barlines: bool = False,
    minimum_string_spacing: float | None = None,
) -> list[dict]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    black = gray < 160
    height, width = black.shape

    minimum_line_run = max(60, round(width * 0.055))
    row_ink = black.sum(axis=1)
    longest_runs = np.zeros(height, dtype=np.int32)
    for row_index, row in enumerate(black):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        if edges.size:
            longest_runs[row_index] = int((edges[1::2] - edges[::2]).max(initial=0))
    # Chords can erase/break a staff line at many x positions at once. GP8
    # then has no single long run even though the total row ink still spans
    # most of the system. Accept either a long continuous run or strong total
    # horizontal support.
    minimum_row_ink = max(minimum_line_run, round(width * 0.25))
    row_groups = group_runs(
        np.where((longest_runs >= minimum_line_run) | (row_ink >= minimum_row_ink))[0]
    )
    candidate_rows = [max(group, key=lambda y: int(longest_runs[y])) for group in row_groups]
    chains: list[list[int]] = []
    minimum_string_spacing = (
        float(minimum_string_spacing)
        if minimum_string_spacing is not None
        else max(8.0, height * 0.006)
    )
    for row_index, first_y in enumerate(candidate_rows):
        for second_y in candidate_rows[row_index + 1 :]:
            spacing = second_y - first_y
            if spacing < minimum_string_spacing:
                continue
            if spacing > 40:
                break
            # A nominal 16 px GP8 staff is rasterised as alternating 15/17 px
            # gaps on some systems.  One-pixel tolerance broke the chain into
            # a four-line suffix and the page-wide modal filter then removed
            # the whole system.
            tolerance = max(2.0, spacing * 0.14)
            chain = [first_y]
            current = first_y
            while len(chain) < 8:
                expected = current + spacing
                options = [
                    value for value in candidate_rows
                    if value > current and abs(value - expected) <= tolerance
                ]
                if not options:
                    break
                current = min(options, key=lambda value: abs(value - expected))
                chain.append(current)
            if len(chain) >= 4:
                # Keep prefixes as candidates. Dense TAB beams below the
                # sixth string can continue at exactly one string spacing and
                # otherwise turn a real six-string staff into a false 7/8-line
                # chain. True string rows have comparable page-wide ink,
                # whereas appended beam rows are much shorter.
                for length in range(4, len(chain) + 1):
                    chains.append(chain[:length])

    def chain_quality(chain: list[int]) -> tuple[float, int, float, int]:
        strengths = np.asarray([row_ink[value] for value in chain], dtype=np.float32)
        median = float(np.median(strengths))
        consistency = float(strengths.min() / max(1.0, median))
        return consistency, -abs(len(chain) - 6), median, len(chain)

    chains.sort(
        key=lambda chain: (
            -len(chain),
            -chain_quality(chain)[0],
            -chain_quality(chain)[2],
            chain[0],
        )
    )
    selected: list[list[int]] = []
    for chain in chains:
        # Dense digits can interrupt one middle string much more heavily than
        # its neighbours (observed support ratio about 0.73 on GP8).  The
        # page-wide string-count vote and horizontal-density test below are
        # stronger guards against beam/chord-diagram chains.
        if chain_quality(chain)[0] < 0.68:
            continue
        if any(set(chain) & set(existing) for existing in selected):
            continue
        selected.append(chain)
    selected.sort(key=lambda chain: chain[0])

    # A printed page uses one string count for a single-track score. Dense
    # rhythm beams can form a short four/five-line chain at exactly the TAB
    # spacing; keep the page-wide modal line count instead of treating that
    # beam fragment as another staff. This is especially important for GP8,
    # whose TAB stems and beams are longer than TuxGuitar's.
    if selected:
        line_counts = Counter(len(chain) for chain in selected)
        dominant_count = max(line_counts, key=lambda count: (line_counts[count], count == 6, count))
        selected = [chain for chain in selected if len(chain) == dominant_count]

    staffs: list[dict] = []
    measure_number = 1
    for staff_index, string_y in enumerate(selected):
        spacing = float(np.median(np.diff(string_y)))
        top_y, bottom_y = string_y[0], string_y[-1]
        boundaries = _detect_tab_boundaries(
            black, string_y, preserve_cross_staff_barlines
        )
        if not boundaries:
            continue

        measures: list[dict] = []
        staff_top = top_y - spacing / 2.0
        staff_height = (bottom_y - top_y) + spacing
        for local_index in range(len(boundaries) - 1):
            left, right = boundaries[local_index], boundaries[local_index + 1]
            measures.append(
                {
                    "measure_number": measure_number,
                    "system_measure_index": local_index,
                    "bbox": [float(left), staff_top, float(right - left), staff_height],
                    "symbols": [],
                    "events": [],
                }
            )
            measure_number += 1
        staffs.append(
            {
                "staff_index": staff_index,
                "string_count": len(string_y),
                "string_y": [float(value) for value in string_y],
                "spacing": spacing,
                "boundaries": [float(value) for value in boundaries],
                "measures": measures,
            }
        )
    return staffs
