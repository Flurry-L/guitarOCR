from __future__ import annotations

from collections import Counter

from PIL import Image
import numpy as np

from research.inference.layout.tab_geometry import detect_tab_geometry
from research.inference.layout.score_geometry import detect_score_tab_geometry


LAYOUTS = ("score_tab", "tab_only", "score_only")


def part_tab_strings(records: list[dict]) -> int | None:
    """Use several complete measures so short/faint grids cannot drop strings."""
    rows = [r for r in records if r.get('mode') in {'tab', 'both'} and r.get('image')]
    if not rows:
        return None
    indices = np.unique(np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int))
    votes = Counter()
    for index in indices:
        row = rows[index]
        with Image.open(row['image']) as image:
            count = visible_tab_strings(image, row['mode'])
        if count is not None:
            votes[count] += 1
    if not votes:
        return None
    count, support = votes.most_common(1)[0]
    return count if support >= min(3, len(rows)) and support / votes.total() >= .75 else None


def visible_tab_strings(image: Image.Image, mode: str) -> int | None:
    """Count complete, strongly supported TAB lines in one full staff crop."""
    if mode not in {'tab', 'both'}:
        return None
    gray = np.asarray(image.convert('L'))
    if gray.shape[1] < 100:
        return None
    interior = gray[:, round(gray.shape[1] * .13):round(gray.shape[1] * .9)]
    ink = (interior < 210).mean(1)
    faint = (interior < min(253, float(np.percentile(interior, 95)) - 2)).mean(1)
    threshold = max(.35, float(ink.max()) * .55)
    hits = np.flatnonzero(ink > threshold)
    groups = [g for g in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1) if len(g)]
    ys = [float(np.average(g, weights=ink[g])) for g in groups]
    strengths = [float(ink[g].max()) for g in groups]
    candidates = []
    i = 0
    while i + 3 < len(ys):
        gap = ys[i + 1] - ys[i]
        if not 5 <= gap <= 45:
            i += 1
            continue
        j, count, missing = i + 1, 2, 0
        while j + 1 < len(ys):
            distance = ys[j + 1] - ys[j]
            steps = round(distance / gap)
            if (steps not in {1, 2} or missing + steps - 1 > 1
                    or abs(distance - steps * gap) > max(2.0, gap * .12) * steps):
                break
            if steps == 2:
                expected = round(ys[j] + gap)
                if faint[max(0, expected - 2):expected + 3].max(initial=0) <= .6:
                    break
            count += steps
            missing += steps - 1
            j += 1
            # Fit the whole grid; raster rounding can alternate 14/16/15 pixels.
            gap = (ys[j] - ys[i]) / (count - 1)
        if 4 <= count <= 8 and j - i + 1 >= 4 and min(strengths[i:j + 1]) > .45:
            edges = 0
            for y in (ys[i] - gap, ys[j] + gap):
                y = round(y)
                if (2 <= y < len(ink) - 2
                        and faint[y-2:y+3].max() > .6
                        and ink[y-2:y+3].max() <= threshold):
                    edges += 1
            if edges == 1 and missing == 0 and count < 8:
                count += 1
            elif edges:
                i = j + 1
                continue
            candidates.append((count, gap))
            i = j + 1
        else:
            i += 1
    if mode == 'both':
        # A combined crop must expose the tighter notation grid as well.
        # A single grid is insufficient evidence for choosing the TAB lines.
        minimum = min((gap for _, gap in candidates), default=0)
        candidates = [(count, gap) for count, gap in candidates if gap > minimum * 1.15]
    counts = {count for count, _ in candidates}
    return counts.pop() if len(counts) == 1 else None


def _classify_fixed_scale(page: Image.Image) -> dict:
    paired = detect_score_tab_geometry(page)
    if paired:
        return {
            'layout': "score_tab", "confidence": 0.99,
            "systems": len(paired), "method": "paired_five_and_tab_staffs",
        }
    staffs = detect_tab_geometry(page, minimum_string_spacing=6.0)
    if not staffs:
        return {'layout': "unknown", "confidence": 0.0, "systems": 0, "method": "no_staffs"}
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
        'layout': layout,
        "confidence": support / max(1, len(staffs)),
        "systems": len(staffs),
        "staff_line_counts": dict(sorted(counts.items())),
        "method": "unpaired_staff_spacing_and_line_count",
    }


def classify_notation_layout(page: Image.Image) -> dict:
    """Classify printed notation, normalizing common PDF raster scales."""
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
    if not any(result['layout'] == "score_tab" for result in results):
        # Thin GP8 staff lines in scanned PDFs can be lighter than the
        # geometry detector's ink threshold, including when only the darker
        # TAB staff was found. Enhance only the classification copy; the
        # detector and OCR still receive the original page pixels.
        for scale, candidate in candidates:
            result = _classify_fixed_scale(candidate.convert("L").point(
                lambda value: 0 if value < 230 else 255
            ))
            result["analysis_scale"] = scale
            result["contrast_threshold"] = 230
            results.append(result)
    # Paired score+TAB geometry is stronger evidence than an unpaired staff
    # guess. Otherwise choose the result with the greatest supported fraction.
    return max(
        results,
        key=lambda item: (
            item['layout'] == "score_tab",
            float(item["confidence"]),
            int(item["systems"]),
        ),
    )
