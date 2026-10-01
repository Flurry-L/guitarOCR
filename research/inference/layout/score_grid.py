"""Recover common bar columns from agreement between simultaneous staves."""

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median

from PIL import Image

from research.common.crops import crop_measure


def missing_bar_box(image, row, left, right):
    """Confirm an undetected bar using its own staff lines and two barlines."""
    from research.inference.layout.tab_geometry import detect_tab_geometry

    modes = {r['mode'] for r in row}
    if len(modes) != 1 or next(iter(modes)) not in {'tab', 'notation'}:
        return None
    mode = next(iter(modes))
    top = min(r['bbox'][1] for r in row)
    bottom = max(r['bbox'][1] + r['bbox'][3] for r in row)
    x0, x1 = max(0, int(left) - 12), min(image.width, int(right) + 12)
    y0, y1 = max(0, int(top)), min(image.height, int(bottom))
    staffs = detect_tab_geometry(image.crop((x0, y0, x1, y1)), minimum_string_spacing=5)
    if len(staffs) != 1 or len(staffs[0]['boundaries']) != 2:
        return None
    staff = staffs[0]
    if mode == 'notation' and staff['string_count'] != 5:
        return None
    start, stop = staff['boundaries']
    if start > max(36, (right - left) * .12):
        return None
    a, b = max(x0, x0 + start - 10), min(x1, x0 + stop + 10)
    return [a, y0, b - a, y1 - y0]


def common_columns(rows):
    if len(rows) < 2:
        return [r['bbox'][0] for r in rows[0]], max(r['bbox'][0] + r['bbox'][2] for r in rows[0])
    widths = [r['bbox'][2] for row in rows for r in row]
    tolerance = max(10., median(widths) * .065)
    points = sorted((r['bbox'][0], i, float(r.get('score', 1.))) for i, row in enumerate(rows) for r in row)
    clusters = []
    for point in points:
        if clusters and point[0] - median(p[0] for p in clusters[-1]) <= tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    support = [len({p[1] for p in cluster}) for cluster in clusters]
    quorum = len(rows) // 2 + 1
    selected = [median(p[0] for p in c) for c, votes in zip(clusters, support) if votes >= quorum]
    typical = median(len(row) for row in rows)
    if len(selected) < max(1, typical * .6):
        # Tied two-staff evidence prefers the typical count, not the maximum.
        counts = Counter(len(row) for row in rows)
        common = min(counts, key=lambda n: (-counts[n], n))
        reference = max((r for r in rows if len(r) == common), key=lambda r: sum(b.get('score', 1.) for b in r))
        selected = [r['bbox'][0] for r in reference]
    else:
        # A first bar may contain a large clef/name margin; its left edge varies.
        first = median(row[0]['bbox'][0] for row in rows)
        if selected[0] - first > tolerance * 2:
            selected.insert(0, first)
        # Retain minority boundaries only when two independent staves support
        # them and no majority box crosses their interior.
        for cluster, votes in zip(clusters, support):
            x = median(p[0] for p in cluster)
            if votes < 2 or any(abs(x - s) <= tolerance for s in selected):
                continue
            crossing = sum(any(b['bbox'][0] + tolerance * 2 < x < b['bbox'][0] + b['bbox'][2] - tolerance * 2 for b in row) for row in rows)
            if crossing < votes and x > selected[0]:
                selected.append(x)
    selected.sort()
    right = median(max(r['bbox'][0] + r['bbox'][2] for r in row) for row in rows)
    return selected, max(right, selected[-1] + tolerance * 2)


def align_system(members, output, page, system):
    """Merge false splits and split spanning boxes using the common bar grid.

    New crops retain their source boxes; all downstream stages consume these
    resolved records rather than reconstructing the detector's old ordering.
    """
    rows = [sorted(boxes, key=lambda r: r['bbox'][0]) for boxes, _, _ in members]
    columns, right = common_columns(rows)
    boundaries = [*columns, right]
    result = []
    for row_index, (row, (_, part, staff)) in enumerate(zip(rows, members, strict=True)):
        grouped = defaultdict(list)
        for record in row:
            x, y, width, height = record['bbox']
            end = x + width
            # Only an interior majority boundary can split a detector box.
            cuts = [b for b in columns[1:] if x + max(12, width * .15) < b < end - max(12, width * .15)]
            edges = [x, *cuts, end]
            for left, stop in zip(edges, edges[1:]):
                column = min(range(len(columns)), key=lambda i: abs(left - columns[i]))
                if abs(left - columns[column]) > max(18, (boundaries[column + 1] - columns[column]) * .18):
                    column = max(0, next((i - 1 for i, c in enumerate(columns) if c > left), len(columns) - 1))
                grouped[column].append((record, [left, y, stop - left, height]))
        corrected = []
        for column, fragments in sorted(grouped.items()):
            first = max(fragments, key=lambda pair: pair[0].get('score', 1.))[0]
            left = min(box[0] for _, box in fragments)
            top = min(box[1] for _, box in fragments)
            end = max(box[0] + box[2] for _, box in fragments)
            bottom = max(box[1] + box[3] for _, box in fragments)
            bbox = [left, top, end - left, bottom - top]
            record = deepcopy(first)
            record['source_measure_numbers'] = sorted({r['measure_number'] for r, _ in fragments})
            record['system_measure_index'] = column
            record['grid_column'] = column
            record['grid_columns'] = len(columns)
            changed = len(fragments) != 1 or any(abs(a - b) > 1 for a, b in zip(first['bbox'], bbox))
            if changed:
                path = Path(output) / 'resolved_crops' / f'p{page}-s{system}-r{row_index}-b{column}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(record['source_page']) as image:
                    crop_measure(image, bbox).save(path)
                record.update(bbox=bbox, image=str(path.resolve()), geometry_source='staff_consensus')
            corrected.append((record, column))
        missing = sorted(set(range(len(columns))) - grouped.keys())
        # Three or more simultaneous rows provide a majority bar grid. A
        # missing detection is recovered only if that row visibly contains a
        # complete bounded staff, so blank paper never becomes invented music.
        if missing and len(rows) >= 3 and row[0].get('source_page'):
            with Image.open(row[0]['source_page']) as image:
                for column in missing:
                    bbox = missing_bar_box(image, row, boundaries[column], boundaries[column + 1])
                    if bbox is None:
                        continue
                    record = deepcopy(min(row, key=lambda r: abs(r['bbox'][0] - boundaries[column])))
                    path = Path(output) / 'resolved_crops' / f'p{page}-s{system}-r{row_index}-b{column}-recovered.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    crop_measure(image, bbox).save(path)
                    record.update(bbox=bbox, image=str(path.resolve()),
                                  source_measure_numbers=[], system_measure_index=column,
                                  grid_column=column, grid_columns=len(columns),
                                  geometry_source='staff_lines_and_consensus')
                    corrected.append((record, column))
        corrected.sort(key=lambda pair: pair[1])
        result.append((corrected, part, staff))
    return result, len(columns)


def fuse_paired_staves(aligned, output, page, system):
    """Fuse only notation/TAB rows assigned to the same staff by the model."""
    grouped = defaultdict(list)
    for boxes, part, staff in aligned:
        grouped[(part, staff)].append(boxes)
    result = []
    for (part, staff), groups in grouped.items():
        if len(groups) == 1:
            result.append((groups[0], part, staff))
            continue
        columns = defaultdict(list)
        for boxes in groups:
            for record, column in boxes:
                columns[column].append(record)
        boxes = []
        for column, records in sorted(columns.items()):
            if len(records) == 1:
                boxes.append((records[0], column))
                continue
            if len(records) != 2 or {r['mode'] for r in records} != {'notation', 'tab'}:
                raise ValueError('Only complementary notation and TAB can share a staff')
            left = min(r['bbox'][0] for r in records)
            top = min(r['bbox'][1] for r in records)
            right = max(r['bbox'][0] + r['bbox'][2] for r in records)
            bottom = max(r['bbox'][1] + r['bbox'][3] for r in records)
            bbox = [left, top, right - left, bottom - top]
            record = deepcopy(records[0])
            path = Path(output) / 'resolved_crops' / f'p{page}-s{system}-part{part}-staff{staff}-b{column}-both.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(record['source_page']) as image:
                crop_measure(image, bbox).save(path)
            record.update(mode='both', bbox=bbox, image=str(path.resolve()), geometry_source='notation_tab_pair',
                          source_measure_numbers=sorted({n for r in records for n in r.get('source_measure_numbers', [r['measure_number']])}))
            boxes.append((record, column))
        result.append((boxes, part, staff))
    return result
