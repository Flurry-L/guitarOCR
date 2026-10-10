"""Build the shared source catalog and rendered-page inventory from GP8 exports."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
from pathlib import Path

from research.common.pdf import open_pdf

from research.data.catalog import source_catalog
from research.data.native_supervision import native_pitch_annotations, pdf_spans, visible_text_box


def chord_annotations(page, number, layout):
    """Label GP's actual PDF grids and names, including the header dictionary.

    The grid is a connected set of painted horizontal and vertical rectangles.
    Source chord shapes do not supply coordinates or prove that a grid exists.
    This is supervision for GP exports, not an image-recognition fallback.
    """
    from research.data.native_alignment import chord_words

    drawing = page._text_page()
    ox, oy = drawing.bbox[:2]
    rects = drawing.rects
    remaining = set(range(len(rects)))
    grids = []
    while remaining:
        group, pending = [], [remaining.pop()]
        while pending:
            index = pending.pop()
            item = rects[index]
            group.append(item)
            adjacent = [j for j in remaining
                        if rects[j]['x0'] <= item['x1'] + .01 and rects[j]['x1'] >= item['x0'] - .01
                        and rects[j]['top'] <= item['bottom'] + .01 and rects[j]['bottom'] >= item['top'] - .01]
            remaining.difference_update(adjacent)
            pending.extend(adjacent)
        horizontal = sum(r['width'] > r['height'] for r in group)
        vertical = sum(r['height'] > r['width'] for r in group)
        if horizontal >= 4 and vertical >= 4:
            grids.append([min(r['x0'] for r in group) - ox, min(r['top'] for r in group) - oy,
                          max(r['x1'] for r in group) - ox, max(r['bottom'] for r in group) - oy])
    names = chord_words(page)
    boxes, consumed = [], set()
    for grid in grids:
        x0, y0, x1, y1 = grid
        width = x1 - x0
        above = [(i, word) for i, word in enumerate(names)
                 if abs((word[0] + word[2] - x0 - x1) / 2) < width / 2
                 and 0 <= y0 - word[3] <= width]
        if not above:
            raise ValueError(f'Printed GP chord grid has no visible name: {grid}')
        index, name = min(above, key=lambda pair: y0 - pair[1][3])
        consumed.add(index)
        # Preserve the actual x/o, fret-position and finger glyphs surrounding
        # the grid. Their PDF font boxes may be taller than their ink.
        members = [grid, list(name[:4])]
        for item in drawing.chars + drawing.curves:
            a, b, c, d = item['x0'] - ox, item['top'] - oy, item['x1'] - ox, item['bottom'] - oy
            if item['object_type'] == 'char':
                mark = re.fullmatch('[\ue858-\ue85a\ued10-\ued2f]', item['text'])
                position = (re.fullmatch('[0-9IVX]+', item['text']) and c <= x0
                            and y0 <= (b+d)/2 <= y1)
                if not mark and not position:
                    continue
            elif not y0 <= (b+d)/2 <= y1:
                continue
            if (x0 - width / 2 <= (a+c)/2 <= x1 + width / 4
                    and name[3] <= (b+d)/2 <= y1 + width / 2):
                members.append([a, b, c, d])
        boxes.append([min(b[0] for b in members), min(b[1] for b in members),
                      max(b[2] for b in members), max(b[3] for b in members)])
    # Name-only local chords align with a real note/rest origin. This excludes
    # a track-name letter or header prose that happens to spell a chord.
    anchors = [g for g in layout.get('note_geometry', {}).get('glyphs', []) if g['page'] == number]
    for i, word in enumerate(names):
        if i in consumed:
            continue
        cx, bottom = (word[0] + word[2]) / 2 * 25.4 / 72, word[3] * 25.4 / 72
        if any(abs(g['render_origin_mm'][0] - cx) < .75 and 0 < g['bbox_mm'][1] - bottom < 20
               for g in anchors):
            boxes.append(list(word[:4]))
    return [{'label': 'annotation_region', 'bbox_mm': [x0 * 25.4 / 72, y0 * 25.4 / 72,
             (x1-x0) * 25.4 / 72, (y1-y0) * 25.4 / 72]} for x0, y0, x1, y1 in boxes]


def header_annotations(page, score):
    """Match enabled native header fields to their actually printed PDF text."""
    metadata = score['document']['metadata']
    fields = []
    for key, style in metadata['first_page_header'].items():
        if not style['visible'] or not metadata['properties'].get(key):
            continue
        text = re.sub(r'%([^%]+)%', lambda match: metadata['properties'].get(
            match[1].lower().replace('wordsmusic', 'words_and_music'), match[0]), style['formatted_text'])
        fields.append(({'title': 'title_region', 'subtitle': 'subtitle_region',
                        'album': 'header_text_region'}.get(key, 'credit_region'), text))
    for track in score['tracks']:
        for staff in track['staves']:
            if staff['tuning_label_visible'] and staff['tuning_displayed_label']:
                label = staff['tuning_displayed_label']
                fields.append(('tuning_region', 'Standard tuning' if label == 'Standard' else label))
    result = []
    for label, text in fields:
        box = visible_text_box(page, text)
        if box is None:
            continue
        x0, y0, x1, y1 = [value * 25.4 / 72 for value in box]
        result.append({'label': label, 'bbox_mm': [x0, y0, x1-x0, y1-y0]})
    return result

def _inventory_track(job):
    export_root, output, dpi, sid, assignment, mode = job
    total = 0
    documents = export_root / "native-export" / "documents" / f"{mode}-{sid}" / "tracks"
    layouts = sorted(documents.glob("*/layout.json"))
    if len(layouts) != 1:
        raise ValueError(
            f"Expected one exported {mode} track for {sid}, found {len(layouts)}; run --phase render"
        )
    source = layouts[0].parent
    layout = json.loads(layouts[0].read_text(encoding="utf-8"))
    score = json.loads((source / "official-score.json").read_text())
    if layout["display_mode"] != mode:
        raise ValueError(f"Rendered display mode differs from {mode}: {source}")
    folder = Path("tracks") / mode / sid
    destination = output / folder
    destination.mkdir(parents=True, exist_ok=True)
    annotations = {}
    for system in layout["systems"]:
        for box in system["measure_boxes"]:
            annotations.setdefault(int(system["page"]), []).append(
                {"label": "measure", "bbox_mm": box["bbox_mm"]}
            )
    for tempo in layout.get("tempo_indications", []):
        annotations.setdefault(int(tempo["page"]), []).append(
            {"label": "tempo_region", "bbox_mm": tempo["bbox_mm"]}
        )
    pages = []
    with open_pdf(source / "score.pdf") as pdf:
        if len(pdf) != len(layout["pages"]):
            raise ValueError(f"PDF/layout page counts differ: {source}")
        for index, page in enumerate(pdf):
            number = index + 1
            page_mm = layout["pages"][index]["bbox_mm"]
            if int(layout["pages"][index]["index"]) != number:
                raise ValueError(f"Invalid layout page order: {source}")
            pix = page.render(dpi)
            image = folder / f"page_{number:03d}.png"
            pix.save(output / image)
            sx, sy = pix.width / page_mm[2], pix.height / page_mm[3]
            rows = []
            regions = (annotations.get(number, []) + native_pitch_annotations(page, number, layout, score, assignment)
                       + chord_annotations(page, number, layout))
            drawing = page._text_page()
            for char in drawing.chars:
                if char['text'] in ('\ue06d', '\ue06e'):
                    regions.append({'label': 'clef_region', 'bbox_mm': [
                        (char['x0'] - drawing.bbox[0]) * 25.4 / 72,
                        (char['top'] - drawing.bbox[1]) * 25.4 / 72,
                        char['width'] * 25.4 / 72, char['height'] * 25.4 / 72]})
            for span in pdf_spans(page):
                x0, y0, x1, y1 = span['text_bbox']
                x1 = max(x1, span['end_x'])
                factor = 25.4 / 72
                regions.append({'label': 'annotation_region',
                    'bbox_mm': [x0 * factor, y0 * factor, (x1-x0) * factor, (y1-y0) * factor],
                    'task': 'annotation', 'target': {'kind': 'technique', 'semitones': None,
                        'capo': None, 'text': 'P.M.' if span['kind'] == 'pm' else 'let ring'}})
            if number == 1:
                regions += header_annotations(page, score)
            for region in regions:
                x, y, w, h = region["bbox_mm"]
                rows.append(
                    {
                        **{key: value for key, value in region.items() if key != "bbox_mm"},
                        "box": [
                            (x - page_mm[0]) * sx,
                            (y - page_mm[1]) * sy,
                            (x + w - page_mm[0]) * sx,
                            (y + h - page_mm[1]) * sy,
                        ],
                    }
                )
            pages.append(
                {
                    "page_index": index,
                    "source_id": sid,
                    "mode": mode,
                    "renderer": "guitarpro8",
                    "split": assignment["split"],
                    "family": assignment["family"],
                    "image": image.as_posix(),
                    "image_size": [pix.width, pix.height],
                    "page_bbox_mm": page_mm,
                    "annotations": rows,
                }
            )
            total += 1
    (destination / "layout-pages.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pages),
        encoding="utf-8",
    )
    return (
        {
            **assignment,
            "mode": mode,
            "renderer": "guitarpro8",
            "sequence_id": f"{mode}-{sid}",
            "source_track": str(source),
            "folder": folder.as_posix(),
            "errors": [],
        },
        total,
    )


def build_inventory(
    export_root: Path,
    output: Path,
    dpi: int = 180,
    seed: int = 20260715,
    modes: list[str] | None = None,
    workers: int = 1,
) -> dict:
    export_root, output = export_root.resolve(), output.resolve()
    catalog = source_catalog(export_root, seed)
    available = ("tab", "notation", "both")
    if modes is None:
        documents = export_root / "native-export" / "documents"
        modes = [mode for mode in available if any(documents.glob(f"{mode}-*"))]
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(available):
        raise ValueError(f"Invalid or missing rendered modes: {modes}")
    tracks = []
    total = 0
    jobs = [
        (export_root, output, dpi, sid, assignment, mode)
        for sid, assignment in sorted(catalog.items())
        for mode in modes if mode in assignment.get("modes", modes)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_inventory_track, jobs))
    else:
        results = list(map(_inventory_track, jobs))
    tracks = [track for track, _ in results]
    total = sum(count for _, count in results)
    (output / "track-index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tracks),
        encoding="utf-8",
    )
    return {"tracks": len(tracks), "pages": total, "inventory": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp8-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", action="append", choices=("tab", "notation", "both"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    print(
        json.dumps(
            build_inventory(
                args.gp8_export,
                args.output,
                dpi=args.dpi,
                seed=args.seed,
                modes=args.mode,
                workers=args.workers,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
