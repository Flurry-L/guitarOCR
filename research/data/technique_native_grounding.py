"""Real native TAB pixel supervision from exact native glyph owner traces.

Only existing training/validation source families are read. Image crops and
coordinate targets come from the same PDF/layout export. No nearest-character
matching, invented x positions, final-test inspection or playback owner claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scorelib.score import parse_measure_target

from gpbridge.score_native_note_geometry import group_glyphs_by_owner
from research.common.pdf import open_pdf
from research.common.techniques import event_features
from research.data.technique_span_relations import compile_row, native_spans


def note_glyphs(layout, official):
    """Join drawn frets to musical onsets, excluding separate grace beats."""
    events = {}
    for track in official['tracks']:
        if track['track_index'] != official['source_track_index']:
            continue
        for staff in track['staves']:
            for measure in staff['measures']:
                for voice in measure['voices']:
                    for event in voice['events']:
                        if not event['grace'] and not event['placeholder']:
                            key = (staff['staff_index'], measure['measure_index'],
                                   voice['voice_index'], event['event_index'])
                            events[key] = round(960 * Fraction(*event['offset']))
    glyphs = defaultdict(list)
    for glyph in layout['note_geometry']['glyphs']:
        key = tuple(glyph[name] for name in ('staff_index', 'measure_index', 'voice_index', 'event_index'))
        if glyph['kind'] == 'fret' and key in events:
            glyphs[glyph['measure_index']].append({**glyph, 'onset': events[key]})
    return glyphs


def glyph_matches_note(glyph, note):
    # Trills can draw an extra parenthesized fret using the same owning Note.
    # The anchor must enclose the main note's fret, including ghost/harmonic
    # decorations, rather than that second trill glyph.
    text = glyph['text'].strip().strip('()<>[]').lower().replace('×', 'x')
    fret = 'x' if 'dead' in note.get('effects', []) else str(note.get('fret')).lower()
    return text == fret


def extract(root, source, output, limit=24):
    sid = source["source_id"]
    paths = list(
        (root / "native-export/documents" / f"tab-{sid}").glob("tracks/*/layout.json")
    )
    if len(paths) != 1:
        return [], {"missing_export": 1}
    path = paths[0]
    layout = json.loads(path.read_text())
    official = json.loads(path.with_name("official-score.json").read_text())
    group_glyphs_by_owner(official, layout, require_complete=True)
    glyphs = note_glyphs(layout, official)
    label = json.loads((root / "labels" / f"{sid}.json").read_text())
    strings = len(label["track"]["tuning_midi_high_to_low"])
    if strings > 6:
        return [], {"more_than_six_strings": 1}
    boxes = {
        box["measure_index"]: (system["page"], box["bbox_mm"])
        for system in layout["systems"]
        for box in system["measure_boxes"]
    }
    chosen = {
        int(value) for value in np.linspace(
            0, len(label["measures"]) - 1, min(limit, len(label["measures"])), dtype=int)
    }
    destination = output / source["split"] / sid
    destination.mkdir(parents=True, exist_ok=True)
    rows, reasons, pages = [], Counter(), {}
    with open_pdf(path.with_name("score.pdf")) as pdf:
        for mi in sorted(chosen):
            target = label["measures"][mi]["targets"]["tab"]
            parsed = parse_measure_target(target)
            features, anchors = event_features(parsed, "tab")
            if not 0 < len(anchors) <= 32:
                reasons["unsupported_event_count"] += 1
                continue
            if mi not in boxes:
                reasons["missing_measure_bounds"] += 1
                continue
            page, bbox = boxes[mi]
            by_event = defaultdict(list)
            events = {
                (v["voice"], e["start"]): e
                for v in parsed["voices"]
                for e in v["events"]
            }
            for glyph in glyphs[mi]:
                key = (glyph["voice_index"], glyph["onset"])
                event = events.get(key)
                if event is None:
                    continue
                string = strings - glyph["native_string_index"]
                note = next(
                    (n for n in event["notes"] if n.get("string") == string), None
                )
                if note and glyph_matches_note(glyph, note):
                    by_event[key].append(glyph)
            if any(
                len({g["note_index"] for g in by_event[tuple(anchor)]})
                != len(events[tuple(anchor)]["notes"])
                for anchor in anchors
            ):
                reasons["incomplete_or_mismatched_native_note_owner"] += 1
                continue
            if page not in pages:
                pages[page] = pdf[page - 1].render(dpi=180, grayscale=False)
            image = pages[page]
            factor = 180 / 25.4
            x, y, w, h = bbox
            crop = [
                max(0, int((x - 1.5) * factor)),
                max(0, int((y - 4) * factor)),
                min(image.width, int((x + w + 1.5) * factor) + 1),
                min(image.height, int((y + h + 4) * factor) + 1),
            ]
            positions = []
            for anchor in anchors:
                bounds = [g["bbox_mm"] for g in by_event[tuple(anchor)]]
                left, top = min(b[0] for b in bounds), min(b[1] for b in bounds)
                right, bottom = (
                    max(b[0] + b[2] for b in bounds),
                    max(b[1] + b[3] for b in bounds),
                )
                positions.append(
                    [
                        ((left + right) / 2 * factor - crop[0]) / (crop[2] - crop[0]),
                        ((top + bottom) / 2 * factor - crop[1]) / (crop[3] - crop[1]),
                    ]
                )
            if (
                min(np.array(positions).reshape(-1)) < 0
                or max(np.array(positions).reshape(-1)) > 1
            ):
                reasons["native_glyph_outside_crop"] += 1
                continue
            image_path = destination / f"measure-{mi}.png"
            image.crop(crop).save(image_path)
            rows.append(
                {
                    "id": f"native-{sid}-tab-{mi}",
                    "family": source["family"],
                    "source_id": sid,
                    "split": source["split"],
                    "mode": "tab",
                    "measure_index": mi,
                    "image": str(image_path),
                    "source_pdf": str(path.with_name("score.pdf")),
                    "source_layout": str(path),
                    "page": page,
                    "bbox": crop,
                    "target": target,
                    "features": features.tolist(),
                    "anchors": anchors,
                    "positions": positions,
                    "position_source": "native draw-time glyph bounds joined to exact official event owner",
                    "instrument": source.get("instrument"),
                    "variant": "native",
                    "layout": "native",
                }
            )
    return rows, dict(reasons)



def visible_event(event, spans, bbox):
    """Annotate native engraving, without treating playback bits as visual truth.

    These bounds are native PDF typographic points, solely for generating
    training truth. A dashed span directly supports interior event coverage,
    including tied notes without a repeated source flag. A word without a dash
    does not specify an interval: its downstream scope stays unknown unless a
    visible dashed span supplies direct coverage.
    """
    scale = 72 / 180
    x, y = [(event['pixel'][i] + bbox[i]) * scale for i in (0, 1)]
    nearby = [s for s in spans if bbox[1] * scale <= s['line_y'] <= bbox[3] * scale
              and y > s['line_y']]
    matched = [s for s in nearby if s['start_x'] - 3 <= x <= s['end_x'] + 1]
    source_label = event.get('source_label', event['label'])
    source = {'pm', 'let'} if source_label == 'both' else {source_label} - {'none', 'unknown'}
    observed = {s['kind'] for s in matched}
    interior = {s['kind'] for s in matched if s['style'] == 'labelled_dash'
                and s['start_x'] <= x <= s['end_x']}
    # A printed word without a terminal mark does not reveal where its scope
    # ends. Even source-unmarked notes to its right are not visual negatives.
    open_scope = {s['kind'] for s in nearby if s['style'] == 'label_only'
                  and x >= s['start_x'] - 3} - interior
    fields = {}
    for effect in ('pm', 'let'):
        if effect in interior:
            fields[effect] = True
        elif (effect in open_scope or source_label == 'unknown'
              or effect in source - observed or effect in observed - source):
            fields[effect] = None
        else:
            fields[effect] = effect in observed
    supported = all(value is not None for value in fields.values())
    visible = {effect for effect, value in fields.items() if value is True}
    label = ('both' if len(visible) == 2 else next(iter(visible), 'none')) if supported else 'unknown'
    return {**event, 'source_label': source_label, 'label': label, 'visible_effect_labels': fields,
            'visible_scope_evidence': dict(source='native_pdf_annotation_commands',
                matched_spans=matched, direct_interior_effects=sorted(interior),
                open_label_only_scope=sorted(open_scope),
                status='observed' if supported else 'ambiguous_source_or_short_label_scope')}


def system_image(system, folder, document, pages):
    page_number = system['page']
    if page_number not in pages:
        image = document[page_number - 1].render(dpi=180, grayscale=False)
        page_path = folder / f'page-{page_number}.png'
        image.save(page_path)
        pages[page_number] = image, page_path
    image, page_path = pages[page_number]
    x,y,w,h = system['bbox_mm']
    factor = 180/25.4
    crop = [max(0,int((x-2)*factor)), max(0,int((y-4)*factor)),
            min(image.width,int((x+w+2)*factor)+1), min(image.height,int((y+h+4)*factor)+1)]
    path = folder / f'system-{system["system_index"]}.png'
    if not path.exists():
        image.crop(crop).save(path)
    return path, page_path, crop


def relation_cases(original, output):
    groups = defaultdict(list)
    for row in original:
        groups[row["source_pdf"]].append(row)
    cases = []
    for pdf_path, rows in groups.items():
        layout = json.loads(Path(rows[0]["source_layout"]).read_text())
        folder = output / rows[0]["source_id"]
        folder.mkdir(parents=True, exist_ok=True)
        with open_pdf(pdf_path) as document:
            pages = {}
            for row in rows:
                system = next(
                    system
                    for system in layout["systems"]
                    if system["first_measure_index"]
                    <= row["measure_index"]
                    < system["last_measure_index"]
                )
                case_path, page_path, crop = system_image(system, folder, document, pages)
                previous = [s for s in layout['systems']
                            if s['last_measure_index'] == system['first_measure_index']]
                previous_path = (system_image(previous[0], folder, document, pages)[0]
                                 if len(previous) == 1 else None)
                music = parse_measure_target(row["target"])
                notes = {
                    (v["voice"], e["start"]): e
                    for v in music["voices"]
                    for e in v["events"]
                    if e["notes"]
                }
                events = []
                width, height = (
                    row["bbox"][2] - row["bbox"][0],
                    row["bbox"][3] - row["bbox"][1],
                )
                for (voice, onset), (px, py) in zip(
                    row["anchors"], row["positions"], strict=True
                ):
                    effects = {
                        e
                        for n in notes[voice, onset]["notes"]
                        for e in n["effects"]
                        if e in ("pm", "let")
                    }
                    label = "both" if len(effects) == 2 else next(iter(effects), "none")
                    events.append(
                        {
                            "voice": voice,
                            "onset": onset,
                            "measure_index": row["measure_index"],
                            "pixel": [
                                px * width + row["bbox"][0] - crop[0],
                                py * height + row["bbox"][1] - crop[1],
                            ],
                            "label": label,
                        }
                    )
                cases.append(
                    {
                        "id": row["id"],
                        "image": str(case_path),
                        "source_page": str(page_path),
                        "source_pdf": pdf_path,
                        "source_layout": rows[0]['source_layout'],
                        "previous_system_image": str(previous_path) if previous_path else None,
                        "bbox": crop,
                        "mode": row["mode"],
                        "events": events,
                        "family": row["family"],
                        "split": row["split"],
                        "page": row["page"],
                        "measures": [
                            {
                                "measure_index": row["measure_index"],
                                "target": row["target"],
                                "query_image": row["image"],
                                "query_bbox": row["bbox"],
                            }
                        ],
                        "label_provenance": "source event union, not automatically verified visual coverage",
                        "note_pixels": "exact native draw-time note owner bounds",
                        "renderer": "guitar_pro",
                    }
                )
    return cases


def build(output, source_root, limit=0, measures_per_source=24, splits=('train', 'validation')):
    output = output.resolve()
    source_root = source_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((source_root / 'source_catalog.json').read_text())['sources']
    families, reports = {}, {}
    for kind in ('grounding', 'relations'):
        (output / kind).mkdir()
    for split in splits:
        rows, selected, reasons = [], 0, Counter()
        for source in sorted(catalog, key=lambda r: r['source_id']):
            if source['split'] != split:
                continue
            if families.setdefault(source['family'], split) != split:
                raise ValueError('Native source family crosses dataset splits')
            current, excluded = extract(source_root, source, output, limit=measures_per_source)
            reasons.update(excluded)
            if not current:
                continue
            rows.extend(current)
            selected += 1
            if limit and selected >= limit:
                break
        cases = relation_cases(rows, output / split / 'systems')
        compiled = []
        for case in cases:
            spans = native_spans(case['source_pdf'], case['page'])
            case['events'] = [visible_event(e, spans, case['bbox']) for e in case['events']]
            case['label_provenance'] = 'Native PDF text and dashed spans; ambiguous fields remain unknown'
            compiled.append(compile_row(case))
        for kind, values in [('grounding', rows), ('relations', compiled)]:
            (output / kind / f'manifest_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in values))
        reports[split] = dict(sources=selected, measures=len(rows), systems=len(compiled), exclusions=dict(reasons))
        print(split, reports[split], flush=True)
    for kind in ('grounding', 'relations'):
        (output / kind / 'provenance.json').write_text(json.dumps(reports, indent=2))
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Maximum sources per split; 0 uses all sources')
    p.add_argument('--measures-per-source', type=int, default=24)
    p.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
