"""Build native measure labels from the pitch state and content visible in each crop."""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image
from scorelib.pitch_context import convert_pitch_target
from scorelib.score import format_measure_target, parse_measure_target

from research.data.catalog import native_document
from research.data.technique_visibility import (
    native_pm_evidence,
    prepare_visual_techniques,
)


def pdf_spans(page):
    anchors = []
    words = page.words()
    for index, word in enumerate(words):
        left, top, right, bottom, text = word
        text = text.strip().lower().replace(" ", "")
        if text == "let" and index + 1 < len(words):
            x0, y0, x1, y1, following = words[index + 1]
            if following.casefold() == "ring" and abs(top - y0) < 1 and 0 <= x0 - right <= 6:
                right, bottom, text = x1, max(bottom, y1), "letring"
        kind = "pm" if text in ("p.m.", "pm") else "let" if text == "letring" else None
        if kind:
            anchors.append({"kind": kind, "text_bbox": [left, top, right, bottom]})
    dashes = page.dashed_lines()
    spans = []
    for anchor in anchors:
        left, top, right, bottom = anchor["text_bbox"]
        matching = [
            line
            for line in dashes
            # Text bounds include italic overhang and need not end before the
            # actual dash begins. Accept overlap with the label's trailing box.
            if left <= line[0] <= right + 6 and line[2] > right and top <= line[1] <= bottom
        ]
        if len(matching) == 1:
            spans.append(
                {
                    **anchor,
                    "start_x": (left + right) / 2,
                    "end_x": matching[0][2],
                    "line_y": matching[0][1],
                    "style": "labelled_dash",
                }
            )
        else:
            # A visible word supplies technique identity but no scope endpoint.
            # Multiple plausible strokes are also unresolved; never discard
            # the actual word or choose a line from source-effect switches.
            spans.append({**anchor, 'start_x': (left+right)/2,
                          'end_x': right, 'line_y': (top+bottom)/2,
                          'style': 'label_only'})
    return spans


@lru_cache(maxsize=2048)
def native_layout(label_path, mode):
    matches = list((native_document(label_path, mode) / 'tracks').glob('*/layout.json'))
    if len(matches) != 1:
        return None, None
    return json.loads(matches[0].read_text()), str(matches[0].resolve())


def prepare_visual_metadata(row):
    """Remove tempo only when native geometry proves it is outside the image.

    This deliberately leaves partially visible or unknown crops unchanged. The
    crop dimensions must agree with the repository's 180-DPI native crop rule
    (1.5 mm horizontal / 4 mm vertical padding); scans and resized derivatives
    are not inferred from stale source coordinates. Playback labels are kept.
    """
    if (row.get('renderer') != 'guitar_pro' or 'tempo=' not in row['target']
            or not row.get('label_json') or not row.get('bbox_mm') or not row.get('image')):
        return row
    index = row.get('bar_index', row.get('measure_index'))
    if index is None or int(index) < 0:
        # Without a known source measure, an empty indication lookup cannot
        # establish that this crop's tempo was never rendered.
        return row
    index = int(index)
    layout, path = native_layout(str(row['label_json']), row['mode'])
    if (not layout or layout.get('schema') != 'gpomr.render-layout'
            or 'tempo_indications' not in layout):
        return row
    x, y, width, height = map(float, row['bbox_mm'])
    crop = [max(0, x - 1.5), max(0, y - 4), x + width + 1.5, y + height + 4]
    page_number = int(row.get('page', 1))
    pages = layout.get('pages', [])
    if 0 < page_number <= len(pages) and pages[page_number - 1].get('bbox_mm'):
        px, py, pw, ph = pages[page_number - 1]['bbox_mm']
        crop[2], crop[3] = min(crop[2], px + pw), min(crop[3], py + ph)
    with Image.open(row['image']) as image:
        size = image.size
    expected_size = [(crop[2] - crop[0]) * 180 / 25.4, (crop[3] - crop[1]) * 180 / 25.4]
    if any(abs(actual - expected) > 2 for actual, expected in zip(size, expected_size)):
        return row
    indications = [r for r in layout['tempo_indications'] if r.get('master_measure_index') == index]
    for indication in indications:
        if int(indication['page']) != page_number:
            continue
        tx, ty, tw, th = indication['bbox_mm']
        # A conservative sub-mm margin retains uncertain edge/clipping cases.
        if (max(crop[0] - .3, tx) < min(crop[2] + .3, tx + tw)
                and max(crop[1] - .3, ty) < min(crop[3] + .3, ty + th)):
            return row
    measure = parse_measure_target(row['target'])
    if measure.get('tempo_quarter') is None:
        return row
    result = dict(row)
    result.setdefault('source_target', row['target'])
    result['metadata_visibility'] = {
        'rule': 'native_tempo_outside_measure_crop' if indications else 'native_tempo_not_rendered',
        'removed_tempo_quarter': measure['tempo_quarter'], 'layout': path,
        'crop_xyxy_mm': crop, 'crop_size_px': size, 'tempo_indications': indications,
    }
    measure['tempo_quarter'] = None
    result['target'] = format_measure_target(measure, row['mode'], preserve_playback=True)
    if 'target_utf8_bytes' in result:
        result['target_utf8_bytes'] = len(result['target'].encode('utf-8'))
    return result


NATIVE_OCTAVES = {None: 0, "8va": 12, "8vb": -12, "15ma": 24, "15mb": -24}


def native_pitch_contexts(score, capo):
    if len(score['tracks']) != 1 or len(score['tracks'][0]['staves']) != 1:
        raise ValueError('Pitch supervision requires one native track and staff')
    track = score['tracks'][0]
    return {
        bar['measure_index']: {
            'instrument_transpose': track['view_transposition_offset'],
            'clef': bar['clef'],
            'clef_octave': NATIVE_OCTAVES[bar['ottavia_name']],
            'capo': capo,
            'octave_spans': [],
        }
        for bar in track['staves'][0]['measures']
    }


def prepare_written_target(row):
    """Convert a declared sounding target once; never infer missing pitch state."""
    result = dict(row)
    if row['mode'] == 'tab' or row['instrument'] == 'drums':
        return result
    context = row.get('pitch_context')
    if not isinstance(context, dict) or any(
        type(context.get(key)) is not int
        for key in ('instrument_transpose', 'clef_octave', 'capo')
    ):
        raise ValueError('Regenerate labels with explicit native pitch context and capo')
    reference = row.get('pitch_reference')
    if reference not in {'written', 'sounding'}:
        raise ValueError('Regenerate labels with an explicit pitch_reference')
    if reference == 'sounding':
        result['sounding_target'] = row['target']
        result['target'] = convert_pitch_target(
            row['target'], context, to_written=True, mode=row['mode'])
    result.update(visual_pitch=True, written_pitch=True, pitch_reference='written')
    return result


def prepare_native_measure(row):
    """Keep crop labels and serialized training targets on the same evidence."""
    if row.get('renderer') != 'guitar_pro':
        return row
    label = row.get('label_json')
    if not label:
        return row
    result = dict(row)
    layout, path = native_layout(str(label), row['mode'])
    if (not layout or layout.get('schema') != 'gpomr.render-layout'
            or layout.get('display_mode') != row['mode'] or not layout.get('pdf_ok')):
        raise ValueError(f"Unverified native render: {row['id']}")
    index = row.get('bar_index', row.get('measure_index'))
    boxes = [box['bbox_mm'] for system in layout['systems']
             if system['page'] == row['page'] for box in system['measure_boxes']
             if box['measure_index'] == index]
    if len(boxes) != 1:
        raise ValueError(f"Missing or ambiguous source measure: {row['id']}")
    if row.get('bbox') and any(abs(a - b * 180 / 25.4) > 2
                               for a, b in zip(row['bbox'], boxes[0])):
        raise ValueError(f"Native crop no longer matches its source: {row['id']}")
    result.update(renderer='guitar_pro', bbox_mm=boxes[0], native_layout=path)
    evidence = native_pm_evidence(str(label), row['mode'])
    if evidence is not None:
        result['technique_evidence'] = evidence
    return prepare_visual_metadata(prepare_visual_techniques(result))


def measure_boxes(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if layout.get("schema") != "gpomr.render-layout":
        raise ValueError("Regenerate layout with the current native renderer")
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
    return {"beat_unit": "quarter", "dotted": False, "bpm": int(tempo["value"])}


@lru_cache(maxsize=8192)
def source_profile(label_path):
    label = json.loads(Path(label_path).read_text())
    track = label['track']
    instrument = track.get('instrument')
    program = track.get('midi_program')
    tuning = list(track.get('tuning_midi_high_to_low') or [])
    if instrument not in {'guitar', 'bass', 'drums', 'pitched'} or program is None:
        raise ValueError(f'Regenerate the source label with instrument metadata: {label_path}')
    if instrument in {'guitar', 'bass'} and not tuning:
        raise ValueError(f'Regenerate the source label with its actual tuning: {label_path}')
    if instrument not in {'guitar', 'bass'}:
        tuning = []
    return {'instrument': instrument, 'midi_program': int(program),
            'tuning': tuning, 'string_count': len(tuning)}


def visible_clef(page, region, bar):
    """BarView also exposes the TAB symbol as a clef element in mixed mode."""
    x, y, w, h = region["bbox_mm"]
    printed = page.text((x * 72 / 25.4, y * 72 / 25.4,
                         (x + w) * 72 / 25.4, (y + h) * 72 / 25.4))
    if any(glyph in printed for glyph in ("\ue06d", "\ue06e")):
        return {"clef": "tab", "clef_octave": 0}
    clef = "percussion" if bar["clef"] == "Neutral" else bar["clef"]
    if clef not in {"G2", "F4", "C3", "C4", "percussion"}:
        raise ValueError(f"Unmapped native clef: {clef}")
    return {"clef": clef, "clef_octave": NATIVE_OCTAVES[bar.get("ottavia_name")]}


def visible_text_box(page, text):
    """Use native PDF character positions, never a guessed header rectangle."""
    words = page.words()
    target = "".join(text.lower().split())
    for start in range(len(words)):
        found = ""
        for end in range(start, min(len(words), start + 18)):
            found += "".join(words[end][4].lower().split())
            if not target.startswith(found):
                break
            if found == target:
                group = words[start:end + 1]
                return [min(w[0] for w in group), min(w[1] for w in group),
                        max(w[2] for w in group), max(w[3] for w in group)]
    return None


def native_pitch_annotations(page, page_number, layout, score, assignment):
    """Read visible clefs and pitch instructions once for layout and OCR data."""
    track = score['tracks'][0]
    bars = {(staff_index, bar['measure_index']): bar
            for staff_index, staff in enumerate(track['staves']) for bar in staff['measures']}
    if ('expected_native_transpose' in assignment
            and track['view_transposition_offset'] != assignment['expected_native_transpose']):
        raise ValueError(f"Native transposition differs: {assignment['source_id']}")
    rows = []
    for region in layout.get('pitch_regions', []):
        if region['page'] != page_number:
            continue
        if region['kind'] == 'clef':
            task, label = 'clef', 'clef_region'
            target = visible_clef(page, region, bars[region['staff_index'], region['measure_index']])
        elif region['type_code'] == 23:
            x, y, w, h = region['bbox_mm']
            printed = page.text((x * 72 / 25.4, y * 72 / 25.4, (x + w) * 72 / 25.4, (y + h) * 72 / 25.4))
            shifts = {shift for glyph, shift in {'\ue511': 12, '\ue51c': -12, '\ue515': 24, '\ue51d': -24}.items() if glyph in printed}
            shift = next(iter(shifts)) if len(shifts) == 1 else None
            task, label = 'annotation', 'annotation_region'
            target = dict(kind='ottava', semitones=shift, capo=None,
                          text={12: '8va', -12: '8vb', 24: '15ma', -24: '15mb'}.get(shift))
        else:
            continue
        rows.append(dict(label=label, bbox_mm=region['bbox_mm'], task=task, target=target))
    if page_number == 1 and (instruction := assignment.get('instruction')):
        box = visible_text_box(page, instruction)
        if box is None:
            raise ValueError(f"Pitch instruction is not visible: {assignment['source_id']}")
        x0, y0, x1, y1 = [v * 25.4 / 72 for v in box]
        capo = assignment.get('instruction_kind') == 'capo'
        rows.append(dict(label='annotation_region', bbox_mm=[x0, y0, x1-x0, y1-y0], task='annotation',
                         target=dict(kind='capo' if capo else 'instrument',
                                     semitones=None if capo else track['view_transposition_offset'],
                                     capo=assignment['capo'] if capo else None, text=instruction)))
    return rows

def native_scope_evidence(page, spans, grouped, staff_index, system, index, crop_mm, strings=None):
    """Preserve drawn scopes even when their endpoint or voice is not visible."""
    left, top, right, bottom = [v * 72 / 25.4 for v in crop_mm]
    glyphs_by_owner = {}
    for key, glyphs in grouped.items():
        if key[:2] != (staff_index, index):
            continue
        visible = []
        for glyph in glyphs:
            x, y, width, height = glyph['bbox_mm']
            if (glyph.get('kind') != 'hidden_fret'
                    and crop_mm[0] <= x and x + width <= crop_mm[2]
                    and crop_mm[1] <= y and y + height <= crop_mm[3]):
                visible.append(glyph)
        if visible:
            glyphs_by_owner.setdefault(key[:4], []).extend(visible)
    # An endpoint outside this crop cannot select between simultaneous voices
    # inside it. Cross-measure voice continuity belongs to the later assembler.
    local_voices = {key[2] for key in glyphs_by_owner}

    def candidates(position):
        x = position * 25.4 / 72
        matches = [(key, glyph) for key, glyphs in glyphs_by_owner.items() for glyph in glyphs
                   if glyph['bbox_mm'][0] - 1e-4 <= x
                   <= glyph['bbox_mm'][0] + glyph['bbox_mm'][2] + 1e-4]
        # A two-digit fret can reach a dash endpoint that a simultaneous
        # one-digit fret does not. Their shared drawing origin keeps both
        # voices possible; glyph width is not evidence of scope ownership.
        origins = [glyph['render_origin_mm'][0] for _, glyph in matches
                   if glyph.get('render_origin_mm')]
        return {key for key, _ in matches} | {
            key for key, glyphs in glyphs_by_owner.items()
            if any(glyph.get('render_origin_mm')
                   and any(abs(glyph['render_origin_mm'][0] - origin) <= 1e-4 for origin in origins)
                   for glyph in glyphs)}

    # Native glyph origins supply the actual TAB string spacing. This is PDF
    # annotation geometry, not a raster line detector or inference heuristic.
    origins = {}
    for glyphs in glyphs_by_owner.values():
        for glyph in glyphs:
            if glyph.get('native_string_index') is not None and glyph.get('render_origin_mm'):
                origins.setdefault(glyph['native_string_index'], []).append(glyph['render_origin_mm'][1])
    origin_y = {string: sum(ys) / len(ys) for string, ys in origins.items()}
    staff_geometry = None
    if strings and len(origin_y) >= 2:
        lo, hi = min(origin_y), max(origin_y)
        spacing = (origin_y[lo] - origin_y[hi]) / (hi - lo)
        if spacing > 0 and all(abs(origin_y[n] - (origin_y[lo] - (n - lo) * spacing)) < 1e-3
                               for n in origin_y):
            staff_geometry = (origin_y[lo] - (strings - 1 - lo) * spacing, (strings - 1) * spacing)

    relations, labels, paths = [], [], []
    for span in spans:
        x0, y0, x1, y1 = span['text_bbox']
        label_intersects = x0 < right and x1 > left and y0 < bottom and y1 > top
        line_visible = (span['style'] == 'labelled_dash' and top <= span['line_y'] <= bottom
                        and span['start_x'] < right and span['end_x'] > left)
        if not label_intersects and not line_visible:
            continue
        label_visible = left <= x0 and x1 <= right and top <= y0 and y1 <= bottom
        if label_intersects:
            labels.append(span['text_bbox'])
        start_candidates = candidates(span['start_x'])
        end_candidates = candidates(span['end_x']) if span['style'] == 'labelled_dash' else set()
        start_voices = {owner[2] for owner in start_candidates}
        end_voices = {owner[2] for owner in end_candidates}
        possible_voices = start_voices & end_voices if start_voices and end_voices else start_voices or end_voices
        voice = next(iter(possible_voices)) if len(possible_voices) == 1 else None
        if voice is None and not possible_voices and len(local_voices) == 1:
            voice = next(iter(local_voices))
        if voice is not None:
            start_candidates = {owner for owner in start_candidates if owner[2] == voice}
            end_candidates = {owner for owner in end_candidates if owner[2] == voice}
        start = next(iter(start_candidates)) if len(start_candidates) == 1 else None
        end = next(iter(end_candidates)) if len(end_candidates) == 1 else None
        if start and end and (start[2] != end[2] or (start[1], start[3]) > (end[1], end[3])):
            start = end = voice = None
        lines = [line for line in page.dashed_lines()
                 if abs(line[1] - span['line_y']) < 1e-6 and abs(line[2] - span['end_x']) < 1e-6]
        line = lines[0] if len(lines) == 1 and span['style'] == 'labelled_dash' else None
        relation = {'kind': span['kind'] if label_visible else None,
                    'start': [start[2], start[3]] if start and start[1] == index and label_visible
                             else 'left' if line and line[0] < left < line[2] else None,
                    'end': [end[2], end[3]] if end and end[1] == index
                           else 'right' if line and line[0] < right < line[2] else None}
        if voice is not None:
            relation['voice'] = voice
        if relation['kind'] is None:
            relation['shape'] = 'extension'
        if staff_geometry:
            staff_top, staff_height = staff_geometry
            relation['lane'] = round((span['line_y'] * 25.4 / 72 - staff_top) / staff_height, 3)
        relations.append(relation)
        if line:
            paths.append((line[0], line[1], line[2], line[1]))
            for stroke in page._text_page().lines:
                if (abs(stroke['x0'] - span['end_x']) < 1e-6 and abs(stroke['x1'] - span['end_x']) < 1e-6
                        and (abs(stroke['top'] - span['line_y']) < 1e-6
                             or abs(stroke['bottom'] - span['line_y']) < 1e-6)
                        and y0 <= stroke['top'] <= stroke['bottom'] <= y1):
                    paths.append((stroke['x0'], stroke['top'], stroke['x1'], stroke['bottom']))
    return relations, labels, paths


def native_slide_owners(points_mm, grouped, staff_index, system):
    """Join a drawn slide to its visibly adjacent fret glyphs.

    GP leaves space between the stroke and digits. Read adjacency within the
    stroke endpoint's vertical band instead of assuming that spacing in mm.
    Multiple owners at the same visible position remain unresolved; a stroke
    that runs to the system edge has no invented owner on the next system.
    """
    if len(points_mm) != 2:
        return None, None
    measures = {box['measure_index'] for box in system['measure_boxes']}
    visible = [(owner, glyph) for owner, glyphs in grouped.items()
               if owner[0] == staff_index and owner[1] in measures
               for glyph in glyphs]

    def adjacent(point, before):
        px, py = point
        candidates = []
        for owner, glyph in visible:
            x, y, width, height = glyph['bbox_mm']
            if not y - 1e-5 <= py <= y + height + 1e-5:
                continue
            distance = px - x - width if before else x - px
            if distance >= -1e-5:
                candidates.append((max(0, distance), owner, glyph))
        if not candidates:
            return None
        nearest = min(candidate[0] for candidate in candidates)
        neighbors = [(owner, glyph) for distance, owner, glyph in candidates
                     if abs(distance - nearest) <= 1e-5]
        owners = {owner for owner, glyph in neighbors if glyph['kind'] == 'fret'}
        if len(owners) != 1 or any(glyph['kind'] != 'fret' for _, glyph in neighbors):
            return None
        # Different digit widths at one printed onset cannot distinguish
        # overlapping voice owners. Compare their actual drawing anchors too.
        anchor = neighbors[0][1]
        origin_x = anchor.get('render_origin_mm', [anchor['bbox_mm'][0] + anchor['bbox_mm'][2] / 2])[0]
        coincident = {owner for owner, glyph in visible if glyph['kind'] == 'fret'
                      and glyph['bbox_mm'][1] - 1e-5 <= py <= glyph['bbox_mm'][1] + glyph['bbox_mm'][3] + 1e-5
                      and abs(glyph.get('render_origin_mm', [glyph['bbox_mm'][0] + glyph['bbox_mm'][2] / 2])[0]
                              - origin_x) <= 1e-5}
        if len(coincident) != 1:
            return None
        return next(iter(owners))

    start, end = sorted(points_mm, key=lambda point: point[0])
    return adjacent(start, True), adjacent(end, False)


def native_single_slides(page, grouped, staff_index, system, index, crop_mm, consumed_lines):
    """Read GP's short, single-ended diagonal slide symbols from printed ink.

    The native symbol is a square diagonal beside one fret, with an open other
    end. Paired slide strokes have already been consumed by draw-time tracing.
    Contact is measured in the symbol's own extent, so scaling or font size
    cannot change a label through a fixed pixel or millimetre cutoff.
    """
    effects, consumed, unknown = {}, set(), []
    scale = 25.4 / 72
    left, top, right, bottom = crop_mm
    for li, line in enumerate(page._text_page().lines):
        if li in consumed_lines or not line.get('stroke'):
            continue
        points = sorted([[x * scale, y * scale] for x, y in line.get('pts', [])])
        if len(points) != 2:
            continue
        (x0, y0), (x1, y1) = points
        width, height = x1 - x0, abs(y1 - y0)
        if width <= 1e-5 or height <= 1e-5:
            continue
        if not (x0 < right and x1 > left and min(y0, y1) < bottom and max(y0, y1) > top):
            continue
        complete = left <= x0 < x1 <= right and top <= min(y0, y1) and max(y0, y1) <= bottom
        owners = native_slide_owners(points, grouped, staff_index, system)
        contacts = []
        visible_sides = []
        for before, owner in zip((True, False), owners):
            px, py = points[0 if before else 1]
            near = [g for key, glyphs in grouped.items() if key[:2] == (staff_index, index)
                    for g in glyphs if g['kind'] == 'fret'
                    and g['bbox_mm'][1] <= py <= g['bbox_mm'][1] + g['bbox_mm'][3]
                    and -1e-5 <= (px - g['bbox_mm'][0] - g['bbox_mm'][2] if before
                                  else g['bbox_mm'][0] - px) <= min(width, g['bbox_mm'][3])]
            if near:
                visible_sides.append(before)
            if owner is None or owner[1] != index:
                continue
            glyphs = [g for g in grouped[owner] if g['kind'] == 'fret']
            if any((x0 - g['bbox_mm'][0] - g['bbox_mm'][2] if before
                    else g['bbox_mm'][0] - x1) <= min(width, g['bbox_mm'][3]) for g in glyphs):
                contacts.append((before, owner))
        single = (complete and abs(width - height) <= 1e-4 and len(contacts) == 1
                  and len(visible_sides) == 1
                  and any(width <= g['bbox_mm'][3] for g in grouped[contacts[0][1]]))
        if not single:
            if visible_sides:
                unknown.append({'kind': None, 'shape': 'diagonal',
                                'start': list(owners[0][2:]) if owners[0] is not None
                                         and owners[0][1] == index and True in visible_sides else None,
                                'end': list(owners[1][2:]) if owners[1] is not None
                                       and owners[1][1] == index and False in visible_sides else None})
                consumed.add(li)
            continue
        outgoing, owner = contacts[0]
        effect = ('sod' if y1 > y0 else 'sou') if outgoing else ('sia' if y1 > y0 else 'sib')
        effects.setdefault(owner, set()).add(effect)
        consumed.add(li)
    return effects, consumed, unknown


def native_tuplet_evidence(page, grouped, staff_index, system, index, crop_mm):
    """Return drawn tuplet counts, consumed bracket paths and number boxes.

    Counts come from the printed number between two actual bracket halves;
    owners come from bracket endpoints inside draw-time fret/rest bounds. The
    caller checks the native rhythmic ratio against this visible count before
    using it. No source tuplet switch can create evidence here.
    """
    left, top, right, bottom = [v * 72 / 25.4 for v in crop_mm]
    drawing = page._text_page()
    measures = {box['measure_index'] for box in system['measure_boxes']}
    owners = {key[:4] for key in grouped if key[0] == staff_index and key[1] in measures}

    def candidates(position):
        x = position * 25.4 / 72
        return {key[:4] for key, glyphs in grouped.items()
                if key[0] == staff_index and key[1] in measures
                and any(g['bbox_mm'][0] - 1e-4 <= x <= g['bbox_mm'][0] + g['bbox_mm'][2] + 1e-4
                        for g in glyphs)}

    halves = []
    for ci, curve in enumerate(drawing.curves):
        path = curve.get('path', [])
        if ''.join(command[0] for command in path) != 'mll' or not curve.get('stroke'):
            continue
        a, b, c = [command[1] for command in path]
        if abs(a[0] - b[0]) < 1e-5 or abs(b[0] - c[0]) > 1e-5 or abs(b[1] - c[1]) < 1e-5:
            continue
        if not (left <= min(a[0], b[0]) and max(a[0], b[0]) <= right
                and top <= min(a[1], b[1], c[1]) and max(a[1], b[1], c[1]) <= bottom):
            continue
        halves.append((ci, a, b, c))
    counts, consumed, labels = {}, set(), []
    for li, la, lb, lc in halves:
        if lb[0] >= la[0]:
            continue
        for ri, ra, rb, rc in halves:
            slope = (la[1] - lb[1]) / (la[0] - lb[0])
            if not (ra[0] < rb[0] and la[0] < ra[0]
                    and abs((rb[1] - ra[1]) / (rb[0] - ra[0]) - slope) < 1e-5
                    and abs(ra[1] - (la[1] + slope * (ra[0] - la[0]))) < 1e-5
                    and (lc[1] - lb[1]) * (rc[1] - rb[1]) > 0):
                continue
            if any(la[0] < other_a[0] < ra[0]
                   and abs(other_a[1] - (la[1] + slope * (other_a[0] - la[0]))) < 1e-5
                   for _, other_a, _, _ in halves):
                continue
            digits = [char for char in drawing.chars
                      if char['text'].isascii() and char['text'].isdigit()
                      and la[0] <= (char['x0'] + char['x1']) / 2 <= ra[0]
                      and char['top'] <= la[1] + slope * ((char['x0'] + char['x1']) / 2 - la[0]) <= char['bottom']
                      and left <= char['x0'] and char['x1'] <= right
                      and top <= char['top'] and char['bottom'] <= bottom]
            digits.sort(key=lambda char: char['x0'])
            printed = ''.join(char['text'] for char in digits)
            if not printed or int(printed) < 2:
                continue
            starts, ends = candidates(lb[0]), candidates(rb[0])
            pairs = [(a, b) for a in starts for b in ends if a[2] == b[2]
                     and (a[1], a[3]) <= (b[1], b[3])]
            if len(pairs) != 1:
                continue
            first, last = pairs[0]
            covered = {owner for owner in owners if owner[1] == index and owner[2] == first[2]
                       and (first[1], first[3]) <= (owner[1], owner[3]) <= (last[1], last[3])}
            count = int(printed)
            if any(owner in counts and counts[owner] != count for owner in covered):
                raise ValueError('overlapping_native_tuplet_brackets')
            counts.update(dict.fromkeys(covered, count))
            consumed.update((li, ri))
            labels.append([min(char['x0'] for char in digits), min(char['top'] for char in digits),
                           max(char['x1'] for char in digits), max(char['bottom'] for char in digits)])
    return counts, consumed, labels


def native_event_symbol_evidence(drawing, grouped, staff_index, index, crop_mm):
    """Place printed dynamics and picking signs at their visible event anchor."""
    from research.data.observations import DYNAMIC_SYMBOLS, PICK_SYMBOLS

    symbols = {code: f'dyn:{value}' for code, value in DYNAMIC_SYMBOLS.items()} | PICK_SYMBOLS

    left, top, right, bottom = crop_mm
    effects, boxes = {}, []
    for char in drawing.chars:
        if len(char['text']) != 1 or ord(char['text']) not in symbols:
            continue
        x0, y0, x1, y1 = [char[k] * 25.4 / 72 for k in ('x0', 'top', 'x1', 'bottom')]
        if not (left <= x0 <= x1 <= right and top <= y0 <= y1 <= bottom):
            continue
        effect = symbols[ord(char['text'])]
        candidates = {key[:4] for key, glyphs in grouped.items()
                      if key[:2] == (staff_index, index)
                      and any(x0 <= (g['bbox_mm'][0] + g['bbox_mm'][2] / 2
                                      if effect.startswith('pick_') else g['render_origin_mm'][0]) <= x1
                              for g in glyphs)}
        # Simultaneous voices share a staff dynamic, but a picking sign
        # requires a unique event owner.
        events = {(key[1], key[3]) for key in candidates}
        if not candidates or len(events) != 1 or effect.startswith('pick_') and len(candidates) != 1:
            raise ValueError('printed_event_symbol_without_visible_onset')
        owner = min(candidates)
        if effect not in effects.setdefault(owner, []):
            effects[owner].append(effect)
        boxes.append([char[k] for k in ('x0', 'top', 'x1', 'bottom')])
    return effects, boxes


def native_connection_representation(raw, grouped, staff_index, system):
    """Locate a painted connection in its actual TAB or notation drawing."""
    owners = [owner for owner in (raw.get('start_owner'), raw.get('end_owner')) if owner]
    kinds = {'tab' if owner['fret_view'] else 'notation' for owner in owners}
    if len(kinds) == 1:
        return kinds.pop()
    points = raw.get('points_mm', [])
    if not points:
        return None
    measures = {box['measure_index'] for box in system['measure_boxes']}
    distances = {}
    for key, glyphs in grouped.items():
        if key[0] != staff_index or key[1] not in measures:
            continue
        for glyph in glyphs:
            if glyph['kind'] not in ('fret', 'notehead'):
                continue
            representation = 'tab' if glyph['kind'] == 'fret' else 'notation'
            x, y, w, h = glyph['bbox_mm']
            distance = min(max(x - px, 0, px - x - w) ** 2 + max(y - py, 0, py - y - h) ** 2
                           for px, py in (points[0], points[-1]))
            distances[representation] = min(distance, distances.get(representation, float('inf')))
    if not distances or len(set(distances.values())) < len(distances):
        return None
    return min(distances, key=distances.get)


def native_connection_evidence(page, layout, grouped, staff_index, system, index, crop_mm,
                               other_staff_relations=()):
    """Read note connections from actual draw paths and local PDF lettering.

    Native owner pointers locate notes; class names and hidden fret strings do
    not supply visual technique labels. Hidden anchors contain no fret value.
    """
    import re

    drawing = page._text_page()
    factor = 72 / 25.4
    left, top, right, bottom = [v * factor for v in crop_mm]
    measures = {box['measure_index'] for box in system['measure_boxes']}
    geometry = layout.get('relation_geometry', {})
    relations, effects, hidden, curves, lines, labels = [], {}, {}, set(), set(), []

    def owner_key(owner):
        if owner is None or owner.get('staff_index') != staff_index or owner.get('note_index') is None:
            return None
        return tuple(owner[field] for field in ('staff_index', 'measure_index', 'voice_index', 'event_index', 'note_index'))

    def intersects(box):
        return box[0] < right and box[2] > left and box[1] < bottom and box[3] > top

    def bbox_of(points):
        return [min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points)]

    def matching_points(a, b):
        # Both exports serialize the same painter coordinates. This tolerance
        # covers numerical serialization, less than 0.001 pixel at training DPI.
        return len(a) == len(b) and all(abs(x - y) < 1e-4 for p, q in zip(a, b) for x, y in zip(p, q))

    def head_owner(point):
        # GP's notation beat-slur painter omits owner pointers. Its endpoints
        # still align with actual painted heads; ambiguous unisons stay unknown.
        px, py = point
        candidates = []
        for key, glyphs in grouped.items():
            if key[0] != staff_index or key[1] not in measures:
                continue
            for glyph in glyphs:
                if glyph['kind'] != 'notehead':
                    continue
                x, y, width, height = glyph['bbox_mm']
                if x - 1e-4 <= px <= x + width + 1e-4:
                    candidates.append((max(y - py, 0, py - y - height), key))
        if not candidates:
            return None
        nearest = min(distance for distance, _ in candidates)
        owners = {key for distance, key in candidates if abs(distance - nearest) < 1e-4}
        return next(iter(owners)) if len(owners) == 1 else None

    drawn = []
    for raw in geometry.get('relations', []) if geometry.get('status') == 'complete' else []:
        if raw.get('page') != system['page'] or raw.get('system_index') != system['system_index']:
            continue
        points = [[x * factor, y * factor] for x, y in raw.get('points_mm', [])]
        if len(points) not in (2, 4):
            continue
        matched = []
        if len(points) == 4:
            for ci, curve in enumerate(drawing.curves):
                path = curve.get('path', [])
                if len(path) >= 2 and path[0][0] == 'm' and path[1][0] == 'c':
                    actual = [path[0][1], *path[1][1:]]
                    if matching_points(points, actual) or matching_points(points, list(reversed(actual))):
                        matched.append(ci)
        else:
            for li, line in enumerate(drawing.lines):
                actual = line.get('pts', [])
                if matching_points(points, actual) or matching_points(points, list(reversed(actual))):
                    matched.append(li)
        if len(matched) != 1:
            continue
        start, end = owner_key(raw.get('start_owner')), owner_key(raw.get('end_owner'))
        if len(points) == 4 and start is None and end is None:
            a, b = head_owner(raw['points_mm'][0]), head_owner(raw['points_mm'][-1])
            if a and b and a[2] == b[2] and (a[1], a[3]) < (b[1], b[3]):
                start, end = a, b
        if len(points) == 2 and start is None and end is None:
            start, end = native_slide_owners(raw['points_mm'], grouped, staff_index, system)
        record = dict(raw=raw, points=points, box=bbox_of(points), start=start, end=end,
                      shape='curve' if len(points) == 4 else 'diagonal', primitive=matched[0])
        drawn.append(record)
        for owner in (raw.get('start_owner'), raw.get('end_owner')):
            key = owner_key(owner)
            if (key is not None and key not in grouped and owner.get('visibility_code') == 1
                    and owner.get('render_origin_mm') and owner.get('layout_bbox_mm')
                    and owner.get('page') == system['page']):
                hidden[key] = {**{field: value for field, value in owner.items() if field != 'fret_text'},
                               'kind': 'hidden_fret', 'text': '',
                               'bbox_mm': list(owner['layout_bbox_mm'])}

    local_glyphs = {key: glyphs[0] for key, glyphs in grouped.items() if len(glyphs) == 1}
    local_glyphs.update(hidden)
    system_x = [glyph['render_origin_mm'][0] * factor for key, glyph in local_glyphs.items()
                if key[0] == staff_index and key[1] in measures and glyph.get('render_origin_mm')]

    def visible_text(key):
        glyph = local_glyphs.get(key)
        if glyph is None or glyph.get('kind') == 'hidden_fret':
            return None
        x, y, w, h = glyph['bbox_mm']
        if not (crop_mm[0] <= x and x + w <= crop_mm[2] and crop_mm[1] <= y and y + h <= crop_mm[3]):
            return None
        return (glyph.get('text') or '').strip()

    def numeric(text):
        match = re.fullmatch(r'\(?([0-9]+)\)?', text or '')
        return int(match[1]) if match else None

    # A diagonal and a curved arc with the same visible note owners form one
    # legato slide. Keep the accompanying ink, rather than two duplicate marks.
    merged, used = [], set()
    for di, item in enumerate(drawn):
        if di in used:
            continue
        members = [item]
        if item['start'] is not None or item['end'] is not None:
            companions = [(oi, other) for oi, other in enumerate(drawn) if oi != di and oi not in used
                          and other['shape'] != item['shape']
                          and (other['start'], other['end']) == (item['start'], item['end'])]
            if len(companions) == 1:
                oi, other = companions[0]
                members.append(other)
                used.add(oi)
        used.add(di)
        merged.append(members)
    sy = system['bbox_mm'][1] * factor
    sb = (system['bbox_mm'][1] + system['bbox_mm'][3]) * factor
    word_owners = {}
    for word in page.words():
        if word[4] not in ('H', 'P', 'sl.', 'sl') or not sy <= word[1] <= word[3] <= sb:
            continue
        center = (word[0] + word[2]) / 2
        matches = [gi for gi, members in enumerate(merged)
                   if any(min([item['box'][0], *(local_glyphs[key]['render_origin_mm'][0] * factor
                              for key in (item['start'], item['end']) if key in local_glyphs)]) <= center
                          <= max([item['box'][2], *(local_glyphs[key]['render_origin_mm'][0] * factor
                                  for key in (item['start'], item['end']) if key in local_glyphs)])
                          for item in members)]
        # Two staves can print the same H/P at the same x. Attach its actual
        # ink to the nearer drawn curve, independently for each staff.
        def distance(box):
            return max(box[1] - word[3], 0, word[1] - box[3])
        if matches:
            nearest = min(distance(item['box']) for gi in matches for item in merged[gi])
            other = [bbox_of([[x * factor, y * factor] for x, y in raw['points_mm']])
                     for raw in other_staff_relations
                     if raw.get('page') == system['page'] and raw.get('system_index') == system['system_index']
                     and raw.get('points_mm')]
            if any(box[0] <= center <= box[2] and distance(box) <= nearest for box in other):
                continue
        if len(matches) == 1:
            word_owners.setdefault(matches[0], []).append(word)

    for gi, members in enumerate(merged):
        visible = [item for item in members if intersects(item['box'])]
        if not visible:
            continue
        main = next((item for item in members if item['shape'] == 'diagonal'), members[0])
        start, end = main['start'], main['end']
        for item in visible:
            (curves if item['shape'] == 'curve' else lines).add(item['primitive'])
        words = word_owners.get(gi, [])
        printed = [word[4] for word in words if left <= word[0] and word[2] <= right
                   and top <= word[1] and word[3] <= bottom]
        labels.extend(list(word[:4]) for word in words if intersects(word[:4]))
        has_line = any(item['shape'] == 'diagonal' for item in visible)
        has_curve = any(item['shape'] == 'curve' for item in visible)
        kind = None
        if has_line:
            kind = 'sl' if has_curve else None if any(item['shape'] == 'curve' for item in members) else 'ss'
        elif any(text in ('H', 'P') for text in printed):
            kind = 'hammer'
        elif 'sl.' in printed or 'sl' in printed:
            kind = 'sl'
        elif end is not None and end[1] == index:
            a, b = visible_text(start), visible_text(end)
            if (local_glyphs.get(end, {}).get('kind') == 'hidden_fret' or (b and b.startswith('(') and b.endswith(')'))
                    or (a is not None and numeric(a) is not None and numeric(a) == numeric(b)
                        and local_glyphs[start].get('native_string_index') == local_glyphs[end].get('native_string_index'))):
                kind = 'tie'
        local_start = start is not None and start[1] == index
        local_end = end is not None and end[1] == index
        if kind in ('hammer', 'sl', 'ss') and local_start and local_end:
            origins = {start}
            hopo_words = [word for word in words if word[4] in ('H', 'P')
                          and left <= word[0] and word[2] <= right
                          and top <= word[1] and word[3] <= bottom]
            if kind == 'hammer' and len(hopo_words) > 1:
                # One arc can carry H/P labels for several consecutive
                # connections. Each visible label belongs to its own gap.
                anchors = [(glyph['render_origin_mm'][0] * factor, key)
                           for key, glyph in local_glyphs.items()
                           if key[:3] == start[:3] and start[3] <= key[3] <= end[3]
                           and visible_text(key) is not None]
                origins = set()
                for word in hopo_words:
                    center = (word[0] + word[2]) / 2
                    before = [(x, key) for x, key in anchors if x < center]
                    after = [(x, key) for x, key in anchors if x > center]
                    if not before or not after:
                        raise ValueError('hopo_label_without_visible_note_gap')
                    a = {key for x, key in before if abs(x - max(x for x, _ in before)) < 1e-4}
                    b = {key for x, key in after if abs(x - min(x for x, _ in after)) < 1e-4}
                    if len(a) != 1 or len(b) != 1:
                        raise ValueError('ambiguous_hopo_label_note_gap')
                    origins.update(a)
                if len(origins) != len(hopo_words):
                    raise ValueError('duplicate_hopo_label_note_gap')
            for origin in origins:
                effects.setdefault(origin, set()).add(kind)
            continue
        # At a system edge the painter can expose only one owner. Its open
        # segment is still visible even when it starts after the clef margin.
        minimum_x = min(item['box'][0] for item in visible)
        maximum_x = max(item['box'][2] for item in visible)
        enters = minimum_x < left or (index == min(measures) and system_x and minimum_x < min(system_x))
        leaves = maximum_x > right or (index == max(measures) and system_x and maximum_x > max(system_x))
        relation = {'kind': kind,
                    'start': list(start[2:]) if local_start else 'left'
                             if start is not None and start[1] < index or start is None and enters else None,
                    'end': list(end[2:]) if local_end else 'right'
                           if end is not None and end[1] > index or end is None and leaves else None,
                    'shape': 'curve' if has_curve else 'diagonal'}
        if start is None and end is None:
            # Unavailable ownership does not itself imply a crop crossing.
            # A fully visible unattached slur keeps unknown local endpoints.
            relation['start'] = 'left' if any(item['box'][0] < left for item in visible) else None
            relation['end'] = 'right' if any(item['box'][2] > right for item in visible) else None
        if not local_start and not local_end:
            voices = {key[2] for key in local_glyphs if key[:2] == (staff_index, index)
                      and visible_text(key) is not None}
            if len(voices) == 1:
                relation['voice'] = next(iter(voices))
        relations.append(relation)
    single_effects, single_lines, unknown_diagonals = native_single_slides(
        page, grouped, staff_index, system, index, crop_mm, lines)
    for owner, visible_effects in single_effects.items():
        effects.setdefault(owner, set()).update(visible_effects)
    lines.update(single_lines)
    relations.extend(unknown_diagonals)
    return relations, effects, hidden, curves, lines, labels


def native_chord_evidence(pdf_page, grouped, staff, measure, page, crop):
    """Read printed chord names and bind their centres to native beat origins.

    Source chord flags identify possible owners, never the answer's spelling.
    Page-header dictionaries have no beat anchor and stay outside this task.
    A partly cropped name is accounted for as print, but is not guessed.
    """
    from urllib.parse import quote
    from research.data.native_alignment import chord_words

    left, top, right, bottom = crop
    index = measure['measure_index']
    candidates = []
    for other in staff['measures']:
        for voice in other['voices']:
            for event in voice['events']:
                if not event.get('chord_present') or event['placeholder']:
                    continue
                owner = (staff['staff_index'], other['measure_index'], voice['voice_index'], event['event_index'])
                anchors = [g for key, glyphs in grouped.items() if key[:4] == owner
                           for g in glyphs if g['page'] == page]
                if anchors:
                    candidates.append((owner, sum(g['render_origin_mm'][0] for g in anchors) / len(anchors),
                                       min(g['bbox_mm'][1] for g in anchors)))
    effects, labels, evidence, diagram_labels, assigned = {}, [], [], [], set()
    for word in chord_words(pdf_page):
        x0, y0, x1, y1 = [v * 25.4 / 72 for v in word[:4]]
        if not (y0 < bottom and y1 > top):
            continue
        inside = x0 < right and x1 > left
        owners = [(owner, origin, note_y) for owner, origin, note_y in candidates
                  if abs(origin - (x0 + x1) / 2) <= .75 and note_y > y1]
        # Repeated x coordinates on later systems are not competing owners.
        if owners:
            nearest_y = min(v[2] for v in owners)
            owners = [v for v in owners if v[2] == nearest_y]
        if len(owners) != 1:
            # A chord spelling is only a candidate: H can be a traced hammer
            # label. Unbound words remain for the common printed-glyph check.
            continue
        owner, origin, _ = owners[0]
        diagram_labels.append(list(word[:4]))
        if not inside:
            continue
        labels.append(list(word[:4]))
        # A neighbour's full name may overlap this rectangular crop. It stays
        # context; its event belongs to the neighbour's own observation.
        if owner[1] != index:
            continue
        if owner in assigned:
            raise ValueError('multiple_printed_chords_for_native_beat')
        assigned.add(owner)
        complete = left <= x0 and x1 <= right and top <= y0 and y1 <= bottom
        if complete:
            effects[owner] = ['chord:' + quote(word[4], safe='')]
        evidence.append({'voice': owner[2], 'event': owner[3],
                         'name': word[4] if complete else None,
                         'bbox_mm': [x0, y0, x1 - x0, y1 - y0],
                         'beat_origin_mm': origin, 'complete': complete})
    return effects, labels, evidence, diagram_labels


def native_barre_path(curve):
    """A painted rounded horizontal bar has a vertical edge at both ends.

    GP ties can also be filled paths, but taper to a single point at their
    endpoints; fill/stroke flags alone cannot distinguish them from barres.
    """
    points = curve.get('pts', [])
    if (not curve.get('fill') or curve.get('stroke') or not points
            or not 0 < curve['bottom'] - curve['top'] < curve['x1'] - curve['x0']):
        return False
    return all(len({round(y, 6) for x, y in points if abs(x - edge) < 1e-5}) >= 2
               for edge in (curve['x0'], curve['x1']))


def native_staff_lines(drawing, system, mode, strings):
    """Read the staff grid from the PDF's actual horizontal drawing segments."""
    from collections import defaultdict

    x, y, w, h = system['bbox_mm']
    rows = defaultdict(float)
    for line in drawing.lines:
        x0, y0, x1, y1 = [line[k] * 25.4 / 72 for k in ('x0', 'top', 'x1', 'bottom')]
        if line.get('stroke') and abs(y0 - y1) < 1e-5 and y <= y0 <= y + h:
            rows[round(y0, 4)] += max(0, min(x1, x + w) - max(x0, x))
    lines = sorted(y for y, width in rows.items() if width > w * .65)
    expected = 5 + strings if mode == 'both' else 5
    if len(lines) != expected:
        raise ValueError('native_pdf_staff_line_count')
    notation = lines[:5]
    spacing = (notation[-1] - notation[0]) / 4
    if spacing <= 0 or any(abs(y - notation[0] - i * spacing) > 1e-3 for i, y in enumerate(notation)):
        raise ValueError('native_pdf_staff_spacing')
    return notation


def native_beam_path(curve, drawing):
    """A filled beam is a straight quadrilateral attached to printed stems."""
    path = curve.get('path', [])
    if not curve.get('fill') or curve.get('stroke') or ''.join(p[0] for p in path) != 'mllllh':
        return False
    a, b, c, d, close = [p[1] for p in path[:-1]]
    if (a != close or abs(a[0] - d[0]) > 1e-4 or abs(b[0] - c[0]) > 1e-4
            or abs(a[0] - b[0]) <= abs(a[1] - d[1])
            or abs((b[1] - a[1]) - (c[1] - d[1])) > 1e-4):
        return False
    return all(any(abs(line['x0'] - line['x1']) < 1e-5
                   and abs(line['x0'] - p[0]) <= line['linewidth'] / 2 + 1e-4
                   and line['top'] <= max(p[1], q[1]) and line['bottom'] >= min(p[1], q[1])
                   for line in drawing.lines) for p, q in ((a, d), (b, c)))


def native_shared_tab_rhythm(drawing, grouped, staff_index, measure, crop_mm, tuplet_counts):
    """Reject competing source rhythms unsupported by a shared drawn stem.

    TAB can print different source voices on one rhythmic column. Fret owners
    identify the notes, but do not give each voice its own stem. Printed dots
    and traced tuplet brackets may disambiguate the shared rhythm; otherwise
    conflicting durations remain unknown.
    """
    index = measure['measure_index']
    left, top, right, bottom = crop_mm
    events, columns, origins = {}, [], []
    for voice in measure['voices']:
        for event in voice['events']:
            owner = (staff_index, index, voice['voice_index'], event['event_index'])
            glyphs = [g for key, gs in grouped.items() if key[:4] == owner for g in gs]
            if not glyphs:
                continue
            x = glyphs[0]['render_origin_mm'][0]
            origins.append(x)
            if (event['rest'] or event['placeholder']
                    or any(abs(g['render_origin_mm'][0] - x) > 1e-4 for g in glyphs)):
                continue
            rhythm = event['rhythm']
            events[owner] = (rhythm['note_value_code'], rhythm['augmentation_dots'],
                             tuple(rhythm['tuplets']['primary'] or (1, 1)))
            column = next((c for c in columns if abs(c[0] - x) < 1e-4), None)
            if column is None:
                column = (x, [], [])
                columns.append(column)
            column[1].append(owner)
            column[2].extend(glyphs)
    unknown = set()
    for x, owners, glyphs in columns:
        if len({o[2] for o in owners}) < 2 or len({events[o] for o in owners}) < 2:
            continue
        band_top = min(g['bbox_mm'][1] for g in glyphs)
        band_bottom = max(g['bbox_mm'][1] + g['bbox_mm'][3] for g in glyphs)
        # Internal pieces between fret digits belong to the same column.
        # An independent stem above and below the notes is not a shared stem.
        stems = [(line['top'] * 25.4 / 72, line['bottom'] * 25.4 / 72)
                 for line in drawing.lines
                 if line.get('stroke') and abs(line['x0'] - line['x1']) < 1e-6
                 and abs(line['x0'] * 25.4 / 72 - x) < 1e-4
                 and top <= line['top'] * 25.4 / 72 <= line['bottom'] * 25.4 / 72 <= bottom
                 and (line['top'] * 25.4 / 72 < band_top
                      or line['bottom'] * 25.4 / 72 > band_bottom)]
        if len(stems) != 1:
            continue
        stem_top, stem_bottom = stems[0]
        next_x = min((other for other in origins if other > x + 1e-4), default=right)
        dots = sum(char['text'].count('\ue1e7') for char in drawing.chars
                   if x < char['x0'] * 25.4 / 72 < char['x1'] * 25.4 / 72 < next_x
                   and top <= char['top'] * 25.4 / 72 <= char['bottom'] * 25.4 / 72 <= bottom
                   and (char['top'] * 25.4 / 72 <= stem_top <= char['bottom'] * 25.4 / 72
                        or char['top'] * 25.4 / 72 <= stem_bottom <= char['bottom'] * 25.4 / 72))
        counts = {tuplet_counts[o] for o in owners if o in tuplet_counts}
        # An unprinted source tuplet was already unreadable before considering
        # shared stems; it must not erase another voice's visible plain rhythm.
        matches = set()
        for owner in owners:
            _, source_dots, (enters, times) = events[owner]
            ratio_visible = (enters == times or enters > 2 and enters & (enters - 1) != 0
                and times == 1 << (enters.bit_length() - 1) and tuplet_counts.get(owner) == enters)
            if ratio_visible and source_dots == dots and (not counts or counts == {enters}):
                matches.add(owner)
        # A symbol can rule out a source duration without proving which of two
        # remaining base values owns the only stem. Do not choose by voice id.
        if len({events[o] for o in matches}) != 1:
            matches.clear()
        unknown.update(set(owners) - matches)
    return unknown


def native_observations(layout, score, document, dpi=180):
    """Compile drawn notes, rhythm and connections, retaining visible unknowns.

    Numeric frets come only from FretView draw text. The rendered native score
    supplies the owning event and its rhythm, never an invisible fret or effect.
    A missing glyph needs an actually hidden FretView attached to a drawn
    connector; note_geometry.complete alone does not prove invisibility.
    """
    from fractions import Fraction
    import math
    from gpbridge.score_native_note_geometry import group_glyphs_by_owner
    from scorelib.score import _duration_token, duration_ticks

    mode = layout['display_mode']
    key_names = ['CbMajor', 'GbMajor', 'DbMajor', 'AbMajor', 'EbMajor', 'BbMajor', 'FMajor',
                 'CMajor', 'GMajor', 'DMajor', 'AMajor', 'EMajor', 'BMajor', 'F#Major', 'C#Major']
    if not layout.get('pdf_ok') or not score.get('pdf_ok'):
        raise ValueError('native_pdf_export_incomplete')
    if len(score['tracks']) != 1 or len(score['tracks'][0]['staves']) != 1:
        raise ValueError('native_observation_requires_one_staff')
    all_glyphs = group_glyphs_by_owner(score, layout, require_complete=True)
    primary = 'notehead' if mode == 'notation' else 'fret'
    grouped = {key: [g for g in glyphs if g['kind'] in (primary, 'rest')]
               for key, glyphs in all_glyphs.items()}
    grouped = {key: glyphs for key, glyphs in grouped.items() if glyphs}
    track = score['tracks'][0]
    staff = track['staves'][0]
    strings = staff['string_count']
    percussion = all(m['clef'] == 'Neutral' for m in staff['measures'])
    if mode != 'notation' and not 4 <= strings <= 8:
        raise ValueError('unsupported_tab_string_count')
    systems = {box['measure_index']: system for system in layout['systems']
               for box in system['measure_boxes']}
    masters = {m['master_measure_index']: m for m in score['document']['master_measures']}
    note_marks = ('bend', 'harmonic', 'trill_present', 'vibrato_present',
        'accent_flags', 'anti_accent', 'tapped', 'left_hand_tapped', 'pizzicato',
        'show_string_number', 'sustain_pedal', 'ornament_present', 'left_fingering_code',
        'right_fingering_code')
    event_marks = ('grace', 'ottavia_present', 'barre_present', 'fade_present', 'hairpin_present',
        'golpe_present', 'wah_present', 'slashed', 'timer_present', 'free_text_present',
        'lyrics',
        'is_brushed', 'has_arpeggio', 'slapped', 'popped',
        'rasgueado_present', 'tremolo', 'whammy', 'whammy_extend', 'tremolo_bar_vibrato_present',
        'dead_slapped', 'sustain_pedal')
    page_objects, page_spans, result = {}, {}, {}
    for measure in staff['measures']:
        index = measure['measure_index']
        try:
            if track.get('let_ring_throughout'):
                raise ValueError('untraced_track_let_ring')
            if measure.get('simile_present') or measure.get('ottavia_present'):
                raise ValueError('untraced_measure_mark')
            master = masters[measure['master_measure_index']]
            if master.get('fermata_present') or master.get('section_present'):
                raise ValueError('untraced_master_mark')
            system = systems[index]
            box = next(b['bbox_mm'] for b in system['measure_boxes'] if b['measure_index'] == index)
            x, y, w, h = box
            page = system['page']
            pdf_page = document[page - 1]
            if page not in page_objects:
                page_objects[page] = pdf_page._text_page()
                page_spans[page] = pdf_spans(pdf_page)
            drawing = page_objects[page]
            notation_lines = native_staff_lines(drawing, system, mode, strings) if mode != 'tab' else None
            factor = dpi / 25.4
            crop = [max(0, round(x * factor)), max(0, round((y - 4) * factor)),
                    min(math.ceil(pdf_page.width * dpi / 72), round((x + w) * factor)),
                    min(math.ceil(pdf_page.height * dpi / 72), round((y + h + 4) * factor))]
            for indication in layout.get('tempo_indications', []):
                tx, ty, tw, th = indication['bbox_mm']
                if (indication['page'] == page and indication['master_measure_index'] == index
                        and x <= tx and tx + tw <= x + w):
                    crop[1] = min(crop[1], math.floor(ty * factor))
            for neighbour in layout['systems']:
                if neighbour['page'] != page or neighbour['system_index'] == system['system_index']:
                    continue
                _, ny, _, nh = neighbour['bbox_mm']
                if ny + nh <= y:
                    crop[1] = max(crop[1], round((ny + nh + y) / 2 * factor))
                elif ny >= y + h:
                    crop[3] = min(crop[3], round((y + h + ny) / 2 * factor))
            # Padding is not score content. Stop it before unrelated text
            # below the native system, including a page counter/footer.
            system_bottom = (system['bbox_mm'][1] + system['bbox_mm'][3]) * 72 / 25.4
            external_text = [char['top'] for char in drawing.chars
                             if char['top'] >= system_bottom
                             and char['x0'] < (x + w) * 72 / 25.4 and char['x1'] > x * 72 / 25.4]
            if external_text:
                crop[3] = min(crop[3], math.floor(min(external_text) * dpi / 72))
            left, top, right, bottom = [v / factor for v in crop]
            for glyphs in all_glyphs.values():
                for glyph in glyphs:
                    if (glyph['page'] != page or glyph['measure_index'] == index
                            or glyph['kind'] == 'hidden_fret'):
                        continue
                    gx, gy, gw, gh = glyph['bbox_mm']
                    if gx < right and gx + gw > left and gy < bottom and gy + gh > top:
                        raise ValueError('crop_contains_foreign_measure_glyph')
            connections, note_effects = [], {}
            connection_curves, connection_lines, connection_labels = set(), set(), []
            representations = ('tab', 'notation') if mode == 'both' else (mode,)
            for representation in representations:
                kind = 'fret' if representation == 'tab' else 'notehead'
                visible = {key: [g for g in glyphs if g['kind'] in (kind, 'rest')]
                           for key, glyphs in all_glyphs.items()}
                visible = {key: glyphs for key, glyphs in visible.items() if glyphs}
                connection_layout = layout
                other_staff_relations = []
                if mode == 'both':
                    geometry = layout['relation_geometry']
                    other_staff_relations = [raw for raw in geometry.get('relations', [])
                        if native_connection_representation(raw, all_glyphs, staff['staff_index'], system) != representation]
                    connection_layout = {**layout, 'relation_geometry': {**geometry, 'relations': [
                        raw for raw in geometry.get('relations', [])
                        if native_connection_representation(raw, all_glyphs, staff['staff_index'], system) == representation]}}
                found, effects, hidden, curves, lines, labels = native_connection_evidence(
                    pdf_page, connection_layout, visible, staff['staff_index'], system,
                    index, (left, top, right, bottom), other_staff_relations)
                if mode == 'both':
                    for relation in found:
                        relation['representation'] = representation
                connections.extend(found)
                note_effects[representation] = effects
                connection_curves.update(curves)
                connection_lines.update(lines)
                connection_labels.extend(labels)
                if representation == 'tab':
                    for owner, anchor in hidden.items():
                        grouped.setdefault(owner, [anchor])
            relations, scope_labels, scope_lines = native_scope_evidence(
                pdf_page, page_spans[page], grouped, staff['staff_index'], system, index,
                (left, top, right, bottom), strings)
            tuplet_counts, tuplet_curves, tuplet_labels = native_tuplet_evidence(
                pdf_page, grouped, staff['staff_index'], system, index, (left, top, right, bottom))
            symbol_effects, symbol_labels = native_event_symbol_evidence(
                drawing, grouped, staff['staff_index'], index, (left, top, right, bottom))
            chord_effects, chord_labels, chord_evidence, diagram_labels = native_chord_evidence(
                pdf_page, grouped, staff, measure, page, (left, top, right, bottom))
            note_top = min((g['bbox_mm'][1] for key, glyphs in grouped.items()
                            if key[:2] == (staff['staff_index'], index)
                            for g in glyphs if g['kind'] != 'hidden_fret'), default=top)
            # Only an actual rounded horizontal bar below a traced name and
            # above the note band belongs to the diagram. Filled ties/slurs
            # still require their own native connection trace.
            diagram_curves = {ci for ci, curve in enumerate(drawing.curves)
                if score['document'].get('diagram_style', {}).get('score_visible')
                and native_barre_path(curve)
                and curve['bottom'] * 25.4 / 72 < note_top
                and any(curve['top'] >= y1 and x0 - 12 <= (curve['x0'] + curve['x1']) / 2 <= x1 + 12
                        for x0, y0, x1, y1 in diagram_labels)}
            tempo_labels, tempo_effects = [], {}
            for indication in layout.get('tempo_indications', []):
                tx, ty, tw, th = indication['bbox_mm']
                if (indication['page'] != page or indication['master_measure_index'] != index
                        or not (left <= tx and tx + tw <= right and top <= ty and ty + th <= bottom)):
                    continue
                import re
                bounds = [v * 72 / 25.4 for v in (tx, ty, tx + tw, ty + th)]
                printed = pdf_page.text(bounds)
                match = re.search(r'=\s*(\d+)', printed)
                if not match or '\ue1d5' not in printed:
                    raise ValueError('unreadable_native_quarter_tempo')
                tempo_labels.append(bounds)
                tempo_effects[round(float(indication['position']) * 3840)] = [f'tempo:{int(match[1])}']
            def intersects(item):
                return (item['x1'] * 25.4 / 72 > left and item['x0'] * 25.4 / 72 < right
                        and item['bottom'] * 25.4 / 72 >= top and item['top'] * 25.4 / 72 <= bottom)
            # These paths may enter from a neighbouring measure whose source
            # flags are absent here. Never label their missing relations empty.
            relations.extend(connections)
            if any(intersects(c) for ci, c in enumerate(drawing.curves)
                   if ci not in tuplet_curves | connection_curves | diagram_curves
                   and not (mode != 'tab' and native_beam_path(c, drawing))):
                raise ValueError('untraced_visible_curve')
            if any(intersects(line) and line.get('dash') and any(line['dash'][0])
                   and not any(abs(line['x0'] - x0) < 1e-6 and abs(line['x1'] - x1) < 1e-6
                               and abs(line['top'] - y0) < 1e-6 and abs(line['bottom'] - y1) < 1e-6
                               for x0, y0, x1, y1 in scope_lines)
                   for li, line in enumerate(drawing.lines) if li not in connection_lines):
                raise ValueError('untraced_visible_dashed_line')
            # Actual PDF glyphs also catch implicit dynamics and textual marks
            # that are not represented by the note's source-effect switches.
            chars = [c for c in drawing.chars if intersects(c)]
            for char in chars:
                if any(x0 - 1e-6 <= char['x0'] and char['x1'] <= x1 + 1e-6
                       and y0 - 1e-6 <= char['top'] and char['bottom'] <= y1 + 1e-6
                       for x0, y0, x1, y1 in scope_labels + tuplet_labels + symbol_labels + tempo_labels + connection_labels + chord_labels):
                    continue
                text = char['text']
                for letter in text:
                    code = ord(letter)
                    if (letter in '0123456789xX× .()' or code in (0xE06D, 0xE06E)
                            or mode != 'tab' and (0xE050 <= code <= 0xE069 or 0xE260 <= code <= 0xE264
                                or any(g['smufl'] == code for gs in all_glyphs.values() for g in gs))
                            or 0xE080 <= code <= 0xE089 or 0xE4E2 <= code <= 0xE4EC
                            or 0xE1D0 <= code <= 0xE1E9 or 0xE240 <= code <= 0xE24F):
                        continue
                    # Standard SMuFL fretboard marks and fingering digits are
                    # diagram content, never TAB notes. Accept them only above
                    # the actual note/rest band alongside a traced chord name.
                    if (chord_evidence and char['top'] * 25.4 / 72 < note_top
                            and (0xE858 <= code <= 0xE85A or 0xED10 <= code <= 0xED15)):
                        continue
                    raise ValueError(f'untraced_printed_glyph_U+{code:04X}')
            state = {'time': {'status': 'absent'}, 'key': {'status': 'absent'},
                     'clef': {'status': 'absent'}}
            if mode == 'tab' and any(any(ord(c) in (0xE06D, 0xE06E) for c in char['text']) for char in chars):
                state['clef'] = {'status': 'observed', 'value': 'tab'}
            if mode != 'tab':
                for region in layout.get('pitch_regions', []):
                    if region['kind'] == 'clef' and region['measure_index'] == index and region['staff_index'] == staff['staff_index']:
                        cx, cy, cw, ch = region['bbox_mm']
                        if region['page'] == page and left <= cx and cx + cw <= right and top <= cy and cy + ch <= bottom:
                            clef = visible_clef(pdf_page, region, measure)
                            if clef['clef'] != 'tab':
                                state.update({k: {'status': 'observed', 'value': v} for k, v in clef.items()})
            for signature in system['time_signatures']:
                if signature['measure_index'] == index and signature['staff_index'] == staff['staff_index']:
                    state['time'] = {'status': 'observed', 'value': f"{signature['numerator']}/{signature['denominator']}"}
            for signature in system['key_signatures']:
                if signature['measure_index'] == index and signature['staff_index'] == staff['staff_index']:
                    if mode == 'tab':
                        raise ValueError('unsupported_printed_tab_key')
                    state['key'] = {'status': 'observed', 'value': signature['accidental_count']}
            shared_rhythm_unknown = (native_shared_tab_rhythm(
                drawing, grouped, staff['staff_index'], measure, (left, top, right, bottom), tuplet_counts)
                if mode == 'tab' else set())
            voices, reference_voices, event_indices, note_indices = [], [], {}, {}
            event_origins = {}
            for voice in measure['voices']:
                events, reference_events = [], []
                next_start = 0
                for event in voice['events']:
                    if event['placeholder']:
                        continue
                    marked = next((key for key in event_marks if event.get(key)), None)
                    if marked:
                        raise ValueError('untraced_event_' + marked)
                    owner = (staff['staff_index'], index, voice['voice_index'], event['event_index'])
                    rhythm = event['rhythm']
                    code, dots = rhythm['note_value_code'], rhythm['augmentation_dots']
                    if not 2 <= code <= 8 or dots not in (0, 1, 2) or rhythm['tuplets']['secondary']:
                        raise ValueError('unsupported_native_rhythm')
                    enters, times = rhythm['tuplets']['primary'] or (1, 1)
                    # A lone printed count encodes the conventional group
                    # (3 in 2, 5/6/7 in 4, etc.), not an arbitrary source ratio.
                    conventional = enters > 2 and enters & (enters - 1) != 0 and times == 1 << (enters.bit_length() - 1)
                    rhythm_visible = (owner not in shared_rhythm_unknown
                        and (enters == times or (conventional and tuplet_counts.get(owner) == enters)))
                    duration = dict(value=1 << (code - 2), dotted=dots == 1,
                                    double_dotted=dots == 2, tuplet_enters=enters, tuplet_times=times)
                    onset = Fraction(*event['offset']) * 960
                    if onset.denominator != 1:
                        raise ValueError('non_integral_native_onset')
                    start = int(onset) if int(onset) == next_start else None
                    notes, notation_notes, reference_notes, native_note_indices = [], [], [], []
                    notation_owner_indices, notation_glyph_indices = {}, {}
                    if event['rest']:
                        glyphs = grouped.get((*owner, None), [])
                        if len(glyphs) != 1 or glyphs[0]['kind'] != 'rest':
                            raise ValueError('rest_without_unique_visible_glyph')
                    else:
                        for note in event['notes']:
                            # A clipped label or ambiguous voice remains unknown.
                            # Source switches are reference semantics, never a
                            # requirement to invent missing visual scope labels.
                            marked = next((key for key in note_marks if note.get(key)), None)
                            if marked:
                                raise ValueError('untraced_note_' + marked)
                            glyphs = grouped.get((*owner, note['note_index']), [])
                            if len(glyphs) != 1 or glyphs[0]['kind'] not in (primary, 'hidden_fret'):
                                raise ValueError('note_without_unique_visible_glyph')
                            glyph = glyphs[0]
                            observed_note, notation_note = {}, {}
                            if mode != 'notation':
                                text = glyph['text'].strip().removeprefix('(').removesuffix(')')
                                if glyph['kind'] != 'hidden_fret' and not (
                                        text.isascii() and text.isdigit() or text.lower() in ('x', '×')):
                                    raise ValueError('unsupported_visible_fret_text')
                                observed_note.update(string=strings - glyph['native_string_index'],
                                    fret=None if glyph['kind'] == 'hidden_fret' else int(text) if text.isdigit() else 'x')
                            if mode != 'tab':
                                evidence = all_glyphs.get((*owner, note['note_index']), [])
                                heads = [g for g in evidence if g['kind'] == 'notehead']
                                marks = [g for g in evidence if g['kind'] == 'accidental']
                                if len(heads) != 1 or len(marks) > 1:
                                    raise ValueError('note_without_unique_drawn_head')
                                position = (notation_lines[-1] - heads[0]['render_origin_mm'][1]) / ((notation_lines[-1] - notation_lines[0]) / 8)
                                if abs(position - round(position)) > 1e-3:
                                    raise ValueError('drawn_head_off_staff_grid')
                                accidentals = {0xE260: -1, 0xE261: 0, 0xE262: 1, 0xE263: 2, 0xE264: -2}
                                if marks and marks[0]['smufl'] not in accidentals:
                                    raise ValueError('unsupported_drawn_accidental')
                                notation_note.update(staff_position=round(position),
                                    accidental=accidentals[marks[0]['smufl']] if marks else None)
                                if percussion:
                                    from research.data.observations import painted_notehead
                                    notation_note['notehead'] = painted_notehead(heads[0]['smufl'])
                                if effects := note_effects['notation'].get((*owner, note['note_index'])):
                                    notation_note['effects'] = sorted(effects)
                                # Unison notes on different strings can paint
                                # the same head twice. The bitmap contains one
                                # head; source ownership must not duplicate it.
                                head_key = (tuple(heads[0]['render_origin_mm']), heads[0]['smufl'])
                                if head_key not in notation_glyph_indices:
                                    notation_glyph_indices[head_key] = len(notation_notes)
                                    notation_notes.append(notation_note)
                                else:
                                    existing = notation_notes[notation_glyph_indices[head_key]]
                                    if existing['accidental'] is None:
                                        existing['accidental'] = notation_note['accidental']
                                    elif notation_note['accidental'] not in (None, existing['accidental']):
                                        raise ValueError('conflicting_coincident_accidentals')
                                    if notation_note.get('effects'):
                                        existing['effects'] = sorted(set(existing.get('effects', [])) | set(notation_note['effects']))
                                notation_owner_indices[note['note_index']] = notation_glyph_indices[head_key]
                            if mode != 'notation':
                                notes.append(observed_note)
                                if effects := note_effects['tab'].get((*owner, note['note_index'])):
                                    notes[-1]['effects'] = sorted(effects)
                            native_note_indices.append(note['note_index'])
                            effects = [name for field, name in (
                                ('dead', 'dead'), ('tie_destination', 'tie'),
                                ('hopo_origin', 'hammer'), ('palm_muted', 'pm'), ('let_ring', 'let'))
                                if note.get(field)]
                            effects += [name for field, name in (
                                ('in_above', 'sia'), ('in_below', 'sib'), ('legato', 'sl'),
                                ('shift', 'ss'), ('out_down', 'sod'), ('out_up', 'sou'))
                                if note.get('slides', {}).get(field)]
                            ref = {'effects': effects}
                            if mode != 'notation':
                                ref.update(string=strings - note['native_string_index'], fret=note['fret'])
                            if mode != 'tab':
                                ref['pitch'] = staff['tuning_pitches_low_to_high'][note['native_string_index']] + note['fret'] - track['view_transposition_offset']
                            reference_notes.append(ref)
                        if not notes and not notation_notes:
                            raise ValueError('normal_event_without_visible_notes')
                        # A tied TAB note can omit its fret while still printing
                        # a stem at the connector's traced note anchor.
                        rhythm_glyphs = [g for key, gs in (grouped if mode == 'tab' else all_glyphs).items()
                                         if key[:4] == owner for g in gs
                                         if g['kind'] in (('fret', 'hidden_fret') if mode == 'tab' else ('notehead',))]
                        glyph = max(rhythm_glyphs, key=lambda g: g['bbox_mm'][1])
                        gx, gy, gw, gh = glyph['bbox_mm']
                        origin_x = glyph['render_origin_mm'][0]
                        # Match actual PDF stems to the native drawing origin;
                        # the tolerance only covers PDF coordinate rounding.
                        if code > 2 and not any(abs(line['x0'] - line['x1']) < 1e-6
                            and (abs(line['x0'] * 25.4 / 72 - origin_x) < 1e-4 if mode == 'tab'
                                 else gx - .05 <= line['x0'] * 25.4 / 72 <= gx + gw + .05)
                            and (gy + gh < line['bottom'] * 25.4 / 72 <= bottom
                                 or top <= line['top'] * 25.4 / 72 < gy)
                            and top <= line['top'] * 25.4 / 72
                            for line in drawing.lines):
                            rhythm_visible = False
                    # Every used glyph must actually lie inside this measure's
                    # unpadded horizontal crop and its printed vertical band.
                    for glyph in (g for key, gs in all_glyphs.items() if key[:4] == owner for g in gs):
                        if glyph['kind'] == 'hidden_fret':
                            continue
                        gx, gy, gw, gh = glyph['bbox_mm']
                        if glyph['page'] != page or gx < left or gx + gw > right or gy < top or gy + gh > bottom:
                            raise ValueError('visible_glyph_clipped_by_crop')
                    order = sorted(range(len(notes)), key=lambda ni: notes[ni]['string'])
                    notes = [notes[ni] for ni in order]
                    reference_notes.sort(key=lambda note: note['string'] if mode != 'notation' else -note['pitch'])
                    for ni, previous in enumerate(order):
                        note_indices[voice['voice_index'], event['event_index'], native_note_indices[previous],
                                     'tab'] = ni
                    if mode != 'tab':
                        notation_order = sorted(range(len(notation_notes)), key=lambda ni: -notation_notes[ni]['staff_position'])
                        notation_notes = [notation_notes[ni] for ni in notation_order]
                        for native_index, previous in notation_owner_indices.items():
                            note_indices[voice['voice_index'], event['event_index'], native_index, 'notation'] = notation_order.index(previous)
                    if mode == 'notation':
                        notes = notation_notes
                    status = 'rest' if event['rest'] else 'normal'
                    event_indices[voice['voice_index'], event['event_index']] = len(events)
                    anchors = [g['render_origin_mm'][0] for key, gs in grouped.items()
                               if key[:4] == owner for g in gs]
                    if anchors:
                        event_origins[voice['voice_index'], len(events)] = sum(anchors) / len(anchors)
                    observed_event = dict(start=start, duration=_duration_token(duration) if rhythm_visible else None,
                                          status=status, notes=notes)
                    if mode == 'both':
                        observed_event['notation_notes'] = notation_notes
                    printed_effects = list(symbol_effects.get(owner, []))
                    printed_effects += chord_effects.get(owner, [])
                    if voice['voice_index'] == min(v['voice_index'] for v in measure['voices']):
                        printed_effects += tempo_effects.get(int(onset), [])
                    if printed_effects:
                        observed_event['effects'] = printed_effects
                    events.append(observed_event)
                    next_start = start + duration_ticks(duration) if start is not None and rhythm_visible else None
                    reference_events.append(dict(start=int(onset), duration=duration, status=status,
                        notes=reference_notes, effects=printed_effects))
                if events:
                    voices.append({'voice': voice['voice_index'], 'events': events})
                    reference_voices.append({'voice': voice['voice_index'], 'events': reference_events})
            if not voices:
                raise ValueError('measure_without_visible_events')
            # Coincident stems in another visible voice can establish an onset
            # even after this voice's unreadable rhythmic subdivision.
            for voice in voices:
                for ei, event in enumerate(voice['events']):
                    if event['start'] is not None:
                        continue
                    origin = event_origins.get((voice['voice'], ei))
                    matches = {other['start'] for v in voices for j, other in enumerate(v['events'])
                               if other['start'] is not None and origin is not None
                               and abs(event_origins.get((v['voice'], j), float('inf')) - origin) < 1e-4}
                    if len(matches) == 1:
                        event['start'] = matches.pop()
            for relation in relations:
                for endpoint in ('start', 'end'):
                    if isinstance(relation[endpoint], list):
                        vi, ei, *ni = relation[endpoint]
                        relation[endpoint] = [vi, event_indices[vi, ei]]
                        if ni:
                            representation = relation.get('representation', 'notation' if mode == 'notation' else 'tab')
                            relation[endpoint].append(note_indices[vi, ei, ni[0], representation])
            reference = parse_measure_target('M | V0{@0:w:r}')
            reference.update(voices=reference_voices, time_signature=state['time'].get('value'),
                             key_signature=key_names[state['key']['value'] + 7] if state['key']['status'] == 'observed' else None)
            time = master['time_signature']
            result[index] = {'observation': {'state': state, 'voices': voices, 'relations': relations},
                'bbox': crop,
                'visible_chords': chord_evidence,
                'visible_diagram_marks': sum(0xE858 <= ord(letter) <= 0xE85A
                    for char in chars for letter in char['text']),
                'reference_target': format_measure_target(reference, mode),
                'oracle_auxiliary': {'staff_start': index == 0, 'state': {'time': {'status': 'observed', 'value': f"{time['numerator']}/{time['denominator']}"},
                    'tuning': {'status': 'observed', 'value': list(reversed(staff['tuning_pitches_low_to_high']))}},
                    'voice_map': {str(v['voice']): v['voice'] for v in voices}}}
            if mode != 'tab':
                result[index]['oracle_auxiliary']['state'].update(
                    clef={'status': 'observed', 'value': 'percussion' if percussion else measure['clef']},
                    clef_octave={'status': 'observed', 'value': NATIVE_OCTAVES[measure['ottavia_name']]},
                    instrument_transpose={'status': 'observed', 'value': track['view_transposition_offset']},
                    instrument_transpose_explicit={'status': 'observed', 'value': False},
                    capo={'status': 'observed', 'value': staff['capo_fret'] if staff['capo_present'] else 0},
                    key={'status': 'observed', 'value': measure['drawing_key_signature']['accidental_count']})
        except (ValueError, KeyError) as error:
            result[index] = {'error': str(error)}
    return result
