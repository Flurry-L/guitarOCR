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
    return {"tempo_quarter": int(tempo["value"])}


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
