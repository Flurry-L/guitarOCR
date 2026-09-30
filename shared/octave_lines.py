"""Link visibly connected octave-line fragments to an explicit printed mark."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from shared.pitch_context import _distance, _staff


def _runs(values):
    return [group for group in np.split(values, np.flatnonzero(np.diff(values) > 1) + 1) if len(group)]


def dashed_line(path):
    """Return a line's position, terminal hook and whether it has no text."""
    if not path or not Path(path).is_file():
        return None
    with Image.open(path) as image:
        ink = np.asarray(image.convert('L')) < 180
    height, width = ink.shape
    if width < 60 or width < height * 2:
        return None
    strength = ink.sum(1)
    if strength.max() < width * .2:
        return None
    peak = int(strength.argmax())
    band = next(group for group in _runs(np.flatnonzero(strength >= strength.max() * .65)) if peak in group)
    center = float(np.average(band, weights=strength[band]))
    radius = max(2, len(band))
    upper, lower = max(0, round(center - radius)), min(height, round(center + radius) + 1)
    columns = ink[upper:lower].any(0)
    strokes = _runs(np.flatnonzero(columns))
    thin = [group for group in strokes if 1 <= len(group) <= max(18, width * .045)]
    if len(thin) < 4 or sum(len(group) for group in thin) < width * .2:
        return None
    left, right = int(strokes[0][0]), int(strokes[-1][-1])
    if right - left < width * .6:
        return None
    outside = ink.copy()
    outside[upper:lower] = False
    # A terminal hook is allowed; letters elsewhere disqualify a continuation.
    terminal_width = max(4, round(width * .015))
    terminal = ink[:, max(0, right - terminal_width):right + 1]
    hook = bool(terminal.sum(0).max(initial=0) >= max(5, len(band) * 2 + 1))
    outside[:, max(0, right - terminal_width):right + 1] = False
    bare = outside.sum() <= max(4, ink.sum() * .06)
    return {'y': center, 'left': left, 'right': right, 'hook': hook, 'bare': bool(bare)}


def link_octave_continuations(records, predictions):
    """Keep pitch changes anchored in explicit text and a visible dashed line.

    A new printed label, a terminal hook, a gap, or a staff change ends the
    chain. Row/page continuation additionally requires both page-margin ends.
    """
    predictions = deepcopy(predictions)
    # TAB numbers already locate sounding strings/frets; octave brackets are
    # a notation convention and must not change those positions.
    records = [r for r in records if r.get('mode') != 'tab']
    rows = defaultdict(list)
    for record in records:
        rows[(record.get('page'), record.get('system_index'), _staff(record))].append(record)
    segments = defaultdict(list)
    for prediction in predictions:
        if prediction.get('kind') not in {'annotation', 'transposition'} or not prediction.get('bbox'):
            continue
        parsed = prediction.get('parsed', {})
        explicit = parsed.get('kind') == 'ottava' and parsed.get('semitones') in {-24, -12, 12, 24}
        line = dashed_line(prediction.get('image'))
        candidates = [r for r in records if r.get('page') == prediction.get('page')]
        if not candidates:
            continue
        box = prediction['bbox']
        anchor = min(candidates, key=lambda r: (_distance(r['bbox'], box), abs(r['bbox'][0] - box[0])))
        key = (anchor.get('page'), anchor.get('system_index'), _staff(anchor))
        # Crop padding means raster positions are relative to the annotation
        # image, so use bbox centers consistently for geometric comparisons.
        segments[key].append({'prediction': prediction, 'explicit': explicit, 'line': line,
                              'left': box[0], 'right': box[0] + box[2], 'y': box[1] + box[3] / 2,
                              'height': box[3]})
    previous = {}
    for key, measures in sorted(rows.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        staff = key[2]
        left = min(r['bbox'][0] for r in measures)
        right = max(r['bbox'][0] + r['bbox'][2] for r in measures)
        center = float(np.median([r['bbox'][1] + r['bbox'][3] / 2 for r in measures]))
        gap = max(32, float(np.median([r['bbox'][2] for r in measures])) * .08)
        carried = previous.pop(staff, None)
        active = None
        for segment in sorted(segments.get(key, []), key=lambda s: s['left']):
            if segment['explicit']:
                # Crops can include the next system's heading below the TAB.
                # An upward octave mark there cannot start this row's chain.
                upward = segment['prediction']['parsed']['semitones'] > 0
                active = segment if segment['line'] and (not upward or segment['y'] < center) else None
                carried = None
                continue
            if segment['line'] is None or not segment['line']['bare']:
                if active and abs(segment['y'] - active['y']) <= max(8, segment['height'] * .55):
                    active = None
                if segment['left'] <= left + gap:
                    carried = None
                continue
            predecessor = active
            connected = (predecessor is not None and not predecessor['line']['hook']
                         and -8 <= segment['left'] - predecessor['right'] <= max(gap, segment['height'] * 2)
                         and abs(segment['y'] - predecessor['y']) <= max(8, segment['height'] * .55))
            if predecessor is None and carried is not None:
                predecessor = carried
                connected = (segment['left'] <= left + gap
                             and (segment['y'] < center) == carried['above'])
            carried = None
            if not connected:
                active = None
                continue
            reference = predecessor['prediction']
            prediction = segment['prediction']
            prediction['model_parsed'] = prediction.get('parsed', {})
            prediction['kind'] = 'transposition'
            prediction['parsed'] = {'kind': 'ottava', 'semitones': reference['parsed']['semitones'],
                                    'capo': None, 'text': None}
            prediction['octave_continuation'] = {k: reference[k] for k in ('page', 'bbox')}
            active = segment
        if active and not active['line']['hook'] and active['right'] >= right - gap:
            previous[staff] = {**active, 'above': active['y'] < center}
    return predictions
