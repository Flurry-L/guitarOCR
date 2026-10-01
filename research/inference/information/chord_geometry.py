"""Resolve chord dots and barres against a confidently recovered raster grid."""

import cv2
import numpy as np
from PIL import Image

from scorelib.chords import normalize_diagram


def _runs(mask):
    values = np.flatnonzero(mask)
    return [group for group in np.split(values, np.flatnonzero(np.diff(values) > 1) + 1) if len(group)]


def refine_diagram(path, diagram):
    """Keep the model's printed base/fingers; read positions from visible ink.

    Skewed, incomplete or ambiguous grids retain the model prediction. Chord
    names never supply fret positions, and an absent dot is not assumed open.
    """
    if not isinstance(diagram, dict) or not isinstance(diagram.get('frets'), list):
        return None
    base = diagram.get('base_fret')
    count = len(diagram['frets'])
    if type(base) is not int or not 1 <= base <= 36 or not 3 <= count <= 12:
        return None
    with Image.open(path) as image:
        ink = (np.asarray(image.convert('L')) < 175).astype(np.uint8)
    height, width = ink.shape
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(12, round(height * .23)), 1), np.uint8))
    strength = vertical.sum(0)
    columns = _runs(strength >= max(12, strength.max() * .7))
    xs = [float(np.average(c, weights=strength[c])) for c in columns]
    if len(xs) != count:
        return None
    gap = float(np.median(np.diff(xs)))
    if np.max(np.abs(np.diff(xs) - gap)) > max(1.5, gap * .12):
        return None
    bands = []
    for x in xs:
        ys = np.flatnonzero(vertical[:, round(x)])
        if not len(ys):
            return None
        bands.append((ys[0], ys[-1]))
    top, bottom = np.median(bands, axis=0)
    if max(abs(a - top) for a, b in bands) > 2 or max(abs(b - bottom) for a, b in bands) > 2:
        return None
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(12, round((xs[-1] - xs[0]) * .7))), np.uint8))
    strength = horizontal[:, round(xs[0]):round(xs[-1]) + 1].sum(1)
    strokes = [(float(np.average(c, weights=strength[c])), len(c))
               for c in _runs(strength > (xs[-1] - xs[0]) * .7)]
    grids = []
    for frets in range(3, 8):
        candidates = []
        # A thick nut extends above the true first fret boundary. Fit the
        # thin fret lines instead of treating that ink edge as the origin.
        for origin in np.arange(top - 1, top + max(4, (bottom - top) / frets * .3) + .25, .25):
            spacing = (bottom - origin) / frets
            if spacing < 5:
                continue
            lines = [y for y, thickness in strokes if thickness <= max(2, round(spacing * .15))]
            errors = [min((abs(y - (origin + i * spacing)) for y in lines), default=float('inf'))
                      for i in range(1, frets + 1)]
            if max(errors) <= max(1.5, spacing * .1):
                candidates.append((sum(errors), float(origin), float(spacing)))
        if candidates:
            error, origin, spacing = min(candidates)
            grids.append((frets, -error, origin, spacing))
    grids.sort(reverse=True)
    if not grids:
        return None
    fret_count, _, top, step = grids[0]
    positions = []
    for x in xs:
        dots = []
        for i in range(fret_count):
            y = top + (i + .5) * step
            radius = min(gap, step) * .28
            left, right = round(x - radius), round(x + radius) + 1
            upper, lower = round(y - radius), round(y + radius) + 1
            if min(left, upper) < 0 or right > width or lower > height:
                return None
            if float(ink[upper:lower, left:right].mean()) > .48:
                dots.append(base + i)
        if dots:
            positions.append(dots[-1])
            continue
        left, right = max(0, round(x - gap * .34)), round(x + gap * .34) + 1
        upper, lower = max(0, round(top - step * 1.1)), round(top - step * .18)
        patch = ink[upper:lower, left:right]
        if not patch.size or patch.sum() < 4:
            return None
        contours, hierarchy = cv2.findContours(patch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [cv2.contourArea(c) for c, h in zip(contours, hierarchy[0]) if h[3] >= 0]
        positions.append(0 if max(holes, default=0) >= 1 else 'x')
    barres = []
    for fret in range(fret_count):
        y = top + (fret + .5) * step
        bridges = []
        for index, (a, b) in enumerate(zip(xs, xs[1:])):
            middle = (a + b) / 2
            # A dot's rounded cap can reach the midpoint of the preceding
            # gap. Require ink across most of the gap, not just its center.
            patch = ink[round(y - step * .12):round(y + step * .12) + 1,
                        round(middle - gap * .3):round(middle + gap * .3) + 1]
            center = ink[round(y - step * .12):round(y + step * .12) + 1,
                         round(middle - gap * .1):round(middle + gap * .1) + 1]
            compatible = all(type(p) is not int or p >= base + fret for p in positions[index:index + 2])
            bridges.append(compatible and patch.size > 0 and center.size > 0
                           and float(patch.mean()) > .7 and float(center.mean()) > .7)
        for group in _runs(bridges):
            barres.append([base + fret, int(group[0]), int(group[-1] + 1)])
    return normalize_diagram({**diagram, 'frets': positions, 'barres': barres})
