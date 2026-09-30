"""Read chord supervision from GP's painted PDF glyphs and grid paths."""

from collections import defaultdict
import re

import numpy as np

from shared.chords import normalize_diagram


def chord_words(page):
    """Chord spellings actually painted by GP, including altered extensions."""
    translate = str.maketrans({'\ue260': 'b', '\ue262': '#', '♭': 'b', '♯': '#'})
    pattern = (r'[A-GH][#b]?(?:(?:maj|min|dim|aug|sus|add|omit|no|alt|dom|m|M)'
               r'|[0-9#b+°øΔ(),/\-])*(?:/[A-GH][#b]?)?')
    return [(*word[:4], text) for word in page.words()
            if re.fullmatch(pattern, text := word[4].translate(translate))]


def visible_chords(page):
    words = page.words()
    names = chord_words(page)
    rectangles = []
    for path in page.line_paths():
        if len(path) < 4:
            continue
        points = [p for segment in path for p in segment]
        left, top = min(p[0] for p in points), min(p[1] for p in points)
        right, bottom = max(p[0] for p in points), max(p[1] for p in points)
        rectangles.append((left, top, right, bottom))
    vertical = defaultdict(list)
    for left, top, right, bottom in rectangles:
        if right - left < .8 and 15 < bottom - top < 55:
            vertical[(round(top), round(bottom))].append((left + right) / 2)
    grids = []
    for (top, bottom), xs in vertical.items():
        xs = sorted(set(round(x, 2) for x in xs))
        i = 0
        while i + 6 <= len(xs):
            group = xs[i:i + 6]
            gaps = np.diff(group)
            if 2 < float(np.median(gaps)) < 8 and np.ptp(gaps) < .35:
                grids.append((group, float(top), float(bottom)))
                i += 6
            else:
                i += 1
    chars = page._text_page().chars
    result, diagram_names = [], set()
    for xs, top, bottom in grids:
        left, right = xs[0], xs[-1]
        candidates = [(i, word) for i, word in enumerate(names)
                      if left - 3 <= (word[0] + word[2]) / 2 <= right + 3 and 0 <= top - word[3] < 20]
        if not candidates:
            continue
        i, name = min(candidates, key=lambda p: top - p[1][3])
        step = (bottom - top) / 5
        numbers = [w for w in words if w[4].isdigit() and left - 13 < w[0] < left and w[2] < left + 1 and abs(w[1] - top) < step * 1.1]
        base = int(min(numbers, key=lambda w: abs(w[2] - left))[4]) if numbers else 1
        frets, fingers, barres = [None] * 6, [None] * 6, []
        for x0, y0, x1, y1 in rectangles:
            if (1.1 < y1 - y0 < step * .85 and x1 - x0 > xs[1] - left
                    and top < (y0 + y1) / 2 < bottom and x0 >= left - 4 and x1 <= right + 4):
                radius = (xs[1] - left) / 2
                low = min(range(6), key=lambda s: abs(xs[s] - x0 - radius))
                high = min(range(6), key=lambda s: abs(xs[s] - x1 + radius))
                fret = base + round(((y0 + y1) / 2 - top) / step - .5)
                if low < high:
                    barres.append([fret, low, high])
                    for s in range(low, high + 1):
                        frets[s] = fret
        for char in chars:
            x = (char['x0'] + char['x1']) / 2
            if not left - 2 <= x <= right + 2:
                continue
            s = min(range(6), key=lambda s: abs(xs[s] - x))
            glyph = char['text']
            # GP's diagram-dot font has its painted centre at this baseline
            # offset; its font em bounding box extends well beyond the ink.
            center = char['top'] + char['size'] * .426
            if glyph == '\ue858' and top < center < bottom:
                frets[s] = base + round((center - top) / step - .5)
            elif glyph in {'\ue859', '\ue85a'} and top - 9 < center < top:
                frets[s] = 'x' if glyph == '\ue859' else 0
            elif len(glyph) == 1 and '\ued10' <= glyph <= '\ued14' and bottom - 3 <= char['top'] < bottom + 6:
                fingers[s] = ord(glyph) - 0xed10
            elif glyph in {'\ued16', '\ued17'} and bottom - 3 <= char['top'] < bottom + 6:
                fingers[s] = 0
        diagram = normalize_diagram({'base_fret': base, 'frets': frets, 'fingers': fingers,
                                     'barres': [list(b) for b in sorted(set(tuple(b) for b in barres))]})
        # Unresolved glyphs are not converted into guessed open/muted strings.
        if diagram is None:
            continue
        box = [min(left - 12, name[0] - 2), name[1] - 2, max(right + 4, name[2] + 2), bottom + 12]
        result.append((box, {'kind': 'chord_diagram', 'semitones': None, 'capo': None, 'text': name[4], 'diagram': diagram}))
        diagram_names.add(i)
    for i, word in enumerate(names):
        if i not in diagram_names:
            result.append(([word[0] - 2, word[1] - 2, word[2] + 2, word[3] + 2],
                           {'kind': 'chord', 'semitones': None, 'capo': None, 'text': word[4]}))
    return result
