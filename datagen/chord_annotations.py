"""Draw chord diagrams from the same frets, barres and fingers used as labels."""

from PIL import ImageDraw, ImageFont


# Absolute frets, low to high. Movable closed shapes have a true barre.
SHAPES = [
    ('C', ['x', 3, 2, 0, 1, 0], [None, 3, 2, None, 1, None], []),
    ('G', [3, 2, 0, 0, 0, 3], [2, 1, None, None, None, 3], []),
    ('D', ['x', 'x', 0, 2, 3, 2], [None, None, None, 1, 3, 2], []),
    ('Am', ['x', 0, 2, 2, 1, 0], [None, None, 2, 3, 1, None], []),
    ('Em', [0, 2, 2, 0, 0, 0], [None, 2, 3, None, None, None], []),
    ('A7', ['x', 0, 2, 0, 2, 0], [None, None, 1, None, 2, None], []),
    ('D7', ['x', 'x', 0, 2, 1, 2], [None, None, None, 2, 1, 3], []),
    ('Cmaj7', ['x', 3, 2, 0, 0, 0], [None, 3, 2, None, None, None], []),
    ('Dsus4', ['x', 'x', 0, 2, 3, 3], [None, None, None, 1, 2, 3], []),
    ('C/G', [3, 3, 2, 0, 1, 0], [3, 4, 2, None, 1, None], []),
    ('D/F#', [2, 'x', 0, 2, 3, 2], [0, None, None, 1, 3, 2], []),
    ('F', [1, 3, 3, 2, 1, 1], [1, 3, 4, 2, 1, 1], [[1, 0, 5]]),
    ('Fm', [1, 3, 3, 1, 1, 1], [1, 3, 4, 1, 1, 1], [[1, 0, 5]]),
    ('F7', [1, 3, 1, 2, 1, 1], [1, 3, 1, 2, 1, 1], [[1, 0, 5]]),
    ('Fm7', [1, 3, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1], [[1, 0, 5]]),
    ('B', ['x', 2, 4, 4, 4, 2], [None, 1, 3, 3, 3, 1], [[2, 1, 5], [4, 2, 4]]),
    ('Bm', ['x', 2, 4, 4, 3, 2], [None, 1, 3, 4, 2, 1], [[2, 1, 5]]),
    ('B7', ['x', 2, 4, 2, 4, 2], [None, 1, 3, 1, 4, 1], [[2, 1, 5]]),
]
NAMES = [('C',), ('C#', 'Db'), ('D',), ('D#', 'Eb'), ('E',), ('F',),
         ('F#', 'Gb'), ('G',), ('G#', 'Ab'), ('A',), ('A#', 'Bb'), ('B',)]


def sample_diagram(rng, string_counts=(6,)):
    count = rng.choice(string_counts)
    shapes = SHAPES if count >= 6 else [
        ('F', [1, 3, 3, 2], [1, 3, 4, 2], []),
        ('Fm', [1, 3, 3, 1], [1, 3, 4, 1], [[1, 0, 3]]),
        ('F7', [1, 3, 1, 2], [1, 3, 1, 2], [[1, 0, 2]]),
        ('Fm7', [1, 3, 1, 1], [1, 3, 1, 1], [[1, 0, 3]]),
        ('Fsus4', [1, 3, 3, 3], [1, 3, 3, 3], [[3, 1, 3]]),
    ]
    name, frets, fingers, barres = rng.choice(shapes)
    frets, fingers, barres = list(frets), list(fingers), [list(b) for b in barres]
    if barres:
        shift = rng.randrange(11)
        base_pc = 5 if name.startswith('F') else 11
        name = rng.choice(NAMES[(base_pc + shift) % 12]) + name[1:]
        frets = [f + shift if type(f) is int else f for f in frets]
        barres = [[f + shift, a, b] for f, a, b in barres]
    extra = count - len(frets)
    if extra:
        frets = ['x'] * extra + frets
        fingers = [None] * extra + fingers
        barres = [[f, a + extra, b + extra] for f, a, b in barres]
    if rng.random() < .4:
        name = name.replace('b', '♭').replace('#', '♯')
    base = min((f for f in frets if type(f) is int and f > 0), default=1)
    if base <= 3 or 0 in frets:
        base = 1
    if rng.random() < .55:
        fingers = [None] * len(frets)
    return name, {'base_fret': base, 'frets': frets, 'fingers': fingers, 'barres': barres}


def draw_diagram(image, diagram, top, font_path, rng):
    draw = ImageDraw.Draw(image)
    step = rng.randrange(18, 24)
    strings = len(diagram['frets'])
    x0, y0 = (image.width - step * (strings - 1)) // 2, top + 21
    base = diagram['base_fret']
    small = ImageFont.truetype(font_path, 12)
    for string in range(strings):
        draw.line((x0 + string * step, y0, x0 + string * step, y0 + step * 5), fill='black', width=1)
    for fret in range(6):
        draw.line((x0, y0 + fret * step, x0 + step * (strings - 1), y0 + fret * step),
                  fill='black', width=3 if fret == 0 and base == 1 else 1)
    if base > 1:
        draw.text((x0 - 24, y0 + 3), str(base), font=small, fill='black')
    for fret, low, high in diagram['barres']:
        y = y0 + (fret - base + .5) * step
        draw.rounded_rectangle((x0 + low * step - 4, y - 4, x0 + high * step + 4, y + 4), radius=4, fill='black')
    for i, fret in enumerate(diagram['frets']):
        x = x0 + i * step
        if fret in {'x', 0}:
            mark = 'x' if fret == 'x' else 'o'
            draw.text((x - 4, y0 - 20), mark, font=small, fill='black')
        else:
            y = y0 + (fret - base + .5) * step
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='black')
        finger = diagram['fingers'][i]
        if finger is not None:
            draw.text((x - 4, y0 + step * 5 + 5), str(finger), font=small, fill='black')
