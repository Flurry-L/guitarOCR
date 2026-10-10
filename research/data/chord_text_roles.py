"""Musically playable chord diagrams with renderer-owned text-role supervision.

The name, fret label, open/muted marks and finger numbers are all ordinary
notation. Roman/Nashville names are included so lexical digit tests cannot
solve the role task. User PDFs are absent from this generator.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


def query_image(image, bounds, size=256):
    scale = min(size / image.width, size / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    ox, oy = (size - w) // 2, (size - h) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image.resize((w, h), Image.Resampling.LANCZOS), (ox, oy))
    mask = Image.new("L", (size, size))
    x, y, bw, bh = bounds
    ImageDraw.Draw(mask).rectangle(
        (
            ox + x * w / image.width,
            oy + y * h / image.height,
            ox + (x + bw) * w / image.width,
            oy + (y + bh) * h / image.height,
        ),
        fill=255,
    )
    pixels = np.asarray(canvas, np.float32).transpose(2, 0, 1) / 127.5 - 1
    return np.concatenate((pixels, np.asarray(mask, np.float32)[None] / 255), axis=0)


SIZE = 256


SHAPES = {
    "train": [
        ("C", ["x", 3, 2, 0, 1, 0], [None, 3, 2, None, 1, None], 1, "1", {0, 4, 7}),
        ("G", [3, 2, 0, 0, 0, 3], [2, 1, None, None, None, 3], 1, "5", {7, 11, 2}),
        ("D", ["x", "x", 0, 2, 3, 2], [None, None, None, 1, 3, 2], 1, "2", {2, 6, 9}),
        ("Em", [0, 2, 2, 0, 0, 0], [None, 2, 3, None, None, None], 1, "3m", {4, 7, 11}),
        ("Am", ["x", 0, 2, 2, 1, 0], [None, None, 2, 3, 1, None], 1, "6m", {9, 0, 4}),
        (
            "B7",
            ["x", 2, 1, 2, 0, 2],
            [None, 2, 1, 3, None, 4],
            1,
            "VII7",
            {11, 3, 6, 9},
        ),
        ("F", [1, 3, 3, 2, 1, 1], [1, 3, 4, 2, 1, 1], 1, "4", {5, 9, 0}),
        ("E", [0, 2, 2, 1, 0, 0], [None, 2, 3, 1, None, None], 1, "3", {4, 8, 11}),
        ("B", ["x", 2, 4, 4, 4, 2], [None, 1, 3, 3, 3, 1], 2, "7", {11, 3, 6}),
        (
            "Cmaj7",
            ["x", 3, 2, 0, 0, 0],
            [None, 3, 2, None, None, None],
            1,
            "1maj7",
            {0, 4, 7, 11},
        ),
        (
            "Amaj7",
            ["x", 0, 2, 1, 2, 0],
            [None, None, 2, 1, 3, None],
            1,
            "6maj7",
            {9, 1, 4, 8},
        ),
        ("Bm7", ["x", 2, 4, 2, 3, 2], [None, 1, 3, 1, 2, 1], 2, "7m7", {11, 2, 6, 9}),
        ("F#sus4", [2, 4, 4, 4, 2, 2], [1, 3, 3, 3, 1, 1], 2, "#4sus4", {6, 11, 1}),
        ("Gsus4", [3, 3, 0, 0, 1, 3], [3, 4, None, None, 1, 2], 1, "5sus4", {7, 0, 2}),
        ("Db/F", [1, 4, 3, 1, 2, 1], [1, 4, 3, 1, 2, 1], 1, "b2/4", {1, 5, 8}),
    ],
    "validation": [
        ("A", ["x", 0, 2, 2, 2, 0], [None, None, 1, 2, 3, None], 1, "6", {9, 1, 4}),
        ("Bm", ["x", 2, 4, 4, 3, 2], [None, 1, 3, 4, 2, 1], 2, "7m", {11, 2, 6}),
        (
            "Dmaj7",
            ["x", "x", 0, 2, 2, 2],
            [None, None, None, 1, 1, 1],
            1,
            "2maj7",
            {2, 6, 9, 1},
        ),
        (
            "Em7",
            [0, 2, 2, 0, 3, 0],
            [None, 1, 2, None, 3, None],
            1,
            "3m7",
            {4, 7, 11, 2},
        ),
        (
            "Asus2",
            ["x", 0, 2, 2, 0, 0],
            [None, None, 1, 2, None, None],
            1,
            "6sus2",
            {9, 11, 4},
        ),
        ("C/E", [0, 3, 2, 0, 1, 0], [None, 3, 2, None, 1, None], 1, "1/3", {0, 4, 7}),
    ],
    "test": [
        ("F#m", [2, 4, 4, 2, 2, 2], [1, 3, 4, 1, 1, 1], 2, "#4m", {6, 9, 1}),
        ("Esus4", ["x", 7, 9, 9, 10, 7], [None, 1, 3, 3, 4, 1], 7, "3sus4", {4, 9, 11}),
        (
            "Fmaj7",
            ["x", "x", 3, 2, 1, 0],
            [None, None, 3, 2, 1, None],
            1,
            "4maj7",
            {5, 9, 0, 4},
        ),
        (
            "Am7",
            ["x", 0, 2, 0, 1, 0],
            [None, None, 2, None, 1, None],
            1,
            "6m7",
            {9, 0, 4, 7},
        ),
        (
            "Dsus2",
            ["x", "x", 0, 2, 3, 0],
            [None, None, None, 1, 3, None],
            1,
            "2sus2",
            {2, 4, 9},
        ),
        (
            "G/B",
            ["x", 2, 0, 0, 3, 3],
            [None, 1, None, None, 3, 4],
            1,
            "5/7",
            {7, 11, 2},
        ),
        (
            "Cadd9",
            ["x", 3, 2, 0, 3, 3],
            [None, 2, 1, None, 3, 4],
            1,
            "1add9",
            {0, 2, 4, 7},
        ),
        (
            "Bbmaj7",
            ["x", 1, 3, 2, 3, 1],
            [None, 1, 3, 2, 4, 1],
            1,
            "b7maj7",
            {10, 2, 5, 9},
        ),
    ],
}


FONT_FAMILIES = {"train": ("lato",), "validation": ("liberation",), "test": ("dejavu",)}


FONTS = {
    split: sorted(
        str(p)
        for family in families
        for p in (Path("/usr/share/fonts/truetype") / family).glob("*.ttf")
        if not any(word in p.name for word in ("Thin", "Hairline", "Light"))
    )
    for split, families in FONT_FAMILIES.items()
}


def validate_voicings():
    tuning = (40, 45, 50, 55, 59, 64)
    for split, shapes in SHAPES.items():
        if not FONTS[split]:
            raise ValueError(
                f"Missing declared {split} font family: {FONT_FAMILIES[split]}"
            )
        for name, frets, fingers, base, _, expected in shapes:
            observed = {
                (pitch + fret) % 12
                for pitch, fret in zip(tuning, frets)
                if isinstance(fret, int)
            }
            if observed != expected:
                raise ValueError(
                    f"{name}: sounding pitches disagree with authored harmony"
                )
            stopped = [fret for fret in frets if isinstance(fret, int) and fret > 0]
            if max(stopped) - min(stopped) > 4 or min(stopped) < base:
                raise ValueError(f"{name}: impractical fret span")
            for fret, finger in zip(frets, fingers):
                if (isinstance(fret, int) and fret > 0) != (finger in (1, 2, 3, 4)):
                    raise ValueError(f"{name}: finger/played-string mismatch")
            for finger in (1, 2, 3, 4):
                strings = [i for i, f in enumerate(fingers) if f == finger]
                if len(strings) < 2:
                    continue
                fret = frets[strings[0]]
                if any(frets[i] != fret for i in strings) or any(
                    not isinstance(f, int) or f < fret
                    for f in frets[min(strings) : max(strings) + 1]
                ):
                    raise ValueError(f"{name}: impossible shared-finger barre")


validate_voicings()


def render(index, split):
    rng = random.Random(
        index + {"train": 1300000, "validation": 2300000, "test": 3300000}[split]
    )
    # Disjoint chord voicing families in addition to disjoint seeds and fonts.
    choices = SHAPES[split]
    name, frets, fingers, base, degree, _ = rng.choice(choices)
    gap = rng.randint(17, 30)
    step = rng.randint(20, 33)
    face = rng.choice(FONTS[split])
    font = ImageFont.truetype(face, rng.randint(14, 21))
    title = ImageFont.truetype(face, rng.randint(18, 29))
    label = rng.choice([name, name, name, degree])
    placement = rng.choice(["top", "top", "bottom", "side"])
    rows = max(4, max(f - base + 1 for f in frets if isinstance(f, int) and f > 0))
    x0 = rng.randint(35, 62)
    y0 = rng.randint(62, 87)
    width = x0 + 5 * gap + rng.randint(45, 67)
    height = y0 + rows * step + 65
    if placement == "side":
        width += max(80, round(title.getlength(label))) + 12
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regions = []
    ink = rng.choice(["black", "#202020", "#404040"])

    def text(xy, words, font, anchor, role):
        draw.text(xy, words, font=font, anchor=anchor, fill=ink)
        x, y, r, b = draw.textbbox(xy, words, font=font, anchor=anchor)
        regions.append(([x, y, r - x, b - y], role))

    for i in range(6):
        draw.line(
            (x0 + i * gap, y0, x0 + i * gap, y0 + rows * step),
            fill=ink,
            width=rng.choice([1, 1, 2]),
        )
    for i in range(rows + 1):
        draw.line(
            (x0, y0 + i * step, x0 + 5 * gap, y0 + i * step),
            fill=ink,
            width=3 if i == 0 and base == 1 else 1,
        )
    side = rng.choice(["left", "right"])
    text(
        (x0 - 10 if side == "left" else x0 + 5 * gap + 10, y0 + step * 0.5),
        str(base),
        font,
        "rm" if side == "left" else "lm",
        0,
    )
    if placement == "top":
        text((x0 + 2.5 * gap, 12), label, title, "mt", 1)
    elif placement == "bottom":
        text((x0 + 2.5 * gap, height - 8), label, title, "mb", 1)
    else:
        text((width - 10, y0 + rows * step / 2), label, title, "rm", 1)
    # Repeated fingers at one fret are playable barres; drawing their owned
    # geometry covers the same notation found in ordinary chord charts.
    for finger in (1, 2, 3, 4):
        strings = [i for i, f in enumerate(fingers) if f == finger]
        if len(strings) > 1 and rng.random() < 0.65:
            y = y0 + (frets[strings[0]] - base + 0.5) * step
            radius = min(gap, step) * 0.22
            b = [
                x0 + min(strings) * gap - radius,
                y - radius,
                x0 + max(strings) * gap + radius,
                y + radius,
            ]
            draw.rounded_rectangle(b, radius=radius, fill=ink)
            regions.append(([b[0], b[1], b[2] - b[0], b[3] - b[1]], 0))
    finger_regions = []
    string_regions = []
    for i, f in enumerate(frets):
        x = x0 + i * gap
        if f in ("x", 0):
            text((x, y0 - 6), "x" if f == "x" else "o", font, "mb", 0)
            string_regions.append(regions[-1][0])
        else:
            y = y0 + (f - base + 0.5) * step
            radius = min(gap, step) * rng.uniform(0.19, 0.28)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ink)
            finger = str(fingers[i])
            text((x, y0 + rows * step + 5), finger, font, "mt", 0)
            finger_regions.append(regions[-1][0])
            regions.append(([x - radius, y - radius, radius * 2, radius * 2], 0))
    # Text detectors may return a whole numeric row instead of one digit.
    # These unions use drawing-owned label groups, never source-image thresholds.
    for group in (finger_regions, string_regions):
        if len(group) > 1:
            for portion in (group, group[: max(2, len(group) // 2)]):
                a = min(b[0] for b in portion)
                b = min(b[1] for b in portion)
                r = max(bx[0] + bx[2] for bx in portion)
                d = max(bx[1] + bx[3] for bx in portion)
                regions.append(([a, b, r - a, d - b], 0))
    # Random, readable crop margins use the renderer's own extents. This models
    # tightly detected diagrams as well as padded ones without inspecting ink.
    if rng.random() < 0.85:
        boxes = [b for b, _ in regions] + [[x0, y0, 5 * gap, rows * step]]
        margin = rng.randint(0, 25)
        left = max(0, int(min(b[0] for b in boxes)) - margin)
        top = max(0, int(min(b[1] for b in boxes)) - margin)
        right = min(image.width, int(np.ceil(max(b[0] + b[2] for b in boxes))) + margin)
        bottom = min(
            image.height, int(np.ceil(max(b[1] + b[3] for b in boxes))) + margin
        )
        image = image.crop((left, top, right, bottom))
        regions = [([x - left, y - top, w, h], role) for (x, y, w, h), role in regions]
    return image, regions


class RoleData(Dataset):
    def __init__(self, split, count):
        self.split = split
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image, regions = render(index // 2, self.split)
        rng = random.Random(index + 731)
        wanted = index % 2
        b, label = rng.choice([r for r in regions if r[1] == wanted])
        x, y, w, h = b
        # A detector may return a tight or slightly shifted ink box. Its
        # ownership is unchanged; all targets still come from this renderer.
        extent = min(w, h)
        pad = rng.uniform(-0.22, 0.6) * extent
        dx, dy = (rng.uniform(-0.1, 0.1) * extent for _ in range(2))
        x0 = max(0, x - pad + dx)
        y0 = max(0, y - pad + dy)
        x1 = min(image.width, x + w + pad + dx)
        y1 = min(image.height, y + h + pad + dy)
        return torch.from_numpy(
            query_image(image, [x0, y0, x1 - x0, y1 - y0]).copy()
        ), label
