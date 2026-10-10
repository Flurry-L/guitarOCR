"""Source/shape-separated native supervision and readable diagram renderings.

Labels originate in source diagram data and drawing commands, never the old
pixel postprocessor. User PDFs are intentionally absent from this module.
"""

import functools
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset

from research.models.chord_evidence import MAX_TEXT_LENGTH, roman, tokens

SIZE = 256
SPLITS = ("train", "validation", "test")
FONT_ROOT = Path("/usr/share/fonts/truetype")
TRAIN_FONTS = sorted(
    str(p)
    for family in ("liberation", "lato")
    for p in (FONT_ROOT / family).glob("*.ttf")
    if not any(word in p.name for word in ("Thin", "Hairline", "Light"))
)


def shape_identity(diagram):
    positive = [f for f in diagram["frets"] if type(f) is int and f > 0]
    origin = min(positive, default=1)
    return json.dumps(
        [
            [
                f - origin + 1 if type(f) is int and f > 0 else f
                for f in diagram["frets"]
            ],
            [[f - origin + 1, a, b] for f, a, b in diagram.get("barres", [])],
        ],
        separators=(",", ":"),
    )


def shape_split(diagram):
    # All transpositions, fonts and finger-label variants of a shape stay in
    # one split. The native document split is additionally respected.
    bucket = (
        int(hashlib.sha256(shape_identity(diagram).encode()).hexdigest()[:8], 16) % 10
    )
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def native_rows(root, split, negative_limit=None):
    rows = []
    negatives = 0
    for line in (Path(root) / f"{split}.jsonl").open():
        row = json.loads(line)
        answer = row['messages'][-1]['content']
        if not answer.lstrip().startswith('{'):
            continue
        value = json.loads(answer)
        if not isinstance(value, dict) or value.get('kind') not in {'chord', 'chord_diagram'}:
            continue
        if len(row['images']) != 1:
            raise ValueError('Chord evidence requires one target crop')
        diagram = value.get("diagram")
        if value['kind'] == 'chord_diagram' and not diagram:
            continue
        if diagram:
            if shape_split(diagram) != split:
                continue
            relative = [
                f - diagram["base_fret"] + 1
                for f in diagram["frets"]
                if type(f) is int and f > 0
            ]
            if max(relative, default=1) > 7:
                continue
        elif negative_limit is not None and negatives >= negative_limit:
            continue
        else:
            negatives += 1
        path = Path(root) / row["images"][0]
        if not path.is_file():
            raise FileNotFoundError(path)
        source = row.get('provenance', {}).get('family') or f'{split}/{path.stem}'
        rows.append(
            {
                "image": str(path),
                "diagram": diagram,
                "kind": 0 if diagram else 1,
                "source_group": source,
                "shape_group": shape_identity(diagram) if diagram else None,
            }
        )
    return rows


@functools.lru_cache(maxsize=256)
def font(path, size):
    face = ImageFont.truetype(path, size)
    try:
        axes = face.get_variation_axes()
    except OSError:
        return face  # Static font files already specify their intended weight.
    # Some variable files default to ExtraLight (200). A score styled with
    # the regular face uses 400; otherwise small reversed digits disappear.
    values = [
        max(axis["minimum"], min(400, axis["maximum"]))
        if axis["name"] == b"Weight"
        else axis["default"]
        for axis in axes
    ]
    face.set_variation_by_axes(values)
    return face


@functools.lru_cache(maxsize=1)
def musical_voicings():
    # Reuse the repository's authored, pitch-checked repertoire.
    from research.data.chord_text_roles import SHAPES

    rows = []
    for shapes in SHAPES.values():
        for name, frets, fingers, base, _, pitches in shapes:
            barres = []
            for finger in range(1, 5):
                indexes = [i for i, value in enumerate(fingers) if value == finger]
                if len(indexes) > 1:
                    barres.append([frets[indexes[0]], min(indexes), max(indexes)])
            diagram = {
                "base_fret": base,
                "frets": list(frets),
                "fingers": list(fingers),
                "barres": barres,
            }
            rows.append(
                {
                    "name": name,
                    "diagram": diagram,
                    "pitch_classes": sorted(pitches),
                    "split": shape_split(diagram),
                    "tuning_midi": [40, 45, 50, 55, 59, 64],
                    "instrument": "guitar",
                    "source": "authored/" + name,
                    "allowed_pitch_classes": sorted(pitches),
                    "required_pitch_classes": sorted(pitches),
                }
            )
    library = json.loads(
        Path(__file__)
        .with_name("resources")
        .joinpath("chord_voicings.json")
        .read_text()
    )
    excluded = set(library["evaluation_shape_exclusions"]["shapes"])
    return [
        row
        for row in [*rows, *library["voicings"]]
        if shape_identity(row["diagram"]) not in excluded
    ]


def voicing_groups(rows, split):
    groups = {}
    for row in rows:
        if row["split"] == split:
            groups.setdefault(len(row["diagram"]["frets"]), []).append(row)
    return groups


def musical_diagram(rng, split, voicings=None, groups=None):
    groups = groups or voicing_groups(voicings or musical_voicings(), split)
    counts = list(groups)
    weights = {4: 0.16, 5: 0.08, 6: 0.42, 7: 0.10, 8: 0.10, 10: 0.07, 12: 0.07}
    row = rng.choice(
        groups[rng.choices(counts, [weights[count] for count in counts])[0]]
    )
    diagram = json.loads(json.dumps(row["diagram"]))
    name = row["name"]
    positive = [f for f in diagram["frets"] if type(f) is int and f > 0]
    instrument = row.get("instrument", "guitar")
    last_fret = {
        "ukulele": 19,
        "5-string-banjo": 22,
        "10-string-guitar": 19,
        "12-string-guitar-paired-courses": 21,
    }.get(instrument, 24)
    shift = 0
    if positive and 0 not in diagram["frets"]:
        shift = rng.randint(1 - min(positive), last_fret - max(positive))
    elif (
        positive
        and "/" not in name
        and max(positive) + 12 <= last_fret
        and rng.random() < 0.25
    ):
        # Fretted notes can move one octave against an open pedal, preserving
        # all pitch classes. Explicit slash-bass inversions are kept intact.
        shift = 12
    if shift:
        diagram["frets"] = [
            f + shift if type(f) is int and f > 0 else f for f in diagram["frets"]
        ]
        diagram["barres"] = [[f + shift, a, b] for f, a, b in diagram["barres"]]
    if shift % 12:
        root = re.match(r"[A-G][#b]?", name).group()
        pitch = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[root[0]]
        pitch += 1 if "#" in root else -1 if "b" in root else 0
        names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        name = names[(pitch + shift) % 12] + name[len(root) :]
        if "/" in name:
            quality, bass = name.split("/")
            pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[bass[0]] + (
                1 if "#" in bass else -1 if "b" in bass else 0
            )
            name = quality + "/" + names[(pc + shift) % 12]
    lowest = min((f for f in diagram["frets"] if type(f) is int and f > 0), default=1)
    # An open bass may accompany high stopped frets. Its O mark stays above
    # the diagram; it does not force the grid back to the first fret.
    diagram["base_fret"] = 1 if lowest <= 3 and rng.random() < 0.5 else lowest
    if rng.random() < 0.3:
        diagram["fingers"] = [None] * len(diagram["frets"])
    return name, diagram


def render_diagram(
    diagram,
    rng,
    style=None,
    fonts=None,
    name="",
    return_positions=False,
    return_base_position=False,
    return_base_box=False,
    return_base_text=False,
    return_grid_box=False,
):
    count = len(diagram["frets"])
    base = diagram["base_fret"]
    compact = rng.random() < 0.45
    gap = rng.randint(16, 24) if compact else rng.randint(22, 30)
    step = rng.randint(18, 27) if compact else rng.randint(25, 34)
    rows = rng.choice([4, 5, 6, 7])
    rows = max(
        rows,
        max(
            [f - base + 1 for f in diagram["frets"] if type(f) is int and f > 0],
            default=1,
        ),
    )
    face = rng.choice(fonts or TRAIN_FONTS)
    size = rng.randint(12, 17) if compact else rng.randint(15, 20)
    small = font(face, size)
    label = (
        (roman(base) if rng.random() < 0.25 else str(base))
        if base > 1 or rng.random() < 0.25
        else None
    )
    side = "left" if compact else rng.choice(["left", "right"])
    label_width = small.getbbox(label)[2] if label else 0
    name_font = font(face, rng.randint(18, 24))
    name_box = name_font.getbbox(name)
    x0 = max(
        rng.randint(21, 32) if compact else rng.randint(36, 53),
        label_width + 14 if side == "left" else 0,
    )
    right = max(
        rng.randint(11, 22) if compact else rng.randint(40, 57),
        label_width + 16 if side == "right" else 0,
    )
    y0 = max(
        rng.randint(40, 56) if compact else rng.randint(55, 78),
        10 + name_box[3] - name_box[1] + size + 15,
    )
    width = x0 + (count - 1) * gap + right
    name_extra = max(0, name_box[2] - name_box[0] + 20 - width)
    x0 += name_extra // 2
    width += name_extra
    height = (
        y0
        + rows * step
        + max(size + 15, rng.randint(18, 27) if compact else rng.randint(28, 45))
    )
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    line = rng.choice([1, 1, 2])
    ink = rng.choice([0, 0, 0, 35, 60])
    color = (ink,) * 3
    for i in range(count):
        draw.line(
            (x0 + i * gap, y0, x0 + i * gap, y0 + rows * step), fill=color, width=line
        )
    for i in range(rows + 1):
        draw.line(
            (x0, y0 + i * step, x0 + (count - 1) * gap, y0 + i * step),
            fill=color,
            width=4 if i == 0 and base == 1 and rng.random() < 0.8 else line,
        )
    radius = rng.uniform(0.20, 0.30) * min(gap, step)
    base_position = np.full(2, -100.0, np.float32)
    base_box = np.full((4, 2), -100.0, np.float32)
    if label:
        # Leave normal text padding beyond the dot's ink, including its
        # antialiasing fringe. A label must never merge with a stopped note.
        label_gap = math.ceil(radius) + 3
        lx = x0 - label_gap if side == "left" else x0 + (count - 1) * gap + label_gap
        anchor = "rm" if side == "left" else "lm"
        xy = (lx, y0 + step * 0.5)
        draw.text(xy, label, font=small, fill=color, anchor=anchor)
        a, b, c, d = draw.textbbox(xy, label, font=small, anchor=anchor)
        base_position[:] = [(a + c) / 2, (b + d) / 2]
        base_box[:] = [(a, b), (c, b), (c, d), (a, d)]
    # The printed name always agrees with the authored/translated voicing.
    name_anchor = (width / 2, 10)
    name_bottom = draw.textbbox(name_anchor, name, font=name_font, anchor="mt")[3]
    draw.text(name_anchor, name, font=name_font, fill=color, anchor="mt")
    barre_style = style or rng.choice(["thick", "thin", "arc", "rounded"])
    for fret, a, b in diagram["barres"]:
        y = y0 + (fret - base + 0.5) * step
        left = x0 + a * gap
        right = x0 + b * gap
        if barre_style == "arc":
            points = [
                (
                    left + (right - left) * i / 32,
                    y - math.sin(math.pi * i / 32) * step * 0.42,
                )
                for i in range(33)
            ]
            draw.line(points, fill=color, width=rng.randint(2, 4))
        elif barre_style == "rounded":
            r = radius * 0.75
            draw.rounded_rectangle(
                (left - r, y - r, right + r, y + r), radius=r, fill=color
            )
        else:
            draw.line(
                (left, y, right, y),
                fill=color,
                width=rng.randint(2, 4)
                if barre_style == "thin"
                else rng.randint(7, 11),
            )
    inside = not compact and rng.random() < 0.2
    positions = np.full((12, 2), -100.0, np.float32)

    def digit(xy, value, face, fill, anchor, index):
        draw.text(xy, str(value), font=face, fill=fill, anchor=anchor)
        x, y, right, bottom = draw.textbbox(xy, str(value), font=face, anchor=anchor)
        positions[index] = [(x + right) / 2, (y + bottom) / 2]

    for i, fret in enumerate(diagram["frets"]):
        x = x0 + i * gap
        finger = diagram["fingers"][i]
        if fret in ("x", 0):
            draw.text(
                (x, y0 - 7),
                "x" if fret == "x" else "o",
                font=small,
                fill=color,
                anchor="mb",
            )
        else:
            y = y0 + (fret - base + 0.5) * step + rng.uniform(-0.07, 0.07) * step
            on_barre = any(fret == f and a <= i <= b for f, a, b in diagram["barres"])
            if not (compact and on_barre and barre_style in ("thick", "rounded")):
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )
            if inside and finger is not None:
                digit(
                    (x, y),
                    finger,
                    font(face, max(10, round(radius * 1.5))),
                    "white",
                    "mm",
                    i,
                )
        if finger is not None and not inside:
            digit((x, y0 + rows * step + 5), finger, small, color, "mt", i)
    if return_base_text:
        return image, positions, base_position, base_box, label or ""
    if return_grid_box:
        # Renderer-owned supervision excludes the chord name above the grid.
        grid_top = max(name_bottom + 2, y0 - size - 12)
        grid_bottom = min(height, y0 + rows * step + size + 18)
        return image, (0, grid_top, width, grid_bottom)
    if return_base_box:
        return image, positions, base_position, base_box
    if return_base_position:
        return image, positions, base_position
    return (image, positions) if return_positions else image


def render_negative(rng, kind, fonts=None):
    width = rng.randint(170, 400)
    height = rng.randint(60, 230)
    face = rng.choice(fonts or TRAIN_FONTS)
    if kind == 1:
        words = rng.choice(
            [
                "Standard tuning",
                "Drop D",
                "let ring",
                "P.M.",
                "8va",
                "Guitar",
                "Allegro",
                "E B G D A E",
                "7",
                "Cmaj7",
                "12 fr.",
                "copyright 2026",
                "tab guitar",
                "piano",
            ]
        )
        text_font = font(face, rng.randint(16, 30))
        box = text_font.getbbox(words)
        width = max(width, box[2] - box[0] + 30)
        height = max(height, box[3] - box[1] + 30)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((15 - box[0], 15 - box[1]), words, font=text_font, fill="black")
    else:
        gap = rng.randint(8, 16)
        top = rng.randint(35, 45)
        lines = rng.choice([4, 5, 6])
        height = max(height, top + max(lines - 1, 3.5) * gap + 15)
        image = Image.new("RGB", (width, int(height)), "white")
        draw = ImageDraw.Draw(image)
        for i in range(lines):
            draw.line(
                (10, top + i * gap, width - 10, top + i * gap),
                fill="black",
                width=rng.choice([1, 2]),
            )
        for x in range(25, width - 20, rng.randint(25, 50)):
            y = top + rng.randrange(8) * gap / 2
            draw.ellipse((x - 5, y - 3, x + 5, y + 3), fill="black")
            draw.line((x + 5, y, x + 5, y - 25), fill="black", width=2)
    return image


def adjacent_context(image, rng, fonts=None):
    """Add ordinary neighboring print while preserving every diagram pixel."""
    face = font(rng.choice(fonts or TRAIN_FONTS), rng.randint(12, 22))
    text = rng.choice(
        [
            str(rng.randint(1, 256)),
            "Am",
            "D/F#",
            "Cadd9",
            "Gsus4",
            "P.M.",
            "let ring",
            "sl.",
            "Guitar",
        ]
    )
    side = rng.choices(["bottom", "top", "left", "right"], [7, 1, 1, 1])[0]
    box = face.getbbox(text)
    gap = rng.randint(3, 14)
    if side in ("bottom", "top"):
        strip = box[3] - box[1] + gap + rng.randint(2, 12)
        offset = (0, strip if side == "top" else 0)
        size = (image.width, image.height + strip)
        location = (
            rng.randint(3, max(3, image.width - box[2] - 3)),
            (image.height if side == "bottom" else 0) + gap - box[1],
        )
    else:
        strip = box[2] - box[0] + gap + rng.randint(2, 12)
        offset = (strip if side == "left" else 0, 0)
        size = (image.width + strip, image.height)
        location = (
            (image.width if side == "right" else 0) + gap - box[0],
            rng.randint(3, max(3, image.height - box[3] - 3)),
        )
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, offset)
    ImageDraw.Draw(canvas).text(location, text, font=face, fill="black")
    return canvas, np.asarray(offset, dtype=np.float32)


def preprocess(image):
    image = image.convert("RGB")
    scale = SIZE / max(image.size)
    w = max(1, int(image.width * scale + 0.5))
    h = max(1, int(image.height * scale + 0.5))
    image = image.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE), "white")
    canvas.paste(image, ((SIZE - w) // 2, (SIZE - h) // 2))
    data = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255
    data = (
        data - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    ) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    return torch.from_numpy(data.copy())


def labels(diagram, kind):
    count = len(diagram["frets"]) if diagram else 0
    frets = np.full(12, -100, np.int64)
    fingers = np.full(12, -100, np.int64)
    barres = np.zeros((7, 11), np.float32)
    if diagram:
        base = diagram["base_fret"]
        for i, f in enumerate(diagram["frets"]):
            frets[i] = 0 if f == "x" else 1 if f == 0 else f - base + 2
        for i, f in enumerate(diagram["fingers"]):
            fingers[i] = 0 if f is None else f + 1
        for fret, a, b in diagram.get("barres", []):
            barres[fret - base, a:b] = 1
    return {
        "kind": torch.tensor(kind),
        "strings": torch.tensor(count),
        "base": torch.tensor(diagram["base_fret"] if diagram else 0),
        "frets": torch.from_numpy(frets),
        "fingers": torch.from_numpy(fingers),
        "barres": torch.from_numpy(barres),
    }


class ChordEvidenceData(Dataset):
    def __init__(
        self,
        root,
        split,
        synthetic=24000,
        augment=False,
        fonts=None,
        voicings=None,
        native_excluded_shapes=None,
        context_probability=0.0,
    ):
        self.split = split
        self.augment = augment
        self.synthetic = synthetic
        self.epoch = 0
        self.fonts = fonts
        if not 0 <= context_probability <= 1:
            raise ValueError("Context probability must be in [0, 1]")
        self.context_probability = context_probability
        self.voicings = voicings or musical_voicings()
        self.voicing_groups = voicing_groups(self.voicings, split)
        if native_excluded_shapes is None:
            library = json.loads(
                Path(__file__)
                .with_name("resources")
                .joinpath("chord_voicings.json")
                .read_text()
            )
            native_excluded_shapes = library["evaluation_shape_exclusions"]["shapes"]
        self.native_excluded_shapes = native_excluded_shapes
        self.rows = native_rows(
            root, split, negative_limit=6000 if split == "train" else 800
        )
        excluded = set(native_excluded_shapes)
        self.rows = [row for row in self.rows if row["shape_group"] not in excluded]

    def __len__(self):
        return len(self.rows) + self.synthetic

    def __getitem__(self, index):
        rng = random.Random(
            index
            + {"train": 1000000, "validation": 2000000, "test": 3000000}[self.split]
        )
        positions = np.full((12, 2), -100.0, np.float32)
        base_position = np.full(2, -100.0, np.float32)
        base_box = np.full((4, 2), -100.0, np.float32)
        base_text = ""
        if index < len(self.rows):
            row = self.rows[index]
            diagram = row["diagram"]
            kind = row["kind"]
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
        else:
            kind = rng.choices([0, 1, 2], [8, 1, 1])[0]
            if kind == 0:
                name, diagram = musical_diagram(
                    rng, self.split, self.voicings, self.voicing_groups
                )
                if rng.random() < 0.65:
                    name = ""  # An ordinary unnamed chart prevents a name-to-shape shortcut.
                image, positions, base_position, base_box, base_text = render_diagram(
                    diagram, rng, fonts=self.fonts, name=name, return_base_text=True
                )
            else:
                diagram = None
                image = render_negative(rng, kind, fonts=self.fonts)
        # Rotations represent ordinary small scan skew. Never stretch or make
        # the printed image unreadable. Held-out skew is deterministic.
        angle = random.uniform(-3, 3) if self.augment else rng.choice([-2, -1, 0, 1, 2])
        positions = np.vstack([positions, base_position, base_box])
        valid = positions[:, 0] >= 0
        if (
            index >= len(self.rows)
            and kind == 0
            and self.context_probability
            and rng.random() < self.context_probability
        ):
            image, offset = adjacent_context(image, rng, self.fonts)
            positions[valid] += offset
        if angle:
            old_size = image.size
            image = image.rotate(
                angle, Image.Resampling.BICUBIC, expand=True, fillcolor="white"
            )
            radians = math.radians(angle)
            rotation = np.array(
                [
                    [math.cos(radians), math.sin(radians)],
                    [-math.sin(radians), math.cos(radians)],
                ]
            )
            positions[valid] = (
                positions[valid] - np.array(old_size) / 2
            ) @ rotation.T + np.array(image.size) / 2
        if self.augment:
            margin = random.randint(0, 12)
            image = ImageOps.expand(image, margin, "white")
            positions[valid] += margin
        scale = SIZE / max(image.size)
        width = max(1, int(image.width * scale + 0.5))
        height = max(1, int(image.height * scale + 0.5))
        positions[valid] = (
            positions[valid] * np.array([width / image.width, height / image.height])
            + np.array([(SIZE - width) // 2, (SIZE - height) // 2])
        ) / SIZE
        target = labels(diagram, kind)
        target["finger_positions"] = torch.from_numpy(positions[:12])
        target["base_position"] = torch.from_numpy(positions[12])
        target["base_supervised"] = torch.tensor(index >= len(self.rows))
        target["base_extent"] = torch.from_numpy(
            positions[13:].max(0) - positions[13:].min(0)
        )
        text_tokens = tokens(base_text)
        target["base_tokens"] = torch.tensor(
            text_tokens + [0] * (MAX_TEXT_LENGTH - len(text_tokens))
        )
        target["base_text_length"] = torch.tensor(len(text_tokens))
        return preprocess(image), target


def diagram_from_labels(labels, index):
    count = int(labels["strings"][index])
    base = int(labels["base"][index])
    frets = labels["frets"][index, :count].tolist()
    fingers = labels["fingers"][index, :count].tolist()
    barres = []
    for row, edges in enumerate(labels["barres"][index, :, : count - 1].tolist(), base):
        start = None
        for i, on in enumerate([*edges, False]):
            if on and start is None:
                start = i
            elif not on and start is not None:
                barres.append([row, start, i])
                start = None
    return {
        "base_fret": base,
        "frets": ["x" if f == 0 else 0 if f == 1 else base + f - 2 for f in frets],
        "fingers": [None if f == 0 else f - 1 for f in fingers],
        "barres": barres,
    }
