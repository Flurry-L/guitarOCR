"""Keep instrument labels legible without shrinking a complete score row."""

import numpy as np
from PIL import Image


def focus_staff(image):
    image = image.convert('RGB')
    ink = np.asarray(image.convert('L')) < 210
    density = ink.mean(1)
    strong = np.flatnonzero(density > max(.55, float(density.max()) * .8))
    label = None
    if len(strong) >= 4:
        columns = np.flatnonzero(ink[strong].mean(0) > .75)
        runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        lines = [run for run in runs if len(run) >= 20]
        if lines:
            left = int(lines[0][0])
            margin = ink[:, :max(0, left - 2)].copy()
            if margin.size:
                # Brackets and system barlines are not part of a printed name.
                margin[:, margin.mean(0) > .65] = False
                ys, xs = np.nonzero(margin)
                if len(xs) >= 15:
                    bounds = (max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2),
                              int(xs.max()) + 3, int(ys.max()) + 3)
                    label = image.crop(bounds)
                    if label.height > label.width * 1.25 and label.width < 45:
                        label = label.transpose(Image.Transpose.ROTATE_270)
                    scale = min(3., 48 / label.height)
                    label = label.resize((max(1, round(label.width * scale)),
                                          max(1, round(label.height * scale))), Image.Resampling.LANCZOS)
    # A first-bar excerpt retains the clef and TAB lines at their real scale.
    excerpt = image.crop((0, 0, min(image.width, max(560, image.height * 2)), image.height))
    if label is None:
        return excerpt
    canvas = Image.new('RGB', (max(excerpt.width, label.width + 24), excerpt.height + label.height + 24), 'white')
    canvas.paste(label, (12, 8))
    canvas.paste(excerpt, (0, label.height + 24))
    return canvas
