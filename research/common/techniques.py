"""Technique image inputs, observable music queries and event metrics."""

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw
from scorelib.score import duration_ticks, parse_measure_target
from scorelib.techniques import EFFECTS


@lru_cache(maxsize=192)
def prepared(path):
    with Image.open(path) as image:
        return _prepared_image(image)


def _prepared_image(image):
    tensor, transform = prepare_image(image)
    return ((tensor[:3] + 1) * 127.5).round().astype(np.uint8), transform


def prepare_pair(row):
    current, transform = prepared(row['image'])
    previous = prepared(row['previous_system_image'])[0] if row.get('previous_system_image') else None
    return _stack(current, previous), transform


def prepare_images(current_image, previous_image=None):
    """The same pair for in-memory production crops, without a file round trip."""
    current, transform = _prepared_image(current_image)
    previous = _prepared_image(previous_image)[0] if previous_image is not None else None
    return _stack(current, previous), transform


def _stack(current, previous):
    if previous is None:
        previous = np.full_like(current, 255)
    rgb = np.concatenate((previous, current), 1).astype(np.float32) / 127.5 - 1
    tensor = np.concatenate((rgb, np.zeros((1, HEIGHT * 2, WIDTH), dtype=np.float32)), 0)
    return tensor


def pair_points(transform, points):
    normalized = transform.to_normalized(points)
    normalized[..., 1] = (normalized[..., 1] + 1) / 2
    return normalized


WIDTH, HEIGHT, MAX_EVENTS = 960, 256, 32


@dataclass(frozen=True)
class ImageTransform:
    original_width: int
    original_height: int
    width: int
    height: int
    offset_x: int
    offset_y: int

    def to_pixels(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return (xy * [WIDTH, HEIGHT] - [self.offset_x, self.offset_y]) * [
            self.original_width / self.width,
            self.original_height / self.height,
        ]

    def to_normalized(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return (
            xy * [self.width / self.original_width, self.height / self.original_height]
            + [self.offset_x, self.offset_y]
        ) / [WIDTH, HEIGHT]


def prepare_image(image, query=None):
    image = image.convert("RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError("A nonempty score image is required")
    scale = min(WIDTH / image.width, HEIGHT / image.height)
    width, height = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    ox, oy = (WIDTH - width) // 2, (HEIGHT - height) // 2
    transform = ImageTransform(image.width, image.height, width, height, ox, oy)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), (ox, oy))
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    if query is not None:
        x, y, w, h = query
        if (
            w <= 0
            or h <= 0
            or x < 0
            or y < 0
            or x + w > image.width
            or y + h > image.height
        ):
            raise ValueError("A query must have positive area within the score image")
        sx, sy = width / image.width, height / image.height
        ImageDraw.Draw(mask).rectangle(
            (ox + x * sx, oy + y * sy, ox + (x + w) * sx, oy + (y + h) * sy), fill=255
        )
    rgb = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    tensor = np.concatenate((rgb, np.asarray(mask, dtype=np.float32)[None] / 255), 0)
    return tensor, transform


def image_tensor(image, query):
    """Torch view of the same pixels and coordinates used by inference."""
    import torch

    pixels, transform = prepare_image(image, query)

    def transform_box(box):
        return transform.to_normalized(np.asarray(box).reshape(2, 2)).reshape(-1).tolist()

    return torch.from_numpy(pixels.copy()), transform_box


FEATURE_DIM = 32
TECHNIQUE_LABELS = ("none", "pm", "let", "both", "unknown")


def event_features(measure, mode):
    """No technique, text, renderer bounds or reference coordinates in queries."""
    if mode not in ("tab", "notation", "both"):
        raise ValueError("Unsupported score display mode")
    events = sorted(
        (
            (voice["voice"], event)
            for voice in measure["voices"]
            for event in voice["events"]
        ),
        key=lambda pair: (pair[0], pair[1]["start"]),
    )
    end = max(
        (
            event["start"] + float(duration_ticks(event["duration"]))
            for _, event in events
        ),
        default=3840,
    )
    sounding = [(voice, event) for voice, event in events if event.get("notes")]
    result, anchors = [], []
    for index, (voice, event) in enumerate(sounding):
        notes = event["notes"]
        # Authoring models can retain latent fingerings in notation and pitch
        # values in TAB. Only use attributes available in that OCR contract.
        pitches = [
            n["pitch"] if mode != "tab" and isinstance(n.get("pitch"), Real) else 60
            for n in notes
        ]
        features = [
            voice / 4,
            event["start"] / end,
            float(duration_ticks(event["duration"])) / end,
            index / max(1, len(sounding) - 1),
            len(sounding) / 32,
            len(notes) / 6,
            min(pitches) / 127,
            max(pitches) / 127,
            sum(pitches) / len(pitches) / 127,
        ]
        features += [float(mode == name) for name in ("tab", "notation", "both")]
        for string in range(1, 7):
            note = next(
                (n for n in notes if mode != "notation" and n.get("string") == string),
                None,
            )
            fret = note.get("fret") if note else None
            features.extend(
                [
                    float(note is not None),
                    float(fret) / 24
                    if isinstance(fret, Real)
                    else -1 / 24
                    if note
                    else 0,
                ]
            )
        features += [0.0] * (FEATURE_DIM - len(features))
        result.append(features)
        anchors.append([voice, event["start"]])
    return np.asarray(result, dtype=np.float32).reshape(-1, FEATURE_DIM), anchors


def located_queries(row, grounding):
    pixels = {}
    with Image.open(row['source_page']) as raw:
        page = raw.convert('RGB')
    for measure in row['measures']:
        if measure.get('query_image'):
            crop = measure['query_bbox']
            with Image.open(measure['query_image']) as raw:
                image = raw.convert('RGB')
        else:
            x, y, w, h = measure['bbox']
            crop = [max(0, int(x) - 3), max(0, int(y) - 3),
                    min(page.width, int(x + w) + 4), min(page.height, int(y + h) + 4)]
            image = page.crop(crop)
        for event in grounding.predict(image, parse_measure_target(measure['target']), row['mode']):
            pixels[measure['measure_index'], event['voice'], event['onset']] = [
                event['pixel'][0] + crop[0] - row['bbox'][0],
                event['pixel'][1] + crop[1] - row['bbox'][1]]
    return [dict(measure_index=e['measure_index'], voice=e['voice'], onset=e['onset'],
                 pixel=pixels[e['measure_index'], e['voice'], e['onset']]) for e in row['events']]


EFFECT_POLICY = 'independent_visible_effects'


def effect_probabilities(probabilities):
    values = [probabilities.get(label) for label in ('none', 'pm', 'let', 'both', 'unknown')]
    if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or not abs(sum(values) - 1) <= 1e-5):
        return {}
    none, pm, let, both, _ = values
    # 1 - present would turn an unknown prediction into confident absence.
    return {'pm': {'present': pm + both, 'absent': none + let},
            'let': {'present': let + both, 'absent': none + pm}}


def effect_decisions(probabilities, thresholds):
    scores = effect_probabilities(probabilities)
    result = {}
    for effect in ('pm', 'let'):
        accepted = [state for state in ('present', 'absent')
                    if thresholds.get(effect, {}).get(state) is not None and effect in scores
                    and scores[effect][state] >= thresholds[effect][state]]
        result[effect] = accepted[0] == 'present' if len(accepted) == 1 else None
    return result


EFFECT_LABELS = {value: key for key, value in EFFECTS.items()}


def metrics(logits, labels):
    prediction = np.argmax(logits, 1)
    confusion = np.zeros((len(TECHNIQUE_LABELS), len(TECHNIQUE_LABELS)), dtype=int)
    for expected, actual in zip(labels, prediction, strict=True):
        confusion[expected, actual] += 1
    per_effect = {}
    for effect in ("pm", "let"):
        indices = [TECHNIQUE_LABELS.index(effect), TECHNIQUE_LABELS.index("both")]
        expected, actual = np.isin(labels, indices), np.isin(prediction, indices)
        tp, fp, fn = (
            int((expected & actual).sum()),
            int((~expected & actual).sum()),
            int((expected & ~actual).sum()),
        )
        per_effect[effect] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
    return {
        "events": len(labels),
        "exact": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).mean()),
        "labels": TECHNIQUE_LABELS,
        "confusion": confusion.tolist(),
        "per_effect": per_effect,
    }
