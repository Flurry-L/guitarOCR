"""Full-image signature inputs and decisions shared by training and inference."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DENOMINATORS = (None, 1, 2, 4, 8, 16, 32, 64)
SPATIAL_ARCHITECTURE = 'mobilenet_v3_small_spatial_v1'
SPATIAL_CONFIG = {'version': 2, 'architecture': SPATIAL_ARCHITECTURE,
                  'input': {'width': 640, 'height': 320, 'views': 1,
                            'normalization': 'minus_one_to_one', 'resize': 'contain'},
                  'evidence_version': 1}


def validate_signature_config(config):
    if not isinstance(config, dict) or config.get('version') != 2 or config.get('architecture') != SPATIAL_ARCHITECTURE:
        raise ValueError('Unsupported signature contract: a version 2 full-image model is required')
    if config.get('key_evidence', 'printed') != 'printed':
        raise ValueError('Signature labels must describe printed evidence')
    shape = config.get('input', {})
    if not isinstance(shape, dict) or shape.get('views') != 1 or shape.get('resize') != 'contain' or shape.get('normalization') != 'minus_one_to_one' or config.get('evidence_version') != 1:
        raise ValueError('Unsupported signature image/evidence contract')
    for field, maximum in (('width', 1536), ('height', 1024)):
        value = shape.get(field)
        if type(value) is not int or not 32 <= value <= maximum or value % 32:
            raise ValueError(f'Invalid signature input {field}: {value}')
    return deepcopy(config)


def signature_config(path):
    path = Path(path)
    config = path / 'config.json' if path.is_dir() else path.with_name('config.json')
    if not config.is_file():
        raise ValueError(f'Missing full-image signature contract: {config}')
    return validate_signature_config(json.loads(config.read_text()))


def full_signature_array(path, augment=False, *, width=640, height=320):
    """Whole-image contract: no line detector, semantic crop, or aspect stretch."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
    if augment:
        # Mild, human-readable scan/engraving variation; never remove content.
        angle = float(np.random.uniform(-1.2, 1.2))
        image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor='white')
        px, py = int(image.width * np.random.uniform(0, .06)), int(image.height * np.random.uniform(0, .06))
        image = ImageOps.expand(image, (px, py, px, py), 'white')
    image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), 'white')
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    if augment:
        x = int(np.random.randint(0, width - image.width + 1))
        y = int(np.random.randint(0, height - image.height + 1))
    canvas.paste(image, (x, y))
    values = np.asarray(canvas, dtype=np.float32) / 255.
    if augment:
        values = np.clip(values * np.random.uniform(.85, 1.) + np.random.uniform(0., .035), 0, 1)
    return (values.transpose(2, 0, 1) - .5) / .5


def predictions_from_logits(outputs, calibration, *, source='visual_classifier'):
    temperatures = calibration['temperatures']
    if len(outputs) != 3 or len(temperatures) != 3:
        raise ValueError('Signature classification requires three heads and temperatures')
    batch = len(outputs[0])
    probabilities = []
    for output, temperature, classes in zip(outputs, temperatures, (16, 33, 8), strict=True):
        if np.asarray(output).shape != (batch, classes) or not np.isfinite(output).all():
            raise ValueError('Malformed or non-finite signature logits')
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid signature temperature')
        value = np.asarray(output, dtype=np.float64) / temperature
        value -= value.max(-1, keepdims=True)
        probability = np.exp(value)
        probabilities.append(probability / probability.sum(-1, keepdims=True))
    labels = [p.argmax(-1) for p in probabilities]
    confidences = [p.max(-1) for p in probabilities]
    results = []
    for i, (key, numerator, denominator) in enumerate(zip(*labels, strict=True)):
        key_value = int(key) - 7 if key < 15 else None
        time_value = f'{numerator}/{DENOMINATORS[denominator]}' if numerator and denominator else None
        # A positive time needs both components; absence can be indicated by
        # either head. Disagreement about presence is always unresolved.
        time_confidence = float(confidences[1][i] * confidences[2][i])
        inconsistent_time = bool(numerator) != bool(denominator)
        evidence = {}
        for field, value, confidence in [('key', key_value, float(confidences[0][i])),
                                          ('time', time_value, time_confidence)]:
            kind = 'present' if value is not None else 'absent'
            threshold = calibration['thresholds'][f'{field}_{kind}']
            known = bool(calibration.get('calibrated')) and confidence >= threshold
            if field == 'time' and inconsistent_time:
                known = False
            evidence[field] = {'status': ('observed' if value is not None else 'absent') if known else 'unknown',
                               'value': value if known else None, 'candidate': value,
                               'confidence': confidence, 'source': source,
                               'calibrated': bool(calibration.get('calibrated'))}
        results.append({'key': evidence['key']['value'], 'time': evidence['time']['value'],
                        'evidence': evidence, 'source': source})
    return results
