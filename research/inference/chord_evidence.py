"""ONNX diagram evidence; no raster heuristic can accept or erase a diagram."""

import json

import numpy as np
from PIL import Image

CONTRACT = "guitarocr.chord-evidence/1"
PREPROCESSING = "rgb_imagenet_letterbox_lanczos_v1"
OUTPUT_NAMES = ("kind", "strings", "base", "frets", "fingers", "barres")


def image_tensor(image):
    image = image.convert("RGB")
    scale = 256 / max(image.size)
    width = max(1, int(image.width * scale + 0.5))
    height = max(1, int(image.height * scale + 0.5))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (256, 256), "white")
    canvas.paste(image, ((256 - width) // 2, (256 - height) // 2))
    values = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255
    return (
        values - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    ) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def probabilities(logits, temperature):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = np.exp(values - values.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def decode(outputs, contract):
    """Decode learned categorical/edge predictions, preserving confidence.

    Contiguous edge labels encode a barre relation. This is the output grammar,
    not a second visual detector or evidence inferred from a chord name.
    """
    temperature = contract["temperature"]
    p = [
        probabilities(v, temperature[name])
        for name, v in zip(OUTPUT_NAMES[:5], outputs[:5])
    ]
    kind, count, base = (int(p[i].argmax()) for i in range(3))
    frets = p[3].argmax(-1)[:count]
    fingers = p[4].argmax(-1)[:count]
    edges = 1 / (
        1
        + np.exp(
            -np.clip(
                np.asarray(outputs[5], np.float64) / temperature["barres"],
                -80,
                80,
            )
        )
    )
    edges = edges[:, : max(0, count - 1)]
    confidence = {
        "kind": float(p[0].max()),
        "strings": float(p[1].max()),
        "base": float(p[2].max()),
        "frets": float(p[3][:count].max(-1).min()) if count else 0.0,
        "fingers": float(p[4][:count].max(-1).min()) if count else 0.0,
        "barres": float(np.maximum(edges, 1 - edges).min()) if edges.size else 0.0,
    }
    relative_barres = []
    for fret, row in enumerate(edges, 1):
        start = None
        for i, active in enumerate([*(row >= 0.5), False]):
            if active and start is None:
                start = i
            elif not active and start is not None:
                relative_barres.append([fret, start, i])
                start = None
    relative = ["x" if f == 0 else 0 if f == 1 else int(f) - 1 for f in frets]
    diagram = None
    if kind == 0 and count > 0 and base > 0:
        diagram = {
            "base_fret": base,
            "frets": [
                base + f - 1 if type(f) is int and f > 0 else f for f in relative
            ],
            "fingers": [None if f == 0 else int(f) - 1 for f in fingers],
            "barres": [[base + f - 1, a, b] for f, a, b in relative_barres],
        }
        from scorelib.chords import normalize_diagram

        diagram = normalize_diagram(diagram)
    thresholds = contract["thresholds"]
    accepted = {key: confidence[key] >= thresholds[key] for key in confidence}
    for name, keys in [
        ("geometry", ["kind", "strings", "base", "frets", "barres"]),
        ("full", list(OUTPUT_NAMES)),
    ]:
        confidence[name] = min(confidence[key] for key in keys)
        accepted[name] = all(accepted[key] for key in keys) and confidence[
            name
        ] >= contract["joint_thresholds"][name]
    return {
        "schema": CONTRACT,
        "reader": contract["reader"],
        "kind": ["diagram", "text", "other"][kind],
        "kind_probabilities": p[0].tolist(),
        "strings": count,
        "base_fret": base,
        "relative_frets": relative,
        "relative_barres": relative_barres,
        "diagram": diagram,
        "confidence": confidence,
        "accepted": accepted,
        "review_required": kind == 0
        and not (diagram is not None and all(accepted.values())),
    }


class ChordEvidenceReader:
    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.contract = json.loads(
            self.session.get_modelmeta().custom_metadata_map["guitarocr.chord_evidence"]
        )
        if (
            self.contract["schema"] != CONTRACT
            or self.contract["preprocessing"] != PREPROCESSING
            or self.contract.get("input_size") != 256
            or not isinstance(self.contract.get("reader"), str)
        ):
            raise ValueError("Unsupported chord evidence model contract")
        for name in OUTPUT_NAMES:
            threshold = self.contract["thresholds"][name]
            temperature = self.contract["temperature"][name]
            if (
                not np.isfinite(threshold)
                or not 0.5 <= threshold <= 1.001
                or not np.isfinite(temperature)
                or temperature <= 0
            ):
                raise ValueError("Invalid chord evidence calibration")
        if any(
            not np.isfinite(v) or not 0.5 <= v <= 1.001
            for v in (self.contract["joint_thresholds"][key] for key in ("geometry", "full"))
        ):
            raise ValueError("Invalid joint chord evidence calibration")
        self.consensus = self.contract.get("consensus") == "calibrated_exact_diagram_v1"
        if self.contract.get("consensus") and not self.consensus:
            raise ValueError("Unsupported chord consensus contract")
        if self.consensus:
            secondary = self.contract["secondary_contract"]
            if (
                secondary["schema"] != CONTRACT
                or secondary["preprocessing"] != PREPROCESSING
                or secondary["input_size"] != 256
            ):
                raise ValueError("Unsupported secondary chord contract")
            for name in OUTPUT_NAMES:
                threshold = secondary["thresholds"][name]
                temperature = secondary["temperature"][name]
                if (
                    not np.isfinite(threshold)
                    or not 0.5 <= threshold <= 1.001
                    or not np.isfinite(temperature)
                    or temperature <= 0
                ):
                    raise ValueError("Invalid secondary chord calibration")
            if any(
                not np.isfinite(v) or not 0.5 <= v <= 1.001
                for v in (secondary["joint_thresholds"][key] for key in ("geometry", "full"))
            ):
                raise ValueError("Invalid secondary joint chord calibration")
        self.output_names = (
            [
                *OUTPUT_NAMES,
                *("secondary_" + name for name in OUTPUT_NAMES),
                "agreement",
            ]
            if self.consensus
            else list(OUTPUT_NAMES)
        )
        if [o.name for o in self.session.get_outputs()] != self.output_names:
            raise ValueError("Unexpected chord evidence outputs")

    def predict(self, images):
        if not images:
            return []
        if len(images) > 32:
            raise ValueError("Chord evidence batch exceeds 32 images")
        values = np.stack([image_tensor(image) for image in images])
        return self.predict_tensors(values)

    def predict_tensors(self, values):
        """Evaluate already-preprocessed tensors for reproducible dataset checks."""
        batch = len(values)
        if (
            not 1 <= batch <= 32
            or values.shape != (batch, 3, 256, 256)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("Malformed chord evidence input")
        outputs = self.session.run(self.output_names, {"images": values})
        shapes = [(3,), (13,), (37,), (12, 9), (12, 6), (7, 11)]
        if self.consensus:
            shapes = [*shapes, *shapes, (2,)]
        for value, shape in zip(outputs, shapes):
            if value.shape != (batch, *shape) or not np.isfinite(value).all():
                raise ValueError("Malformed chord evidence output")
        rows = []
        for i in range(batch):
            evidence = decode([v[i] for v in outputs[:6]], self.contract)
            if self.consensus:
                secondary = decode(
                    [v[i] for v in outputs[6:12]], self.contract["secondary_contract"]
                )
                agreement = {
                    key: bool(outputs[12][i, j] >= 0.5)
                    for j, key in enumerate(["geometry", "full"])
                }
                if evidence["diagram"] is None or secondary["diagram"] is None:
                    agreement = dict.fromkeys(agreement, False)
                evidence.update(
                    agreement=agreement,
                    secondary_diagram=secondary["diagram"],
                    secondary_confidence=secondary["confidence"],
                )
                for key in ["geometry", "full"]:
                    evidence["accepted"][key] &= agreement[key]
                if evidence["kind"] != secondary["kind"]:
                    evidence["accepted"]["kind"] = False
                evidence["review_required"] = (
                    evidence["kind"] == "diagram" or secondary["kind"] == "diagram"
                ) and not evidence["accepted"]["full"]
            rows.append(evidence)
        return rows
