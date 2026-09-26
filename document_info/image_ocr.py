from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from document_info.prompts import HEADER_PROMPT, TEMPO_PROMPT
from shared.tuning import tuning_from_name
from shared.glm_backend import GlmBackend


def parse_info_response(raw: str, kind: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    if kind == "header":
        return {
            key: item.strip() if isinstance(item, str) and item.strip() else None
            for key in ("title", "artist", "tuning_name")
            for item in [value.get(key)]
        }
    tempo = value.get("tempo_quarter")
    if isinstance(tempo, int) and not isinstance(tempo, bool) and 20 <= tempo <= 400:
        return {"tempo_quarter": tempo}
    return {}


def recognize_document_info(
    regions: list[dict[str, Any]], model_path: Path, adapter_path: Path, device: str,
    backend=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not regions:
        return {}, []
    backend = backend or GlmBackend(model_path, adapter_path, device)

    predictions = []
    metadata: dict[str, Any] = {"source": "image_document_info_lora"}
    for region in regions:
        kind = str(region["kind"])
        prompt = HEADER_PROMPT if kind == "header" else TEMPO_PROMPT
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": region["image"]},
                {"type": "text", "text": prompt},
            ],
        }]
        raw, _count = backend.generate(messages, 128)
        raw = raw.strip()
        parsed = parse_info_response(raw, kind)
        predictions.append({**region, "raw": raw, "parsed": parsed})
        if kind == "header" or (kind == "tempo" and metadata.get("tempo_quarter") is None):
            metadata.update(parsed)

    tuning = tuning_from_name(metadata.get("tuning_name"))
    if tuning:
        metadata["tuning_midi_high_to_low"] = tuning
    elif metadata.get("tuning_name"):
        metadata["warnings"] = [
            f"Unsupported visible tuning label: {metadata['tuning_name']}; pass --tuning for exact playback."
        ]
    return metadata, predictions
