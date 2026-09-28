from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from document_info.prompts import (
    HEADER_PROMPT, TEMPO_PROMPT, STAFF_PROMPT, CLEF_PROMPT, TRANSPOSITION_PROMPT,
)
from shared.instruments import INSTRUMENTS, program_from_visible_name
from shared.tuning import tuning_from_name
from shared.glm_backend import create_backend
from shared.pitch_context import explicit_transposition


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
    if kind == "clef":
        clef = value.get("clef")
        octave = value.get("clef_octave")
        return {
            "clef": clef if isinstance(clef, str) and clef in {"G2", "F4", "C3", "C4", "percussion", "tab"} else None,
            "clef_octave": octave if type(octave) is int and octave in {-24, -12, 0, 12, 24} else None,
        }
    if kind == "transposition":
        instruction = value.get("kind")
        semitones, capo, text = value.get("semitones"), value.get("capo"), value.get("text")
        if not isinstance(instruction, str) or instruction not in {"instrument", "ottava", "capo"}:
            return {"kind": None, "semitones": None, "capo": None, "text": None}
        if instruction == "instrument":
            explicit = explicit_transposition(text)
            if explicit is not None:
                semitones = explicit
            elif program_from_visible_name(text) is not None and semitones == 0:
                semitones = 0
            else:
                return {"kind": None, "semitones": None, "capo": None, "text": text}
        if instruction == "ottava":
            mark = re.sub(r'[\s.\-–_]+', '', str(text or '').casefold())
            shifts = {'8va': 12, '8vaalta': 12, '8vb': -12, '8vabassa': -12,
                      '8ba': -12, '15ma': 24, '15maalta': 24, '15mb': -24,
                      '15mabassa': -24, '15ba': -24}
            if mark in shifts:
                semitones = shifts[mark]
            elif mark not in {'8', '15', 'ottava', 'ottavabassa'}:
                return {"kind": None, "semitones": None, "capo": None, "text": text}
        valid_shift = type(semitones) is int and -36 <= semitones <= 36
        if instruction == "ottava":
            valid_shift = valid_shift and semitones in {-24, -12, 12, 24}
        return {
            "kind": instruction,
            "semitones": semitones if valid_shift and instruction != "capo" else None,
            "capo": capo if instruction == "capo" and type(capo) is int and 0 <= capo <= 24 else None,
            "text": text.strip() if isinstance(text, str) and text.strip() else None,
        }
    if kind == "staff":
        instrument = value.get("instrument")
        count = value.get("string_count")
        return {
            "instrument": instrument if isinstance(instrument, str) and instrument in INSTRUMENTS else None,
            "string_count": count if type(count) is int and 1 <= count <= 12 else None,
        }
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
    backend=None, cancelled=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not regions:
        return {}, []
    backend = backend or create_backend(model_path, adapter_path, device)

    predictions = []
    messages = []
    for region in regions:
        kind = str(region['kind'])
        prompt = {'header': HEADER_PROMPT, 'tempo': TEMPO_PROMPT, 'staff': STAFF_PROMPT,
                  'clef': CLEF_PROMPT, 'transposition': TRANSPOSITION_PROMPT}[kind]
        messages.append([{'role': 'user', 'content': [
            {'type': 'image', 'url': region['image']}, {'type': 'text', 'text': prompt},
        ]}])
    outputs = []
    batch_size = 32 if getattr(backend, 'supports_ragged_batch', False) else 8
    for offset in range(0, len(regions), batch_size):
        if cancelled and cancelled():
            from shared.tasks import Cancelled

            raise Cancelled("已取消谱面信息识别")
        batch = messages[offset:offset + batch_size]
        if hasattr(backend, 'generate_batch'):
            outputs.extend(backend.generate_batch(batch, 128))
        else:
            outputs.extend(backend.generate(message, 128) for message in batch)
    for region, (raw, _count) in zip(regions, outputs, strict=True):
        kind = str(region["kind"])
        if cancelled and cancelled():
            from shared.tasks import Cancelled

            raise Cancelled("已取消谱面信息识别")
        raw = raw.strip()
        parsed = parse_info_response(raw, kind)
        predictions.append({**region, "raw": raw, "parsed": parsed})
    return metadata_from_predictions(predictions), predictions


def metadata_from_predictions(predictions):
    metadata = {"source": "image_document_info_lora"}
    for prediction in predictions:
        kind, parsed = prediction['kind'], prediction['parsed']
        if kind in {"header", "staff"} or (kind == "tempo" and metadata.get("tempo_quarter") is None):
            metadata.update(parsed)

    tuning = tuning_from_name(metadata.get("tuning_name"))
    if tuning:
        metadata["tuning_midi_high_to_low"] = tuning
    elif metadata.get("tuning_name"):
        metadata["warnings"] = [
            f"无法根据“{metadata['tuning_name']}”确定调弦，请对照原谱填写各弦音高。"
        ]
    return metadata
