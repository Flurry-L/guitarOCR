"""LLaMA-Factory input format; musical prompts remain with the OCR stage."""

import json
from pathlib import Path

from document_info.prompts import ANNOTATION_PROMPT
from measure_ocr.prompts import recognition_prompt
from shared.score_state import state_prompt


def dataset_entry(filename):
    return {
        "file_name": filename,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def measure_sample(row):
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>"
                + recognition_prompt(
                    row["mode"],
                    row["previous_context"],
                    row.get("instrument", "guitar"),
                    row.get("pitch_context"),
                    written_pitch=row.get("written_pitch", False),
                ),
            },
            {"role": "assistant", "content": row["target"]},
        ],
        "images": [row["image"].replace("\\", "/")],
    }


def visual_measure_sample(row, *, first=None):
    """Serialize a written-pitch target with its previous/next visual context.

    Builders may supply score-start status when their local measure numbering
    differs from the composed page's bar index.
    """
    state = dict(row["score_state"])
    if row["mode"] != "tab":
        state.pop("tuning", None)
    if first is None:
        first = (row["bar_index"] if "bar_index" in row else row["measure_index"]) == 0
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image><image><image>" + state_prompt(
                    row["mode"], row.get("instrument", "guitar"), state,
                    row.get("pitch_context"), first=first, visual_pitch=True,
                ),
            },
            {"role": "assistant", "content": row["target"]},
        ],
        "images": [
            row["image"], row.get("previous_image", row["image"]),
            row.get("next_image", row["image"]),
        ],
    }


def annotation_sample(image, value):
    """Serialize a visible score annotation without changing its label data."""
    return {
        "messages": [
            {"role": "user", "content": "<image>" + ANNOTATION_PROMPT},
            {
                "role": "assistant",
                "content": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": [str(Path(image).resolve())],
    }
