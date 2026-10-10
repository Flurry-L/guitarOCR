"""LLaMA-Factory input format; musical prompts remain with the OCR stage."""

import json
from pathlib import Path

from research.data.prompts import state_prompt



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


def visual_measure_sample(row, *, first=None):
    """Serialize a written-pitch target with its previous/next visual context.

    Builders may supply score-start status when their local measure numbering
    differs from the composed page's bar index.
    """
    from research.data.native_supervision import prepare_visual_metadata
    from research.data.technique_visibility import verified_visual_target
    row = verified_visual_target(row)
    row = prepare_visual_metadata(row)
    sample = visual_measure_inputs(row, first=first)
    sample['messages'].append({'role': 'assistant', 'content': row['target']})
    return sample


def visual_measure_inputs(row, *, first=None):
    """Build inference inputs without consulting or validating answer labels."""
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
        ],
        "images": [
            row["image"], row.get("previous_image", row["image"]),
            row.get("next_image", row["image"]),
        ],
    }


def iter_training_samples(rows, *, review_path, serializer=visual_measure_sample):
    """Skip unverified optical answers, retaining every rejected source record."""
    from research.data.technique_visibility import UnverifiedTechniqueVisibility
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open('w') as review:
        for row in rows:
            try:
                yield serializer(row)
            except UnverifiedTechniqueVisibility as error:
                review.write(json.dumps(error.row, ensure_ascii=False) + '\n')


def chat(images, prompt, target, **provenance):
    return {
        "messages": [
            {"role": "user", "content": "<image>" * len(images) + prompt},
            {
                "role": "assistant",
                "content": json.dumps(
                    target, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ],
        "images": [str(Path(p).resolve()) for p in images],
        "provenance": provenance,
    }
