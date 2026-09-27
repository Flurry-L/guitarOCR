"""LLaMA-Factory input format; musical prompts remain with the OCR stage."""

from measure_ocr.prompts import recognition_prompt


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
