"""Single-image observation samples and shared auxiliary-task serialization."""

import json
from functools import lru_cache
from pathlib import Path

from research.data.prompts import observation_prompt


@lru_cache(maxsize=16)
def _observation_schema(mode, instrument):
    from research.inference.assembly import observation_schema
    return observation_schema(mode, instrument)


def _ordered_target(value, schema):
    """Constrained decoders emit object keys in the schema's declared order."""
    if isinstance(value, dict):
        if 'properties' not in schema:
            # Relation branches share property order; only their endpoint
            # lengths and required fields differ.
            schema = next(branch for branch in schema['anyOf'] if 'properties' in branch)
        properties = schema['properties']
        if extra := value.keys() - properties.keys():
            raise ValueError(f'Unsupported observation fields: {sorted(extra)}')
        return {key: _ordered_target(value[key], child)
                for key, child in properties.items() if key in value}
    if isinstance(value, list):
        if 'items' not in schema:
            schema = next(branch for branch in schema['anyOf'] if branch.get('type') == 'array')
        return [_ordered_target(item, schema['items']) for item in value]
    return value


def dataset_entry(filename):
    return {
        "file_name": filename,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"},
    }


def visual_measure_sample(row):
    """Only renderer-observed facts are answers; complete M is assembly truth."""
    observation = row["observation"]
    if not isinstance(observation, dict) or not {"state", "voices", "relations"} <= observation.keys():
        raise ValueError("A measure sample requires local state, voices and relations")
    sample = visual_measure_inputs(row)
    sample["messages"].append({"role": "assistant", "content": json.dumps(
        _ordered_target(observation, _observation_schema(row['mode'], row.get('instrument'))),
        ensure_ascii=False, separators=(",", ":"), allow_nan=False)})
    sample["provenance"] = {key: row[key] for key in ("family", "split", "renderer") if key in row}
    return sample


def visual_measure_inputs(row):
    """Training and inference read the same single crop, without source answers."""
    return {
        "messages": [{"role": "user", "content": "<image>" + observation_prompt(row["mode"])}],
        "images": [str(row["image"])],
    }


def chat(images, prompt, target, **provenance):
    return {
        "messages": [
            {"role": "user", "content": "<image>" * len(images) + prompt},
            {"role": "assistant", "content": json.dumps(
                target, ensure_ascii=False, separators=(",", ":"))},
        ],
        "images": [str(Path(p).resolve()) for p in images],
        "provenance": provenance,
    }
