"""Assemble recognized records into the shared score IR."""
import json
from scorelib import _native


def score_document(result: dict) -> dict:
    return json.loads(_native.score_document(json.dumps(result)))
