"""MusicXML export through the same implementation used by the client."""
import json
from pathlib import Path
from scorelib import _native


def score_musicxml(score):
    return _native.score_musicxml(json.dumps(score)).encode("utf-8")


def write_musicxml(score, output):
    output = Path(output)
    _native.write_musicxml(json.dumps(score), str(output))
    return output


def duration_ticks(duration):
    from scorelib.m2 import duration_ticks as exact_ticks
    return int(exact_ticks(duration))
