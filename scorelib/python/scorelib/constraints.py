"""Hard score invariants shared with the native editor and recognizer."""
import json
from scorelib import _native


def validate_measure_target(target, mode, *, tuning=None, string_count=None):
    parsed, errors = json.loads(_native.validate_measure_target(target, mode, tuning, string_count))
    return parsed, errors
