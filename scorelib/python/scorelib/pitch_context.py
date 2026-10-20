"""Convert written and sounding pitches using the shared score rules."""

import json
from functools import lru_cache
from scorelib import _native


@lru_cache(maxsize=128)
def key_signature(name):
    from guitarpro.models import KeySignature

    try:
        fifths, minor = _native.key_signature(str(name))
    except ValueError as error:
        raise KeyError(name) from error
    return KeySignature((fifths, int(minor)))


def transpose_key(name, semitones):
    return _native.transpose_key(name, semitones)


def convert_pitch_target(target, context, *, to_written=False, mode="notation"):
    return _native.convert_pitch_target(target, json.dumps(context), mode, to_written)
