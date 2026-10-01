"""GP5 fretboard diagnostics supplied by the shared score library."""
import json
from scorelib import _native
MAX_MELODIC_FRET = 30


def notation_fingering_errors(measure, tuning):
    return _native.notation_fingering_errors(json.dumps(measure), list(tuning))
