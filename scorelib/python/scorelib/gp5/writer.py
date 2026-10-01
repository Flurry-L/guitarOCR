"""GP5 export and a PyGuitarPro object adapter for data generation.

The Rust writer owns fingering, timing, encoding and read-back checks. Source
editing can still use PyGuitarPro objects without maintaining a second writer.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from scorelib.gp5.score import write_score_gp5
from scorelib.gp5.timing import gp5_timing_errors
from scorelib.gp_io import read_gp
from scorelib.instruments import DEFAULT_TUNINGS, DEFAULT_PROGRAMS


class GP5TimingError(ValueError):
    def __init__(self, measures):
        self.measures = list(measures)
        super().__init__(f"GP5 cannot preserve timing in measures {self.measures}")


class GP5ReadbackError(ValueError):
    def __init__(self, measures, locations=()):
        self.measures, self.locations = list(measures), list(locations)
        super().__init__(f"GP5 readback changed measures {self.measures}")


def write_targets_gp5(targets, output, *, mode="both", title="Guitar OCR", artist="",
                      tuning=None, capo=0, instrument="guitar", midi_program=None):
    targets = list(targets)
    invalid = [i for i, t in enumerate(targets, 1) if gp5_timing_errors(t)]
    if invalid:
        raise GP5TimingError(invalid)
    if not targets:
        raise ValueError("At least one measure is required")
    result = {
        "title": title, "artist": artist, "mode": mode, "instrument": instrument,
        "tuning_used": list(tuning if tuning is not None else DEFAULT_TUNINGS[instrument]),
        "capo": capo, "midi_program": midi_program if midi_program is not None else DEFAULT_PROGRAMS[instrument],
        "records": [{"target": t, "measure_number": i, "bar_index": i - 1} for i, t in enumerate(targets, 1)],
    }
    return write_score_gp5(result, output)


def targets_to_song(targets, **options):
    with TemporaryDirectory(prefix="scorelib-") as directory:
        output = write_targets_gp5(targets, Path(directory) / "score.gp5", **options)
        return read_gp(str(output), encoding="cp936")
