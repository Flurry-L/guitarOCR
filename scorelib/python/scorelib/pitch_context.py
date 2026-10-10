"""Convert written and sounding pitches using explicit musical context."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache

from scorelib.instruments import pitch_reference


@lru_cache(maxsize=128)
def key_signature(name):
    from guitarpro.models import KeySignature

    def identity(value):
        mode = "minor" if "Minor" in value else "major"
        tonic = (
            value.replace("Major", "")
            .replace("Minor", "")
            .replace("Sharp", "#")
            .replace("Flat", "b")
            .replace("♯", "#")
            .replace("♭", "b")
        )
        return tonic, mode

    wanted = identity(str(name))
    for candidate in KeySignature:
        if identity(candidate.name) == wanted:
            return candidate
    raise KeyError(name)


def conventional_octave(text):
    if not isinstance(text, str):
        return None
    name = " ".join(text.casefold().strip().rstrip(".").split())
    return {
        "piccolo": 12,
        "picc": 12,
        "contrabass": -12,
        "double bass": -12,
        "guitar": -12,
        "bass guitar": -12,
    }.get(name)


def transpose_key(name, semitones):
    """Transpose a GP key name, choosing the spelling with fewest accidentals."""
    if not semitones % 12:
        return name
    from guitarpro.models import KeySignature

    key = key_signature(name)
    fifths, minor = key.value
    tonic = (7 * fifths + (9 if minor else 0) + semitones) % 12
    choices = [
        k
        for k in KeySignature
        if k.value[1] == minor and (7 * k.value[0] + (9 if minor else 0)) % 12 == tonic
    ]
    return min(choices, key=lambda k: (abs(k.value[0]), abs(k.value[0] - fifths))).name


def convert_pitch_target(target, context, *, to_written=False, mode="notation"):
    """Convert notation pitches once, including pitched grace/trill notes.

    The OCR adapter declares whether its notation output is written or sounding.
    Local octave effects name the affected events; their pitches are converted
    while the effects remain available for GP5 engraving.
    """
    from scorelib.score import format_measure_target, parse_measure_target
    from scorelib.techniques import ornament_position

    measure = parse_measure_target(target)
    direction = -1 if to_written else 1
    instrument_shift = context.get("instrument_transpose") or 0
    base_shift = (
        instrument_shift
        + (context.get("clef_octave") or 0)
        - (context.get("capo") or 0)
    )
    if measure.get("key_signature"):
        measure["key_signature"] = transpose_key(
            measure["key_signature"], direction * instrument_shift
        )
    for voice in measure["voices"]:
        for event in voice["events"]:
            octaves = [
                int(e.partition(":")[2])
                for e in event.get("effects", [])
                if e.startswith("ottava:")
            ]
            if len(octaves) > 1 or any(o not in {-24, -12, 12, 24} for o in octaves):
                raise ValueError("Invalid local octave instruction")
            shift = direction * (base_shift + (octaves[0] if octaves else 0))
            for note in event.get("notes", []):
                if "dead" in note.get("effects", []):
                    continue
                note["pitch"] += shift
                if not 0 <= note["pitch"] <= 127:
                    raise ValueError("Transposed pitch is outside MIDI range")
                effects = []
                for effect in note.get("effects", []):
                    parts = effect.split(":")
                    if parts[0] in {"grace", "trill"} and len(parts) > 1:
                        fret, pitch = ornament_position(parts[1])
                        if pitch is not None:
                            if not 0 <= pitch + shift <= 127:
                                raise ValueError(
                                    "Transposed ornament is outside MIDI range"
                                )
                            parts[1] = (
                                f"f{fret}" if fret is not None else ""
                            ) + f"p{pitch + shift}"
                    effects.append(":".join(parts))
                note["effects"] = effects
    return format_measure_target(measure, mode, preserve_playback=True)


def _staff(record):
    return str(record.get("part_id", "part-1")), str(record.get("staff_id", "staff-1"))


def _distance(box, region):
    _, top, _, height = box
    _, y, _, h = region
    center = y + h / 2
    return max(top - center, center - top - height, 0), abs(center - top - height / 2)


def _apply_clef_changes(state, pending, changes, measure_number):
    clefs = [
        p
        for p in changes
        if p["kind"] == "clef" and p.get("parsed", {}).get("clef") != "tab"
    ]
    if not clefs:
        return
    # The context represents one staff for an entire measure. Different clefs
    # assigned to that scope cannot be resolved by detector iteration order.
    # This also preserves a legitimate within-bar change for review instead of
    # falsely presenting its final clef as applying to every event in the bar.
    unresolved = []
    for field in ("clef", "clef_octave"):
        values = {p.get("parsed", {}).get(field) for p in clefs} - {None}
        # A repeated read of exactly one region may provide its missing field.
        # Another unreadable region in the same bar is a separate observation
        # and cannot disappear merely because some other clef was readable.
        unknown = any(
            p.get("parsed", {}).get(field) is None
            and not any(
                other.get("parsed", {}).get(field) is not None
                and p.get("page") == other.get("page")
                and p.get("bbox") == other.get("bbox")
                for other in clefs
            )
            for p in clefs
        )
        if len(values) == 1 and not unknown:
            state[field] = values.pop()
            pending.discard(field)
        else:
            state[field] = None
            pending.add(field)
            unresolved.append(field)
    if unresolved:
        candidates = [
            {
                key: deepcopy(p[key])
                for key in ("page", "bbox", "parsed", "clef_spatial_evidence")
                if key in p
            }
            for p in clefs
        ]
        state["clef_evidence"] = {
            "status": "unresolved",
            "scope": "staff_measure",
            "measure_number": measure_number,
            "fields": unresolved,
            "candidates": sorted(
                candidates,
                key=lambda p: json.dumps(p, sort_keys=True, separators=(",", ":")),
            ),
        }
    else:
        state.pop("clef_evidence", None)


def apply_pitch_regions(
    records, predictions, *, instrument="guitar", transpose=None, default_transpose=None
):
    """Resolve per-staff state; octave lines keep their horizontal extent.

    semitones means sounding minus written, excluding the separate guitar/bass
    capo offset. This function supplies context without shifting note values.
    """
    records = deepcopy(records)
    changes, spans, unresolved = {}, {}, set()
    for prediction in predictions:
        kind = prediction.get("kind")
        if kind not in {"clef", "transposition"} or not prediction.get("bbox"):
            continue
        if kind == "transposition" and prediction.get("parsed", {}).get("kind") not in {
            "instrument",
            "ottava",
            "capo",
        }:
            continue
        page = prediction.get("page")
        candidates = [(i, r) for i, r in enumerate(records) if r.get("page") == page]
        if not candidates:
            continue
        bbox = prediction["bbox"]
        nearest = min(
            candidates,
            key=lambda pair: (
                _distance(pair[1]["bbox"], bbox),
                abs(pair[1]["bbox"][0] - bbox[0]),
            ),
        )
        index, anchor = nearest
        system = (anchor.get("system_index"), _staff(anchor))
        centre_y = anchor["bbox"][1] + anchor["bbox"][3] / 2
        same = [
            (i, r)
            for i, r in candidates
            if (r.get("system_index"), _staff(r)) == system
            and abs(r["bbox"][1] + r["bbox"][3] / 2 - centre_y)
            < min(r["bbox"][3], anchor["bbox"][3]) / 2
        ]
        parsed = prediction.get("parsed", {})
        is_span = kind == "transposition" and parsed.get("kind") == "ottava"
        if is_span:
            for i, record in same:
                x, _, width, _ = record["bbox"]
                left, right = max(x, bbox[0]), min(x + width, bbox[0] + bbox[2])
                if right - left <= max(2, width * 0.015):
                    continue
                spans.setdefault(i, []).append(
                    {
                        "semitones": parsed.get("semitones"),
                        "start": round((left - x) / width, 4),
                        "end": round((right - x) / width, 4),
                        "region": {
                            key: prediction[key]
                            for key in ("page", "bbox", "image")
                            if key in prediction
                        },
                    }
                )
                if parsed.get("semitones") is None:
                    unresolved.add(i)
        else:
            # Labels to the left of a system begin at its first bar; changes
            # inside a row begin at the bar where the mark starts.
            if kind == "clef":
                centre = bbox[0] + bbox[2] / 2
                index = next(
                    (
                        i
                        for i, r in same
                        if r["bbox"][0] <= centre < r["bbox"][0] + r["bbox"][2]
                    ),
                    index,
                )
            else:
                index = next(
                    (
                        i
                        for i, r in same
                        if r["bbox"][0] <= bbox[0] < r["bbox"][0] + r["bbox"][2]
                    ),
                    min(i for i, _ in same),
                )
            changes.setdefault(index, []).append(prediction)

    states, uncertain, explicit_transposition = {}, {}, set()
    for index, record in enumerate(records):
        local_instrument = record.get("instrument", instrument)
        pending = uncertain.setdefault(_staff(record), set())
        state = states.setdefault(
            _staff(record),
            {
                "instrument_transpose": transpose
                if transpose is not None
                else default_transpose
                if default_transpose is not None
                else (-12 if local_instrument in {"guitar", "bass"} else 0),
                "clef": None,
                "clef_octave": 0,
            },
        )
        _apply_clef_changes(
            state, pending, changes.get(index, []), record.get("measure_number")
        )
        for prediction in changes.get(index, []):
            parsed = prediction.get("parsed", {})
            if prediction["kind"] == "clef":
                continue
            elif parsed.get("kind") == "instrument":
                if transpose is None:
                    if parsed.get("semitones") is None:
                        pending.add("instrument_transpose")
                    else:
                        state["instrument_transpose"] = parsed["semitones"]
                        if conventional_octave(parsed.get("text")) is None:
                            explicit_transposition.add(_staff(record))
                        pending.discard("instrument_transpose")
            elif parsed.get("kind") == "capo":
                if parsed.get("capo") is None:
                    pending.add("capo")
                else:
                    state["capo"] = parsed["capo"]
                    pending.discard("capo")
            else:
                unresolved.add(index)
        effective = dict(state)
        if (
            transpose is None
            and _staff(record) not in explicit_transposition
            and state["clef_octave"]
            and state["clef_octave"] == state["instrument_transpose"]
        ):
            # An octave clef can express the instrument's conventional shift.
            # Explicit numeric instructions and manual overrides stay additive.
            effective["instrument_transpose"] = 0
        record["pitch_context"] = {**effective, "octave_spans": spans.get(index, [])}
        record["pitch_reference"] = pitch_reference(local_instrument)
        if index in unresolved or pending:
            record["pitch_needs_review"] = True
    return records


def prompt_pitch_context(context):
    """Omit paths and detector scores from model input."""
    return {
        key: (
            [{k: span[k] for k in ("semitones", "start", "end")} for span in value]
            if key == "octave_spans"
            else value
        )
        for key in ("clef", "clef_octave", "instrument_transpose", "octave_spans")
        if key in context
        for value in [context[key]]
    }
