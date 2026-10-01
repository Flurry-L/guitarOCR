"""Attach visible pitch instructions to their staff and preserve their scope."""

from copy import deepcopy
import re
from scorelib.instruments import pitch_reference


def conventional_octave(text):
    if not isinstance(text, str):
        return None
    name = ' '.join(text.casefold().strip().rstrip('.').split())
    return {'piccolo': 12, 'picc': 12, 'contrabass': -12, 'double bass': -12,
            'guitar': -12, 'bass guitar': -12}.get(name)


def explicit_transposition(text):
    """Decode an explicit signed instruction without interpreting a key signature."""
    if not isinstance(text, str):
        return None
    text = " ".join(text.strip().lower().replace("−", "-").split())
    conventional = {'concert pitch': 0, 'at concert pitch': 0}
    if text.rstrip('.') in conventional:
        return conventional[text.rstrip('.')]
    if (octave := conventional_octave(text)) is not None:
        return octave
    match = re.fullmatch(r"(?:transpose|written to sounding:)\s*([+-]?\d+)\s+semitones?\.?", text)
    if match:
        shift = int(match[1])
    else:
        match = re.fullmatch(r"sounds\s+(\d+)\s+semitones?\s+(lower|higher)\.?", text)
        if not match:
            # Resolve standard transposing instruments only when both the
            # instrument/register and its key are explicitly printed.
            name = re.sub(r"\b([a-g])[ -]flat\b", r"\1b", text.replace("♭", "b")).rstrip(".")
            for instrument, key, offset in (
                (r"clarinet", "bb", -2),
                (r"clarinet", "a", -3),
                (r"clarinet", "eb", 3),
                (r"bass clarinet", "bb", -14),
                (r"trumpet", "bb", -2),
                (r"trumpet", "a", -3),
                (r"trumpet", "c", 0),
                (r"trumpet", "d", 2),
                (r"trumpet", "eb", 3),
                (r"soprano sax(?:ophone)?", "bb", -2),
                (r"alto sax(?:ophone)?", "eb", -9),
                (r"tenor sax(?:ophone)?", "bb", -14),
                (r"baritone sax(?:ophone)?", "eb", -21),
                (r"(?:french )?horn", "f", -7),
                (r"(?:english horn|cor anglais)", "f", -7),
            ):
                if re.fullmatch(rf"(?:{instrument} in {key}|{key} {instrument})", name):
                    return offset
            return None
        shift = int(match[1]) * (-1 if match[2] == "lower" else 1)
    return shift if -36 <= shift <= 36 else None


def transpose_key(name, semitones):
    """Transpose a GP key name, choosing the spelling with fewest accidentals."""
    if not semitones % 12:
        return name
    from guitarpro.models import KeySignature

    key = KeySignature[name]
    fifths, minor = key.value
    tonic = (7 * fifths + (9 if minor else 0) + semitones) % 12
    choices = [k for k in KeySignature if k.value[1] == minor
               and (7 * k.value[0] + (9 if minor else 0)) % 12 == tonic]
    return min(choices, key=lambda k: (abs(k.value[0]), abs(k.value[0] - fifths))).name


def convert_pitch_target(target, context, *, to_written=False, mode="notation"):
    """Convert notation pitches once, including pitched grace/trill notes.

    The OCR adapter declares whether its notation output is written or sounding.
    Local octave effects name the affected events; their pitches are converted
    while the effects remain available for GP5 engraving.
    """
    from scorelib.m2 import parse_measure_target
    from scorelib.m2 import format_measure_target
    from scorelib.techniques import ornament_position

    measure = parse_measure_target(target)
    direction = -1 if to_written else 1
    instrument_shift = context.get("instrument_transpose") or 0
    base_shift = instrument_shift + (context.get("clef_octave") or 0) - (context.get("capo") or 0)
    if measure.get("key_signature"):
        measure["key_signature"] = transpose_key(measure["key_signature"], direction * instrument_shift)
    for voice in measure["voices"]:
        for event in voice["events"]:
            octaves = [int(e.partition(":")[2]) for e in event.get("effects", []) if e.startswith("ottava:")]
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
                                raise ValueError("Transposed ornament is outside MIDI range")
                            parts[1] = (f"f{fret}" if fret is not None else "") + f"p{pitch + shift}"
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


def apply_pitch_regions(records, predictions, *, instrument="guitar", transpose=None, default_transpose=None):
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
        if kind == 'transposition' and prediction.get('parsed', {}).get('kind') not in {'instrument', 'ottava', 'capo'}:
            continue
        page = prediction.get("page")
        candidates = [(i, r) for i, r in enumerate(records) if r.get("page") == page]
        if not candidates:
            continue
        bbox = prediction["bbox"]
        nearest = min(candidates, key=lambda pair: (_distance(pair[1]["bbox"], bbox),
                                                   abs(pair[1]["bbox"][0] - bbox[0])))
        index, anchor = nearest
        system = (anchor.get("system_index"), _staff(anchor))
        centre_y = anchor["bbox"][1] + anchor["bbox"][3] / 2
        same = [(i, r) for i, r in candidates
                if (r.get("system_index"), _staff(r)) == system
                and abs(r["bbox"][1] + r["bbox"][3] / 2 - centre_y)
                    < min(r["bbox"][3], anchor["bbox"][3]) / 2]
        parsed = prediction.get("parsed", {})
        is_span = kind == "transposition" and parsed.get("kind") == "ottava"
        if is_span:
            for i, record in same:
                x, _, width, _ = record["bbox"]
                left, right = max(x, bbox[0]), min(x + width, bbox[0] + bbox[2])
                if right - left <= max(2, width * 0.015):
                    continue
                spans.setdefault(i, []).append({
                    "semitones": parsed.get("semitones"),
                    "start": round((left - x) / width, 4),
                    "end": round((right - x) / width, 4),
                    "region": {key: prediction[key] for key in ("page", "bbox", "image") if key in prediction},
                })
                if parsed.get("semitones") is None:
                    unresolved.add(i)
        else:
            # Labels to the left of a system begin at its first bar; changes
            # inside a row begin at the bar where the mark starts.
            if kind == "clef":
                centre = bbox[0] + bbox[2] / 2
                index = next((i for i, r in same if r["bbox"][0] <= centre < r["bbox"][0] + r["bbox"][2]), index)
            else:
                index = next((i for i, r in same if r["bbox"][0] <= bbox[0] < r["bbox"][0] + r["bbox"][2]),
                             min(i for i, _ in same))
            changes.setdefault(index, []).append(prediction)

    states, uncertain, explicit_transposition = {}, {}, set()
    for index, record in enumerate(records):
        local_instrument = record.get("instrument", instrument)
        pending = uncertain.setdefault(_staff(record), set())
        state = states.setdefault(_staff(record), {
            "instrument_transpose": transpose if transpose is not None else
                default_transpose if default_transpose is not None else
                (-12 if local_instrument in {"guitar", "bass"} else 0),
            "clef": None, "clef_octave": 0,
        })
        for prediction in changes.get(index, []):
            parsed = prediction.get("parsed", {})
            if prediction["kind"] == "clef":
                if parsed.get("clef") == "tab":
                    continue
                for key in ("clef", "clef_octave"):
                    if parsed.get(key) is not None:
                        state[key] = parsed[key]
                        pending.discard(key)
                    else:
                        pending.add(key)
            elif parsed.get("kind") == "instrument":
                if transpose is None:
                    if parsed.get("semitones") is None:
                        pending.add("instrument_transpose")
                    else:
                        state["instrument_transpose"] = parsed["semitones"]
                        if conventional_octave(parsed.get('text')) is None:
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
        if (transpose is None and _staff(record) not in explicit_transposition
                and state['clef_octave'] and state['clef_octave'] == state['instrument_transpose']):
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
        key: ([{k: span[k] for k in ("semitones", "start", "end")} for span in value]
              if key == "octave_spans" else value)
        for key in ("clef", "clef_octave", "instrument_transpose", "octave_spans")
        if key in context
        for value in [context[key]]
    }
