"""Pitch-checked public voicings and explicit instrument tuning variants.

This module prepares supervised data only. It is never an inference rule or
an image-to-fingering dictionary. Every derived record retains its source.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

PITCHES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Allowed chord tones and the defining tones that must actually sound.
QUALITIES = {
    "major": ("", {0, 4, 7}, {4}),
    "minor": ("m", {0, 3, 7}, {3}),
    "5": ("5", {0, 7}, {0, 7}),
    "dim": ("dim", {0, 3, 6}, {3, 6}),
    "dim7": ("dim7", {0, 3, 6, 9}, {3, 6, 9}),
    "aug": ("aug", {0, 4, 8}, {4, 8}),
    "sus2": ("sus2", {0, 2, 7}, {2}),
    "sus4": ("sus4", {0, 5, 7}, {5}),
    "7sus4": ("7sus4", {0, 5, 7, 10}, {5, 10}),
    "6": ("6", {0, 4, 7, 9}, {4, 9}),
    "m6": ("m6", {0, 3, 7, 9}, {3, 9}),
    "69": ("6add9", {0, 2, 4, 7, 9}, {2, 4, 9}),
    "7": ("7", {0, 4, 7, 10}, {4, 10}),
    "m7": ("m7", {0, 3, 7, 10}, {3, 10}),
    "maj7": ("maj7", {0, 4, 7, 11}, {4, 11}),
    "mmaj7": ("mmaj7", {0, 3, 7, 11}, {3, 11}),
    "9": ("9", {0, 2, 4, 7, 10}, {2, 4, 10}),
    "m9": ("m9", {0, 2, 3, 7, 10}, {2, 3, 10}),
    "maj9": ("maj9", {0, 2, 4, 7, 11}, {2, 4, 11}),
    "m11": ("m11", {0, 2, 3, 5, 7, 10}, {3, 5, 10}),
    "13": ("13", {0, 2, 4, 5, 7, 9, 10}, {4, 9, 10}),
    "maj13": ("maj13", {0, 2, 4, 5, 7, 9, 11}, {4, 9, 11}),
    "7b5": ("7b5", {0, 4, 6, 10}, {4, 6, 10}),
    "m7b5": ("m7b5", {0, 3, 6, 10}, {3, 6, 10}),
    "aug7": ("aug7", {0, 4, 8, 10}, {4, 8, 10}),
    "7b9": ("7b9", {0, 1, 4, 7, 10}, {1, 4, 10}),
    "7#9": ("7#9", {0, 3, 4, 7, 10}, {3, 4, 10}),
    "add9": ("add9", {0, 2, 4, 7}, {2, 4}),
    "madd9": ("madd9", {0, 2, 3, 7}, {2, 3}),
}


def pitch(name):
    match = re.fullmatch(r"([A-G])([#b]?)(-?\d+)?", name)
    if match is None:
        raise ValueError("Unsupported pitch spelling: " + name)
    letter, accidental, octave = match.groups()
    value = PITCHES[letter] + (
        1 if accidental == "#" else -1 if accidental == "b" else 0
    )
    return value % 12 if octave is None else 12 * (int(octave) + 1) + value


def sounding(row):
    return [
        note + fret
        for note, fret in zip(row["tuning_midi"], row["diagram"]["frets"])
        if type(fret) is int
    ]


def barres_from_fingers(frets, fingers):
    result = []
    for finger in range(1, 5):
        positions = [i for i, value in enumerate(fingers) if value == finger]
        if len(positions) < 2:
            continue
        fret = frets[positions[0]]
        if any(frets[i] != fret for i in positions):
            raise ValueError("One finger assigned multiple frets")
        if any(
            type(value) is not int or value < fret
            for value in frets[min(positions) : max(positions) + 1]
        ):
            raise ValueError("Barre crosses muted or lower-fretted string")
        result.append([fret, min(positions), max(positions)])
    return result


def validate(row):
    diagram = row["diagram"]
    frets = diagram["frets"]
    fingers = diagram["fingers"]
    if len(frets) != len(row["tuning_midi"]) or len(fingers) != len(frets):
        raise ValueError("String/tuning count mismatch")
    if any(
        finger is not None and (type(fret) is not int or fret <= 0)
        for fret, finger in zip(frets, fingers)
    ):
        raise ValueError("Finger on muted/open string")
    if any(finger is not None and not 1 <= finger <= 4 for finger in fingers):
        raise ValueError("Unrecognised library finger")
    stopped = [value for value in frets if type(value) is int and value > 0]
    if stopped and max(stopped) - min(stopped) > 4:
        raise ValueError("Uncomfortable stopped-fret span")
    if stopped and max(stopped) - diagram["base_fret"] >= 7:
        raise ValueError("Unsupported visible fret extent")
    if diagram["barres"] != barres_from_fingers(frets, fingers):
        raise ValueError("Barre/finger ownership mismatch")
    actual = set(note % 12 for note in sounding(row))
    if actual != set(row["pitch_classes"]):
        raise ValueError("Recorded pitches differ from playable strings")
    if (
        not set(row["required_pitch_classes"])
        <= actual
        <= set(row["allowed_pitch_classes"])
    ):
        raise ValueError("Voicing does not express its named quality")
    return row


def source_rows(folder):
    rows, rejected = [], Counter()
    for instrument in ("guitar", "ukulele"):
        path = folder / (instrument + ".json")
        source = json.loads(path.read_text())
        tuning = [pitch(name) for name in source["tunings"]["standard"]]
        for chords in source["chords"].values():
            for chord in chords:
                quality = QUALITIES.get(chord["suffix"])
                if quality is None:
                    rejected["quality_not_audited"] += len(chord["positions"])
                    continue
                suffix, allowed, required = quality
                root = pitch(chord["key"])
                for index, position in enumerate(chord["positions"]):
                    base = position["baseFret"]
                    frets = [
                        "x" if fret == -1 else fret + base - 1 if fret else 0
                        for fret in position["frets"]
                    ]
                    fingers = [finger or None for finger in position["fingers"]]
                    try:
                        barres = barres_from_fingers(frets, fingers)
                        if {f - base + 1 for f, _, _ in barres} != set(
                            position.get("barres", [])
                        ):
                            raise ValueError("Source barre and finger labels disagree")
                        row = {
                            "name": chord["key"] + suffix,
                            "instrument": instrument,
                            "tuning_midi": tuning,
                            "diagram": {
                                "base_fret": base,
                                "frets": frets,
                                "fingers": fingers,
                                "barres": barres,
                            },
                            "allowed_pitch_classes": sorted(
                                (root + value) % 12 for value in allowed
                            ),
                            "required_pitch_classes": sorted(
                                (root + value) % 12 for value in required
                            ),
                            "source": f"chords-db/{instrument}/{chord['key']}/{chord['suffix']}/{index}",
                        }
                        notes = sounding(row)
                        if notes != position["midi"]:
                            raise ValueError("Source MIDI differs from tuning/frets")
                        row["pitch_classes"] = sorted(
                            set(value % 12 for value in notes)
                        )
                        rows.append(validate(row))
                    except ValueError as error:
                        rejected[str(error)] += 1
    return rows, rejected


def instrument_variants(row):
    if row["instrument"] != "guitar":
        return []
    result = []
    # Explicit extended-guitar tunings. Added strings are muted, or remain
    # open only when their actual pitch already belongs to the chord.
    for prefix in ([35], [30, 35], [30, 32, 34, 36]):
        for opens in (False, True):
            extra = [
                0 if opens and note % 12 in row["pitch_classes"] else "x"
                for note in prefix
            ]
            derived = copy.deepcopy(row)
            diagram = derived["diagram"]
            derived["tuning_midi"] = [*prefix, *row["tuning_midi"]]
            diagram["frets"] = [*extra, *diagram["frets"]]
            diagram["fingers"] = [None] * len(prefix) + diagram["fingers"]
            diagram["barres"] = [
                [f, a + len(prefix), b + len(prefix)] for f, a, b in diagram["barres"]
            ]
            derived["instrument"] = f"{len(prefix) + 6}-string-guitar"
            derived["source"] += (
                "/extended-"
                + str(len(prefix))
                + ("-open-chord-tones" if opens else "-muted")
            )
            result.append(validate(derived))
    # A conventional twelve-string guitar has six paired courses. A course
    # shares fret and finger; the four lower courses have octave partners.
    paired = copy.deepcopy(row)
    diagram = paired["diagram"]
    paired["instrument"] = "12-string-guitar-paired-courses"
    paired["tuning_midi"] = [
        note + shift
        for i, note in enumerate(row["tuning_midi"])
        for shift in (0, 12 if i < 4 else 0)
    ]
    diagram["frets"] = [fret for fret in diagram["frets"] for _ in range(2)]
    diagram["fingers"] = [finger for finger in diagram["fingers"] for _ in range(2)]
    diagram["barres"] = barres_from_fingers(diagram["frets"], diagram["fingers"])
    paired["source"] += "/paired-courses"
    result.append(validate(paired))
    # Four lower guitar strings have the same intervals as standard bass,
    # one octave higher. Keep only root-containing voicings whose defining
    # chord tones survive; add the standard low B of a five-string bass.
    for opens in (False, True):
        bass = copy.deepcopy(row)
        diagram = bass["diagram"]
        bass["tuning_midi"] = [23, *[note - 12 for note in row["tuning_midi"][:4]]]
        diagram["frets"] = [
            0 if opens and 11 in row["pitch_classes"] else "x",
            *diagram["frets"][:4],
        ]
        diagram["fingers"] = [None, *diagram["fingers"][:4]]
        diagram["barres"] = barres_from_fingers(diagram["frets"], diagram["fingers"])
        bass["pitch_classes"] = sorted({note % 12 for note in sounding(bass)})
        root = pitch(re.match(r"[A-G][#b]?", row["name"]).group())
        if root not in bass["pitch_classes"] or len(bass["pitch_classes"]) < 2:
            continue
        bass["instrument"] = "5-string-bass"
        bass["source"] += "/lower-strings-bass" + ("-open-B" if opens else "-muted-B")
        try:
            result.append(validate(bass))
        except ValueError:
            pass  # Missing defining tones means this is not the named chord.
    return result


def banjo_rows():
    rows = []
    for name, frets, fingers, intervals in [
        ("G", [0, 0, 0, 0, 0], [None] * 5, {2, 7, 11}),
        ("C", [0, 2, 0, 1, 2], [None, 2, None, 1, 3], {0, 4, 7}),
        ("D", ["x", 0, 2, 3, 4], [None, None, 1, 2, 3], {2, 6, 9}),
        ("F", ["x", 3, 2, 1, 3], [None, 3, 2, 1, 4], {0, 5, 9}),
        ("Em", [0, 2, 0, 0, 2], [None, 1, None, None, 2], {4, 7, 11}),
        ("Am7", [0, 2, 2, 1, 2], [None, 2, 3, 1, 4], {0, 4, 7, 9}),
    ]:
        row = {
            "name": name,
            "instrument": "5-string-banjo",
            "tuning_midi": [67, 50, 55, 59, 62],
            "diagram": {
                "base_fret": 1,
                "frets": frets,
                "fingers": fingers,
                "barres": [],
            },
            "pitch_classes": sorted(intervals),
            "allowed_pitch_classes": sorted(intervals),
            "required_pitch_classes": sorted(intervals),
            "source": "authored-open-g-banjo/" + name,
        }
        rows.append(validate(row))
    return rows


def build(folder, output, exclude_shapes=None):
    from research.data.chord_evidence import shape_identity, shape_split

    exclusions = (
        json.loads(exclude_shapes.read_text()) if exclude_shapes else {"shapes": []}
    )
    excluded = set(exclusions["shapes"])
    rows, rejected = source_rows(folder)
    rows += banjo_rows()
    expanded = []
    for row in rows:
        if shape_identity(row["diagram"]) in excluded:
            rejected["previously_evaluated_shape"] += 1
            continue
        row["split"] = shape_split(row["diagram"])
        expanded.append(row)
        # Derived records stay with the source shape's split as well as their
        # own geometry split. A test/validation template cannot produce train.
        for derived in instrument_variants(row):
            derived["split"] = shape_split(derived["diagram"])
            if (
                derived["split"] == row["split"]
                and shape_identity(derived["diagram"]) not in excluded
            ):
                expanded.append(derived)
    unique = {}
    for row in expanded:
        key = (
            row["name"],
            tuple(row["tuning_midi"]),
            json.dumps(row["diagram"], sort_keys=True),
        )
        unique.setdefault(key, row)
    rows = list(unique.values())
    record = {
        "source": "https://github.com/tombatossals/chords-db",
        "license": "MIT",
        "source_files": ["guitar.json", "ukulele.json"],
        "evaluation_shape_exclusions": exclusions,
        "policy": "Source MIDI and named chord quality checked; source and derived normalized shapes keep their declared split",
        "rejections": dict(rejected),
        "counts": dict(
            Counter(
                (row["split"] + ":" + str(len(row["diagram"]["frets"]))) for row in rows
            )
        ),
        "shape_counts": {
            split: len(
                {
                    shape_identity(row["diagram"])
                    for row in rows
                    if row["split"] == split
                }
            )
            for split in ["train", "validation", "test"]
        },
        "voicings": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(record, separators=(",", ":")))
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in ["voicings", "evaluation_shape_exclusions"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-shapes", type=Path)
    args = parser.parse_args()
    build(args.source, args.output, args.exclude_shapes)
