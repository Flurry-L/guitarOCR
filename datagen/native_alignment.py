"""Check source note labels against the score actually imported by Guitar Pro."""

from fractions import Fraction
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from shared.m2 import parse_measure_target
from shared.percussion import DRUM_KEY_ALIASES


def note_differences(label, native):
    if len(native["tracks"]) != 1 or len(native["tracks"][0]["staves"]) != 1:
        raise ValueError("Single-staff labels require one native track and staff")
    staff = native["tracks"][0]["staves"][0]
    tuning = staff["tuning_pitches_low_to_high"]
    percussion = label["track"].get("instrument") == "drums"
    capo = 0 if percussion else label["track"].get("capo", 0)
    expected, rendered = {}, {}
    for index, measure in enumerate(label["measures"]):
        for voice in parse_measure_target(measure["targets"]["notation"])["voices"]:
            for event in voice["events"]:
                pitches = sorted(
                    n["pitch"] + capo
                    for n in event.get("notes", [])
                    if "dead" not in n.get("effects", [])
                )
                if pitches:
                    expected[index, voice["voice"], event["start"]] = pitches
    for measure in staff["measures"]:
        for voice in measure["voices"]:
            for event in voice["events"]:
                if event["grace"] or event["rest"] or event["placeholder"]:
                    continue
                pitches = []
                for note in event["notes"]:
                    if note["dead"]:
                        continue
                    pitch = tuning[note["native_string_index"]] + note["fret"]
                    if not percussion:
                        pitch += staff["total_capo_frets_low_to_high"][note["native_string_index"]]
                    # Retain unsupported native values so an import that turns
                    # a valid source drum into an invisible note is rejected.
                    pitches.append(
                        DRUM_KEY_ALIASES.get(pitch, pitch) if percussion else pitch
                    )
                if pitches:
                    onset = int(Fraction(*event["offset"]) * 960)
                    rendered[measure["measure_index"], voice["voice_index"], onset] = (
                        sorted(pitches)
                    )
    return [
        {"location": key, "source": expected.get(key), "native": rendered.get(key)}
        for key in sorted(expected.keys() | rendered.keys())
        if expected.get(key) != rendered.get(key)
    ]


def verify_source(job):
    root, row = job
    label = json.loads((root / "labels" / (row["source_id"] + ".json")).read_text())
    folder = (
        root
        / "native-export"
        / "documents"
        / ("notation-" + row["source_id"])
        / "tracks"
    )
    paths = list(folder.glob("*/official-score.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one native notation export: {row['source_id']}")
    differences = note_differences(label, json.loads(paths[0].read_text()))
    return {
        "source_id": row["source_id"],
        "split": row["split"],
        "instrument": row.get("instrument", "guitar"),
        "different_events": len(differences),
        "examples": differences[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = json.loads((args.source / "source_catalog.json").read_text())["sources"]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(
            pool.map(verify_source, ((args.source, row) for row in rows), chunksize=8)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(
        {
            "sources": len(results),
            "excluded_sources": sum(bool(row["different_events"]) for row in results),
            "different_events": sum(row["different_events"] for row in results),
        }
    )


if __name__ == "__main__":
    main()
