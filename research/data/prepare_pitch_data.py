"""Create native Guitar Pro sources with visible transposition and octave spans."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import guitarpro

from research.data.gp_sources import analyze_source
from scorelib.gp5.writer import targets_to_song


# Verified with the bundled native GP8 importer. These are data-generation
# settings, never an inference lookup from a hidden MIDI program.
PROGRAM_OFFSETS = {0: 0, 24: -12, 33: -12, 56: -2, 60: -7,
                   64: -2, 65: -9, 66: -14, 67: -21, 69: -7, 71: 0}
INSTRUMENT_LABELS = {
    56: ["Trumpet in Bb", "B-flat Trumpet"],
    60: ["Horn in F", "F Horn"],
    64: ["Soprano Sax in Bb", "Bb Soprano Saxophone"],
    65: ["Alto Sax in Eb", "E-flat Alto Saxophone"],
    66: ["Tenor Sax in Bb", "Bb Tenor Saxophone"],
    67: ["Baritone Sax in Eb", "E-flat Baritone Saxophone"],
    69: ["English Horn in F", "Cor anglais in F"],
}


def prepare_named(source: Path, output: Path, per_program: int = 8):
    """Replace numeric instructions with printed names, retaining family splits."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    (output / "labels").mkdir()
    rows = json.loads((source / "source_catalog.json").read_text())["sources"]
    counts, catalog, manifest = Counter(), [], []
    for row in rows:
        if row["instrument"] != "pitched" or row["variant"] != "marked":
            continue
        song = guitarpro.parse(row["source_path"])
        track = song.tracks[0]
        program = track.channel.instrument
        if program not in INSTRUMENT_LABELS:
            continue
        group = (row["split"], program)
        if counts[group] >= (per_program if row["split"] == "train" else max(3, per_program // 3)):
            continue
        counts[group] += 1
        name = INSTRUMENT_LABELS[program][counts[group] % 2]
        identifier = sha256((row["source_id"] + ":named").encode()).hexdigest()[:16]
        song.measureHeaders = song.measureHeaders[:12]
        track.measures = track.measures[:12]
        track.measures[0].voices[0].beats[0].text = name
        path = output / "sources" / (identifier + ".gp5")
        guitarpro.write(song, str(path))
        label = analyze_source(path, 0)
        label.update(source_id=identifier, family=row["family"], modes=["notation"])
        (output / "labels" / (identifier + ".json")).write_text(json.dumps(label, ensure_ascii=False))
        catalog.append({**row, "source_id": identifier, "source_path": str(path.resolve()),
                        "instruction": name, "modes": ["notation"]})
        manifest.append(dict(document_id="notation-" + identifier, source_family_id=row["family"],
                             split="dev" if row["split"] == "validation" else row["split"],
                             instrument_kind="pitched", source="sources/" + identifier + ".gp5", display_mode="notation"))
    (output / "source_catalog.json").write_text(json.dumps({"sources": catalog}, indent=2))
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    print({"sources": len(catalog), "documents": len(manifest)}, flush=True)


def prepare(source: Path, output: Path, excluded: Path, per_group: int):
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    (output / "labels").mkdir()
    rejected = {r["source_id"] for r in json.loads(excluded.read_text()) if r["different_events"]}
    rows = json.loads((source / "source_catalog.json").read_text())["sources"]
    selected, counts = [], Counter()
    for row in sorted(rows, key=lambda r: sha256(r["source_id"].encode()).digest()):
        group = (row["split"], row["instrument"])
        limit = per_group if row["split"] == "train" else max(20, per_group // 8)
        if row["source_id"] not in rejected and counts[group] < limit:
            selected.append(row)
            counts[group] += 1
    catalog, manifest, failures = [], [], []
    for row in selected:
        label = json.loads((source / "labels" / (row["source_id"] + ".json")).read_text())
        instrument = row["instrument"]
        for variant in ("plain", "marked"):
            identifier = sha256(f"{row['source_id']}:{variant}:pitch".encode()).hexdigest()[:16]
            seed = int(identifier[:8], 16)
            program = label["track"]["midi_program"]
            if instrument == "pitched" and (variant == "marked" or program not in PROGRAM_OFFSETS):
                program = (0, 56, 60, 64, 65, 66, 67, 69, 71)[seed % 9]
            if instrument == "guitar":
                program = 24
            elif instrument == "bass":
                program = 33
            shift = 0 if instrument == "drums" else PROGRAM_OFFSETS.get(program, 0)
            try:
                song = targets_to_song(
                    [m["targets"]["notation"] for m in label["measures"][:48]],
                    mode="notation", instrument=instrument, midi_program=program,
                    title=label["song"]["title"], artist=label["song"].get("artist", ""),
                    tuning=label["track"]["tuning_midi_high_to_low"],
                    capo=label["track"].get("capo", 0),
                )
                track = song.tracks[0]
                track.name = label["track"]["name"]
                instruction = None
                if variant == "marked" and instrument != "drums":
                    instruction = (
                        (f"Sounds {abs(shift)} semitones lower", f"Transpose {shift} semitones",
                         f"Written to sounding: {shift:+d} semitones")[seed % 3]
                        if shift else "Concert pitch"
                    )
                    first = next(b for m in track.measures for v in m.voices for b in v.beats)
                    first.text = instruction
                    octaves = [guitarpro.Octave.none, guitarpro.Octave.ottava,
                               guitarpro.Octave.ottavaBassa, guitarpro.Octave.none,
                               guitarpro.Octave.quindicesima, guitarpro.Octave.quindicesimaBassa]
                    for index, measure in enumerate(track.measures):
                        for voice in measure.voices:
                            for beat_index, beat in enumerate(voice.beats):
                                choice = (index // 2 + seed) % 6
                                # Include spans beginning/ending inside a measure.
                                if index % 2 == 0 and beat_index == 0:
                                    choice = 0
                                if choice >= 4 and seed % 5:
                                    choice = 0
                                beat.octave = octaves[choice]
                path = output / "sources" / (identifier + ".gp5")
                guitarpro.write(song, str(path))
                revised = analyze_source(path, 0)
                revised.update(source_id=identifier, family=row["family"])
                modes = ["notation"] if instrument in {"pitched", "drums"} else ["notation", "both", "tab"]
                revised["modes"] = modes
                (output / "labels" / (identifier + ".json")).write_text(json.dumps(revised, ensure_ascii=False))
                catalog.append({**row, "source_id": identifier, "source_path": str(path.resolve()),
                                "original_source_id": row["source_id"], "modes": modes,
                                "variant": variant, "instruction": instruction,
                                "expected_native_transpose": shift})
                for mode in modes:
                    manifest.append(dict(document_id=f"{mode}-{identifier}", source_family_id=row["family"],
                                         split="dev" if row["split"] == "validation" else row["split"], instrument_kind=instrument,
                                         source=f"sources/{identifier}.gp5", display_mode=mode))
            except (ValueError, StopIteration) as error:
                failures.append({"source_id": row["source_id"], "variant": variant, "error": str(error)})
        if len(catalog) % 50 == 0:
            print(f"Prepared {len(catalog)} variants", flush=True)
    (output / "source_catalog.json").write_text(json.dumps({"sources": catalog}, indent=2))
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (output / "preparation-failures.json").write_text(json.dumps(failures, indent=2))
    print({"sources": len(catalog), "documents": len(manifest), "failures": len(failures)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, help="Native note-audit report required for numeric instructions")
    parser.add_argument("--per-group", type=int, default=200)
    parser.add_argument("--named-instruments", action="store_true", help="Use previously prepared pitch sources to make printed instrument names")
    parser.add_argument("--per-program", type=int, default=8)
    args = parser.parse_args()
    if args.per_group < 1 or args.per_program < 1:
        parser.error("Sample limits must be positive")
    if args.named_instruments:
        prepare_named(args.source, args.output, args.per_program)
    elif args.excluded is None:
        parser.error("--excluded is required unless --named-instruments is selected")
    else:
        prepare(args.source, args.output, args.excluded, args.per_group)


if __name__ == "__main__":
    main()
