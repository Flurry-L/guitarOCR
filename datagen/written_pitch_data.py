"""Teach notation OCR to read written pitches; resolve transposition in code."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from datagen.training_samples import measure_sample
from shared.pitch_context import convert_pitch_target


@lru_cache(maxsize=16384)
def source_capo(path):
    return json.loads(Path(path).read_text())["track"].get("capo", 0) if path else 0


def prepare(row):
    row = dict(row)
    instrument = row.get("instrument", "guitar")
    if row["mode"] == "notation" and instrument != "drums":
        if not row.get("pitch_context") and instrument in {"guitar", "bass"}:
            row["pitch_context"] = {
                "instrument_transpose": -12, "clef_octave": 0,
                "clef": "G2" if instrument == "guitar" else "F4", "octave_spans": [],
            }
        if row.get("pitch_context"):
            row["pitch_context"] = {**row["pitch_context"], "capo": source_capo(row.get("label_json"))}
            row["written_pitch"] = True
            row["target"] = convert_pitch_target(row["target"], row["pitch_context"], to_written=True)
    return row


def build(source, output, extra_validation=(), replay_samples=80000):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    info, report, families = {}, {}, {}
    source_info = json.loads((source / "llamafactory/dataset_info.json").read_text())
    for split in ("train", "validation", "test"):
        manifest = source / "manifests" / (split + ("_sample" if split != "train" else "") + ".jsonl")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        for path in extra_validation if split == "validation" else ():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line]
        rows = list({row["id"]: row for row in rows}.values())
        for row in rows:
            if families.setdefault(row["family"], split) != split:
                raise ValueError("Song family crosses splits")
        if split == "train":
            native = [r for r in rows if r.get("pitch_context")]
            replay = sorted((r for r in rows if not r.get("pitch_context")),
                            key=lambda r: sha256(r["id"].encode()).digest())[:replay_samples]
            rows = native + replay
        chats, errors, groups = [], [], Counter()
        for row in rows:
            try:
                prepared = prepare(row)
            except ValueError as error:
                errors.append({"id": row["id"], "reason": str(error)})
                continue
            chats.append(measure_sample(prepared))
            groups[f"{row.get('instrument', 'guitar')}/{row['mode']}"] += 1
        name = f"measure_{split}"
        (output / f"{name}.json").write_text(json.dumps(chats, ensure_ascii=False) + "\n")
        info[name] = {**source_info[name], "file_name": f"{name}.json"}
        report[split] = {"samples": len(chats), "groups": dict(groups), "excluded": errors,
                         "manifest_sha256": sha256(manifest.read_bytes()).hexdigest()}
    for kind in ("hardcases", "scan"):
        name = f"measure_train_{kind}"
        path = source / "llamafactory" / source_info[name]["file_name"]
        rows = sorted(json.loads(path.read_text()), key=lambda r: sha256(r["images"][0].encode()).digest())[:20000]
        (output / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        info[name] = {**source_info[name], "file_name": f"{name}.json"}
        report[kind] = len(rows)
    (output / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print({key: value.get("samples") if isinstance(value, dict) else value for key, value in report.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-validation", type=Path, action="append", default=[])
    parser.add_argument("--replay-samples", type=int, default=80000)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
