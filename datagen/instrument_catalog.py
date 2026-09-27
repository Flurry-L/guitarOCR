"""Select instrument-balanced GP tracks while keeping related songs in one split."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from datagen.gp_sources import analyze_source


def alias(artist, title):
    title = re.sub(r"\s*\(\d+\)\s*$", "", title)
    text = unicodedata.normalize("NFKD", artist + " " + title).casefold()
    return "".join(c for c in text if c.isalnum())


def identity(label):
    return sha256(
        "\n".join(m["targets"]["notation"] for m in label["measures"]).encode()
    ).hexdigest()


def candidate(row):
    try:
        label = analyze_source(Path(row["path"]), 0)
        return row, label, None
    except Exception as error:
        return row, None, repr(error)


def build(audit, splits, previous, output, workers=16):
    """Use a parsed corpus inventory; preserve the previous model's held-out songs."""
    assignments = json.loads(splits.read_text())["source_splits"]
    assignments = {k: {"dev": "validation"}.get(v, v) for k, v in assignments.items()}
    known_aliases, known_music = defaultdict(set), defaultdict(set)
    old_sources = set()
    if previous:
        old = json.loads((previous / "source_catalog.json").read_text())["sources"]
        for entry in old:
            assignments[entry["family"]] = entry["split"]
            label = json.loads(
                (previous / "labels" / (entry["source_id"] + ".json")).read_text()
            )
            old_sources.add(label["sha256"])
            a = alias(label["song"]["artist"], label["song"]["title"])
            if a:
                known_aliases[a].add(entry["split"])
            known_music[identity(label)].add(entry["split"])
    rows = []
    for line in audit.read_text().splitlines():
        row = json.loads(line)
        if "error" in row or not (8 <= row["measures"] <= 128 and row["notes"] >= 32):
            continue
        if row["category"] == "guitar" and row["strings"] == 6:
            continue  # The mixed training set also replays the existing guitar corpus.
        row["split"] = assignments.get(row["family"])
        if row["split"] not in {"train", "validation", "test"}:
            continue
        row["bucket"] = (
            (row["category"] + "-" + str(row["strings"]))
            if row["category"] in {"guitar", "bass"}
            else row["category"]
        )
        rows.append(row)
    rows.sort(key=lambda r: sha256(("20260928:" + r["path"]).encode()).digest())
    counts, rejected = Counter(), Counter()
    selected, selected_hashes, selected_music = [], set(), set()
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for row, label, error in pool.map(candidate, rows, chunksize=8):
            bucket, split = row["bucket"], row["split"]
            limit = (
                (
                    700
                    if bucket == "drums"
                    else 500
                    if bucket == "piano"
                    else 100
                    if bucket in {"strings", "other"}
                    else 240
                )
                if split == "train"
                else (80 if bucket in {"drums", "piano"} else 40)
            )
            if counts[(bucket, split)] >= limit:
                continue
            if error:
                rejected["label_error"] += 1
                continue
            fingerprint = identity(label)
            a = alias(row["artist"], row["title"])
            if (known_music.get(fingerprint, set()) | known_aliases.get(a, set())) - {
                split
            }:
                rejected["related_song_in_another_split"] += 1
                continue
            if (
                label["sha256"] in old_sources
                or label["sha256"] in selected_hashes
                or fingerprint in selected_music
            ):
                rejected["duplicate"] += 1
                continue
            selected.append(
                dict(
                    source_path=row["path"],
                    family=row["family"],
                    split=split,
                    track_index=0,
                    instrument=label["track"]["instrument"],
                    string_count=row["strings"],
                    bucket=bucket,
                    source_sha256=label["sha256"],
                    music_sha256=fingerprint,
                )
            )
            selected_hashes.add(label["sha256"])
            selected_music.add(fingerprint)
            known_aliases[a].add(split)
            known_music[fingerprint].add(split)
            counts[(bucket, split)] += 1
    payload = dict(
        sources=selected,
        counts={f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        rejected=dict(rejected),
    )
    (output / "input_catalog.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "sources"}, ensure_ascii=False
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
