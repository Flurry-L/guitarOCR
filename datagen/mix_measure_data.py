"""Mix new instruments with guitar rehearsal and fixed, stratified evaluations."""

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from datagen.training_samples import dataset_entry, measure_sample


def stable(row):
    return sha256(str(row.get("id", row.get("images"))).encode()).digest()


def stratified(rows, count):
    buckets = defaultdict(list)
    for row in rows:
        instrument = row.get("instrument", "guitar")
        key = (
            instrument,
            row["mode"],
            row.get("string_count", 6) if instrument in {"guitar", "bass"} else 0,
        )
        buckets[key].append(row)
    result = []
    for key in sorted(buckets):
        by_source = defaultdict(list)
        for row in sorted(buckets[key], key=stable):
            by_source[row["source_id"]].append(row)
        selected = []
        while len(selected) < count:
            progressed = False
            for source in sorted(by_source):
                if by_source[source]:
                    selected.append(by_source[source].pop())
                    progressed = True
                    if len(selected) == count:
                        break
            if not progressed:
                break
        result.extend(selected)
    return result


def build(
    new,
    replay,
    output,
    replay_samples=180000,
    evaluation_per_group=100,
    exclude_report=(),
):
    excluded = {
        row["source_id"]
        for path in exclude_report
        for row in json.loads(path.read_text())
        if row["different_events"]
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    llama = output / "llamafactory"
    llama.mkdir(exist_ok=True)
    info, report, families, source_ids = {}, {}, {}, set()

    def save(name, rows):
        (llama / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        info[name] = dataset_entry(f"{name}.json")
        report[name] = len(rows)

    for split in ("train", "validation", "test"):
        groups = []
        for root in (new, replay):
            rows = [
                json.loads(line)
                for line in (root / "manifests" / f"{split}.jsonl")
                .read_text()
                .splitlines()
                if line
            ]
            for row in rows:
                source_ids.add(row["source_id"])
                row.setdefault("instrument", "guitar")
                row.setdefault("string_count", 6)
                if families.setdefault(row["family"], split) != split:
                    raise ValueError(f"Source family crosses splits: {row['family']}")
            if split == "train" and root == replay:
                rows = sorted(rows, key=stable)[:replay_samples]
            rows = [row for row in rows if row["source_id"] not in excluded]
            groups.extend(rows)
        path = output / "manifests" / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in groups)
        )
        chosen = (
            groups if split == "train" else stratified(groups, evaluation_per_group)
        )
        if split != "train":
            (output / "manifests" / f"{split}_sample.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen)
            )
        save(f"measure_{split}", [measure_sample(row) for row in chosen])
        report[f"{split}_groups"] = dict(
            Counter(
                f"{r['instrument']}/{r['mode']}/{r['string_count']}" for r in chosen
            )
        )
    for kind, limit in (("hardcases", 30000), ("scan", 45000)):
        added = new / "llamafactory" / f"measure_train_{kind}.json"
        rows = json.loads(added.read_text()) if added.is_file() else []
        old = json.loads(
            (replay / "llamafactory" / f"measure_train_{kind}.json").read_text()
        )
        rows.extend(sorted(old, key=stable)[:limit])
        if excluded:

            def allowed(row):
                matches = {
                    part.split("_")[0] for part in Path(row["images"][0]).parts
                } & source_ids
                if len(matches) != 1:
                    raise ValueError(
                        f"Cannot resolve augmented sample source: {row['images']}"
                    )
                return not (matches & excluded)

            rows = [row for row in rows if allowed(row)]
        save(f"measure_train_{kind}", rows)
    report["excluded_native_mismatch_sources"] = len(excluded)
    report["alignment_reports"] = {
        str(path): sha256(path.read_bytes()).hexdigest() for path in exclude_report
    }
    (llama / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-samples", type=int, default=180000)
    parser.add_argument("--evaluation-per-group", type=int, default=100)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="Native note-alignment reports; exclude sources whose printed notes differ",
    )
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
