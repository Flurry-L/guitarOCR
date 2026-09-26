"""Mix native metadata datasets and deterministic scan variants without split leakage."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from datagen.scan_augment import _save


def build(source: Path, extra: Path, output: Path, repeats: int = 6,
          seed: int = 20260929, workers: int = 8):
    if output.exists():
        raise FileExistsError(output)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    groups = {}
    hashes = {}
    families = {split: set() for split in ("train", "validation", "test")}
    for name, directory in (("base", source), ("extra", extra)):
        for split in families:
            path = directory / f"document_info_{split}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                provenance = row["provenance"]
                if provenance["split"] != split or not provenance.get("family"):
                    raise ValueError(f"Missing or inconsistent source family: {path}")
                families[split].add(provenance["family"])
            groups[name, split] = rows
            hashes[f"{name}_{split}"] = sha256(path.read_bytes()).hexdigest()
    if any(families[a] & families[b] for a in families for b in families if a != b):
        raise ValueError("Source families overlap across splits")
    output.mkdir(parents=True)
    jobs = []
    report = {"seed": seed, "extra_header_repeats": repeats, "source_sha256": hashes,
              "splits": {}, "augmentation": "scan crop=True; generated degradation, not real scans"}
    for split in families:
        base, added = groups["base", split], groups["extra", split]
        rows = list(base)
        # Preserve all base examples and add deterministic degradation to 25% of training.
        selected = sorted(base, key=lambda r: sha256(f"{seed}:{r['images'][0]}".encode()).digest())[:round(len(base) / 4)] if split == "train" else []
        for name, originals in (("base", selected), ("extra", added)):
            for index, row in enumerate(originals):
                header = "title" in json.loads(row["messages"][1]["content"])
                repeat = repeats if split == "train" and name == "extra" and header else 1
                if name == "extra":
                    rows.extend([row] * repeat)
                image = Path(row["images"][0])
                destination = (output / "images" / f"{split}_{name}_{index:05d}.png").resolve()
                augmented = deepcopy(row)
                augmented["images"] = [str(destination)]
                augmented["provenance"].update(augmentation="scan", original_image=str(image))
                rows.extend([augmented] * repeat)
                jobs.append((image, destination, f"{seed}:{split}:{name}:{index}", True))
        (output / f"document_info_{split}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        report["splits"][split] = {"base": len(base), "extra": len(added), "samples": len(rows),
                                   "families": len(families[split])}
    (output / "dataset_info.json").write_bytes((source / "dataset_info.json").read_bytes())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_save, jobs, chunksize=8))
    report["generated_images"] = len(jobs)
    report["dataset_sha256"] = {p.name: sha256(p.read_bytes()).hexdigest()
                                for p in sorted(output.glob("document_info_*.json"))}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
