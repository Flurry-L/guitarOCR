"""Mix metadata and visible staff profiles while checking source-family splits."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from research.data.scan_augment import _save
from research.data.training_samples import dataset_entry


def build(sources, output, workers):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    groups, families, seen, jobs = (
        {s: [] for s in ("train", "validation", "test")},
        {},
        set(),
        [],
    )
    for root in sources:
        for split in groups:
            for prefix in ('document_info', "staff", "pitch_info"):
                path = root / f"{prefix}_{split}.json"
                if not path.exists():
                    continue
                for row in json.loads(path.read_text()):
                    provenance = row["provenance"]
                    if (
                        provenance["split"] != split
                        or families.setdefault(provenance["family"], split) != split
                    ):
                        raise ValueError(
                            "Source family crosses information dataset splits"
                        )
                    identity = json.dumps(
                        [row["images"], row["messages"]], sort_keys=True
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    groups[split].append(row)
                    if (
                        split == "train"
                        and prefix in {"staff", "pitch_info"}
                        and int(sha256(identity.encode()).hexdigest()[:4], 16) % 4 == 0
                    ):
                        augmented = deepcopy(row)
                        image = (
                            output
                            / "images"
                            / f"{sha256(identity.encode()).hexdigest()[:24]}.png"
                        )
                        augmented["images"] = [str(image.resolve())]
                        augmented["provenance"]["augmentation"] = "scan"
                        groups[split].append(augmented)
                        jobs.append((Path(row["images"][0]), image, identity, True))
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_save, jobs, chunksize=16))
    info = {}
    for split, rows in groups.items():
        if not rows:
            raise ValueError(f"No {split} information samples")
        name = f"document_info_{split}"
        (output / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False) + "\n"
        )
        info[name] = dataset_entry(f"{name}.json")
    (output / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    summary = {split: len(rows) for split, rows in groups.items()}
    summary["augmented_staff_samples"] = len(jobs)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    build(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
