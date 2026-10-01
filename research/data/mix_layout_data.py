"""Mix native page datasets with bounded guitar replay and preserved family splits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def build(new, replay, output, replay_pages=12000, new_repeat=1):
    if new_repeat < 1 or replay_pages < 1:
        raise ValueError("Repeat count and replay page limit must be positive")
    if output.exists():
        raise FileExistsError(output)
    (output / "images").mkdir(parents=True)
    (output / "annotations").mkdir()
    families, report = {}, {}
    for split in ("train", "val", "test"):
        images, annotations, categories = [], [], None
        for index, root in enumerate((new, replay)):
            data = json.loads(
                (root / "annotations" / f"instance_{split}.json").read_text()
            )
            if categories is not None and categories != data["categories"]:
                raise ValueError("Layout category meanings differ")
            categories = data["categories"]
            source_catalog = root.parent.parent / "source_catalog.json"
            instruments = (
                {
                    r["source_id"]: r.get("instrument", "guitar")
                    for r in json.loads(source_catalog.read_text())["sources"]
                }
                if source_catalog.exists()
                else {}
            )
            selected = data["images"]
            # Check all available families before sampling replay pages.
            for row in selected:
                if families.setdefault(row["family"], split) != split:
                    raise ValueError("Song family crosses layout splits")
            if split == "train" and index:
                selected = sorted(
                    selected,
                    key=lambda row: sha256(
                        f"{row['source_id']}:{row['file_name']}".encode()
                    ).digest(),
                )[:replay_pages]
            mapping = {}
            for row in selected * (new_repeat if split == "train" and index == 0 else 1):
                identifier = len(images) + 1
                name = f"{index}-{split}-{identifier}.png"
                source = (root / "images" / row["file_name"]).resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                (output / "images" / name).symlink_to(source)
                mapping.setdefault(row["id"], []).append(identifier)
                images.append(
                    {
                        **row,
                        "id": identifier,
                        "file_name": name,
                        "instrument": instruments.get(
                            row.get("source_id"), row.get("instrument", "guitar")
                        ),
                    }
                )
            for row in data["annotations"]:
                for identifier in mapping.get(row["image_id"], []):
                    annotations.append(
                        {
                            **row,
                            "id": len(annotations) + 1,
                            "image_id": identifier,
                        }
                    )
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        (output / "annotations" / f"instance_{split}.json").write_text(
            json.dumps(payload) + "\n"
        )
        report[split] = {"pages": len(images), "boxes": len(annotations)}
    (output / "images_mask").symlink_to("images", target_is_directory=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-pages", type=int, default=12000)
    parser.add_argument("--new-repeat", type=int, default=1, help="Repeat new training pages while retaining one copy in validation and test")
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
