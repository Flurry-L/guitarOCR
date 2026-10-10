"""Build or combine the current layout labels while preserving source splits."""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

from research.common.images import MODES, TYPED_CATEGORIES

CATEGORIES = [
    {"id": index, "name": name, "supercategory": "score"}
    for index, name in enumerate(TYPED_CATEGORIES, 1)
]
CATEGORY_IDS = {row["name"]: row["id"] for row in CATEGORIES}
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def inventory_pages(source):
    """Read native geometry once, through the shared inventory."""
    source = Path(source).resolve()
    for line in (source / "track-index.jsonl").read_text().splitlines():
        track = json.loads(line)
        if any(error["phase"] in {"assets", "layout"} for error in track["errors"]):
            raise ValueError(f"Unresolved layout source: {track['sequence_id']}")
        for line in (source / track["folder"] / "layout-pages.jsonl").read_text().splitlines():
            page = json.loads(line)
            if page["split"] != track["split"] or page["family"] != track["family"]:
                raise ValueError(f"Source assignment mismatch: {page['image']}")
            mode = page["mode"]
            if mode not in MODES or mode != track["mode"]:
                raise ValueError(f"Invalid display mode: {page['image']}")
            width, height = page["image_size"]
            image = (source / page["image"]).resolve()
            if not image.is_relative_to(source):
                raise ValueError(f"Image is outside the inventory: {image}")
            annotations = []
            for row in sorted(page["annotations"], key=lambda r: (r["box"][1], r["box"][0])):
                label = f"measure_{mode}" if row["label"] == "measure" else row["label"]
                x0, y0, x1, y1 = map(float, row["box"])
                x0, y0, x1, y1 = max(0., x0), max(0., y0), min(width, x1), min(height, y1)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                annotations.append({
                    "category_id": CATEGORY_IDS[label], "bbox": [x0, y0, x1-x0, y1-y0],
                    "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                    "area": (x1-x0) * (y1-y0), "iscrowd": 0, "read_order": len(annotations),
                })
            yield track["split"], image, {
                "width": width, "height": height, "source_id": track["source_id"],
                "family": track["family"], "mode": mode, "renderer": track["renderer"],
                "page_index": page["page_index"],
            }, annotations


def coco_pages(source):
    """Combine prepared datasets, including mixed-staff and short-page variants."""
    source = Path(source).resolve()
    for split, name in SPLITS.items():
        path = source / "annotations" / f"instance_{name}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        names = {row["id"]: row["name"] for row in payload["categories"]}
        if set(names.values()) != set(CATEGORY_IDS):
            raise ValueError(f"Layout data must use the current 11 categories: {path}")
        by_image = defaultdict(list)
        for row in payload["annotations"]:
            by_image[row["image_id"]].append({**row, "category_id": CATEGORY_IDS[names[row["category_id"]]]})
        for row in payload["images"]:
            yield split, source / "images" / row["file_name"], row, by_image.pop(row["id"], [])
        if by_image:
            raise ValueError(f"Annotations reference missing images: {path}")


def build_dataset(sources, output_root, *, coco_sources=(), include_test=False):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    splits = {split: {"images": [], "annotations": [], "categories": CATEGORIES}
              for split in SPLITS if split != "test" or include_test}
    families, image_splits = {}, {}
    image_id, annotation_id = 0, 0
    pages = chain.from_iterable([
        *(inventory_pages(source) for source in sources),
        *(coco_pages(source) for source in coco_sources),
    ])
    for split, source, metadata, annotations in pages:
        source = source.resolve(strict=True)
        source_families = metadata.get("source_families") or [metadata.get("family")]
        if split not in SPLITS or not source_families or any(not family for family in source_families):
            raise ValueError(f"Missing source family or invalid split: {source}")
        for family in source_families:
            if families.setdefault(family, split) != split:
                raise ValueError(f"Source family crosses dataset splits: {family}")
        if image_splits.setdefault(source, split) != split:
            raise ValueError(f"Image crosses dataset splits: {source}")
        if split not in splits:
            continue
        image_id += 1
        filename = f"page_{image_id:07d}{source.suffix.lower()}"
        for folder in ("images", "images_mask"):
            destination = output_root / folder / filename
            destination.parent.mkdir(exist_ok=True)
            try:
                destination.symlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
        splits[split]["images"].append({**metadata, "id": image_id, "file_name": filename})
        for row in annotations:
            annotation_id += 1
            splits[split]["annotations"].append({**row, "id": annotation_id, "image_id": image_id})
    (output_root / "annotations").mkdir()
    for split, payload in splits.items():
        (output_root / "annotations" / f"instance_{SPLITS[split]}.json").write_text(
            json.dumps(payload, ensure_ascii=False) + "\n")
    summary = {
        **{split: len(payload["images"]) for split, payload in splits.items()},
        "by_mode": {split: dict(Counter(row.get("mode", "mixed") for row in payload["images"]))
                    for split, payload in splits.items()},
        "annotations": {split: len(payload["annotations"]) for split, payload in splits.items()},
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", default=[], help="Native inventory directories")
    parser.add_argument("--coco", type=Path, nargs="+", default=[], help="Prepared layout dataset directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if not args.source and not args.coco:
        parser.error("Provide --source inventories or --coco datasets")
    print(json.dumps(build_dataset(args.source, args.output, coco_sources=args.coco,
                                   include_test=args.include_test), ensure_ascii=False))


if __name__ == "__main__":
    main()
