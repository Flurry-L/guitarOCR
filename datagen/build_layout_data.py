from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import shutil
from pathlib import Path

from shared.pdf import open_pdf

from datagen.inventory import build_inventory, source_catalog
from shared.layout_labels import MODES, typed_annotations


CATEGORIES = ("measure", "tempo_region")


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"Image link points to another source: {destination}")
    elif destination.exists():
        raise ValueError(f"Image destination already exists: {destination}")
    else:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def _annotation_box(row: dict, size: list[int]) -> list[float] | None:
    x0, y0, x1, y1 = (float(value) for value in row["box"])
    x0, x1 = max(0.0, x0), min(float(size[0]), x1)
    y0, y1 = max(0.0, y0), min(float(size[1]), y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


def _append_page(
    splits: dict,
    split: str,
    filename: str,
    size: list[int],
    rows: list[dict],
    image_id: int,
    annotation_id: int,
) -> int:
    width, height = size
    splits[split]["images"].append(
        {"id": image_id, "file_name": filename, "width": width, "height": height}
    )
    ordered = []
    for row in rows:
        if row["label"] not in CATEGORIES:
            continue
        box = _annotation_box(row, size)
        if box is not None:
            ordered.append((row["label"], box))
    ordered.sort(key=lambda item: (item[1][1], item[1][0], CATEGORIES.index(item[0])))
    for read_order, (label, (x0, y0, x1, y1)) in enumerate(ordered):
        width_box, height_box = x1 - x0, y1 - y0
        splits[split]["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CATEGORIES.index(label) + 1,
                "bbox": [x0, y0, width_box, height_box],
                "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                "area": width_box * height_box,
                "iscrowd": 0,
                "read_order": read_order,
            }
        )
        annotation_id += 1
    return annotation_id


def _gp8_pages(
    export_root: Path,
    output_root: Path,
    splits: dict,
    families: dict,
    image_id: int,
    annotation_id: int,
) -> tuple[int, int, int]:
    added = 0
    catalog = source_catalog(export_root)
    for document in sorted((export_root / "native-export" / "documents").iterdir()):
        mode, _, source_id = document.name.partition("-")
        if mode not in {"tab", "notation", "both"} or not document.is_dir():
            continue
        label_path = export_root / "labels" / f"{source_id}.json"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        assignment = catalog[source_id]
        family, split = assignment["family"], assignment["split"]
        if split == "test":
            continue
        if family in families["validation" if split == "train" else "train"]:
            raise ValueError(f"Source family crosses dataset splits: {family}")
        families[split].add(family)
        tracks = sorted((document / "tracks").glob("*/layout.json"))
        if len(tracks) != 1:
            raise ValueError(f"Expected one rendered track: {document}")
        layout = json.loads(tracks[0].read_text(encoding="utf-8"))
        if layout.get("display_mode", "tab") != mode:
            raise ValueError(f"Rendered display mode differs from {mode}: {document}")
        pdf_path = tracks[0].with_name("score.pdf")
        boxes_by_page: dict[int, list[dict]] = {}
        for system in layout["systems"]:
            for measure in system["measure_boxes"]:
                boxes_by_page.setdefault(int(system["page"]), []).append(
                    {"label": "measure", "bbox_mm": measure["bbox_mm"]}
                )
        for tempo in layout["tempo_indications"]:
            boxes_by_page.setdefault(int(tempo["page"]), []).append(
                {"label": "tempo_region", "bbox_mm": tempo["bbox_mm"]}
            )
        with open_pdf(pdf_path) as pdf:
            if len(layout["pages"]) != len(pdf):
                raise ValueError(f"Page count differs from PDF: {pdf_path}")
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                pixmap = page.render(180)
                filename = f"page_{image_id:07d}.png"
                destination = output_root / "images" / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                pixmap.save(destination)
                _link(destination, output_root / "images_mask" / filename)
                page_mm = layout["pages"][page_index]["bbox_mm"]
                if int(layout["pages"][page_index]["index"]) != page_number:
                    raise ValueError(f"Page ordering differs from PDF: {pdf_path}")
                rows = []
                for row in boxes_by_page.get(page_number, []):
                    x, y, width, height = (float(value) for value in row["bbox_mm"])
                    x0 = (x - page_mm[0]) * pixmap.width / page_mm[2]
                    y0 = (y - page_mm[1]) * pixmap.height / page_mm[3]
                    rows.append(
                        {
                            "label": row["label"],
                            "box": [
                                x0,
                                y0,
                                x0 + width * pixmap.width / page_mm[2],
                                y0 + height * pixmap.height / page_mm[3],
                            ],
                        }
                    )
                annotation_id = _append_page(
                    splits,
                    split,
                    filename,
                    [pixmap.width, pixmap.height],
                    rows,
                    image_id,
                    annotation_id,
                )
                splits[split]["images"][-1].update({
                    "source_id": source_id, "family": family, "mode": mode,
                    "renderer": "guitarpro8", "page_index": page_index,
                })
                image_id += 1
                added += 1
    return image_id, annotation_id, added


def _sparse_native_pages(
    native_root: Path,
    output_root: Path,
    splits: dict,
    families: dict,
    image_id: int,
    annotation_id: int,
    train_limit: int,
    validation_limit: int,
) -> tuple[int, int, dict[str, int]]:
    candidates = {split: {"first": [], "later": []} for split in splits}
    excluded = families["train"] | families["validation"]
    assignments_path = native_root / "sources.json"
    assignments = (
        json.loads(assignments_path.read_text(encoding="utf-8"))
        if assignments_path.exists()
        else {}
    )
    for document in sorted(p for p in native_root.iterdir() if p.is_dir()):
        assignment = assignments.get(document.name, {})
        family = str(assignment.get("family", document.name))
        if family in excluded:
            continue
        tracks = sorted((document / "tracks").glob("*/layout.json"))
        if len(tracks) != 1:
            continue
        layout_path = tracks[0]
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if not layout.get("tab_only") or not layout.get("systems"):
            continue
        by_page: dict[int, list[dict]] = {}
        for system in layout["systems"]:
            by_page.setdefault(int(system["page"]), []).append(system)
        sparse = {
            page_number: systems
            for page_number, systems in by_page.items()
            if any(len(system["measure_boxes"]) <= 2 for system in systems)
        }
        if not sparse:
            continue
        score_path = layout_path.with_name("official-score.json")
        if not score_path.is_file():
            continue
        score = json.loads(score_path.read_text(encoding="utf-8"))
        rest_only = {
            int(measure["measure_index"])
            for measure in score["tracks"][0]["staves"][0]["measures"]
            if any(voice["events"] for voice in measure["voices"])
            and all(
                not event.get("notes")
                for voice in measure["voices"]
                for event in voice["events"]
            )
        }
        split_value = (
            int.from_bytes(
                sha256(f"sparse:{family}".encode("utf-8")).digest()[:8], "big"
            )
            / 2**64
        )
        split = assignment.get("split", "validation" if split_value >= 0.9 else "train")
        if split == "test":
            continue
        if split not in splits:
            raise ValueError(f"Invalid split: {split}")
        for page_number, systems in sparse.items():
            singleton = sum(len(system["measure_boxes"]) == 1 for system in systems)
            double = sum(len(system["measure_boxes"]) == 2 for system in systems)
            rests = sum(
                int(measure["measure_index"] in rest_only)
                for system in systems
                for measure in system["measure_boxes"]
            )
            priority = 8 * singleton + 4 * double + 2 * rests
            bucket = "first" if page_number == 1 else "later"
            candidates[split][bucket].append(
                (priority, family, page_number, layout_path)
            )

    counts = {"train": 0, "validation": 0, "first": 0, "later": 0}
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        pools = {
            bucket: sorted(values, key=lambda row: (-row[0], row[1], row[2]))
            for bucket, values in candidates[split].items()
        }
        positions = {bucket: 0 for bucket in pools}
        selected: list[tuple[str, tuple]] = []
        used = set()
        while len(selected) < limit:
            progressed = False
            for bucket in ("first", "later"):
                pool = pools[bucket]
                while (
                    positions[bucket] < len(pool) and pool[positions[bucket]][1] in used
                ):
                    positions[bucket] += 1
                if positions[bucket] >= len(pool):
                    continue
                candidate = pool[positions[bucket]]
                positions[bucket] += 1
                used.add(candidate[1])
                selected.append((bucket, candidate))
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break
        if len(selected) < limit:
            raise ValueError(
                f"Only {len(selected)} source-disjoint sparse {split} pages found"
            )
        for bucket, (_priority, family, page_number, layout_path) in selected:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            with open_pdf(layout_path.with_name("score.pdf")) as pdf:
                if (
                    page_number < 1
                    or page_number > len(pdf)
                    or page_number > len(layout["pages"])
                ):
                    raise ValueError(f"Invalid page {page_number}: {layout_path}")
                page = pdf[page_number - 1]
                pixmap = page.render(180)
            filename = f"page_{image_id:07d}.png"
            destination = output_root / "images" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            pixmap.save(
                destination
            )
            _link(destination, output_root / "images_mask" / filename)
            page_mm = layout["pages"][page_number - 1]["bbox_mm"]
            if int(layout["pages"][page_number - 1]["index"]) != page_number:
                raise ValueError(f"Page ordering differs from PDF: {layout_path}")
            rows = []
            for system in layout["systems"]:
                if int(system["page"]) != page_number:
                    continue
                for measure in system["measure_boxes"]:
                    x, y, width, height = (float(value) for value in measure["bbox_mm"])
                    left = (x - page_mm[0]) * pixmap.width / page_mm[2]
                    top = (y - page_mm[1]) * pixmap.height / page_mm[3]
                    rows.append(
                        {
                            "label": "measure",
                            "box": [
                                left,
                                top,
                                left + width * pixmap.width / page_mm[2],
                                top + height * pixmap.height / page_mm[3],
                            ],
                        }
                    )
            for indication in layout["tempo_indications"]:
                if int(indication["page"]) != page_number:
                    continue
                x, y, width, height = (float(value) for value in indication["bbox_mm"])
                left = (x - page_mm[0]) * pixmap.width / page_mm[2]
                top = (y - page_mm[1]) * pixmap.height / page_mm[3]
                rows.append(
                    {
                        "label": "tempo_region",
                        "box": [
                            left,
                            top,
                            left + width * pixmap.width / page_mm[2],
                            top + height * pixmap.height / page_mm[3],
                        ],
                    }
                )
            annotation_id = _append_page(
                splits,
                split,
                filename,
                [pixmap.width, pixmap.height],
                rows,
                image_id,
                annotation_id,
            )
            splits[split]["images"][-1].update({
                "source_id": family, "family": family, "mode": "tab",
                "renderer": "guitarpro8", "page_index": page_number - 1,
            })
            image_id += 1
            counts[split] += 1
            counts[bucket] += 1
            families[split].add(family)
    return image_id, annotation_id, counts


def build_dataset(
    source_root: Path | None,
    output_root: Path,
    gp8_export: Path | None = None,
    sparse_native_root: Path | None = None,
    sparse_train_pages: int = 1500,
    sparse_validation_pages: int = 150,
    include_test: bool = False,
    typed_measures: bool = False,
) -> dict:
    if source_root is None:
        if gp8_export is None:
            raise ValueError("Provide --source inventory or --gp8-export")
        source_root = output_root / "inventory"
        build_inventory(gp8_export, source_root)
        gp8_export = None
    source_root = source_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tracks = [
        json.loads(line)
        for line in (source_root / "track-index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    splits = {
        "train": {"images": [], "annotations": []},
        "validation": {"images": [], "annotations": []},
    }
    if include_test:
        splits["test"] = {"images": [], "annotations": []}
    assigned_families = {}
    for track in tracks:
        previous = assigned_families.setdefault(track["family"], track["split"])
        if previous != track["split"]:
            raise ValueError(f"Source family crosses dataset splits: {track['family']}")
    families = {split: set() for split in splits}
    annotation_id = 1
    image_id = 1
    for track in tracks:
        split = track["split"]
        if split not in splits:
            continue
        if any(error["phase"] in {"assets", "layout"} for error in track["errors"]):
            raise ValueError(f"Unresolved layout source: {track['sequence_id']}")
        families[split].add(track["family"])
        pages_file = source_root / track["folder"] / "layout-pages.jsonl"
        for line in pages_file.read_text(encoding="utf-8").splitlines():
            page = json.loads(line)
            if page["split"] != split or page["family"] != track["family"]:
                raise ValueError(f"Split mismatch: {pages_file}")
            page_mode = page.get("mode") or track.get("mode")
            if typed_measures and page_mode not in MODES:
                raise ValueError(f"Typed labels require explicit page display mode: {pages_file}")
            if page.get("mode") and track.get("mode") and page["mode"] != track["mode"]:
                raise ValueError(f"Display mode mismatch: {pages_file}")
            image_source = (source_root / page["image"]).resolve()
            if (
                not image_source.is_relative_to(source_root)
                or not image_source.is_file()
            ):
                raise ValueError(f"Missing or escaping page image: {image_source}")
            filename = f"page_{image_id:07d}{image_source.suffix.lower()}"
            _link(image_source, output_root / "images" / filename)
            _link(image_source, output_root / "images_mask" / filename)
            annotation_id = _append_page(
                splits,
                split,
                filename,
                page["image_size"],
                page["annotations"],
                image_id,
                annotation_id,
            )
            splits[split]["images"][-1].update({
                "source_id": track.get("source_id", track["sequence_id"]),
                "family": track["family"],
                "mode": page_mode or "tab",
                "renderer": page.get("renderer", track.get("renderer", "guitarpro8")),
                "page_index": page["page_index"],
            })
            image_id += 1
    gp8_pages = 0
    if gp8_export is not None:
        image_id, annotation_id, gp8_pages = _gp8_pages(
            gp8_export.resolve(), output_root, splits, families, image_id, annotation_id
        )
    sparse_pages = {"train": 0, "validation": 0, "first": 0, "later": 0}
    if sparse_native_root is not None:
        image_id, annotation_id, sparse_pages = _sparse_native_pages(
            sparse_native_root.resolve(),
            output_root,
            splits,
            families,
            image_id,
            annotation_id,
            sparse_train_pages,
            sparse_validation_pages,
        )
    if families["train"] & families["validation"]:
        raise ValueError("Source families overlap between train and validation")
    categories = [
        {"id": index, "name": name, "supercategory": "score"}
        for index, name in enumerate(CATEGORIES, start=1)
    ]
    annotations_root = output_root / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)
    for split, payload in splits.items():
        name = "val" if split == "validation" else split
        payload = {**payload, "categories": categories}
        if typed_measures:
            payload = typed_annotations(payload)
        (annotations_root / f"instance_{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    summary = {
        "typed_measures": typed_measures,
        **{split: len(payload["images"]) for split, payload in splits.items()},
        "gp8_pages": gp8_pages,
        "sparse_pages": sparse_pages,
        "by_mode": {split: dict(Counter(image.get("mode", "tab") for image in payload["images"])) for split, payload in splits.items()},
        "annotations": {split: len(payload["annotations"]) for split, payload in splits.items()},
        "inventory_sha256": sha256((source_root / "track-index.jsonl").read_bytes()).hexdigest(),
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PP-DocLayoutV3 score page labels"
    )
    parser.add_argument(
        "--source", type=Path, help="Layout inventory root; omit with --gp8-export"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gp8-export", type=Path)
    parser.add_argument("--sparse-native-root", type=Path)
    parser.add_argument("--sparse-train-pages", type=int, default=1500)
    parser.add_argument("--sparse-validation-pages", type=int, default=150)
    parser.add_argument("--include-test", action="store_true", help="Write a held-out instance_test.json; never used by training")
    parser.add_argument("--typed-measures", action="store_true", help="Label measures by native display mode (four detection classes)")
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.source,
                args.output,
                args.gp8_export,
                args.sparse_native_root,
                args.sparse_train_pages,
                args.sparse_validation_pages,
                args.include_test,
                args.typed_measures,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
