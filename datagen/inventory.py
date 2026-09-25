"""Build the shared source catalog and rendered-page inventory from GP8 exports."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pymupdf

from shared.artifacts import write_json

SPLITS = ("train", "validation", "test")


def source_catalog(root: Path, seed: int = 20260715) -> dict[str, dict]:
    path = root / "source_catalog.json"
    labels = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((root / "labels").glob("*.json"))
    ]
    if not labels:
        raise ValueError(
            f"No source labels in {root / 'labels'}; run datagen.run --phase select first"
        )
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))["sources"]
        catalog = {row["source_id"]: row for row in rows}
        if len(catalog) != len(rows) or set(catalog) != {
            r["source_id"] for r in labels
        }:
            raise ValueError(
                "source_catalog.json must contain each selected source exactly once"
            )
    else:
        old = root / "source_splits.json"
        assignments = (
            json.loads(old.read_text(encoding="utf-8")) if old.exists() else None
        )
        groups = {}
        for label in labels:
            family = str(
                label.get("family") or label.get("sha256") or label["source_id"]
            )
            if family not in groups:
                groups[family] = deepcopy(label)
                groups[family]["source_id"] = family
            else:
                tags = groups[family]["statistics"].get("tags", [])
                groups[family]["statistics"]["tags"] = sorted(
                    set(tags) | set(label["statistics"].get("tags", []))
                )
        if assignments is None:
            from datagen.build_measure_data import _stratified_source_splits

            family_splits, _ = _stratified_source_splits(list(groups.values()), seed)
        catalog = {}
        for label in labels:
            sid = label["source_id"]
            family = str(label.get("family") or label.get("sha256") or sid)
            catalog[sid] = {
                "source_id": sid,
                "family": family,
                "split": assignments[sid]
                if assignments is not None
                else family_splits[family],
                "source_path": label["source_path"],
            }
    families = {}
    for row in catalog.values():
        if row["split"] not in SPLITS or not row["family"]:
            raise ValueError(f"Invalid source assignment: {row}")
        previous = families.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Source family crosses dataset splits: {row['family']}")
    write_json(
        path, {"schema_version": "1.0", "seed": seed, "sources": list(catalog.values())}
    )
    write_json(
        root / "source_splits.json", {sid: row["split"] for sid, row in catalog.items()}
    )
    return catalog


def build_inventory(
    export_root: Path, output: Path, dpi: int = 180, seed: int = 20260715
) -> dict:
    export_root, output = export_root.resolve(), output.resolve()
    catalog = source_catalog(export_root, seed)
    tracks = []
    total = 0
    for sid, assignment in sorted(catalog.items()):
        documents = (
            export_root / "native-export" / "documents" / f"tab-{sid}" / "tracks"
        )
        layouts = sorted(documents.glob("*/layout.json"))
        if len(layouts) != 1:
            raise ValueError(
                f"Expected one exported TAB track for {sid}, found {len(layouts)}; run --phase render"
            )
        source = layouts[0].parent
        layout = json.loads(layouts[0].read_text(encoding="utf-8"))
        folder = Path("tracks") / sid
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        annotations = {}
        for system in layout["systems"]:
            for box in system["measure_boxes"]:
                annotations.setdefault(int(system["page"]), []).append(
                    ("measure", box["bbox_mm"])
                )
        for tempo in layout.get("tempo_indications", []):
            annotations.setdefault(int(tempo["page"]), []).append(
                ("tempo_region", tempo["bbox_mm"])
            )
        pages = []
        with pymupdf.open(source / "score.pdf") as pdf:
            if len(pdf) != len(layout["pages"]):
                raise ValueError(f"PDF/layout page counts differ: {source}")
            for index, page in enumerate(pdf):
                number = index + 1
                page_mm = layout["pages"][index]["bbox_mm"]
                if int(layout["pages"][index]["index"]) != number:
                    raise ValueError(f"Invalid layout page order: {source}")
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(dpi / 72, dpi / 72), colorspace=pymupdf.csGRAY
                )
                image = folder / f"page_{number:03d}.png"
                pix.save(output / image)
                sx, sy = pix.width / page_mm[2], pix.height / page_mm[3]
                rows = []
                for kind, (x, y, w, h) in annotations.get(number, []):
                    rows.append(
                        {
                            "label": kind,
                            "box": [
                                (x - page_mm[0]) * sx,
                                (y - page_mm[1]) * sy,
                                (x + w - page_mm[0]) * sx,
                                (y + h - page_mm[1]) * sy,
                            ],
                        }
                    )
                pages.append(
                    {
                        "page_index": index,
                        "split": assignment["split"],
                        "family": assignment["family"],
                        "image": image.as_posix(),
                        "image_size": [pix.width, pix.height],
                        "page_bbox_mm": page_mm,
                        "annotations": rows,
                    }
                )
                total += 1
        (destination / "layout-pages.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pages),
            encoding="utf-8",
        )
        tracks.append(
            {
                **assignment,
                "sequence_id": f"tab-{sid}",
                "source_track": str(source),
                "folder": folder.as_posix(),
                "errors": [],
            }
        )
    (output / "track-index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tracks),
        encoding="utf-8",
    )
    return {"tracks": len(tracks), "pages": total, "inventory": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp8-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    print(
        json.dumps(
            build_inventory(args.gp8_export, args.output, seed=args.seed),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
