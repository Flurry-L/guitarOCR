from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata

import pymupdf
from PIL import Image

from datagen.inventory import build_inventory, source_catalog

from document_info.prompts import HEADER_PROMPT, TEMPO_PROMPT


def _printed_text(value: str) -> str:
    # PDF font mappings may encode Chinese glyphs as compatibility radicals.
    # These supplemental radical glyphs used by GP8's CJK font have no NFKC
    # decomposition. Normalize only the visibility check, never the OCR label.
    value = value.translate(str.maketrans({"⻓": "长", "⻘": "青", "⻛": "风"}))
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _target(field: str, value: object) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": f"<image>{HEADER_PROMPT if field == 'header' else TEMPO_PROMPT}",
            },
            {
                "role": "assistant",
                "content": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def _crop(image: Image.Image, box: list[float], pad: int) -> Image.Image:
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, int(left) - pad),
            max(0, int(top) - pad),
            min(image.width, int(right) + pad + 1),
            min(image.height, int(bottom) + pad + 1),
        )
    ).convert("RGB")


def _header_target(score: dict) -> dict:
    document = score["document"]
    metadata = document["metadata"]
    properties = metadata["properties"]
    header = metadata["first_page_header"]
    staff = score["tracks"][0]["staves"][0]
    return {
        "title": (properties["title"] or "").strip() or None
        if header["title"]["visible"]
        else None,
        "artist": (properties["artist"] or "").strip() or None
        if header["artist"]["visible"]
        else None,
        "tuning_name": (staff["tuning_displayed_label"] or "").strip() or None
        if staff["tuning_label_visible"]
        else None,
    }


def _tempo_target(score: dict, indication: dict) -> dict | None:
    document = score["document"]
    if indication["source"] == "initial":
        tempo = document["tempo"]
    else:
        matches = [
            row
            for row in document["tempo_automations"]
            if row["bar_index"] == indication["master_measure_index"]
            and abs(float(row["position"]) - float(indication["position"])) < 1e-5
        ]
        if len(matches) != 1:
            return None
        tempo = matches[0]
    if not tempo["visible"] or tempo["unit_name"] != "Quarter":
        return None
    return {"tempo_quarter": int(tempo["value"])}


def _overlapping_tempos(indications: list[dict]) -> set[int]:
    ambiguous = set()
    for index, first in enumerate(indications):
        first_x, first_y, first_width, first_height = first["bbox_mm"]
        for other_index in range(index + 1, len(indications)):
            second = indications[other_index]
            if first["page"] != second["page"]:
                continue
            second_x, second_y, second_width, second_height = second["bbox_mm"]
            if max(first_x, second_x) < min(
                first_x + first_width, second_x + second_width
            ) and max(first_y, second_y) < min(
                first_y + first_height, second_y + second_height
            ):
                ambiguous.update((index, other_index))
    return ambiguous


def _gp8_samples(
    export_root: Path, output_root: Path, samples: dict, counts: dict, families: dict
) -> tuple[int, int]:
    added = 0
    rejected = 0
    catalog = source_catalog(export_root)
    for document in sorted((export_root / "native-export" / "documents").glob("tab-*")):
        source_id = document.name.removeprefix("tab-")
        assignment = catalog[source_id]
        family, split = assignment["family"], assignment["split"]
        if split == "test":
            continue
        if family in families["validation" if split == "train" else "train"]:
            raise ValueError(f"Source family crosses dataset splits: {family}")
        families[split].add(family)
        tracks = sorted((document / "tracks").glob("*/layout.json"))
        if len(tracks) != 1:
            raise ValueError(f"Expected one exported track: {document}")
        layout = json.loads(tracks[0].read_text(encoding="utf-8"))
        score = json.loads(
            tracks[0].with_name("official-score.json").read_text(encoding="utf-8")
        )
        by_page: dict[int, list[dict]] = {}
        for row in layout["tempo_indications"]:
            by_page.setdefault(int(row["page"]), []).append(row)
        with pymupdf.open(tracks[0].with_name("score.pdf")) as pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                indications = by_page.get(page_number, [])
                if page_number != 1 and not indications:
                    continue
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(180 / 72, 180 / 72), colorspace=pymupdf.csGRAY
                )
                image = Image.frombytes(
                    "L", (pixmap.width, pixmap.height), pixmap.samples
                )
                page_mm = layout["pages"][page_index]["bbox_mm"]

                def pixel_box(bbox_mm: list[float]) -> list[float]:
                    x, y, width, height = (float(value) for value in bbox_mm)
                    x0 = (x - page_mm[0]) * image.width / page_mm[2]
                    y0 = (y - page_mm[1]) * image.height / page_mm[3]
                    return [
                        x0,
                        y0,
                        x0 + width * image.width / page_mm[2],
                        y0 + height * image.height / page_mm[3],
                    ]

                if page_number == 1:
                    first_measure_tops = [
                        pixel_box(box["bbox_mm"])[1]
                        for system in layout["systems"]
                        if system["page"] == 1
                        for box in system["measure_boxes"]
                    ]
                    if first_measure_tops:
                        header_bottom = min(first_measure_tops)
                        target = _header_target(score)
                        if any(value is not None for value in target.values()):
                            printed = page.get_textbox(
                                pymupdf.Rect(
                                    0,
                                    0,
                                    page.rect.width,
                                    header_bottom / image.height * page.rect.height,
                                )
                            )
                            if any(
                                _printed_text(value) not in _printed_text(printed)
                                for value in target.values()
                                if value
                            ):
                                rejected += 1
                            else:
                                destination = (
                                    output_root
                                    / "images"
                                    / f"gp8_header_{source_id}.png"
                                )
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                _crop(
                                    image, [0, 0, image.width, header_bottom], 0
                                ).save(destination)
                                sample = _target("header", target)
                                sample["images"] = [str(destination.resolve())]
                                samples[split].append(sample)
                                counts["header"] += 1
                                added += 1
                ambiguous = _overlapping_tempos(indications)
                for indication_index, indication in enumerate(indications):
                    if indication_index in ambiguous:
                        continue
                    target = _tempo_target(score, indication)
                    if target is None:
                        continue
                    destination = (
                        output_root
                        / "images"
                        / f"gp8_tempo_{source_id}_p{page_number:03d}_{indication_index:03d}.png"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _crop(image, pixel_box(indication["bbox_mm"]), 6).save(destination)
                    sample = _target("tempo", target)
                    sample["images"] = [str(destination.resolve())]
                    samples[split].append(sample)
                    counts["tempo"] += 1
                    added += 1
    return added, rejected


def build_dataset(
    source_root: Path | None, output_root: Path, gp8_export: Path | None = None,
    *, include_test: bool = False,
) -> dict[str, int]:
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
    samples = {split: [] for split in (("train", "validation", "test") if include_test else ("train", "validation"))}
    counts = {"header": 0, "tempo": 0}
    rejected_headers = 0
    families = {split: set() for split in samples}
    for track in tracks:
        split = track["split"]
        if split not in samples:
            continue
        families[split].add(track["family"])
        folder = Path(track["source_track"])
        if not folder.is_dir():
            raise ValueError(f"Untrusted source track: {folder}")
        score = json.loads((folder / "official-score.json").read_text(encoding="utf-8"))
        layout = json.loads((folder / "layout.json").read_text(encoding="utf-8"))
        pages = [
            json.loads(line)
            for line in (source_root / track["folder"] / "layout-pages.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        pages_by_index = {int(page["page_index"]) + 1: page for page in pages}
        first = pages_by_index.get(1)
        if first:
            measures = [
                row for row in first["annotations"] if row["label"] == "measure"
            ]
            if measures:
                header_bottom = min(float(row["box"][1]) for row in measures)
                with Image.open(source_root / first["image"]) as image:
                    header_image = _crop(image, [0, 0, image.width, header_bottom], 0)
                target = _header_target(score)
                if any(value is not None for value in target.values()):
                    with pymupdf.open(folder / "score.pdf") as pdf:
                        header_bottom_points = (
                            header_bottom
                            / float(first["image_size"][1])
                            * pdf[0].rect.height
                        )
                        printed = (
                            pdf[0]
                            .get_textbox(
                                pymupdf.Rect(
                                    0, 0, pdf[0].rect.width, header_bottom_points
                                )
                            )
                        )
                    if any(
                        _printed_text(value) not in _printed_text(printed)
                        for value in target.values()
                        if value
                    ):
                        rejected_headers += 1
                    else:
                        destination = (
                            output_root
                            / "images"
                            / f"header_{track['family']}_{counts['header']:05d}.png"
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        header_image.save(destination)
                        sample = _target("header", target)
                        sample["images"] = [str(destination.resolve())]
                        sample["provenance"] = {key: track.get(key) for key in ("source_id", "family", "mode", "split")}
                        samples[split].append(sample)
                        counts["header"] += 1
        ambiguous = _overlapping_tempos(layout["tempo_indications"])
        for indication_index, indication in enumerate(layout["tempo_indications"]):
            if indication_index in ambiguous:
                continue
            target = _tempo_target(score, indication)
            page = pages_by_index.get(indication["page"])
            if target is None or page is None:
                continue
            page_width_mm = float(page["page_bbox_mm"][2])
            page_height_mm = float(page["page_bbox_mm"][3])
            width_px, height_px = page["image_size"]
            x, y, width, height = (float(value) for value in indication["bbox_mm"])
            box = [
                (x - page["page_bbox_mm"][0]) * width_px / page_width_mm,
                (y - page["page_bbox_mm"][1]) * height_px / page_height_mm,
                (x + width - page["page_bbox_mm"][0]) * width_px / page_width_mm,
                (y + height - page["page_bbox_mm"][1]) * height_px / page_height_mm,
            ]
            with Image.open(source_root / page["image"]) as image:
                tempo_image = _crop(image, box, 6)
            destination = (
                output_root
                / "images"
                / f"tempo_{track['family']}_{counts['tempo']:05d}.png"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            tempo_image.save(destination)
            sample = _target("tempo", target)
            sample["images"] = [str(destination.resolve())]
            sample["provenance"] = {key: track.get(key) for key in ("source_id", "family", "mode", "split")}
            samples[split].append(sample)
            counts["tempo"] += 1
    gp8_added = 0
    if gp8_export is not None:
        gp8_added, gp8_rejected = _gp8_samples(
            gp8_export.resolve(), output_root, samples, counts, families
        )
        rejected_headers += gp8_rejected
    if any(families[left] & families[right] for left in families for right in families if left != right):
        raise ValueError("Source families overlap between dataset splits")
    llamafactory = output_root / "llamafactory"
    llamafactory.mkdir(exist_ok=True)
    metadata = {}
    for split, rows in samples.items():
        filename = f"document_info_{split}.json"
        (llamafactory / filename).write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        metadata[f"document_info_{split}"] = {
            "file_name": filename,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    (llamafactory / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        **{split: len(rows) for split, rows in samples.items()},
        "gp8_samples": gp8_added,
        "rejected_headers": rejected_headers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build visible document metadata crops"
    )
    parser.add_argument(
        "--source", type=Path, help="Inventory root; omit with --gp8-export"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gp8-export", type=Path)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(args.source, args.output, args.gp8_export, include_test=args.include_test), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
