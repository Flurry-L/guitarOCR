from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "measure": "#d33232",
    "tempo_region": "#a2640b",
}


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans.ttf", size)


def visualize_layout(source_root: Path, family: str, output: Path) -> None:
    tracks = [json.loads(line) for line in (source_root / "track-index.jsonl").read_text().splitlines()]
    track = next(row for row in tracks if row["family"] == family)
    page = json.loads((source_root / track["folder"] / "layout-pages.jsonl").read_text().splitlines()[0])
    with Image.open(source_root / page["image"]) as opened:
        canvas = opened.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for row in page["annotations"]:
        label = row["label"]
        if label not in COLORS:
            continue
        box = list(row["box"])
        if box[3] <= box[1]:
            continue
        draw.rectangle(box, outline=COLORS[label], width=4 if label == "measure" else 3)
        if label == "tempo_region":
            draw.text((box[0] + 3, box[1] + 3), label, font=_font(18), fill=COLORS[label])
    legend_height = 55
    result = Image.new("RGB", (canvas.width, canvas.height + legend_height), "white")
    result.paste(canvas, (0, legend_height))
    draw = ImageDraw.Draw(result)
    x = 20
    for label, color in COLORS.items():
        draw.rectangle((x, 14, x + 24, 38), outline=color, width=4)
        draw.text((x + 32, 13), label, font=_font(19), fill="black")
        x += 220
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def visualize_info(dataset_root: Path, family: str, output: Path) -> None:
    rows = []
    for split in ("train", "validation"):
        path = dataset_root / "llamafactory" / f"document_info_{split}.json"
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    selected = {}
    for row in rows:
        image_path = Path(row["images"][0])
        if f"_{family}_" in image_path.name:
            selected[image_path.name.split("_", 1)[0]] = row
    if set(selected) != {"header", "tempo"}:
        raise ValueError(f"Missing header or tempo sample for {family}")
    margin, width = 25, 1560
    panels = []
    for task in ("header", "tempo"):
        row = selected[task]
        with Image.open(row["images"][0]) as opened:
            crop = opened.convert("RGB")
        if crop.width > width - 2 * margin:
            ratio = (width - 2 * margin) / crop.width
            crop = crop.resize((int(crop.width * ratio), int(crop.height * ratio)))
        panels.append((task, crop, row["messages"][1]["content"]))
    total_height = sum(image.height + 120 for _, image, _ in panels) + margin
    canvas = Image.new("RGB", (width, total_height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    y = margin
    for task, crop, target in panels:
        draw.text((margin, y), f"{task.upper()} INPUT", font=_font(22), fill="#1466ac")
        y += 36
        canvas.paste(crop, (margin, y))
        y += crop.height + 13
        draw.text((margin, y), f"GT: {target}", font=_font(20), fill="#168253")
        y += 71
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def visualize_coco_page(dataset_root: Path, image_id: int, output: Path) -> None:
    for split in ("train", "val"):
        data = json.loads(
            (dataset_root / "annotations" / f"instance_{split}.json")
            .read_text(encoding="utf-8")
        )
        image_record = next((row for row in data["images"] if row["id"] == image_id), None)
        if image_record is None:
            continue
        categories = {row["id"]: row["name"] for row in data["categories"]}
        with Image.open(dataset_root / "images" / image_record["file_name"]) as opened:
            canvas = opened.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for annotation in data["annotations"]:
            if annotation["image_id"] != image_id:
                continue
            x, y, width, height = annotation["bbox"]
            draw.rectangle(
                (x, y, x + width, y + height),
                outline=COLORS[categories[annotation["category_id"]]],
                width=4,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        return
    raise ValueError(f"Unknown image id: {image_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize page and metadata training targets")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-dataset", type=Path)
    parser.add_argument("--image-id", type=int)
    args = parser.parse_args()
    visualize_layout(args.source, args.family, args.output / "layout_gt.png")
    visualize_info(args.info, args.family, args.output / "info_gt.png")
    if args.layout_dataset is not None and args.image_id is not None:
        visualize_coco_page(
            args.layout_dataset, args.image_id, args.output / "gp8_layout_gt.png"
        )


if __name__ == "__main__":
    main()
