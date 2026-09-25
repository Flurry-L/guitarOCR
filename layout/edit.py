"""Validate user rectangles in original page pixels and regenerate stage crops."""

from __future__ import annotations

import math
from pathlib import Path
from PIL import Image
from layout.crops import _crop
from shared.artifacts import write_result


def save_layout(pages: list[dict], boxes: list[dict], output: Path, mode: str) -> Path:
    pages = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in page.items()
        }
        for page in pages
    ]
    if mode not in {"tab", "notation", "both"}:
        raise ValueError("请选择 TAB、五线谱或混合谱")
    if len(boxes) > 5000:
        raise ValueError("框数量过多")
    validated = []
    for box in boxes:
        page = int(box["page"])
        if not 1 <= page <= len(pages) or box["kind"] not in {
            "measure",
            "header",
            "tempo",
        }:
            raise ValueError("无效的页码或区域类型")
        bbox = [float(v) for v in box["bbox"]]
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError("框坐标必须是四个有限数值")
        x, y, w, h = bbox
        with Image.open(pages[page - 1]["image"]) as image:
            if (
                min(x, y) < 0
                or min(w, h) < 2
                or x + w > image.width + 0.01
                or y + h > image.height + 0.01
            ):
                raise ValueError("框必须位于页面内，宽高至少为 2 像素")
        validated.append({"page": page, "kind": box["kind"], "bbox": bbox})
    if not any(b["kind"] == "measure" for b in validated):
        raise ValueError("请至少添加一个小节框")
    if sum(b["kind"] == "header" for b in validated) > 1:
        raise ValueError("请只保留一个标题信息框")
    # The editor supplies reading order; preserve it within each page.
    validated.sort(key=lambda b: b["page"])
    output.mkdir(parents=True, exist_ok=True)
    records, regions = [], []
    for box in validated:
        page = pages[box["page"] - 1]
        with Image.open(page["image"]) as image:
            x, y, w, h = box["bbox"]
            if box["kind"] == "measure":
                path = output / f"m{len(records) + 1:04d}.png"
                _crop(image, box["bbox"]).save(path)
                records.append(
                    {
                        "measure_number": len(records) + 1,
                        "page": box["page"],
                        "system_index": 0,
                        "system_measure_index": len(records),
                        "bbox": box["bbox"],
                        "geometry_source": "manual",
                        "image": str(path.resolve()),
                        "source_page": str(page["image"]),
                        "source_pdf": page.get("source_pdf"),
                        "pdf_page": page.get("pdf_page"),
                    }
                )
            else:
                path = output / f"region_{len(regions)}.png"
                image.crop((round(x), round(y), round(x + w), round(y + h))).convert(
                    "RGB"
                ).save(path)
                regions.append({**box, "image": str(path.resolve())})
    return write_result(
        output,
        "layout",
        mode=mode,
        inputs=[str(p["image"]) for p in pages],
        pages=pages,
        records=records,
        regions=regions,
        info_source="image" if regions else "pdf",
    )
