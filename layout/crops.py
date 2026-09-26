from __future__ import annotations

import json
from pathlib import Path

from shared.defaults import LAYOUT_MODEL, paddle_python
import subprocess
import sys
from typing import Any
from PIL import Image, ImageDraw

from layout.pages import expand_inputs
from layout.classifier import classify_notation_layout
from layout.geometry import _notation_mode, _measure_boxes, _hybrid_tab_pdf_boxes
from layout.pdf_geometry import (
    extract_pdf_vector_measure_boxes, extract_pdf_vector_tab_measure_boxes,
    extract_pdf_vector_tab_systems,
)
from layout.pdf_renderer import MODEL_RENDER_DPI
from shared.layout_labels import MODES, mode_vote


def _crop(page: Image.Image, bbox: list[float]) -> Image.Image:
    left, top, width, height = (float(value) for value in bbox)
    # Match the physical padding used to build the GP8 measure-sequence
    # training crops: 1.5 mm horizontally and 4 mm vertically at 180 DPI.
    # Omitting the vertical context makes dense chords and technique labels
    # materially out-of-distribution during full-document inference.
    pixels_per_mm = MODEL_RENDER_DPI / 25.4
    pad_x = 1.5 * pixels_per_mm
    pad_y = 4.0 * pixels_per_mm
    x0 = max(0, round(left - pad_x))
    y0 = max(0, round(top - pad_y))
    x1 = min(page.width, round(left + width + pad_x))
    y1 = min(page.height, round(top + height + pad_y))
    return page.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).convert("L")


def _document_region_crops(
    pages: list[dict[str, Any]],
    detected_layout: dict[Path, dict[str, Any]],
    records: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if not records:
        return regions
    crop_root = output / "document_crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    first_page = Path(pages[0]["image"])
    first_boxes = [record for record in records if record["page"] == 1]
    if first_boxes:
        with Image.open(first_page) as image:
            header_bottom = max(1, round(min(float(box["bbox"][1]) for box in first_boxes)))
            header_path = crop_root / "header.png"
            image.crop((0, 0, image.width, min(image.height, header_bottom))).convert("RGB").save(header_path)
        regions.append({"kind": "header", "page": 1, "bbox": [0, 0, image.width, min(image.height, header_bottom)], "image": str(header_path.resolve())})

    for page_number, source in enumerate(pages, start=1):
        page_path = Path(source["image"])
        regions_on_page = detected_layout.get(page_path, {}).get("tempo_regions", [])
        if not regions_on_page:
            continue
        with Image.open(page_path) as image:
            for index, region in enumerate(sorted(
                regions_on_page,
                key=lambda row: (row["coordinate"][1], row["coordinate"][0]),
            )):
                left, top, right, bottom = (float(value) for value in region["coordinate"])
                if right <= left or bottom <= top:
                    continue
                path = crop_root / f"tempo_p{page_number:03d}_{index:03d}.png"
                image.crop((
                    max(0, int(left) - 6), max(0, int(top) - 6),
                    min(image.width, int(right) + 7), min(image.height, int(bottom) + 7),
                )).convert("RGB").save(path)
                regions.append({
                    "kind": "tempo", "page": page_number,
                    "bbox": [left, top, right - left, bottom - top],
                    "score": float(region["score"]),
                    "image": str(path.resolve()),
                })
    return regions


def prepare_document_crops(
    inputs: list[Path], output: Path, requested_mode: str, force_pdf_render: bool,
    layout_model_dir: Path | None = None, layout_python: Path | None = None,
    layout_source: str = "auto",
    pages: list[dict] | None = None, allow_empty: bool = False,
    detector=None,
) -> tuple[str, list[dict[str, Any]]]:
    if layout_source not in {"auto", "image", "geometry"}:
        raise ValueError("Unknown layout source")
    if requested_mode not in {"auto", *MODES}:
        raise ValueError("Unknown notation mode")
    if layout_source != "geometry":
        layout_model_dir = layout_model_dir or LAYOUT_MODEL
        candidate = paddle_python()
        layout_python = layout_python or (candidate if candidate.exists() else Path(sys.executable))
    output.mkdir(parents=True, exist_ok=True)
    rendered_root = output / "rendered_pages"
    temp_root = output / "tmp"
    pages = pages if pages is not None else expand_inputs(inputs, rendered_root, temp_root, force_pdf_render)
    if not pages:
        raise ValueError("No supported PDF pages or images were found")
    detected_layout: dict[Path, dict[str, Any]] = {}
    detected_pages = (
        pages if layout_source == "image" or requested_mode == "auto"
        else [source for source in pages if not source.get("vector", bool(source.get("source_pdf")))]
    )
    if layout_source == "image" and layout_model_dir is None:
        raise ValueError("--layout-source image requires --layout-model-dir")
    if layout_source != "geometry" and detected_pages and detector is not None:
        detected_layout = {
            Path(source["image"]): prediction for source, prediction in zip(
                detected_pages, detector([source["image"] for source in detected_pages]), strict=True
            )
        }
    elif layout_source != "geometry" and detected_pages:
        page_manifest = temp_root / "layout_pages.json"
        result_path = temp_root / "layout_boxes.json"
        temp_root.mkdir(parents=True, exist_ok=True)
        page_manifest.write_text(json.dumps([str(source["image"]) for source in detected_pages]))
        subprocess.run([
            str(layout_python or Path(sys.executable)), "-m",
            "layout.detector",
            "--pages", str(page_manifest), "--model-dir", str(layout_model_dir.resolve()),
            "--output", str(result_path), "--include-tempo",
        ], check=True)
        detected_layout = {
            Path(source["image"]): prediction for source, prediction in zip(
                detected_pages, json.loads(result_path.read_text(encoding="utf-8")), strict=True
            )
        }
    records = []
    measure_number = 1
    vector_cache: dict[tuple[Path, str], dict[int, list[dict[str, Any]]]] = {}
    tab_measure_cache: dict[Path, dict[int, list[dict[str, Any]]]] = {}
    tab_system_cache: dict[Path, dict[int, list[dict[str, Any]]]] = {}
    for page_index, source in enumerate(pages, start=1):
        with Image.open(source["image"]) as opened:
            page = opened.convert("L")
        prediction = detected_layout.get(Path(source["image"]), {})
        vote = mode_vote(prediction.get("measures", []))
        mode = requested_mode
        mode_source = "manual"
        if mode == "auto":
            mode, mode_source = vote["mode"], "pp_doclayout"
            if mode is None:
                # Compatibility for old two-class models and an explicitly
                # requested geometry-only run. Typed model predictions never
                # depend on the staff-line classifier.
                page_layout = classify_notation_layout(page)
                if page_layout["layout"] != "unknown":
                    mode = _notation_mode(page_layout["layout"])
                mode_source = "staff_geometry"
        if mode is None and prediction.get("measures"):
            raise ValueError(f"Cannot determine notation type on page {page_index}; select a mode explicitly")
        source["notation_mode"] = mode
        source["notation_mode_source"] = mode_source
        if vote["mode"]:
            source["model_mode"] = vote["mode"]
            source["mode_vote_fraction"] = vote["mode_vote_fraction"]
        boxes = []
        source_pdf = source.get("source_pdf")
        pdf_page = source.get("pdf_page")
        if (
            layout_source == "image"
            or Path(source["image"]) in detected_layout
        ):
            boxes = [
                {**box, "geometry_source": box.get("geometry_source", "pp_doclayout")}
                for box in detected_layout.get(Path(source["image"]), {}).get("measures", [])
            ]
        elif source_pdf is not None and pdf_page is not None:
            resolved_pdf = Path(source_pdf).resolve()
            if mode in {"notation", "both"}:
                cache_key = (resolved_pdf, mode)
                if cache_key not in vector_cache:
                    vector_cache[cache_key] = extract_pdf_vector_measure_boxes(
                        cache_key[0], mode
                    )
                boxes = vector_cache[cache_key].get(int(pdf_page), [])
            elif mode == "tab":
                if resolved_pdf not in tab_measure_cache:
                    tab_measure_cache[resolved_pdf] = (
                        extract_pdf_vector_tab_measure_boxes(resolved_pdf)
                    )
                boxes = tab_measure_cache[resolved_pdf].get(int(pdf_page), [])
                if not boxes:
                    if resolved_pdf not in tab_system_cache:
                        tab_system_cache[resolved_pdf] = extract_pdf_vector_tab_systems(
                            resolved_pdf
                        )
                    boxes = _hybrid_tab_pdf_boxes(
                        page,
                        _measure_boxes(page, mode),
                        tab_system_cache[resolved_pdf].get(int(pdf_page), []),
                    )
        if not boxes and mode in MODES:
            boxes = _measure_boxes(page, mode)
            for box in boxes:
                box["geometry_source"] = "pixel_staff_fallback"
        if not boxes and not allow_empty:
            raise ValueError(f"No {mode} measures detected on page {page_index}")
        overlay = page.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            crop_path = output / "measure_crops" / f"m{measure_number:04d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            _crop(page, box["bbox"]).save(crop_path, format="PNG", compress_level=3)
            left, top, width, height = box["bbox"]
            draw.rectangle((left, top, left + width, top + height), outline=(220, 35, 35), width=2)
            draw.text((left + 2, top + 2), str(measure_number), fill=(220, 35, 35))
            records.append({
                "measure_number": measure_number,
                "mode": (box.get("mode") or mode) if requested_mode == "auto" else requested_mode,
                "mode_source": mode_source,
                **({"detected_mode": box["mode"]} if box.get("mode") else {}),
                **({"score": float(box["score"])} if "score" in box else {}),
                "page": page_index,
                "system_index": box["system_index"],
                "system_measure_index": box["system_measure_index"],
                "bbox": box["bbox"],
                "geometry_source": box["geometry_source"],
                "image": str(crop_path.resolve()),
                "source_page": str(Path(source["image"]).resolve()),
                "source_pdf": str(Path(source_pdf).resolve()) if source_pdf else None,
                "pdf_page": pdf_page,
            })
            measure_number += 1
        overlay_path = output / "overlays" / f"page_{page_index:03d}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path, format="PNG", compress_level=3)
    use_info_images = bool(detected_layout) or (layout_source != "geometry" and any(not p.get("vector") for p in pages))
    regions = _document_region_crops(pages, detected_layout, records, output) if use_info_images else []
    (output / "document_regions.json").write_text(
        json.dumps(regions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "pages.json").write_text(json.dumps(pages, default=str))
    document_mode = requested_mode
    if document_mode == "auto":
        document_mode = mode_vote(records)["mode"] or mode_vote([
            {"mode": page.get("notation_mode")} for page in pages
        ])["mode"] or "auto"
    return document_mode, records
