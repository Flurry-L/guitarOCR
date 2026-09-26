"""Deterministic scan-like degradation that preserves score coordinates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def degrade(image: Image.Image, key: str, *, crop: bool = False) -> Image.Image:
    rng = np.random.default_rng(int.from_bytes(sha256(key.encode()).digest()[:8], "big"))
    image = image.convert("L")
    size = image.size
    if crop:
        # Vary context without deleting any original symbols.
        x, y = max(2, round(size[0] * 0.018)), max(2, round(size[1] * 0.025))
        border = tuple(int(rng.integers(0, v + 1)) for v in (x, y, x, y))
        image = ImageOps.expand(image, border, fill=255).resize(size, Image.Resampling.LANCZOS)
    scale = float(rng.uniform(0.65, 0.95))
    image = image.resize((max(1, round(size[0] * scale)), max(1, round(size[1] * scale))), Image.Resampling.BILINEAR)
    image = image.resize(size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(float(rng.uniform(0.15, 0.45))))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.7, 1.05)))
    pixels = np.asarray(image, dtype=np.float32)
    shade = np.linspace(float(rng.uniform(0, 18)), float(rng.uniform(0, 18)), size[0], dtype=np.float32)
    pixels -= shade[None, :]
    pixels += rng.normal(0, rng.uniform(0.4, 1.8), pixels.shape).astype(np.float32)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(rng.integers(55, 91)))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.copy()


def _save(job):
    source, destination, key, crop = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        degrade(image, key, crop=crop).save(destination, compress_level=3)


def augment_measure_manifest(source: Path, output: Path, max_samples: int, seed: int, workers: int):
    from measure_ocr.evaluate import _load_rows

    if output.exists():
        raise FileExistsError(output)
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative; use 0 for complete sequences")
    rows = _load_rows(source, max_samples, seed)
    if not rows:
        raise ValueError("No measure samples found")
    output.mkdir(parents=True)
    augmented = [{**row, "image": str((output / "images" / f"{row['id']}.png").resolve()),
                  "original_image": row["image"], "augmentation": "scan"} for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_save, ((Path(r["original_image"]), Path(r["image"]),
                              f"{seed}:{source.stem}:{r['id']}", True) for r in augmented), chunksize=16))
    destination = output / source.name
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in augmented))
    report = {"samples": len(rows), "seed": seed, "max_samples": max_samples,
              "source_manifest_sha256": sha256(source.read_bytes()).hexdigest(),
              "manifest_sha256": sha256(destination.read_bytes()).hexdigest(),
              "augmentation": "scan crop=True; separate stress evaluation, not real scans"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def augment_layout(source: Path, output: Path, splits: list[str], fraction: float, seed: int, replace: bool, workers: int):
    if not 0 < fraction <= 1 or not splits or set(splits) - {"train", "val", "test"}:
        raise ValueError("Use train/val/test and fraction in (0, 1]")
    if output.exists():
        raise FileExistsError(output)
    (output / "images").mkdir(parents=True)
    (output / "images/original").symlink_to((source / "images").resolve(), target_is_directory=True)
    (output / "annotations").mkdir()
    summary = {"source": str(source.resolve()), "seed": seed, "fraction": fraction, "replace": replace, "splits": {}}
    for path in sorted((source / "annotations").glob("instance_*.json")):
        split = path.stem.removeprefix("instance_")
        payload = json.loads(path.read_text())
        images = [{**row, "file_name": "original/" + row["file_name"]} for row in payload["images"]]
        annotations = list(payload["annotations"])
        generated = []
        if split in splits:
            chosen = sorted(payload["images"], key=lambda r: sha256(f"{seed}:{split}:{r['id']}".encode()).digest())[:round(len(images) * fraction)]
            max_id = max(r["id"] for r in images)
            remapping = {}
            for index, row in enumerate(chosen, 1):
                identifier = row["id"] if replace else max_id + index
                name = f"scan/{split}_{row['id']}.png"
                remapping[row["id"]] = identifier
                generated.append({**row, "id": identifier, "file_name": name, "augmentation_parent": row["id"], "augmentation": "scan"})
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_save, ((source / "images" / original["file_name"], output / "images" / aug["file_name"],
                                      f"{seed}:{split}:{original['id']}", False) for original, aug in zip(chosen, generated)), chunksize=8))
            max_ann = max((a["id"] for a in annotations), default=0)
            augmented_annotations = [{**row, "image_id": remapping[row["image_id"]]} for row in annotations if row["image_id"] in remapping]
            if replace:
                images, annotations = generated, augmented_annotations
            else:
                images.extend(generated)
                annotations.extend({**row, "id": max_ann + index} for index, row in enumerate(augmented_annotations, 1))
        (output / "annotations" / path.name).write_text(json.dumps({**payload, "images": images, "annotations": annotations}) + "\n")
        summary["splits"][split] = {"pages": len(images), "augmented_pages": len(generated), "source_annotations_sha256": sha256(path.read_bytes()).hexdigest()}
        print(split, summary["splits"][split], flush=True)
    (output / "augmentation.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="Layout COCO dataset directory")
    inputs.add_argument("--measure-manifest", type=Path, help="OCR JSONL manifest for a separate degraded evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--replace", action="store_true", help="Evaluate degraded pages separately from clean originals")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=900, help="OCR source/mode stratified limit; 0 retains complete sequences")
    args = parser.parse_args()
    if args.measure_manifest:
        if args.split or args.replace:
            parser.error("--split and --replace apply only to layout datasets")
        print(augment_measure_manifest(args.measure_manifest, args.output, args.max_samples, args.seed, args.workers))
    else:
        augment_layout(args.source, args.output, args.split or ["train"], args.fraction, args.seed, args.replace, args.workers)


if __name__ == "__main__":
    main()
