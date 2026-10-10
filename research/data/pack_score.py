"""Pack visible-supervision samples into one portable score training dataset."""

import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

from research.data.training_samples import dataset_entry


def verify_sample(sample):
    """Do not mix complete-M targets back into single-image observation training."""
    if len(sample.get("images", [])) != 1:
        raise ValueError("Observation training requires exactly one image per sample")
    messages = sample.get("messages", [])
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("Expected one image question and one observation answer")
    if messages[0]["content"].count("<image>") != 1:
        raise ValueError("The prompt must contain exactly one image marker")
    answer = messages[-1]["content"]
    if answer.startswith("S2 "):
        from research.data.prompts import parse_signature
        parse_signature(answer)
        return
    value = json.loads(answer)
    if not isinstance(value, dict):
        raise ValueError("An observation or auxiliary answer must be a JSON object")
    if "voices" in value and not {"state", "voices", "relations"} <= value.keys():
        raise ValueError("Measure observations need state, voices and relations")


SPLITS = ("train", "validation", "test")


def samples(path):
    path = Path(path)
    with path.open() as stream:
        if path.suffix == ".json":
            yield from json.load(stream)
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def pack(inputs, output):
    """Pack independently assigned source families and their single-image tasks."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    images, counts = {}, {}
    registry, families, image_splits = {}, {}, {}
    for split in SPLITS:
        counts[split] = 0
        with (output / f"{split}.jsonl").open("w") as target:
            for source in inputs[split]:
                for row in samples(source):
                    row = deepcopy(row)
                    provenance = row.get("provenance")
                    if provenance:
                        family = provenance.get("family")
                        if not family or provenance.get("split") != split:
                            raise ValueError(f"Missing or inconsistent source assignment: {source}")
                        if families.setdefault(family, split) != split:
                            raise ValueError(f"Source family crosses dataset splits: {family}")
                    row["images"] = [str((Path(source).resolve().parent / name).resolve())
                                     if not Path(name).is_absolute() else name for name in row["images"]]
                    for name in row["images"]:
                        original = Path(name).resolve()
                        if image_splits.setdefault(original, split) != split:
                            raise ValueError(f"Image crosses dataset splits: {original}")
                    verify_sample(row)
                    packed = []
                    for name in row["images"]:
                        original = Path(name).resolve(strict=True)
                        if original not in images:
                            index = len(images)
                            relative = Path("images") / f"{index // 10000:04d}" / f"{index:08d}{original.suffix.lower()}"
                            destination = output / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            # The dataset is independent of generated work directories.
                            shutil.copyfile(original, destination)
                            images[original] = relative.as_posix()
                        packed.append(images[original])
                    messages = [dict(message) for message in row["messages"]]
                    target.write(json.dumps({"messages": messages, "images": packed},
                                            ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[split] += 1
        registry[split] = dataset_entry(f"{split}.jsonl")
    summary = dict(complete=False, supervision="visible_observation", measure_images=1,
                   image_geometry="preserve_aspect_ratio", splits=counts, images=len(images),
                   token_cache="rebuild_before_training")
    (output / "dataset_info.json").write_text(json.dumps(registry, indent=2) + "\n")
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    validate(output)
    summary["complete"] = True
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def validate(directory):
    """Check the whole bundle, including every referenced image and split size."""
    directory = Path(directory).resolve()
    summary = json.loads((directory / "dataset.json").read_text())
    seen, counts = set(), {}
    for split in SPLITS:
        count = 0
        for row in samples(directory / f"{split}.jsonl"):
            verify_sample(row)
            for name in row["images"]:
                if name in seen:
                    continue
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "images":
                    raise ValueError(f"Image is outside the dataset: {name}")
                image = directory / path
                if image.is_symlink() or not image.is_file() or image.stat().st_size == 0:
                    raise ValueError(f"Missing or empty image: {name}")
                seen.add(name)
            count += 1
        if count != summary["splits"][split] or count == 0:
            raise ValueError(f"Dataset row count differs: {split}")
        counts[split] = count
    actual = sum(len(names) for _, _, names in os.walk(directory / "images"))
    if len(seen) != summary["images"] or actual != len(seen):
        raise ValueError("Dataset has missing or unreferenced images")
    return {"splits": counts, "images": len(seen), "self_contained": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path, help="Validate an existing dataset without changing it")
    parser.add_argument("--output", type=Path, default=Path("database/observations_packed"))
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=Path, nargs="+", help="Task samples in JSON or JSONL")
    args = parser.parse_args()
    if args.validate:
        result = validate(args.validate)
    else:
        inputs = {split: getattr(args, split) for split in SPLITS}
        if not all(inputs.values()):
            parser.error("Packing requires all three source-disjoint splits")
        result = pack(inputs, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
