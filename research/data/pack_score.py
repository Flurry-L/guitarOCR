"""Pack visible-supervision samples into one portable score training dataset."""

import argparse
import json
import os
import re
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

from scorelib.techniques import visible_event_effects

from research.data.technique_visibility import (
    UnverifiedTechniqueVisibility,
    has_technique,
    verified_visual_target,
)
from research.data.training_samples import dataset_entry


class MeasureVisibilityIndex:
    """Keep only technique-bearing source records; prompts are never evidence."""

    def __init__(self, manifests=()):
        self.records = {}
        self.conflicts = set()
        for manifest in manifests:
            with Path(manifest).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if (row.get('image_supervision')
                            or has_technique(row.get('target', ''), 'pm')
                            or has_technique(row.get('source_target', ''), 'pm')):
                        image = str(Path(row['image']).resolve())
                        previous = self.records.get(image)
                        if previous is not None and not self.compatible(previous, row):
                            self.conflicts.add(image)
                        if previous and previous.get('technique_evidence') and not row.get('technique_evidence'):
                            row = {**row, 'technique_evidence': previous['technique_evidence']}
                        self.records[image] = row

    @staticmethod
    def identity(row):
        return (row['mode'], str(Path(row.get('label_json') or '').resolve()),
                visible_event_effects(row.get('source_target', row['target'])))

    @classmethod
    def compatible(cls, previous, row):
        if cls.identity(previous) != cls.identity(row):
            return False
        if previous.get('source_target') and row.get('source_target') and previous['source_target'] != row['source_target']:
            return False
        left, right = previous.get('technique_evidence'), row.get('technique_evidence')
        return left is None or right is None or left == right

    def verify(self, sample):
        answer = sample['messages'][-1]['content']
        if answer.split(maxsplit=1)[0] in {'M2', 'MEASURE', 'C2', 'CONTEXT'}:
            raise ValueError('Training targets must use the current M text protocol')
        if not answer.startswith('M '):
            return sample
        image = sample['images'][0]
        for path in sample['images']:
            context = self.records.get(str(Path(path).resolve()))
            if context and context.get('image_supervision', {}).get('status') == 'review':
                raise UnverifiedTechniqueVisibility({
                    **context, 'image': image, 'target': answer, 'serialized_sample': sample,
                    'visual_supervision_status': 'review', 'rejected_input_image': path,
                    'review_reason': context['image_supervision']['reason'],
                })
        resolved = str(Path(image).resolve())
        source = self.records.get(resolved)
        if source is None and not has_technique(answer, 'pm'):
            return sample
        if resolved in self.conflicts:
            raise UnverifiedTechniqueVisibility({
                'image': image, 'target': answer, 'serialized_sample': sample,
                'visual_supervision_status': 'review',
                'review_reason': 'conflicting_measure_provenance',
            })
        if source is None:
            raise UnverifiedTechniqueVisibility({
                'image': image, 'target': answer, 'serialized_sample': sample,
                'visual_supervision_status': 'review',
                'review_reason': 'missing_measure_provenance',
            })
        try:
            # Verify the source even when its serialized answer lacks PM.
            if 'target' in source:
                if (has_technique(source.get('source_target', ''), 'pm')
                        and source.get('technique_evidence', {}).get('pm', {}).get('state') == 'unknown'
                        and 'pm' not in source.get('printed_techniques', ())):
                    raise UnverifiedTechniqueVisibility({**source, 'visual_supervision_status': 'review'})
                source = verified_visual_target(source)
            verified = verified_visual_target({**source, 'target': answer})
            if 'target' in source:
                expected = {key for key, effects in visible_event_effects(source['target']).items() if 'pm' in effects}
                actual = {key for key, effects in visible_event_effects(verified['target']).items() if 'pm' in effects}
                if expected != actual:
                    raise UnverifiedTechniqueVisibility({
                        **verified, 'visual_supervision_status': 'review',
                        'review_reason': 'serialized_answer_disagrees_with_verified_pm_scope',
                    })
        except UnverifiedTechniqueVisibility as error:
            error.row = {**error.row, 'serialized_sample': sample}
            raise
        result = deepcopy(sample)
        result['messages'][-1]['content'] = verified['target']
        return result


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


def pack(inputs, output, manifests):
    """Keep accepted answers, copy their images, and discard rejected samples.

    Inputs are the source-disjoint task mixtures produced by the generators.
    Measure manifests supply visible-technique and image-geometry evidence.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    visibility = MeasureVisibilityIndex(manifests)
    images, counts, rejected = {}, {}, Counter()
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
                    try:
                        row = visibility.verify(row)
                    except UnverifiedTechniqueVisibility as error:
                        rejected[error.row.get("review_reason", "unverified_visibility")] += 1
                        continue
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
    summary = dict(complete=False, supervision="visible_score", pitch="written_before_transposition",
                   image_geometry="preserve_aspect_ratio", splits=counts, images=len(images),
                   rejected=dict(rejected), token_cache="rebuild_before_training")
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
            if not row.get("images") or not row.get("messages"):
                raise ValueError(f"Empty sample in {split} at row {count + 1}")
            answer = row["messages"][-1]
            if answer.get("role") != "assistant" or not answer.get("content", "").strip():
                raise ValueError(f"Missing target in {split} at row {count + 1}")
            if re.match(r"^(M2|MEASURE)(?:\s|$)", answer["content"]):
                raise ValueError("Dataset targets must use M text")
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
    parser.add_argument("--output", type=Path, default=Path("database/score"))
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=Path, nargs="+", help="Task samples in JSON or JSONL")
    parser.add_argument("--measure-manifests", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    if args.validate:
        result = validate(args.validate)
    else:
        inputs = {split: getattr(args, split) for split in SPLITS}
        if not all(inputs.values()) or not args.measure_manifests:
            parser.error("Packing requires all three splits and source measure manifests")
        result = pack(inputs, args.output, args.measure_manifests)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
