"""Notation labels shared by layout data, detection and evaluation."""

MODES = ("tab", "notation", "both")
MEASURE_LABELS = {f"measure_{mode}": mode for mode in MODES}
TYPED_CATEGORIES = (*MEASURE_LABELS, "tempo_region")


def measure_mode(label: str) -> str | None:
    return MEASURE_LABELS.get(label)


def is_measure(label: str) -> bool:
    return label == "measure" or label in MEASURE_LABELS


def mode_vote(boxes: list[dict]) -> dict:
    """Aggregate model labels; the vote fraction is not calibrated confidence."""
    votes = dict.fromkeys(MODES, 0.0)
    for box in boxes:
        mode = measure_mode(box.get("label", "")) or box.get("mode")
        if mode in votes:
            votes[mode] += max(0.0, float(box.get("score", 1.0)))
    total = sum(votes.values())
    mode = max(votes, key=votes.get) if total else None
    return {
        "mode": mode,
        "mode_vote_fraction": votes[mode] / total if mode else 0.0,
        "mode_votes": votes,
    }


def typed_annotations(payload: dict) -> dict:
    """Replace measure labels using explicit per-page native display modes."""
    modes = {row["id"]: row.get("mode") for row in payload["images"]}
    if set(modes.values()) - set(MODES):
        raise ValueError("Typed layout annotations require a valid mode on every page")
    original = {row["id"]: row["name"] for row in payload["categories"]}
    categories = [
        {"id": index, "name": label, "supercategory": "score"}
        for index, label in enumerate(TYPED_CATEGORIES, 1)
    ]
    ids = {row["name"]: row["id"] for row in categories}
    annotations = []
    for row in payload["annotations"]:
        label = original[row["category_id"]]
        if is_measure(label):
            expected = f"measure_{modes[row['image_id']]}"
            if label != "measure" and label != expected:
                raise ValueError("Measure category disagrees with page display mode")
            label = expected
        annotations.append({**row, "category_id": ids[label]})
    return {**payload, "annotations": annotations, "categories": categories}
