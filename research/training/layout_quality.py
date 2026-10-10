"""Train and export the layout proposal-quality model with runtime parity checks."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from research.common.images import MEASURE_LABELS, resize_rgb_cubic
from research.models.auxiliary import make_model

FEATURE_NAMES = (
    "is_full_page_view",
    "other_coverable_view_fraction",
    "same_label_mean_iou",
    "same_label_mean_score_iou",
    "same_label_max_iou",
    "same_label_max_score_iou",
    "any_label_mean_iou",
    "any_label_max_iou",
)

FEATURE_CONTRACT = {
    "version": "view_agreement_v1",
    "features": list(FEATURE_NAMES),
    "edge_guard_fraction": 0.05,
    "minimum_reference_score": 0.25,
    "reference_eligibility": "typed_measure; raw_score_strictly_greater_than_0.25; existing_edge_guard_accepted",
    "views": "deduplicated_actual_executed_measure_views_including_full_page_and_empty_views; annotations_excluded",
    "view_geometry": "[scale,x,y,width,height]; full_page=[-1,0,0,page_width,page_height]",
    "box_geometry": "page_xyxy_half_open_clipped_to_page; IoU=intersection/union; no_plus_one",
    "visibility": "existing_internal_edge_guard",
    "query_origin": "exclude_entire_origin_view; absent_view_means_full_page",
    "per_view": "independent_max_iou_and_max_score_times_iou",
    "cross_view": "mean_and_max_over_other_coverable_actual_views_including_empty_views",
    "coverable_fraction": "number_other_coverable_views/max(1,number_all_other_actual_views)",
    "empty_evidence": 0,
    "arithmetic": "float64_then_cast_float32",
    "candidate_scope": "All raw eligible measure candidates before quality scoring or batching.",
}

def _view(box, full):
    return tuple(box.get("view") or full)

def agreement_features(queries, references, page_size, views):
    """Return eight view features per query, in FEATURE_NAMES order.

    Training may query perturbed boxes against unchanged raw model references.
    In production queries and references are the same complete candidate list.
    """
    width, height = page_size
    full = (-1, 0, 0, width, height)
    actual_views = list(dict.fromkeys(tuple(view) for view in views))
    if not actual_views:
        actual_views = [full]
    for view in actual_views:
        if (
            len(view) != 5
            or any(int(v) != v for v in view)
            or view[1] < 0
            or view[2] < 0
            or view[3] <= 0
            or view[4] <= 0
            or view[1] + view[3] > width
            or view[2] + view[4] > height
        ):
            raise ValueError("Invalid actual layout view")
    references = [box for box in references if box.get("label") in MEASURE_LABELS]
    scores = np.asarray([float(box["score"]) for box in references], dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid reference detector score")
    references = [box for box in references if float(box["score"]) > 0.25]
    result = np.zeros((len(queries), len(FEATURE_NAMES)), dtype=np.float32)
    if not queries:
        return result

    def coordinates(boxes):
        array = np.asarray(
            [box["coordinate"] for box in boxes], dtype=np.float64
        ).reshape(-1, 4)
        if not np.isfinite(array).all():
            raise ValueError("Non-finite candidate geometry")
        array[:, :2] = np.maximum(array[:, :2], 0)
        array[:, 2] = np.minimum(array[:, 2], width)
        array[:, 3] = np.minimum(array[:, 3], height)
        if np.any(array[:, 2:] <= array[:, :2]):
            raise ValueError("Empty candidate geometry")
        return array

    query_coordinates = coordinates(queries)
    reference_coordinates = coordinates(references)
    reference_scores = np.asarray(
        [float(box["score"]) for box in references], dtype=np.float64
    )
    reference_labels = np.asarray([box["label"] for box in references], dtype=object)
    reference_views = [_view(box, full) for box in references]
    if any(view not in actual_views for view in reference_views):
        raise ValueError("Reference source view was not evaluated")
    grouped = [
        np.asarray(
            [i for i, source in enumerate(reference_views) if source == view], dtype=int
        )
        for view in actual_views
    ]
    reference_areas = np.prod(
        reference_coordinates[:, 2:] - reference_coordinates[:, :2], axis=1
    )
    for start in range(0, len(queries), 64):
        chunk = query_coordinates[start : start + 64]
        qareas = np.prod(chunk[:, 2:] - chunk[:, :2], axis=1)
        intersection = np.maximum(
            0,
            np.minimum(chunk[:, None, 2:], reference_coordinates[None, :, 2:])
            - np.maximum(chunk[:, None, :2], reference_coordinates[None, :, :2]),
        )
        intersection = np.prod(intersection, axis=2)
        overlap = intersection / np.maximum(
            1e-9, qareas[:, None] + reference_areas[None, :] - intersection
        )
        for local, coordinate in enumerate(chunk):
            query_index = start + local
            box = queries[query_index]
            source = _view(box, full)
            if source not in actual_views:
                raise ValueError("Query source view was not evaluated")
            a, b, c, d = coordinate
            visible = []
            for i, view in enumerate(actual_views):
                if view == source:
                    continue
                _, x, y, w, h = view
                guard = min(w, h) * 0.05
                if (
                    (x > 0 and a - x <= guard)
                    or (y > 0 and b - y <= guard)
                    or (x + w < width and c - x >= w - guard)
                    or (y + h < height and d - y >= h - guard)
                ):
                    continue
                visible.append(i)
            values = np.zeros((len(visible), 3), dtype=np.float64)
            for destination, view_index in enumerate(visible):
                indices = grouped[view_index]
                if not len(indices):
                    continue
                agreements = overlap[local, indices]
                values[destination, 2] = agreements.max()
                same_label = reference_labels[indices] == box["label"]
                if same_label.any():
                    values[destination, 0] = agreements[same_label].max()
                    values[destination, 1] = (
                        agreements[same_label] * reference_scores[indices][same_label]
                    ).max()
            means = values.mean(axis=0) if len(values) else np.zeros(3)
            maxima = values.max(axis=0) if len(values) else np.zeros(3)
            denominator = max(1, sum(view != source for view in actual_views))
            result[query_index] = [
                float(source[0] == -1),
                len(visible) / denominator,
                means[0],
                means[1],
                maxima[0],
                maxima[1],
                means[2],
                maxima[2],
            ]
    return result

WIDTH, HEIGHT = 768, 256

LABELS = ("measure_tab", "measure_notation", "measure_both")

CONTRACT = {
    'schema': 'layout_candidate_quality',
    'preprocessing': {'context_fraction': 0.2, 'resize': 'contain_portable_cubic',
                      'rounding': 'half_up', 'rgb_padding': 255, 'mask': 'half_open_binary'},
    'selection': {'minimum_quality': 0.25, 'score': 'rank_product'},
    'image_input': {'name': 'image', 'dtype': 'float32', 'shape': ['N', 4, HEIGHT, WIDTH],
                    'channels': ['R', 'G', 'B', 'candidate_mask'], 'range': [0, 1]},
    'proposal_input': {'name': 'proposal', 'dtype': 'float32', 'shape': ['N', 12],
                       'columns': ['raw_detector_score', *LABELS, *FEATURE_NAMES]},
    'output': {'name': 'quality', 'dtype': 'float32', 'shape': ['N']},
    'maximum_batch': 32,
    'view_features': FEATURE_CONTRACT,
}

def prepare_crop(rgb, coordinate):
    """Return four uint8 channels; mask is 0/255 to simplify tensor conversion."""
    page_height, page_width = rgb.shape[:2]
    a, b, c, d = map(int, coordinate)
    a, b, c, d = max(0, a), max(0, b), min(page_width, c), min(page_height, d)
    if c <= a or d <= b:
        raise ValueError("Candidate is empty")
    padx, pady = max(1, math.ceil((c - a) * 0.2)), max(1, math.ceil((d - b) * 0.2))
    x, y, r, s = (
        max(0, a - padx),
        max(0, b - pady),
        min(page_width, c + padx),
        min(page_height, d + pady),
    )
    crop = rgb[y:s, x:r]
    scale = min(WIDTH / (r - x), HEIGHT / (s - y))
    nw, nh = (
        max(1, math.floor((r - x) * scale + 0.5)),
        max(1, math.floor((s - y) * scale + 0.5)),
    )
    resized = resize_rgb_cubic(crop, nw, nh)
    left, top = (WIDTH - nw) // 2, (HEIGHT - nh) // 2
    canvas = np.full((HEIGHT, WIDTH, 4), 255, dtype=np.uint8)
    canvas[:, :, 3] = 0
    canvas[top : top + nh, left : left + nw, :3] = resized
    mx1 = left + math.floor((a - x) * nw / (r - x) + 0.5)
    mx2 = left + math.floor((c - x) * nw / (r - x) + 0.5)
    my1 = top + math.floor((b - y) * nh / (s - y) + 0.5)
    my2 = top + math.floor((d - y) * nh / (s - y) + 0.5)
    canvas[my1:my2, mx1:mx2, 3] = 255
    return canvas

def proposal_features(box):
    return np.array(
        [float(box["score"]), *(float(box["label"] == label) for label in LABELS)],
        dtype=np.float32,
    )

def box_iou(a: tuple[int, ...] | None, b: tuple[int, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def export(checkpoint, output):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    from research.models.auxiliary import make_model

    torch.set_num_threads(2)
    model = make_model().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = (torch.rand(2, 4, HEIGHT, WIDTH), torch.rand(2, 12))
    torch.onnx.export(model, sample, str(output), input_names=['image', 'proposal'], output_names=['quality'],
                      dynamic_axes={name: {0: 'batch'} for name in ('image', 'proposal', 'quality')},
                      opset_version=17, dynamo=False)
    graph = onnx.load(output)
    onnx.helper.set_model_props(graph, {'guitarocr.layout_quality': json.dumps(CONTRACT)})
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(*sample).numpy()
    actual = session.run(['quality'], dict(zip(('image', 'proposal'), (v.numpy() for v in sample))))[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    print(json.dumps({'output': str(output), 'maximum_error': float(np.abs(actual - expected).max())}))


class CandidateDataset(Dataset):
    """JSONL pages: image, family, split, views, candidates and typed truth boxes."""
    def __init__(self, manifest, split):
        self.rows, self.families = [], set()
        for line in Path(manifest).read_text().splitlines():
            page = json.loads(line)
            if page['split'] != split or not page.get('family') or not page.get('views'):
                raise ValueError('Quality training requires explicit source families, splits and executed views')
            self.families.add(page['family'])
            with Image.open(page['image']) as image:
                size = image.size
            candidates = [box for box in page['candidates'] if box['label'] in LABELS and box['score'] > .25]
            agreements = agreement_features(candidates, candidates, size, page['views'])
            for box, agreement in zip(candidates, agreements, strict=True):
                overlap = max((box_iou(box['coordinate'], truth['coordinate'])
                               for truth in page['truth'] if truth['label'] == box['label']), default=0)
                target = np.float32(np.clip((overlap - .5) / .3, 0, 1))
                features = np.concatenate((proposal_features(box), agreement))
                self.rows.append((page['image'], box['coordinate'], features, target))
        if not self.rows:
            raise ValueError(f'No eligible candidates in {manifest}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, box, features, target = self.rows[index]
        with Image.open(path) as image:
            pixels = prepare_crop(np.asarray(image.convert('RGB')), box)
        return torch.from_numpy(pixels.transpose(2, 0, 1).astype(np.float32) / 255), features, target


def train(args):
    torch.manual_seed(args.seed)
    train_data = CandidateDataset(args.train, 'train')
    validation = CandidateDataset(args.validation, 'validation')
    if train_data.families & validation.families:
        raise ValueError('Training and validation must use different source families')
    # A fine-tuned detector proposes mostly correct boxes. Keep its genuine
    # mistakes represented so the quality head does not learn to accept all.
    negatives = [row for row in train_data.rows if row[3] == 0]
    repeats = max(0, math.ceil((len(train_data) - len(negatives)) / (9 * len(negatives))) - 1) if negatives else 0
    train_data.rows.extend(negatives * repeats)
    print({'training_candidates': len(train_data), 'negative_repetitions': repeats}, flush=True)
    model = make_model().to(args.device)
    if args.initial:
        model.load_state_dict(torch.load(args.initial, map_location='cpu', weights_only=True)['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    best, history = float('inf'), []
    for epoch in range(args.epochs):
        model.train()
        for pixels, features, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model.logits(pixels.to(args.device), features.to(args.device)), target.to(args.device))
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.
        with torch.inference_mode():
            for pixels, features, target in validation_loader:
                total += float(F.binary_cross_entropy_with_logits(model.logits(pixels.to(args.device), features.to(args.device)), target.to(args.device), reduction='sum'))
        loss = total / len(validation)
        history.append({'epoch': epoch + 1, 'validation_loss': loss})
        if loss < best:
            best = loss
            torch.save({'state_dict': model.state_dict(), 'contract': CONTRACT}, args.output / 'model.pt')
        print(history[-1], flush=True)
    (args.output / 'training.json').write_text(json.dumps({'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, 'history': history}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parser_train = commands.add_parser('train')
    parser_train.add_argument('--train', type=Path, required=True)
    parser_train.add_argument('--validation', type=Path, required=True)
    parser_train.add_argument('--initial', type=Path, default=Path('weights/checkpoints/layout_quality/model.pt'))
    parser_train.add_argument('--output', type=Path, default=Path('output/layout_quality'))
    parser_train.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser_train.add_argument('--epochs', type=int, default=5)
    parser_train.add_argument('--batch-size', type=int, default=16)
    parser_train.add_argument('--workers', type=int, default=4)
    parser_train.add_argument('--learning-rate', type=float, default=1e-5)
    parser_train.add_argument('--seed', type=int, default=42)
    parser_export = commands.add_parser('export')
    parser_export.add_argument('--checkpoint', type=Path, default=Path('weights/checkpoints/layout_quality/model.pt'))
    parser_export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'export':
        export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
