"""Encode repeated context images once per training batch, retaining gradients."""

from contextvars import ContextVar
from functools import wraps


def install_shared_vision():
    import torch
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrModel

    original_collate = MultiModalDataCollatorForSeq2Seq.__call__
    original_forward = GlmOcrModel.forward
    original_features = GlmOcrModel.get_image_features
    current_images = ContextVar('shared_training_images', default=None)

    @wraps(original_collate)
    def collate(self, features):
        paths = [path for row in features for path in (row.get('images') or [])]
        first, inverse, lookup = [], [], {}
        if paths and all(isinstance(path, str) for path in paths):
            for index, path in enumerate(paths):
                if path not in lookup:
                    lookup[path] = len(first)
                    first.append(index)
                inverse.append(lookup[path])
        batch = original_collate(self, features)
        if inverse and len(first) < len(paths):
            batch['image_unique_indices'] = torch.tensor(first, dtype=torch.long)
            batch['image_inverse_indices'] = torch.tensor(inverse, dtype=torch.long)
        return batch

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        first = kwargs.pop('image_unique_indices', None)
        inverse = kwargs.pop('image_inverse_indices', None)
        context = current_images.set((first, inverse) if first is not None else None)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            current_images.reset(context)

    @wraps(original_features)
    def features(self, pixel_values, image_grid_thw=None, **kwargs):
        shared = current_images.get()
        if shared is None:
            return original_features(self, pixel_values, image_grid_thw, **kwargs)
        first, inverse = shared
        first, inverse = first.tolist(), inverse.tolist()
        if len(inverse) != len(image_grid_thw):
            raise ValueError('Shared image map does not match the image grid')
        chunks = torch.split(pixel_values, image_grid_thw.prod(-1).tolist())
        unique_pixels = torch.cat([chunks[i] for i in first])
        unique_grid = image_grid_thw[first]
        result = original_features(self, unique_pixels, unique_grid, **kwargs)
        # Repeated tensors stay attached to the same graph: all measure losses
        # contribute to the gradient of their shared image representation.
        unique_features = tuple(value.float() for value in result.pooler_output)
        # Accumulate contributions from repeated uses in FP32 before casting
        # their combined gradient back to the vision tower's compute dtype.
        result.pooler_output = tuple(unique_features[i] for i in inverse)
        return result

    MultiModalDataCollatorForSeq2Seq.__call__ = collate
    GlmOcrModel.forward = forward
    GlmOcrModel.get_image_features = features
