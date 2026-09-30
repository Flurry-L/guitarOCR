"""Shared OCR model loading and generation for all recognition tasks."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any


def create_backend(model_path, adapter_path, device):
    """Use a published accelerated model when its runtime is available."""
    import json
    import os

    config = Path(adapter_path) / 'inference.json' if adapter_path else Path(model_path) / 'inference.json'
    settings = json.loads(config.read_text()) if config.exists() else {}
    merged = config.parent / settings['model'] if settings.get('model') else None
    if os.environ.get('GUITAROCR_BACKEND') == 'llamacpp':
        from shared.llamacpp_backend import LlamaCppBackend

        resolved = merged if merged and merged.is_dir() else Path(model_path)
        task = 'MEASURE' if (resolved / 'music_vocabulary.json').is_file() else 'INFO'
        variable = f'GUITAROCR_LLAMA_{task}_URL'
        endpoint = os.environ.get('GUITAROCR_LLAMA_SCORE_URL') or os.environ.get(variable)
        return LlamaCppBackend(resolved, endpoint, device=device, options=settings.get('options'))
    if device.startswith('cuda') and merged and merged.is_dir() and os.environ.get('GUITAROCR_BACKEND', 'auto') != 'transformers':
        from shared.vllm_backend import VllmBackend, engine_python

        if engine_python().is_file():
            return VllmBackend(merged, device, options=settings.get('options'))
    if os.environ.get('GUITAROCR_BACKEND') == 'vllm':
        raise ValueError('所选 vLLM 环境不可用，请重新运行安装脚本。')
    if merged and merged.is_dir():
        return GlmBackend(merged, None, device)
    return GlmBackend(model_path, adapter_path, device)


class GlmBackend:
    def __init__(
        self, model_path: Path, adapter_path: Path | None, device: str,
        *, merge_adapter: bool = True,
    ) -> None:
        import json
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Task adapters may have been trained on an already fine-tuned base.
        # Direct Transformers callers must use the same published model as vLLM.
        config = Path(adapter_path or model_path) / 'inference.json'
        settings = json.loads(config.read_text()) if config.is_file() else {}
        if settings.get('model'):
            merged = config.parent / settings['model']
            if merged.is_dir():
                model_path, adapter_path = merged, None
        dtype = torch.float32
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        from shared.score_image import load_policy
        self.image_policy = load_policy(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path)
        model = model.to(device=device, dtype=dtype).eval()
        self.model = model.merge_and_unload().eval() if adapter_path is not None and merge_adapter else model
        self.batch_limit = None

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        *,
        skip_special_tokens: bool = True,
    ) -> tuple[str, int]:
        import torch

        from shared.score_image import normalize_messages
        messages = normalize_messages(messages, self.image_policy)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        length = inputs["input_ids"].shape[1]
        text = self.processor.decode(
            generated[0][length:], skip_special_tokens=skip_special_tokens
        )
        return text, int(generated.shape[1] - length)

    def generate_batch(
        self,
        messages: list[list[dict[str, Any]]],
        max_new_tokens: int,
        *,
        skip_special_tokens: bool = True,
    ) -> list[tuple[str, int]]:
        """Decode independent crops together, with left padding for generation."""
        import torch

        if not messages:
            return []
        if self.batch_limit is not None and len(messages) > self.batch_limit:
            limit = self.batch_limit
            return [result for start in range(0, len(messages), limit)
                    for result in self.generate_batch(messages[start:start + limit], max_new_tokens,
                                                      skip_special_tokens=skip_special_tokens)]
        try:
            return self._generate_batch(messages, max_new_tokens, skip_special_tokens)
        except torch.cuda.OutOfMemoryError:
            if len(messages) == 1:
                raise
            self.batch_limit = max(1, len(messages) // 2)
        # Leave the exception frame before freeing the failed generation's KV cache.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        return self.generate_batch(messages, max_new_tokens, skip_special_tokens=skip_special_tokens)

    def _generate_batch(self, messages, max_new_tokens, skip_special_tokens):
        import torch
        from shared.score_image import normalize_messages
        messages = [normalize_messages(m, self.image_policy) for m in messages]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "padding_side": "left"},
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        result = []
        for tokens in generated:
            values = tokens.tolist()
            end = next((i + 1 for i, token in enumerate(values) if token in eos_ids), len(values))
            result.append((self.processor.decode(tokens[:end], skip_special_tokens=skip_special_tokens), end))
        return result


class BackendPool:
    """Share one engine across task prompts and serialize GPU generation."""

    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path.resolve()
        self.device = device
        self.lock = RLock()
        self.backend = None
        self.backends = OrderedDict()

    def adapter(self, path: Path | None):
        path = path.resolve() if path is not None else None
        return _AdapterBackend(self, path)

    def _key(self, path):
        import json

        config = (path or self.model_path) / 'inference.json'
        settings = json.loads(config.read_text()) if config.is_file() else {}
        if settings.get('model'):
            merged = (config.parent / settings['model']).resolve()
            if merged.is_dir():
                return ('merged', merged, json.dumps(settings.get('options', {}), sort_keys=True))
        return ('adapter', path)

    def prepare(self, paths):
        """Resolve task aliases before loading resident engines."""
        from concurrent.futures import ThreadPoolExecutor

        paths = {self._key(path): path for p in paths
                 for path in [Path(p).resolve() if p is not None else None]}
        if len(paths) > 2:
            raise ValueError('A recognition workspace supports at most two distinct OCR models')
        with self.lock:
            missing = [(key, path) for key, path in paths.items() if key not in self.backends]
            for path in list(self.backends):
                if path not in paths:
                    previous = self.backends.pop(path)
                    if self.backend is previous:
                        self.backend = None
                    if hasattr(previous, 'close'):
                        previous.close()
            failures = []
            with ThreadPoolExecutor(max_workers=2) as workers:
                futures = [(key, workers.submit(create_backend, self.model_path, path, self.device)) for key, path in missing]
                for key, future in futures:
                    try:
                        self.backends[key] = future.result()
                    except Exception as error:
                        failures.append(error)
            if failures:
                raise failures[0]

    def _get_backend(self, path: Path | None):
        with self.lock:
            key = self._key(path)
            if key not in self.backends:
                while len(self.backends) >= 2:
                    _, previous = self.backends.popitem(last=False)
                    if self.backend is previous:
                        self.backend = None
                    if hasattr(previous, 'close'):
                        previous.close()
                    del previous
                self.backends[key] = create_backend(self.model_path, path, self.device)
            self.backends.move_to_end(key)
            self.backend = self.backends[key]
            return self.backend

    def _generate(self, path: Path | None, method: str, *args, **kwargs):
        with self.lock:
            return getattr(self._get_backend(path), method)(*args, **kwargs)

    def close(self):
        with self.lock:
            self.backend = None
            for backend in self.backends.values():
                if hasattr(backend, 'close'):
                    backend.close()
            self.backends.clear()


@dataclass(frozen=True)
class _AdapterBackend:
    pool: BackendPool
    path: Path | None

    @property
    def supports_ragged_batch(self):
        with self.pool.lock:
            return getattr(self.pool._get_backend(self.path), 'supports_ragged_batch', False)

    @property
    def supports_json_schema(self):
        with self.pool.lock:
            return getattr(self.pool._get_backend(self.path), 'supports_json_schema', False)

    def generate(self, *args, **kwargs):
        return self.pool._generate(self.path, "generate", *args, **kwargs)

    def generate_batch(self, *args, **kwargs):
        return self.pool._generate(self.path, "generate_batch", *args, **kwargs)
