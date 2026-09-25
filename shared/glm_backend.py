"""GLM loading and generation shared by the two OCR stages and their evaluations."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any


class GlmBackend:
    def __init__(
        self, model_path: Path, adapter_path: Path | None, device: str
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.float32
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(device=device, dtype=dtype).eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        *,
        skip_special_tokens: bool = True,
    ) -> tuple[str, int]:
        import torch

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


class BackendPool:
    """One lazy base model; lock adapter selection and generation together."""

    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path.resolve()
        self.device = device
        self.lock = RLock()
        self.backend = None
        self.adapters = {}

    def adapter(self, path: Path | None):
        path = path.resolve() if path is not None else None
        return _AdapterBackend(self, path)

    def _load_adapter(self, path: Path) -> None:
        if path in self.adapters:
            return
        name = f"adapter_{len(self.adapters)}"
        if self.adapters:
            self.backend.model.load_adapter(path, adapter_name=name)
        else:
            from peft import PeftModel

            self.backend.model = PeftModel.from_pretrained(
                self.backend.model, path, adapter_name=name
            ).eval()
        self.adapters[path] = name

    def _generate(self, path: Path | None, *args, **kwargs):
        with self.lock:
            if self.backend is None:
                self.backend = GlmBackend(self.model_path, path, self.device)
                if path is not None:
                    self.adapters[path] = "default"
            if path is not None:
                self._load_adapter(path)
                self.backend.model.set_adapter(self.adapters[path])
            context = (
                self.backend.model.disable_adapter()
                if path is None and self.adapters
                else nullcontext()
            )
            with context:
                return self.backend.generate(*args, **kwargs)


@dataclass(frozen=True)
class _AdapterBackend:
    pool: BackendPool
    path: Path | None

    def generate(self, *args, **kwargs):
        return self.pool._generate(self.path, *args, **kwargs)
