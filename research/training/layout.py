"""Aspect-preserving readers for the fixed-size DocLayoutV3 graph."""

import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np

from research.common.images import resize_rgb_cubic


def install_cubic_resize() -> None:
    """Install the RGB8 contract in the actual Paddle train/eval worker.

    Keep Paddle's annotation, mask and export geometry. Its augmentation and
    Pad operators store RGB8-valued pixels as float32; validate that boundary
    before converting. Other interpolation modes remain controlled by the
    upstream augmentation configuration.
    """
    from ppdet.data.transform.operators import Resize

    if getattr(Resize.apply_image, "_guitarocr_cubic", False):
        return
    original = Resize.apply_image

    def apply_image(self, image, scale):
        if self.interp != 2:  # OpenCV INTER_CUBIC
            return original(self, image, scale)
        if image.dtype != np.uint8:
            if (not np.issubdtype(image.dtype, np.floating)
                    or not np.isfinite(image).all()
                    or np.any((image < 0) | (image > 255))
                    or np.any(image != np.floor(image))):
                raise ValueError("Layout cubic resize requires RGB8-valued pixels before normalization")
            image = image.astype(np.uint8)
        width = max(1, int(image.shape[1] * scale[0] + 0.5))
        height = max(1, int(image.shape[0] * scale[1] + 0.5))
        return resize_rgb_cubic(image, width, height)

    apply_image._guitarocr_cubic = True
    Resize.apply_image = apply_image


@contextmanager
def cubic_training_worker(runner):
    """Route PaddleX's single/distributed launch through our explicit worker."""
    original = runner.run_cmd
    worker = Path(__file__).with_name("layout_worker.py").resolve()

    def run_cmd(cmd, **kwargs):
        cmd = list(cmd)
        indices = [i for i, value in enumerate(cmd) if value == "tools/train.py"]
        if len(indices) != 1:
            raise ValueError("Unrecognized Paddle layout training command")
        upstream = Path(runner.runner_root_path).resolve() / "tools/train.py"
        if not upstream.is_file():
            raise ValueError(f"Missing Paddle training entry point: {upstream}")
        cmd[indices[0]] = str(worker)
        env = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        env["GUITAROCR_LAYOUT_UPSTREAM_TRAIN"] = str(upstream)
        kwargs["env"] = env
        return original(cmd, **kwargs)

    runner.run_cmd = run_cmd
    try:
        yield
    finally:
        runner.run_cmd = original


def letterbox_transforms(size: int = 800) -> list[dict]:
    """Resize, then pad white at the right/bottom without moving annotations.

    DocLayoutV3 scales normalized boxes using im_shape, ignoring pad_shape.
    The final identity Resize records the full canvas while retaining the
    first Resize's scale factors. Pad mode -1 also pads segmentation masks;
    Paddle's mode 0 returns early before handling masks.
    """
    return [
        {"Resize": {"target_size": [size, size], "keep_ratio": True, "interp": 2}},
        {"Pad": {"size": [size, size], "pad_mode": -1, "offsets": [0, 0],
                 "fill_value": [255.0, 255.0, 255.0]}},
        {"Resize": {"target_size": [size, size], "keep_ratio": False, "interp": 2}},
    ]


def letterbox_reader_updates(config, size: int = 800) -> dict:
    """Reuse the existing model/readers; replace only their image geometry."""
    result = {}
    for name in ("TrainReader", "EvalReader", "TestReader"):
        reader = deepcopy(config[name])
        if name == 'TrainReader':
            # Partial notation/TAB crops would retain an incorrect full-staff label.
            reader['sample_transforms'] = [item for item in reader['sample_transforms'] if 'RandomCrop' not in item]
        field = "batch_transforms" if name == "TrainReader" else "sample_transforms"
        transforms = reader[field]
        indices = [i for i, transform in enumerate(transforms)
                   if next(iter(transform)) in {"Resize", "BatchRandomResize"}]
        if len(indices) != 1:
            raise ValueError(f"Expected one resize in {name}, got {len(indices)}")
        i = indices[0]
        reader[field] = transforms[:i] + letterbox_transforms(size) + transforms[i + 1:]
        if name == "TestReader":
            reader.setdefault("inputs_def", {})["image_shape"] = [3, size, size]
        if name == "TrainReader":
            for transform in reader["sample_transforms"]:
                if "RandomExpand" in transform:
                    transform["RandomExpand"]["fill_value"] = [255.0] * 3
        result[name] = reader
    result["eval_size"] = [size, size]
    return result


def main() -> None:
    from paddlex.engine import Engine
    from paddlex.modules.base.trainer import build_trainer
    from paddlex.utils.config import get_config, parse_args
    from paddlex.utils.lazy_loader import disable_pir_bydefault

    args = parse_args()
    config = get_config(args.config, overrides=args.override, show=False)
    if config.Global.mode != "train" or config.Global.model != "PP-DocLayoutV3":
        Engine().run()
        return
    disable_pir_bydefault()
    trainer = build_trainer(config)
    trainer.update_config()
    # PaddleX's instance segmentation defaults select mask AP even though
    # DocLayoutV3 disables mask evaluation, and omit reading-order labels.
    # Apply these settings after its dataset update, before launching workers.
    trainer.pdx_config.update(
        {
            "target_metrics": "bbox",
            "TrainDataset": {
                "data_fields": [
                    "image",
                    "gt_bbox",
                    "gt_class",
                    "gt_poly",
                    "is_crowd",
                    "gt_read_order",
                ]
            },
        }
    )
    trainer.pdx_config.update_num_workers(config.Train.get("num_workers", 4))
    trainer.pdx_config.update(letterbox_reader_updates(
        trainer.pdx_config, int(config.Train.get("input_size", 800))))
    Path(config.Global.output).mkdir(parents=True, exist_ok=True)
    trainer.dump_config()
    kwargs = trainer.get_train_kwargs()
    kwargs.update(
        {
            "uniform_output_enabled": True,
            "export_with_pir": True,
            "num_workers": config.Train.get("num_workers", 4),
        }
    )
    with cubic_training_worker(trainer.pdx_model.runner):
        result = trainer.pdx_model.train(**kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Layout training failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
