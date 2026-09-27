from pathlib import Path


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
    result = trainer.pdx_model.train(**kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Layout training failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
