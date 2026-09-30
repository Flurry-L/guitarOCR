from pathlib import Path

from shared.training import llamafactory_main


def main() -> None:
    config = Path(__file__).resolve().parents[1] / 'measure_ocr/configs/train.yaml'
    llamafactory_main(config, worker_module='measure_ocr.train_worker')


if __name__ == "__main__":
    main()
