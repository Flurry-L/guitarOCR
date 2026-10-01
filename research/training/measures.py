from pathlib import Path

from research.training.training import llamafactory_main


def main() -> None:
    llamafactory_main(Path(__file__).parents[1] / "configs/measures/train.yaml", worker_module='research.training.worker')


if __name__ == "__main__":
    main()
