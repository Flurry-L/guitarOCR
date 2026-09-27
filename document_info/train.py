from pathlib import Path

from shared.training import llamafactory_main


def main() -> None:
    llamafactory_main(Path(__file__).with_name("configs") / "train.yaml")


if __name__ == "__main__":
    main()
