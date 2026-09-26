from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.paths import DATABASE_ROOT, PROJECT_ROOT
from datagen.select_sources import select_sources, relabel_selected_sources
from datagen.render import render_modes
from datagen.build_measure_data import MODES, crop_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-disjoint official GP8 measure-image to event-sequence data."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "music-scores-collection" / "files" / "guitar_pro",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATABASE_ROOT / "gp8_measure_sequence_v2",
    )
    parser.add_argument("--source-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--minimum-measures", type=int, default=8)
    parser.add_argument("--maximum-measures", type=int, default=128)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--runtime", type=Path, help="Guitar Pro 8 runtime directory")
    parser.add_argument("--wine-prefix-template", type=Path)
    parser.add_argument("--wine-python", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--typed-measures", action="store_true", help="Build four-class layout labels including notation type")
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "select",
            "relabel",
            "relabel-labels",
            "render",
            "crop",
            "datasets",
        ),
        default="all",
    )
    args = parser.parse_args()
    modes = args.mode or list(MODES)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"all", "select"}:
        select_sources(
            args.corpus.resolve(),
            output,
            source_count=args.source_count,
            seed=args.seed,
            minimum_measures=args.minimum_measures,
            maximum_measures=args.maximum_measures,
            modes=modes,
        )
    if args.phase == "relabel":
        relabel_selected_sources(output, modes)
    if args.phase == "relabel-labels":
        relabel_selected_sources(output, modes, prepare=False)
    if args.phase in {"all", "render"}:
        if (
            args.runtime is None
            or args.wine_prefix_template is None
            or args.wine_python is None
        ):
            parser.error(
                "rendering requires --runtime, --wine-prefix-template and --wine-python"
            )
        render_modes(
            output,
            modes,
            args.runtime,
            args.wine_prefix_template,
            args.wine_python,
            args.workers,
        )
    if args.phase in {"all", "crop"}:
        summary = crop_and_manifest(output, modes, args.dpi, args.seed, args.workers)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.phase in {"all", "datasets"}:
        from datagen.inventory import build_inventory
        from datagen.build_layout_data import build_dataset as build_layout
        from datagen.build_info_data import build_dataset as build_info

        inventory = output / "inventory"
        print(
            json.dumps(
                build_inventory(output, inventory, dpi=args.dpi, seed=args.seed, modes=modes, workers=args.workers), ensure_ascii=False
            )
        )
        print(
            json.dumps(
                {
                    "layout": build_layout(inventory, output / "datasets" / "layout", typed_measures=args.typed_measures),
                    "information": build_info(
                        inventory, output / "datasets" / "document_info"
                    ),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
