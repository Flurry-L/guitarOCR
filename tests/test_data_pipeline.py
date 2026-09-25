import json
from pathlib import Path
import shutil
import tempfile
import unittest

from datagen.build_info_data import build_dataset as build_info
from datagen.build_layout_data import build_dataset as build_layout
from datagen.build_measure_data import crop_and_manifest
from datagen.gp_sources import analyze_source
from datagen.inventory import build_inventory, source_catalog
from examples.generate import generate, TARGETS
from gp5_export.writer import write_targets_gp5
from shared.artifacts import write_json


class DataPipelineTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.export = self.root / "exports"
        drawing = self.root / "drawing"
        generate(drawing)
        mm = 25.4 / 72
        layout = {
            "schema": "gpomr.render-layout",
            "tab_only": True,
            "pages": [{"index": 1, "bbox_mm": [0, 0, 595 * mm, 420 * mm]}],
            "systems": [
                {
                    "page": 1,
                    "measure_boxes": [
                        {
                            "measure_index": i,
                            "bbox_mm": [
                                (55 + i % 2 * 245) * mm,
                                (130 + i // 2 * 130) * mm,
                                245 * mm,
                                80 * mm,
                            ],
                        }
                        for i in range(4)
                    ],
                }
            ],
            "tempo_indications": [
                {
                    "page": 1,
                    "source": "initial",
                    "bbox_mm": [50 * mm, 80 * mm, 220 * mm, 18 * mm],
                }
            ],
        }
        score = {
            "document": {
                "metadata": {
                    "properties": {
                        "title": "First Steps",
                        "artist": "GuitarOCR example",
                    },
                    "first_page_header": {
                        "title": {"visible": True},
                        "artist": {"visible": True},
                    },
                },
                "tempo": {"visible": True, "unit_name": "Quarter", "value": 120},
            },
            "tracks": [
                {
                    "staves": [
                        {"tuning_displayed_label": "", "tuning_label_visible": False}
                    ]
                }
            ],
        }
        catalog = []
        for index, split in enumerate(("train", "validation", "test")):
            gp = self.root / "我的曲谱" / f"sample-{index}.gp5"
            write_targets_gp5(
                TARGETS,
                gp,
                mode="tab",
                title=f"Example {index}",
                artist="GuitarOCR",
                tuning=[64, 59, 55, 50, 45, 40],
                capo=0,
            )
            label = analyze_source(gp)
            sid = label["source_id"]
            write_json(self.export / "labels" / f"{sid}.json", label)
            track = (
                self.export
                / "native-export/documents"
                / f"tab-{sid}"
                / "tracks/track-1"
            )
            write_json(track / "layout.json", layout)
            write_json(track / "official-score.json", score)
            shutil.copy2(drawing / "demo.pdf", track / "score.pdf")
            pdf = self.export / "pdf/tab" / f"{sid}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(track / "score.pdf", pdf)
            write_json(self.export / "layout/tab" / f"{sid}.layout.json", layout)
            catalog.append(
                {
                    "source_id": sid,
                    "family": f"曲谱{index}",
                    "split": split,
                    "source_path": str(gp),
                }
            )
        write_json(
            self.export / "source_catalog.json",
            {"schema_version": "1.0", "sources": catalog},
        )

    def test_all_three_datasets_share_source_assignments(self):
        summary = crop_and_manifest(self.export, ["tab"], 180, 20260715)
        self.assertEqual(summary["samples"], 12)
        inventory = self.root / "anywhere" / "inventory"
        self.assertEqual(build_inventory(self.export, inventory)["pages"], 3)
        layout = build_layout(inventory, self.root / "layout-data")
        info = build_info(inventory, self.root / "info-data")
        self.assertEqual((layout["train"], layout["validation"]), (1, 1))
        self.assertEqual((info["train"], info["validation"]), (2, 2))
        catalog = source_catalog(self.export)
        groups = []
        for split in ("train", "validation", "test"):
            rows = [
                json.loads(line)
                for line in (self.export / "manifests" / f"{split}.jsonl")
                .read_text()
                .splitlines()
            ]
            groups.append({catalog[row["source_id"]]["family"] for row in rows})
        self.assertTrue(
            all(not groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        )

    def test_gp8_export_is_a_complete_entry_point(self):
        layout = build_layout(None, self.root / "layout-only", self.export)
        info = build_info(None, self.root / "info-only", self.export)
        self.assertEqual(layout["train"], 1)
        self.assertEqual(info["validation"], 2)

    def test_catalog_rejects_family_leakage(self):
        path = self.export / "source_catalog.json"
        data = json.loads(path.read_text())
        data["sources"][1]["family"] = data["sources"][0]["family"]
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            source_catalog(self.export)


if __name__ == "__main__":
    unittest.main()
