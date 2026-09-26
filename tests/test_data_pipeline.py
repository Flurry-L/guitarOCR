import json
from pathlib import Path
import shutil
import tempfile
import unittest

from datagen.build_info_data import build_dataset as build_info
from datagen.build_layout_data import build_dataset as build_layout
from datagen.build_measure_data import crop_and_manifest
from datagen.gp_sources import analyze_source, encode_measure
from datagen.inventory import build_inventory, source_catalog
from examples.generate import generate, TARGETS
from gp5_export.writer import write_targets_gp5
from shared.artifacts import write_json


class DataPipelineTest(unittest.TestCase):
    def test_curation_rejects_the_invalid_previous_context_of_a_dropped_bar(self):
        from datagen.curate_measure_data import curate

        crop_and_manifest(self.export, ["tab"], 180, 20260715)
        path = self.export / "manifests/train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["target"] = "M2 | V0{@0:d2048:r}"
        rows[1]["previous_context"] = "C2 | V0{@0:d2048:r}"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        destination = self.root / "curated"
        curate(self.export, destination, workers=1)
        report = json.loads((destination / "summary.json").read_text())
        self.assertEqual(report["train"]["samples"], 2)
        self.assertEqual(report["train"]["invalid_targets"], 1)
        self.assertEqual(report["train"]["invalid_contexts"], 1)
        self.assertEqual(report["train"]["rejected_samples"], 2)
        self.assertEqual(report["validation"]["samples"], 4)

    def test_empty_gp_bar_produces_a_valid_full_measure_rest(self):
        from guitarpro import models as gm
        from shared.constraints import validate_measure_target
        from shared.m2 import format_measure_target

        song = gm.Song()
        track = gm.Track(song)
        header = gm.MeasureHeader()
        header.timeSignature.numerator = 3
        header.timeSignature.denominator.value = 4
        measure = gm.Measure(track, header)
        encoded = encode_measure(measure, None, None)
        for mode in ("tab", "notation", "both"):
            target = format_measure_target(encoded, mode)
            self.assertEqual(validate_measure_target(target, mode)[1], [])
            self.assertIn("V0{@0:h.:r}", target)

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


    def test_catalog_rejects_family_leakage(self):
        path = self.export / "source_catalog.json"
        data = json.loads(path.read_text())
        data["sources"][1]["family"] = data["sources"][0]["family"]
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            source_catalog(self.export)

    def test_multimode_inventory_preserves_family_and_test_split(self):
        for track in sorted((self.export / "native-export/documents").glob("tab-*")):
            for mode in ("notation", "both"):
                destination = track.with_name(track.name.replace("tab-", mode + "-", 1))
                shutil.copytree(track, destination)
                layout_path = next(destination.glob("tracks/*/layout.json"))
                layout = json.loads(layout_path.read_text())
                layout.update(display_mode=mode, tab_only=False)
                write_json(layout_path, layout)
        inventory = self.root / "multimode-inventory"
        summary = build_inventory(self.export, inventory, workers=2)
        self.assertEqual((summary["tracks"], summary["pages"]), (9, 9))
        destination = self.root / "multimode-coco"
        build_layout(inventory, destination, include_test=True)
        families = []
        for split in ("train", "val", "test"):
            data = json.loads((destination / "annotations" / f"instance_{split}.json").read_text())
            self.assertEqual({row["mode"] for row in data["images"]}, {"tab", "notation", "both"})
            self.assertEqual(len(data["images"]), 3)
            families.append({row["family"] for row in data["images"]})
        self.assertTrue(all(not families[a] & families[b] for a, b in ((0, 1), (0, 2), (1, 2))))
        typed_destination = self.root / "typed-coco"
        build_layout(inventory, typed_destination, include_test=True, typed_measures=True)
        for split in ("train", "val", "test"):
            data = json.loads((typed_destination / "annotations" / f"instance_{split}.json").read_text())
            labels = {row["id"]: row["name"] for row in data["categories"]}
            modes = {row["id"]: row["mode"] for row in data["images"]}
            self.assertEqual(set(labels.values()), {"measure_tab", "measure_notation", "measure_both", "tempo_region"})
            for row in data["annotations"]:
                if labels[row["category_id"]] != "tempo_region":
                    self.assertEqual(labels[row["category_id"]], f"measure_{modes[row['image_id']]}")

        info_destination = self.root / "multimode-info"
        info = build_info(inventory, info_destination, include_test=True)
        self.assertEqual([info[split] for split in ("train", "validation", "test")], [6, 6, 6])
        info_families = []
        for split in ("train", "validation", "test"):
            rows = json.loads((info_destination / "llamafactory" / f"document_info_{split}.json").read_text())
            self.assertEqual({row["provenance"]["mode"] for row in rows}, {"tab", "notation", "both"})
            self.assertTrue(all(row["provenance"]["split"] == split for row in rows))
            info_families.append({row["provenance"]["family"] for row in rows})
        self.assertTrue(all(not info_families[a] & info_families[b] for a, b in ((0, 1), (0, 2), (1, 2))))

    def test_scan_augmentation_preserves_boxes_and_heldout_images(self):
        from datagen.scan_augment import augment_layout
        from PIL import Image

        inventory = self.root / "scan-inventory"
        build_inventory(self.export, inventory)
        source = self.root / "clean-coco"
        build_layout(inventory, source, include_test=True, typed_measures=True)
        output = self.root / "scan-coco"
        augment_layout(source, output, ["train"], 1.0, 12, False, 1)
        clean = json.loads((source / "annotations/instance_train.json").read_text())
        augmented = json.loads((output / "annotations/instance_train.json").read_text())
        self.assertEqual(len(augmented["images"]), 2 * len(clean["images"]))
        self.assertEqual(len(augmented["annotations"]), 2 * len(clean["annotations"]))
        generated = [row for row in augmented["images"] if "augmentation_parent" in row]
        for row in generated:
            parent = next(image for image in clean["images"] if image["id"] == row["augmentation_parent"])
            with Image.open(output / "images" / row["file_name"]) as image:
                self.assertEqual(image.size, (parent["width"], parent["height"]))
            originals = [a["bbox"] for a in clean["annotations"] if a["image_id"] == parent["id"]]
            copies = [a["bbox"] for a in augmented["annotations"] if a["image_id"] == row["id"]]
            self.assertEqual(originals, copies)
        for split in ("val", "test"):
            before = json.loads((source / "annotations" / f"instance_{split}.json").read_text())
            after = json.loads((output / "annotations" / f"instance_{split}.json").read_text())
            self.assertEqual(before["annotations"], after["annotations"])
            for a, b in zip(before["images"], after["images"]):
                self.assertEqual((source / "images" / a["file_name"]).resolve(), (output / "images" / b["file_name"]).resolve())

    def test_inventory_rejects_mislabeled_display_mode(self):
        track = next((self.export / "native-export/documents").glob("tab-*"))
        layout_path = next(track.glob("tracks/*/layout.json"))
        layout = json.loads(layout_path.read_text())
        layout.update(display_mode="both", tab_only=False)
        write_json(layout_path, layout)
        with self.assertRaisesRegex(ValueError, "display mode differs"):
            build_inventory(self.export, self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
