import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from document_info.image_ocr import parse_info_response, tuning_from_name
from document_info.evaluate_parallel import score
from datagen.build_info_data import _printed_text
from datagen.augment_info import build as build_augmented_info


class DocumentInfoTest(unittest.TestCase):
    def test_header_augmentation_preserves_splits_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "header.png"
            Image.new("RGB", (16, 16), "white").save(image)
            for name in ("base", "extra"):
                folder = root / name
                folder.mkdir()
                (folder / "dataset_info.json").write_text("{}")
                for split in ("train", "validation", "test"):
                    rows = [{"images": [str(image)], "messages": [
                        {"role": "user", "content": "<image>header"},
                        {"role": "assistant", "content": '{"title":"A","artist":null}'},
                    ], "provenance": {"family": split, "split": split}}]
                    (folder / f"document_info_{split}.json").write_text(json.dumps(rows))
            report = build_augmented_info(root / "base", root / "extra", root / "mixed", repeats=2, workers=1)
            self.assertEqual(report["splits"]["train"]["samples"], 5)
            self.assertEqual(report["splits"]["validation"]["samples"], 3)
            heldout = json.loads((root / "mixed/document_info_test.json").read_text())
            self.assertEqual({r["provenance"]["family"] for r in heldout}, {"test"})
            path = root / "extra/document_info_validation.json"
            rows = json.loads(path.read_text())
            rows[0]["provenance"]["family"] = "train"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_augmented_info(root / "base", root / "extra", root / "leaked", workers=1)
            self.assertFalse((root / "leaked").exists())

    def test_native_pdf_compatibility_glyphs_match_visible_chinese(self):
        self.assertEqual(_printed_text("海⻛⼩品  003\n⻘竹⻓川"),
                         _printed_text("海风小品 003 青竹长川"))
        self.assertNotEqual(_printed_text("海风小品"), _printed_text("海凤小品"))

    def test_parallel_metrics_weight_modes_by_samples_and_count_missing_fields(self):
        rows = [
            {"expected": {"title": "A"}, "predicted": {"title": "A"}, "provenance": {"mode": "tab"}},
            {"expected": {"title": "B"}, "predicted": {}, "provenance": {"mode": "notation"}},
            {"expected": {"title": "C"}, "predicted": {"title": "C"}, "provenance": {"mode": "notation"}},
        ]
        result = score(rows)
        self.assertEqual(result["overall"]["title"], {"correct": 2, "total": 3, "accuracy": 2 / 3})
        self.assertEqual(result["notation"]["title"]["accuracy"], 0.5)
        missing = score([{"expected": {"artist": None}, "predicted": {}}])
        self.assertEqual(missing["overall"]["artist"]["accuracy"], 0)

    def test_header_and_tempo_json(self):
        self.assertEqual(
            parse_info_response('{"title":" 夏霞 ","artist":null,"tuning_name":"Standard tuning"}', "header"),
            {"title": "夏霞", "artist": None, "tuning_name": "Standard tuning"},
        )
        self.assertEqual(parse_info_response('{"tempo_quarter":101}', "tempo"), {"tempo_quarter": 101})
        self.assertEqual(parse_info_response('{"tempo_quarter":"fast"}', "tempo"), {})

    def test_visible_tuning_mapping(self):
        self.assertEqual(tuning_from_name("Standard tuning"), [64, 59, 55, 50, 45, 40])
        self.assertIsNone(tuning_from_name("Custom"))


if __name__ == "__main__":
    unittest.main()
