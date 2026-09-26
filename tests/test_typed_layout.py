import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from layout.crops import prepare_document_crops
from layout.evaluate import _type_matches
from layout.postprocess import order_measure_boxes, refine_measure_boxes
from shared.layout_labels import mode_vote, typed_annotations


class TypedLayoutTest(unittest.TestCase):
    def test_type_accuracy_matches_boxes_without_using_the_predicted_class(self):
        result = _type_matches(
            [[0, 0, 100, 50], [100, 0, 100, 50]],
            [
                {"label": "measure_notation", "score": 0.9, "coordinate": [0, 0, 100, 50]},
                {"label": "measure_tab", "score": 0.8, "coordinate": [100, 0, 200, 50]},
                {"label": "measure_notation", "score": 0.7, "coordinate": [100, 0, 200, 50]},
            ],
            "notation",
        )
        self.assertEqual(result["matched_measures"], 2)
        self.assertEqual(result["correct_type_measures"], 1)
        self.assertEqual(result["measure_type_confusion"], {"notation": 1, "tab": 1})

    def test_duplicate_classes_keep_one_box_and_its_type(self):
        boxes = [
            {"label": "measure_tab", "score": 0.4, "coordinate": [10, 10, 100, 80]},
            {"label": "measure_both", "score": 0.95, "coordinate": [10, 10, 100, 80]},
            {"label": "measure_both", "score": 0.9, "coordinate": [100, 10, 200, 80]},
        ]
        result = refine_measure_boxes(Image.new("L", (220, 100), 255), order_measure_boxes(boxes))
        self.assertEqual(len(result), 2)
        self.assertEqual([row["mode"] for row in result], ["both", "both"])
        self.assertEqual(mode_vote(result)["mode"], "both")
        self.assertIsNone(mode_vote([])["mode"])

    def test_page_type_is_required_for_supervision(self):
        payload = {
            "images": [{"id": 1}],
            "categories": [{"id": 1, "name": "measure"}],
            "annotations": [],
        }
        with self.assertRaisesRegex(ValueError, "valid mode"):
            typed_annotations(payload)

    def test_auto_uses_model_types_on_every_page_and_manual_mode_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pages, predictions = [], []
            for index, mode in enumerate(("tab", "notation", "both")):
                image = root / f"page{index}.png"
                Image.new("L", (250, 160), 255).save(image)
                # Vector PDFs must also invoke the model when type is auto.
                pages.append({"image": image, "vector": True, "source_pdf": root / "unused.pdf", "pdf_page": index + 1})
                predictions.append({"measures": order_measure_boxes([
                    {"label": f"measure_{mode}", "score": 0.9, "coordinate": [20, 40, 220, 140]}
                ]), "tempo_regions": []})

            def predict(command, check):
                self.assertTrue(check)
                Path(command[command.index("--output") + 1]).write_text(json.dumps(predictions))

            for setting in ("auto", "notation"):
                with (
                    self.subTest(setting=setting),
                    patch("layout.crops.subprocess.run", side_effect=predict),
                    patch("layout.crops.classify_notation_layout", side_effect=AssertionError("Staff rules were used")),
                    patch("layout.crops.extract_pdf_vector_measure_boxes", side_effect=AssertionError("Geometry was used")),
                ):
                    output = root / setting
                    _, records = prepare_document_crops(
                        [], output, setting, False, root / "model", root / "python",
                        "auto" if setting == "auto" else "image", pages=pages,
                    )
                    self.assertEqual([r["mode"] for r in records], ["tab", "notation", "both"] if setting == "auto" else ["notation"] * 3)
                    self.assertEqual([r["detected_mode"] for r in records], ["tab", "notation", "both"])
                    self.assertTrue(all(Path(r["image"]).is_file() for r in records))
                    saved = json.loads((output / "pages.json").read_text())
                    self.assertEqual([r["notation_mode"] for r in saved], [r["mode"] for r in records])


if __name__ == "__main__":
    unittest.main()
