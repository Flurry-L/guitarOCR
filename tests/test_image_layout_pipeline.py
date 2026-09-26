import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from layout.crops import prepare_document_crops
from pipeline.config import parse_args


class ImageLayoutPipelineTest(unittest.TestCase):
    def test_pdf_page_uses_image_layout_when_requested(self):
        for notation_mode in ("tab", "notation", "both"):
            with self.subTest(mode=notation_mode):
                args = parse_args([
                    "score.pdf", "--output", "output/test",
                    "--mode", notation_mode, "--layout-source", "image",
                ])
                self.assertEqual(args.mode, notation_mode)
                self.check_image_layout(notation_mode)

    def check_image_layout(self, notation_mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "page.png"
            Image.new("L", (200, 120), 255).save(page)
            pdf = root / "score.pdf"
            pdf.touch()

            boxes = [{
                "measures": [{
                    "system_index": 0,
                    "system_measure_index": 0,
                    "bbox": [20, 30, 100, 50],
                }],
                "tempo_regions": [{
                    "label": "tempo_region", "score": 0.9,
                    "coordinate": [10, 10, 40, 22],
                }],
            }]

            def write_boxes(command, check):
                self.assertTrue(check)
                Path(command[command.index("--output") + 1]).write_text(
                    json.dumps(boxes), encoding="utf-8"
                )

            with (
                patch(
                    "layout.crops.expand_inputs",
                    return_value=[{"image": page, "source_pdf": pdf, "pdf_page": 1}],
                ),
                patch(
                    "layout.crops.subprocess.run",
                    side_effect=write_boxes,
                ),
                patch(
                    "layout.crops.extract_pdf_vector_tab_measure_boxes",
                    side_effect=AssertionError("PDF vector path was used"),
                ),
            ):
                mode, records = prepare_document_crops(
                    [pdf], root / "result", notation_mode, False,
                    root / "layout_model", root / "python", "image",
                )

            self.assertEqual(mode, notation_mode)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["geometry_source"], "pp_doclayout")
            self.assertTrue(Path(records[0]["image"]).is_file())
            regions = json.loads((root / "result" / "document_regions.json").read_text())
            self.assertEqual([region["kind"] for region in regions], ["header", "tempo"])
            self.assertTrue(all(Path(region["image"]).is_file() for region in regions))


if __name__ == "__main__":
    unittest.main()
