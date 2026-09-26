from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import tempfile
import unittest

import pypdfium2 as pdfium
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen.canvas import Canvas

from shared.pdf import open_pdf


class PdfTest(unittest.TestCase):
    def test_crop_rotation_text_and_form_paths_align_with_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pdf"
            canvas = Canvas(str(source), pagesize=(200, 180))
            canvas.drawString(45, 95, "Guitar")
            canvas.beginForm("staff")
            canvas.setLineWidth(2)
            canvas.line(0, 0, 80, 0)
            canvas.endForm()
            canvas.translate(40, 80)
            canvas.scale(1.25, 1)
            canvas.doForm("staff")
            canvas.save()
            endpoints = {
                0: [(20, 80), (120, 80)],
                90: [(50, 20), (50, 120)],
                180: [(140, 50), (40, 50)],
                270: [(80, 140), (80, 40)],
            }
            for rotation, expected in endpoints.items():
                path = root / f"rotated-{rotation}.pdf"
                with closing(pdfium.PdfDocument(source)) as pdf:
                    with closing(pdf[0]) as page:
                        page.set_cropbox(20, 30, 180, 160)
                        page.set_rotation(rotation)
                    pdf.save(path)
                with self.subTest(rotation=rotation), open_pdf(path) as pdf:
                    page = pdf[0]
                    image = page.render(72)
                    size = (160, 130) if rotation in (0, 180) else (130, 160)
                    self.assertEqual(image.size, size)
                    self.assertEqual((page.width, page.height), size)
                    self.assertEqual(page.line_paths(), [[tuple(expected)]])
                    midpoint = tuple(round((a + b) / 2) for a, b in zip(*expected))
                    x, y = midpoint
                    self.assertLess(image.crop((x - 1, y - 1, x + 2, y + 2)).getextrema()[0], 100)
                    words = page.words()
                    self.assertTrue(any(word[4] == "Guitar" for word in words))
                    for x0, y0, x1, y1, _ in words:
                        self.assertTrue(0 <= x0 < x1 <= size[0])
                        self.assertTrue(0 <= y0 < y1 <= size[1])
                        self.assertLess(image.crop((x0, y0, x1, y1)).getextrema()[0], 100)
                    self.assertIn("Guitar", page.text())

    def test_password_error_and_concurrent_rendering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encrypted = root / "locked.pdf"
            canvas = Canvas(str(encrypted), encrypt=StandardEncryption("secret"))
            canvas.drawString(10, 10, "Locked")
            canvas.save()
            with self.assertRaisesRegex(ValueError, "密码"):
                with open_pdf(encrypted):
                    self.fail("Encrypted PDF opened without a password")

            def render(_):
                with open_pdf(Path("examples/demo.pdf")) as pdf:
                    return pdf[0].render().tobytes()

            with ThreadPoolExecutor(max_workers=4) as pool:
                images = list(pool.map(render, range(8)))
            self.assertEqual(len(set(images)), 1)


if __name__ == "__main__":
    unittest.main()
