import unittest

from PIL import Image, ImageDraw

from layout.classifier import classify_notation_layout


class NotationClassifierTest(unittest.TestCase):
    def test_light_staff_lines_in_scanned_pages(self):
        for mode, staves in (
            ("score_only", [(50, 5, 8, 205)]),
            ("tab_only", [(50, 6, 12, 205)]),
            ("score_tab", [(50, 5, 8, 205), (140, 6, 12, 205)]),
            ("score_tab", [(50, 5, 8, 205), (140, 6, 12, 20)]),
        ):
            with self.subTest(mode=mode):
                page = Image.new("L", (500, 250), 255)
                draw = ImageDraw.Draw(page)
                for top, lines, spacing, ink in staves:
                    bottom = top + (lines - 1) * spacing
                    for row in range(lines):
                        draw.line((20, top + row * spacing, 480, top + row * spacing), fill=ink)
                    for x in (20, 250, 480):
                        draw.line((x, top, x, bottom), fill=ink)
                self.assertEqual(classify_notation_layout(page)["layout"], mode)

    def test_blank_page_stays_unknown(self):
        self.assertEqual(classify_notation_layout(Image.new("L", (500, 250), 255))["layout"], "unknown")


if __name__ == "__main__":
    unittest.main()
