import unittest

from PIL import Image, ImageDraw

from layout.postprocess import order_measure_boxes, refine_measure_boxes
from layout.tab_geometry import detect_tab_boundaries_for_lines, detect_tab_geometry


def measure(left, top, right, bottom, score=0.9):
    return {
        "label": "measure",
        "score": score,
        "coordinate": [left, top, right, bottom],
    }


def full_rows():
    return [
        measure(index * 100 + 10, top, (index + 1) * 100 + 10, top + 70)
        for top in (150, 250)
        for index in range(4)
    ]


class OrderMeasureBoxesTest(unittest.TestCase):
    def test_tab_boundaries_reject_stems_and_optionally_keep_staff_connectors(self):
        image = Image.new("L", (640, 400), 255)
        draw = ImageDraw.Draw(image)
        rows = list(range(100, 181, 16))
        for y in rows:
            draw.line((20, y, 620, y), fill=0)
        for x in (20, 220, 420, 620):
            draw.line((x, 100, x, 180), fill=0)
        draw.line((300, 100, 300, 195), fill=0)  # Note stem, never a barline.
        draw.line((420, 44, 420, 180), fill=0)  # Connector to a notation staff.
        for keep, expected in (
            (False, [20.0, 220.0, 620.0]),
            (True, [20.0, 220.0, 420.0, 620.0]),
        ):
            with self.subTest(keep=keep):
                self.assertEqual(
                    detect_tab_boundaries_for_lines(
                        image, rows, preserve_cross_staff_barlines=keep
                    ),
                    expected,
                )
                staffs = detect_tab_geometry(image, preserve_cross_staff_barlines=keep)
                self.assertEqual(len(staffs), 1)
                self.assertEqual(staffs[0]["boundaries"], expected)
                self.assertEqual(len(staffs[0]["measures"]), len(expected) - 1)

    def test_rows_are_ordered_and_do_not_overlap(self):
        boxes = [
            measure(98, 10, 202, 40),
            measure(0, 10, 103, 40),
            measure(0, 80, 200, 110),
            measure(0, 10, 202, 40, score=0.3),
            {"label": "tempo_region", "score": 0.9, "coordinate": [0, 0, 30, 8]},
        ]
        original = [box["coordinate"][:] for box in boxes]
        result = order_measure_boxes(boxes)
        self.assertEqual([box["system_index"] for box in result], [0, 0, 1])
        self.assertEqual([box["system_measure_index"] for box in result], [0, 1, 0])
        self.assertLessEqual(result[0]["coordinate"][2], result[1]["coordinate"][0])
        self.assertEqual([box["coordinate"] for box in boxes], original)


    def test_barline_evidence_merges_false_splits(self):
        image = Image.new("L", (420, 130), 255)
        draw = ImageDraw.Draw(image)
        for y in (40, 50, 60, 70, 80):
            draw.line((10, y, 410, y), fill=0)
        draw.line((200, 40, 200, 80), fill=0, width=2)
        boxes = order_measure_boxes(
            [
                measure(10, 30, 100, 100),
                measure(100, 30, 200, 100),
                measure(200, 30, 410, 100),
                *full_rows(),
            ]
        )
        result = refine_measure_boxes(image, boxes)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0]["bbox"], [10.0, 30.0, 190.0, 70.0])

    def test_barlines_recover_missing_middle_measure(self):
        image = Image.new("L", (420, 130), 255)
        draw = ImageDraw.Draw(image)
        for y in (40, 50, 60, 70, 80):
            draw.line((10, y, 410, y), fill=0)
        for x in (100, 200, 300):
            draw.line((x, 40, x, 80), fill=0, width=2)
        boxes = order_measure_boxes(
            [
                measure(10, 30, 100, 100),
                measure(100, 30, 200, 100),
                measure(300, 30, 410, 100),
                *full_rows(),
            ]
        )
        result = refine_measure_boxes(image, boxes)
        self.assertEqual(len(result), 12)
        self.assertEqual(result[2]["geometry_source"], "barline_gap_recovery")
        self.assertEqual(result[2]["bbox"], [200.0, 30.0, 100.0, 70.0])


if __name__ == "__main__":
    unittest.main()
