"""OCR timing review stays soft and survives journal recovery."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from research.inference.measures.parallel import recognize_independent
from research.inference.measures.recognizer import recognize_crops
from research.inference.measures.review import ocr_rhythm_warnings
from scorelib.constraints import validate_measure_target

OVERFULL = "M2 | V0{@0:w:s1f0 @3840:w:s1f3 @7680:w:s1f5 @11520:w:s1f3}"
TUNING = [64, 59, 55, 50, 45, 40]


class OCRRhythmTest(unittest.TestCase):
    def test_real_overfull_prediction_is_preserved_and_review_survives_resume(self):
        for engine in (recognize_crops, recognize_independent):
            with self.subTest(engine=engine.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                image = root / "bar.png"
                Image.new("RGB", (100, 40), "white").save(image)
                backend = Mock(spec=["generate"])
                backend.generate.return_value = (OVERFULL, 50)
                reader = Mock(spec=["predict"])
                reader.predict.side_effect = lambda rows, **kwargs: [{"time": "4/4", "key": 0} for _ in rows]
                options = {"state_reader": reader} if engine is recognize_independent else {}
                args = ("tab", root, None, "cpu", 512, 512, TUNING, 1, root / "recognition.jsonl")
                rows = [{"measure_number": 1, "image": str(image), "mode": "tab"}]
                self.assertEqual(engine(rows, *args, False, backend=backend, **options), [OVERFULL])
                self.assertTrue(rows[0]["needs_review"])
                self.assertIn("超出 4/4", " ".join(rows[0]["fallback_reason"]))
                self.assertEqual(validate_measure_target(OVERFULL, "tab", tuning=TUNING)[1], [])
                backend.generate.assert_called_once()
                # Legacy accepted journals have no soft warning. Recovery must
                # recompute it without inference or changing the saved notes.
                journal = root / "recognition.jsonl"
                entries = [json.loads(line) for line in journal.read_text().splitlines()]
                for entry in entries:
                    entry.pop("needs_review", None)
                    entry.pop("fallback_reason", None)
                journal.write_text("".join(json.dumps(row) + "\n" for row in entries))
                backend.generate.reset_mock()
                restored = [{"measure_number": 1, "image": str(image), "mode": "tab"}]
                self.assertEqual(engine(restored, *args, True, backend=backend, **options), [OVERFULL])
                backend.generate.assert_not_called()
                self.assertTrue(restored[0]["needs_review"])
                self.assertEqual(restored[0]["fallback_reason"], rows[0]["fallback_reason"])

    def test_pickups_and_parallel_voices_do_not_sum_to_an_overfull_bar(self):
        for target in ("M2 | V0{@0:q:s1f0}", "M2 | V0{@0:w:s1f0} || V1{@0:w:s2f0}",
                       "M2 time=3/4 | V0{@0:h.:s1f0}", "M2 | V0{@480:h..:s1f0}",
                       "M2 | V0{@3200:q[3:2]:s1f0}", "M2 | V0{@3291:q[7:4]:s1f0}"):
            with self.subTest(target=target):
                self.assertEqual(ocr_rhythm_warnings(target), [])
        for target, signature in (
            ("M2 | V0{@0:w:s1f0}", "3/4"),
            ("M2 | V0{@3000:q:s1f0}", "4/4"),
            ("M2 | V0{@481:h..:s1f0}", "4/4"),
            ("M2 | V0{@3292:q[7:4]:s1f0}", "4/4"),
        ):
            with self.subTest(target=target):
                self.assertTrue(ocr_rhythm_warnings(target, signature))
                self.assertEqual(validate_measure_target(target, "tab")[1], [])


if __name__ == "__main__":
    unittest.main()
