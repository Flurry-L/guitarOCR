import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from measure_ocr.evaluate import run_inference, evaluate
from shared.m2 import format_previous_measure_context

TARGET = "M2 time=4/4 | V0{@0:w:s1f7}"


class EvaluationTest(unittest.TestCase):
    def test_predicted_context_and_resume_use_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "samples.jsonl"
            rows = [
                {
                    "id": f"{sid}-{i}",
                    "source_id": sid,
                    "mode": "tab",
                    "measure_index": i,
                    "image": "unused.png",
                    "target": TARGET,
                    "previous_context": "GOLD_SENTINEL",
                }
                for sid, i in (("b", 0), ("a", 1), ("a", 0))
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = argparse.Namespace(
                manifest=manifest,
                max_samples=0,
                max_sources=0,
                seed=1,
                context_source="predicted",
                image_ablation="none",
                model=root / "model",
                adapter=root / "adapter",
                device="cpu",
                max_new_tokens=128,
                maximum_attempts=1,
                predictions=root / "predictions.jsonl",
                resume=False,
            )
            with patch("measure_ocr.evaluate.GlmBackend") as backend:
                backend.return_value.generate.return_value = (TARGET, 20)
                run_inference(args)
            predicted = [
                json.loads(line) for line in args.predictions.read_text().splitlines()
            ]
            self.assertEqual([r["id"] for r in predicted], ["a-0", "a-1", "b-0"])
            self.assertEqual(
                [r["previous_context"] for r in predicted],
                ["START", format_previous_measure_context(TARGET, "tab"), "START"],
            )
            self.assertNotIn(
                "GOLD_SENTINEL", str(backend.return_value.generate.call_args_list)
            )
            # Resume in the middle of a sequence with the previous prediction on disk.
            args.predictions.write_text(json.dumps(predicted[0]) + "\n")
            args.resume = True
            with patch("measure_ocr.evaluate.GlmBackend") as backend:
                backend.return_value.generate.return_value = (TARGET, 20)
                run_inference(args)
                self.assertEqual(backend.return_value.generate.call_count, 2)
            restored = [
                json.loads(line) for line in args.predictions.read_text().splitlines()
            ]
            self.assertEqual(
                restored[1]["previous_context"], predicted[1]["previous_context"]
            )
            report = evaluate(args.predictions, root / "metrics.json")
            self.assertEqual(report["context_source"], "predicted")
            args.max_samples = 2
            with self.assertRaisesRegex(ValueError, "complete sequences"):
                run_inference(args)


if __name__ == "__main__":
    unittest.main()
