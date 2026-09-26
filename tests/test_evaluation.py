import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from measure_ocr.evaluate import run_inference, evaluate, _shard_rows
from shared.m2 import format_previous_measure_context, format_history_context
from measure_ocr.metrics import MeasureSequenceMetrics
from measure_ocr.recognizer import recognize_crops
from gp5_export.writer import targets_to_song

TARGET = "M2 time=4/4 | V0{@0:w:s1f7}"


class EvaluationTest(unittest.TestCase):
    def test_swing_survives_failed_bars_but_valid_straight_resets_it(self):
        for reset in ("", " feel=none"):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                outputs = ["M2 time=3/4 feel=eighth | V0{@0:h.:s1f7}",
                           "invalid", "invalid", f"M2{reset} | V0{{@0:h.:s1f7}}",
                           "invalid", "M2 | V0{@0:h.:s1f7}"]
                rows = [{"id": f"a-{i}", "source_id": "a", "mode": "tab", "measure_index": i,
                         "measure_number": i + 1, "image": "unused.png", "target": TARGET}
                        for i in range(len(outputs))]
                manifest = root / "samples.jsonl"
                manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
                args = argparse.Namespace(
                    manifest=manifest, max_samples=0, max_sources=0, seed=1,
                    context_source="predicted", image_ablation="none", model=root, adapter=root,
                    device="cpu", max_new_tokens=128, maximum_attempts=1,
                    predictions=root / "predictions.jsonl", resume=False,
                )
                with patch("measure_ocr.evaluate.GlmBackend") as backend:
                    backend.return_value.generate.side_effect = [(o, 20) for o in outputs]
                    run_inference(args)
                evaluated = [json.loads(line) for line in args.predictions.read_text().splitlines()]
                backend = Mock()
                backend.generate.side_effect = [(o, 20) for o in outputs]
                targets = recognize_crops(
                    rows, "tab", root, root, "cpu", 128, 128, [64, 59, 55, 50, 45, 40],
                    1, root / "recognition.jsonl", False, backend=backend,
                )
                self.assertEqual(targets, [r["predicted"] for r in evaluated])
                self.assertEqual([r["previous_context"] for r in rows], [r["previous_context"] for r in evaluated])
                self.assertEqual([r["needs_review"] for r in rows], [False, True, True, False, True, False])
                self.assertIn("feel=eighth", rows[3]["previous_context"])
                self.assertNotIn("feel=eighth", rows[5]["previous_context"])
                self.assertEqual([h.tripletFeel.name for h in targets_to_song(targets).measureHeaders],
                                 ["eighth", "eighth", "eighth", "none", "none", "none"])
                self.assertIn("@0:h.:r", targets[1])

    def test_rest_fallback_does_not_hide_invalid_raw_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({
                "context_source": "predicted", "mode": "tab", "expected": TARGET,
                "predicted": "M2 time=4/4 | V0{@0:w:r}",
                "raw_prediction": "malformed OCR", "needs_review": True,
            }) + "\n")
            result = evaluate(predictions, root / "metrics.json")
            self.assertEqual(result["overall"]["syntax_valid_rate"], 1)
            self.assertEqual(result["raw_overall"]["syntax_valid_rate"], 0)
            self.assertEqual(result["raw_by_mode"]["tab"]["constraint_valid_rate"], 0)
            self.assertEqual(result["needs_review"], 1)

    def test_invalid_ornament_prediction_does_not_abort_metrics(self):
        metrics = MeasureSequenceMetrics()
        metrics.update("M2 | V0{@0:w:s1f7}", "M2 | V0{@0:w:s1f7(grace:nonsense)}", "tab")
        self.assertEqual(metrics.samples, 1)
        self.assertEqual(metrics.constraint_valid, 0)
        self.assertEqual(metrics.exact, 0)

    def test_notation_context_retains_signature_across_unmarked_bars(self):
        history = ["M2 time=3/4 key=DMajor | V0{@0:h.:p62}", "M2 | V0{@0:h.:p66}"]
        context = format_history_context(history, "notation")
        self.assertIn("time=3/4", context)
        self.assertIn("key=DMajor", context)
        self.assertIn("p66", context)
        self.assertNotIn("p62", context)

    def test_parallel_shards_preserve_complete_prediction_context(self):
        rows = [{"id": f"{source}-{mode}-{index}", "source_id": source, "mode": mode}
                for source in ("a", "b", "c") for mode in ("tab", "notation", "both")
                for index in range(3)]
        for context in ("gold", "predicted"):
            shards = [_shard_rows(rows, index, 2, context) for index in range(2)]
            self.assertEqual(sorted(r["id"] for shard in shards for r in shard), sorted(r["id"] for r in rows))
            if context == "predicted":
                self.assertFalse({r["source_id"] for r in shards[0]} & {r["source_id"] for r in shards[1]})
        with self.assertRaises(ValueError):
            _shard_rows(rows, 2, 2, "gold")

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
