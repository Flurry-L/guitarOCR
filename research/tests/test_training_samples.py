from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from research.data.training_samples import annotation_sample
from research.data.training_samples import visual_measure_sample
from research.inference.information.prompts import ANNOTATION_PROMPT
from scorelib.score_state import state_prompt


class TrainingSamplesTest(unittest.TestCase):
    def test_signature_dataset_retains_labels_order_and_requested_split(self):
        from research.data.signature_data import Signatures

        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ('first', 'second')]
            for root in roots:
                root.mkdir()
                rows = [
                    {'messages': [{'content': 'prompt'}, {'content': target}], 'images': [image]}
                    for image, target in [('printed.png', 'S2 time=6/8 key=-3'),
                                          ('absent.png', 'S2 time=- key=-')]
                ]
                (root / 'state_test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
                (root / 'state_train.jsonl').write_text('not selected')
            dataset = Signatures(roots, 'test')
            self.assertEqual(dataset.rows, [('printed.png', [4, 6, 4]), ('absent.png', [15, 0, 0])] * 2)
            self.assertEqual(len(dataset), 4)
            self.assertFalse(dataset.augment)

    def test_visual_sample_preserves_existing_prompts_images_and_targets(self):
        for mode in ("tab", "notation", "both"):
            with self.subTest(mode=mode):
                row = {
                    "mode": mode, "instrument": "guitar", "measure_index": 4, "bar_index": 0,
                    "score_state": {"time": "3/4", "key": 2, "tuning": [64, 59, 55, 50, 45, 40]},
                    "image": "target.png", "previous_image": "previous.png", "next_image": "next.png",
                    "target": "M2 time=3/4 | V0{@0:h.:r}",
                }
                original = deepcopy(row)
                state = dict(row["score_state"])
                if mode != "tab":
                    state.pop("tuning")
                expected = {
                    "messages": [
                        {"role": "user", "content": "<image><image><image>" + state_prompt(
                            mode, "guitar", state, first=True, visual_pitch=True,
                        )},
                        {"role": "assistant", "content": row["target"]},
                    ],
                    "images": ["target.png", "previous.png", "next.png"],
                }
                self.assertEqual(visual_measure_sample(row), expected)
                self.assertEqual(row, original)
                self.assertNotIn("This is the first measure", visual_measure_sample(row, first=False)["messages"][0]["content"])

    def test_source_builder_retains_its_own_first_measure_policy(self):
        from research.data.score_support_data import prepare_source

        row = {"mode": "tab", "measure_index": 0, "bar_index": 8, "split": "test",
               "score_state": {"time": "4/4", "key": 0}, "image": "single.png", "target": "M2 | V0{@0:w:r}"}
        samples, rows, _, _ = prepare_source(([row], "synthetic", False, False))
        self.assertEqual(samples, [visual_measure_sample(rows[0], first=True)])
        self.assertEqual(samples[0]["images"], ["single.png"] * 3)
        self.assertIn("Independent guitar tab", samples[0]["messages"][0]["content"])
        self.assertIn("This is the first measure", samples[0]["messages"][0]["content"])

    def test_composed_sample_can_use_only_its_bar_index(self):
        row = {"mode": "notation", "bar_index": 0,
               "score_state": {"time": "4/4", "key": 0}, "image": "bar.png", "target": "M2 | V0{@0:w:r}"}
        self.assertIn("This is the first measure", visual_measure_sample(row)["messages"][0]["content"])

    def test_annotation_sample_keeps_unicode_and_compact_json(self):
        label = {"kind": "chord", "text": "B♭", "semitones": None, "diagram": {"frets": [1, 2, None]}}
        original = deepcopy(label)
        sample = annotation_sample("annotations/chord.png", label)
        self.assertEqual(sample, {
            "messages": [
                {"role": "user", "content": "<image>" + ANNOTATION_PROMPT},
                {"role": "assistant", "content": json.dumps(label, ensure_ascii=False, separators=(",", ":"))},
            ],
            "images": [str(Path("annotations/chord.png").resolve())],
        })
        self.assertEqual(label, original)


if __name__ == "__main__":
    unittest.main()
