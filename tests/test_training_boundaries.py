"""Training orchestration checks without Torch, model weights or downloads."""

from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from shared.training import load_training_config, OCR_TRAINING_DEFAULTS


def module(name, **members):
    result = ModuleType(name)
    result.__dict__.update(members)
    return result


class TrainingBoundariesTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config_path = self.root / "training.yaml"
        # JSON is a YAML subset. Stub the optional YAML parser so the core
        # development checks do not require the training dependency stack.
        self.enterContext(patch.dict(sys.modules, {"yaml": module("yaml", safe_load=json.loads)}))

    def write_config(self, **extra):
        config = {"model_name_or_path": "unused-model", "dataset": "unified_train", **extra}
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        return config

    def test_options_are_removed_even_when_their_features_are_disabled(self):
        self.write_config(**OCR_TRAINING_DEFAULTS)
        config, options = load_training_config(self.config_path)
        self.assertEqual(config, {"model_name_or_path": "unused-model", "dataset": "unified_train"})
        self.assertEqual(options, OCR_TRAINING_DEFAULTS)

    def test_overrides_are_applied_before_the_config_is_split(self):
        self.write_config(batch_token_budget=100, vocab_freeze_original=True)
        config, options = load_training_config(self.config_path, [
            "batch_token_budget=null", "maximum_batch_examples=32",
            "vocab_freeze_original=false", 'dataset="new=dataset"',
            "learning_rate=0.0002",
        ])
        self.assertEqual(config["dataset"], "new=dataset")
        self.assertEqual(config["learning_rate"], 0.0002)
        self.assertIsNone(options["batch_token_budget"])
        self.assertEqual(options["maximum_batch_examples"], 32)
        self.assertFalse(options["vocab_freeze_original"])
        self.assertEqual(OCR_TRAINING_DEFAULTS["maximum_batch_examples"], 128)

    def test_non_mapping_config_is_rejected(self):
        self.config_path.write_text("null")
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            load_training_config(self.config_path)

    def test_training_worker_forwards_only_framework_arguments(self):
        from measure_ocr import train_worker

        for enabled in (False, True):
            with self.subTest(enabled=enabled), ExitStack() as stack:
                original = self.write_config(
                    **{**OCR_TRAINING_DEFAULTS, "batch_token_budget": 256 if enabled else None,
                       "maximum_batch_examples": 12, "context_chunk_size": 3,
                       "share_context_images": enabled, "vocab_trainable_from": 50 if enabled else None,
                       "vocab_learning_rate": .002, "vocab_freeze_original": False}
                )
                run_exp, batching, vision = Mock(), Mock(), Mock()
                stack.enter_context(patch.dict(sys.modules, {
                    "llamafactory": module("llamafactory"),
                    "llamafactory.train": module("llamafactory.train"),
                    "llamafactory.train.tuner": module("llamafactory.train.tuner", run_exp=run_exp),
                    "transformers": module("transformers", TrainerCallback=object),
                    "measure_ocr.token_batching": module("measure_ocr.token_batching", install_token_batching=batching),
                    "measure_ocr.shared_vision": module("measure_ocr.shared_vision", install_shared_vision=vision),
                    "shared.score_image": module("shared.score_image", install_training_policy=Mock()),
                }))
                stack.enter_context(patch.object(sys, "argv", ["train_worker", str(self.config_path)]))
                stack.enter_context(patch.dict("os.environ", {"GUITAROCR_DENSE_LOSS": "1"}))
                hooks = {name: stack.enter_context(patch.object(train_worker, name)) for name in (
                    "enable_fused_loss", "materialize_sampler_lengths", "exact_linear_targets",
                    "stage_adapter_weights_on_cpu", "train_music_vocabulary",
                )}
                train_worker.main()
                expected = {key: value for key, value in original.items() if key not in OCR_TRAINING_DEFAULTS}
                self.assertEqual(run_exp.call_args.kwargs["args"], expected)
                self.assertEqual(len(run_exp.call_args.kwargs["callbacks"]), 1)
                if enabled:
                    batching.assert_called_once_with(256, 12, 3)
                    vision.assert_called_once_with()
                    hooks["train_music_vocabulary"].assert_called_once_with(50, .002, False)
                else:
                    batching.assert_not_called()
                    vision.assert_not_called()
                    hooks["train_music_vocabulary"].assert_not_called()

    def test_tokenization_uses_the_same_framework_boundary(self):
        from shared import tokenize_training

        self.write_config(**OCR_TRAINING_DEFAULTS, bf16=True)
        get_train_args = Mock(return_value=(object(), object(), object(), type("Args", (), {"stage": "sft"})(), None))
        get_dataset = Mock(return_value={"train": [1], "validation": []})
        with patch.dict(sys.modules, {
            "torch": module("torch", set_num_threads=Mock()),
            "llamafactory": module("llamafactory"),
            "llamafactory.data": module("llamafactory.data", get_dataset=get_dataset, get_template_and_fix_tokenizer=Mock()),
            "llamafactory.hparams": module("llamafactory.hparams", get_train_args=get_train_args),
            "llamafactory.model": module("llamafactory.model", load_tokenizer=Mock(return_value={"tokenizer": object()})),
            "shared.score_image": module("shared.score_image", install_training_policy=Mock()),
        }), patch.object(sys, "argv", ["tokenize_training", "--config", str(self.config_path), "--output", str(self.root / "cache")]), \
                patch("builtins.print"):
            tokenize_training.main()
        config = get_train_args.call_args.args[0]
        self.assertFalse(set(config) & OCR_TRAINING_DEFAULTS.keys())
        self.assertEqual(config["tokenized_path"], str(self.root / "cache"))
        self.assertTrue(config["use_cpu"])
        self.assertFalse(config["bf16"])
        self.assertFalse(config["fp16"])
        get_dataset.assert_called_once()

    def test_command_help_does_not_import_training_frameworks(self):
        # Use a fresh interpreter: modules cached by other tests must not hide
        # an accidental eager import or require an installed training stack.
        script = """
import importlib.abc
import runpy
import sys
class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'safetensors', 'llamafactory'}:
            raise AssertionError('help imported ' + fullname)
sys.meta_path.insert(0, NoTrainingImports())
name = sys.argv[1]
sys.argv = [name, '--help']
runpy.run_module(name, run_name='__main__')
"""
        for name in ('datagen.run', 'measure_ocr.train', 'document_info.train',
                     'measure_ocr.train_state', 'measure_ocr.evaluate_state',
                     'measure_ocr.train_mtp', 'shared.tokenize_training'):
            with self.subTest(command=name):
                result = subprocess.run([sys.executable, '-c', script, name],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_missing_render_options_fail_before_data_is_written(self):
        from datagen.run import main

        for phase in ('all', 'render'):
            output = self.root / phase
            with self.subTest(phase=phase), \
                    patch.object(sys, 'argv', ['datagen.run', '--phase', phase, '--output', str(output)]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_document_info_entrypoint_is_the_shared_ocr_entrypoint(self):
        from document_info.train import main as information_main
        from measure_ocr.train import main as measure_main

        self.assertIs(information_main, measure_main)


if __name__ == "__main__":
    unittest.main()
