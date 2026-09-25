import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import guitarpro
from PIL import Image, ImageDraw

from datagen.gp_sources import analyze_source
from datagen.select_sources import select_sources
from document_info import run as info_stage
from document_info.image_ocr import recognize_document_info
from gp5_export import run as export_stage
from layout import run as layout_stage
from measure_ocr import run as measure_stage
from pipeline.config import parse_args
from pipeline.run import run as run_pipeline
from shared.artifacts import read_result
from shared.m2 import parse_measure_target


TARGET = "M2 time=4/4 | V0{@0:q:s1f0,s2f1 @960:q:s1f1,s2f2 @1920:h:s2f3,s3f0}"


class StagedPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.page = self.root / "score.png"
        image = Image.new("L", (640, 240), 255)
        draw = ImageDraw.Draw(image)
        for y in range(80, 131, 10):
            draw.line((20, y, 620, y), fill=0)
        for x in (20, 220, 420, 620):
            draw.line((x, 80, x, 130), fill=0, width=2)
        for x in (100, 300, 500):
            draw.text((x, 78), "3", fill=0)
        image.save(self.page)
        self.backend = self.enterContext(patch("measure_ocr.recognizer.GlmBackend"))
        self.enterContext(patch("shared.glm_backend.GlmBackend", self.backend))
        self.backend.return_value.generate.return_value = (TARGET, 30)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def arguments(self, output):
        return parse_args([
            str(self.page), "--output", str(output), "--mode", "tab",
            "--title", "流程测试", "--artist", "Test", "--device", "cpu", "--layout-source", "geometry",
        ])

    def test_pipeline_and_standalone_stages_produce_same_gp5(self):
        combined = run_pipeline(self.arguments(self.root / "combined"))
        self.assertEqual(combined["status"], "complete")
        self.assertEqual(combined["measures"], 3)
        self.assertEqual(list(combined["stages"]), ["layout", "document_info", "measure_ocr", "gp5_export"])
        for entry in combined["stages"].values():
            self.assertEqual(entry["status"], "complete")
            self.assertTrue(Path(entry["manifest"]).is_file())

        layout = layout_stage.run([self.page], self.root / "layout", mode="tab", layout_source="geometry")
        info = info_stage.run(layout, self.root / "info", title="流程测试", artist="Test")
        recognition = measure_stage.run(layout, info, self.root / "ocr", device="cpu")
        exported = export_stage.run(recognition, self.root / "gp5")
        gp5 = Path(read_result(exported, "gp5_export")["gp5"])
        self.assertEqual(gp5.read_bytes(), Path(combined["gp5"]).read_bytes())
        song = guitarpro.parse(str(gp5), encoding="cp936")
        self.assertEqual(song.title, "流程测试")
        self.assertEqual(song.artist, "Test")
        self.assertEqual(len(song.tracks[0].measures), 3)
        self.assertEqual(song.tracks[0].measures[0].voices[0].beats[1].notes[0].value, 1)

        # The reorganized data pipeline must still consume generated Guitar Pro files.
        self.assertEqual(analyze_source(gp5)["statistics"]["measure_count"], 3)
        corpus = self.root / "corpus"
        corpus.mkdir()
        (corpus / "sample.gp5").write_bytes(gp5.read_bytes())
        data = self.root / "dataset"
        select_sources(corpus, data, source_count=1, seed=42, minimum_measures=1, maximum_measures=10, modes=["tab"])
        self.assertEqual(len(list((data / "labels").glob("*.json"))), 1)
        self.assertEqual(len(list((data / "prepared" / "tab").glob("*.gp5"))), 1)

    def test_resume_reuses_accepted_measures_and_rejects_changed_input(self):
        args = self.arguments(self.root / "resume")
        first = run_pipeline(args)
        gp5 = Path(first["gp5"]).read_bytes()
        self.backend.reset_mock()
        args.resume = True
        repeated = run_pipeline(args)
        self.backend.assert_not_called()
        self.assertEqual(Path(repeated["gp5"]).read_bytes(), gp5)
        args.tuning = [64, 59, 55, 50, 45, 38]
        with self.assertRaisesRegex(ValueError, "inputs or options changed"):
            run_pipeline(args)
        failed = json.loads((args.output / "manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["stages"]["measure_ocr"]["status"], "failed")
        self.assertNotIn("gp5_export", failed["stages"])

    def test_invalid_model_output_requires_review_before_export(self):
        self.backend.return_value.generate.return_value = ("invalid", 5)
        args = self.arguments(self.root / "fallback")
        args.maximum_attempts = 1
        result = run_pipeline(args)
        records = [json.loads(line) for line in Path(result["recognition_log"]).read_text().splitlines()]
        accepted = [row for row in records if row["accepted"]]
        self.assertEqual(len(accepted), 3)
        self.assertTrue(all("fallback_full_measure_rest" in row["deterministic_repairs"] for row in accepted))
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["review_measures"], [1, 2, 3])
        self.assertIsNone(result["gp5"])
        args.allow_unreviewed = True
        args.resume = True
        result = run_pipeline(args)
        self.assertEqual(result["review_measures"], [1, 2, 3])
        song = guitarpro.parse(result["gp5"], encoding="cp936")
        self.assertEqual(song.tracks[0].measures[0].voices[0].beats[0].status.name, "rest")

    def test_image_metadata_and_tempo_reach_measure_sequence_and_gp5(self):
        layout = layout_stage.run([self.page], self.root / "layout", mode="tab", layout_source="geometry")
        source = json.loads(layout.read_text())
        source.update(info_source="image", regions=[{"kind": "header", "image": str(self.page)}])
        layout.write_text(json.dumps(source))
        with patch("document_info.run.recognize_document_info", return_value=({
            "title": "Image title", "tempo_quarter": 88,
            "tuning_midi_high_to_low": [64, 59, 55, 50, 45, 38],
        }, [])):
            info = info_stage.run(layout, self.root / "info")
        recognition = measure_stage.run(layout, info, self.root / "ocr", device="cpu")
        result = read_result(recognition, "measure_ocr")
        first = Path(result["m2"]).read_text().splitlines()[0]
        self.assertEqual(parse_measure_target(first)["tempo_quarter"], 88)
        exported = export_stage.run(recognition, self.root / "gp5")
        song = guitarpro.parse(read_result(exported, "gp5_export")["gp5"], encoding="cp936")
        self.assertEqual(song.title, "Image title")
        self.assertEqual(song.tempo, 88)
        self.assertEqual(song.tracks[0].strings[-1].value, 38)

    def test_document_info_backend_reads_header_and_tempo(self):
        with patch("document_info.image_ocr.GlmBackend") as backend:
            backend.return_value.generate.side_effect = [
                ('{"title":"Image title","artist":null,"tuning_name":"Drop D"}', 12),
                ('{"tempo_quarter":88}', 6),
            ]
            metadata, predictions = recognize_document_info([
                {"kind": "header", "image": str(self.page)},
                {"kind": "tempo", "image": str(self.page)},
            ], self.root, self.root, "cpu")
        self.assertEqual(metadata["tempo_quarter"], 88)
        self.assertEqual(metadata["tuning_midi_high_to_low"][-1], 38)
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
