import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import guitarpro

from pipeline.archive import import_project
from pipeline.workspace import Workspace
from scripts.create_demo import create_demo
from shared.artifacts import read_result
from shared.glm_backend import create_backend
from webapp.views import project_view


class SyntheticDemoTest(unittest.TestCase):
    def test_preset_roundtrip_and_two_track_exports_without_model(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            'shared.glm_backend.create_backend', side_effect=AssertionError('Demo must never run OCR')
        ):
            root = Path(directory)
            _, archive = create_demo(root / 'demo')
            workspace = Workspace(root / 'restored', device='cpu')
            self.addCleanup(workspace.close)
            sid = import_project(workspace, archive)
            state = workspace.load(sid)
            view = project_view(workspace, sid)
            self.assertEqual(len(view['metadata']['parts']), 2)
            self.assertEqual(len(view['measures']), 16)
            result = read_result(Path(state['recognition']), 'measure_ocr')
            self.assertFalse(result['provenance']['ocr_executed'])
            self.assertEqual(result['provenance']['license'], 'CC0-1.0')
            for row in view['measures']:
                workspace.correct(sid, row['measure_number'], reviewed=True)
            exported = workspace.export(sid)
            manifest = read_result(Path(exported['export']), 'gp5_export')
            song = guitarpro.parse(manifest['gp5'], encoding='cp936')
            self.assertEqual(len(song.tracks), 2)
            self.assertTrue(all(len(track.measures) == 8 for track in song.tracks))
            self.assertIn('<score-partwise', Path(manifest['musicxml']).read_text())
            score = json.loads(Path(result['score_document']).read_text())
            self.assertEqual(len(score['timeline']), 8)

    def test_metal_never_falls_through_to_torch(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'GUITAROCR_BACKEND': 'transformers'}):
            with self.assertRaisesRegex(ValueError, 'llama.cpp'):
                create_backend(Path(directory), None, 'metal')


if __name__ == '__main__':
    unittest.main()
