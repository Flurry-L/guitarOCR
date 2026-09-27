"""Desktop's editor environment stays useful without inference dependencies."""
from pathlib import Path
import tempfile
import subprocess
import sys
import os
import shutil
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from scripts.desktop_runtime import install_edit
from webapp.app import create_app
from pipeline.workspace import Workspace


class DesktopTest(unittest.TestCase):
    def test_editor_mode_serves_score_assets_and_refuses_model_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Workspace(Path(directory)), inference_enabled=False),
                            base_url='http://127.0.0.1') as client:
                self.assertFalse(client.get('/api/config').json()['inference_enabled'])
                self.assertEqual(client.get('/static/vendor/vexflow.js').status_code, 200)
                for action in ('detect', 'information', 'recognize'):
                    result = client.post(f'/api/sessions/unused/{action}', json={})
                    self.assertEqual(result.status_code, 409, result.text)
                    self.assertIn('校对模式', result.text)

    def test_packaged_backend_starts_without_source_checkout(self):
        from scripts.prepare_desktop import ROOT, backend_files
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "backend"
            for source in backend_files(ROOT):
                destination = bundle / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            # -S and PYTHONPATH prevent an editable install from hiding missing files.
            site_packages = str(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
            result = subprocess.run([sys.executable, "-S", "-c", """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import desktop_runtime
from pipeline.workspace import Workspace
from webapp.app import create_app
from shared.defaults import MODEL
app = create_app(Workspace(Path(sys.argv[2])), inference_enabled=False)
assert Path('webapp/static/metadata-editor.js').is_file()
assert MODEL == Path.cwd() / 'tools/models/GLM-OCR'
assert 'torch' not in sys.modules and 'paddle' not in sys.modules
""", site_packages, str(Path(directory) / "projects")], cwd=bundle,
                env={**os.environ, "PYTHONPATH": str(bundle)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_edit_install_uses_locked_web_dependencies_and_reuses_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'uv.lock').write_text('test lock')
            tools = root / 'tools'
            tools.mkdir()
            with patch('scripts.desktop_runtime.ROOT', root), patch('scripts.launcher.run_uv', return_value='fastapi==0.1\n') as run:
                python = install_edit('uv', tools)
                exported = next(c.args[1] for c in run.call_args_list if 'export' in c.args[1])
                self.assertIn('--frozen', exported)
                self.assertNotIn('glm-ocr', exported)
                self.assertIn('webui', exported)
                python.parent.mkdir(parents=True)
                python.touch()
                run.reset_mock()
                self.assertEqual(install_edit('uv', tools), python)
                run.assert_not_called()
