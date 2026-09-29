from hashlib import sha256
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import package_release
from scripts.model_bundle import restore_bundle


class ReleasePackageTest(unittest.TestCase):
    def init_source(self, root):
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', 'add', '.'], cwd=root, check=True)
        subprocess.run(['git', '-c', 'user.name=Package Fixture', '-c', 'user.email=fixture@localhost',
                        'commit', '-qm', 'Package fixture'], cwd=root, check=True)

    def test_release_keeps_notices_and_excludes_local_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "pyproject.toml": '[project]\nversion = "0.1.0"\n',
                "weights/manifest.json": json.dumps({"models": [], "base_model": {"path": "tools/models/base", "files": []}}),
                "LICENSE": "Owner-supplied test license",
                "NOTICE.txt": "Upstream notice",
                "THIRD_PARTY_NOTICES.md": "Dependency notices",
                "scripts/.env": "PRIVATE",
                "scripts/.env.local": "PRIVATE",
                "scripts/private.key": "PRIVATE",
                "scripts/private.pem": "PRIVATE",
                "scripts/.venv/secret.txt": "PRIVATE",
                "scripts/.env.example": "EXAMPLE=",
                "start.bat": "@echo off\n",
                "使用说明.txt": "解压后运行 start.bat。\n",
                "weights/old/adapter_model.safetensors": "PRIVATE",
            }
            for name, contents in payloads.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            self.init_source(root)
            with patch.object(package_release, "ROOT", root):
                destination = package_release.build(root / "output")
            with ZipFile(destination) as archive:
                names = {
                    name.removeprefix("GuitarOCR-0.1.0/") for name in archive.namelist()
                }
                expected = {
                    name for name, data in payloads.items() if data != "PRIVATE"
                }
                self.assertEqual(names, expected | {"release.json"})
                self.assertEqual(json.loads(archive.read("GuitarOCR-0.1.0/release.json"))["version"], "0.1.0")
                self.assertEqual(
                    archive.read("GuitarOCR-0.1.0/start.bat"), b"@echo off\r\n"
                )

    def test_release_includes_models_and_reuses_installed_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "weights/current/model.safetensors"
            model.parent.mkdir(parents=True)
            data = b"actual model bytes"
            model.write_bytes(data)
            base = root / 'tools/models/base/model.safetensors'
            base.parent.mkdir(parents=True)
            base.write_bytes(b'complete base model')
            (root / "pyproject.toml").write_text('[project]\nversion="0.1.0"\n')
            manifest = {"models": [{"path": "weights/current", "files": [{
                "name": model.name, "bytes": len(data), "sha256": sha256(data).hexdigest(),
            }]}], 'base_model': {'path': 'tools/models/base', 'files': [{
                'name': base.name, 'bytes': base.stat().st_size, 'sha256': sha256(base.read_bytes()).hexdigest(),
            }]}}
            (root / "weights/manifest.json").write_text(json.dumps(manifest))
            self.init_source(root)
            with patch.object(package_release, "ROOT", root):
                destination = package_release.build(root / "output", bundle_models=True)
                with ZipFile(destination) as archive:
                    bundled = root / 'bundle.tar.xz'
                    bundled.write_bytes(archive.read('GuitarOCR-0.1.0/models.tar.xz'))
                installed = root / 'installed'
                with (patch('urllib.request.urlopen', side_effect=AssertionError('No model downloads allowed')),
                      patch('scripts.model_bundle.verify_files', side_effect=AssertionError('Runtime must not hash every model'))):
                    restore_bundle(bundled, installed, manifest)
                    self.assertEqual((installed / model.relative_to(root)).read_bytes(), data)
                    self.assertEqual((installed / base.relative_to(root)).read_bytes(), base.read_bytes())
                    # Reuse valid files without even opening the archive.
                    restore_bundle(root / 'absent.tar.xz', installed, manifest)
                    missing = installed / base.relative_to(root)
                    missing.unlink()
                    restore_bundle(bundled, installed, manifest)
                    self.assertEqual(missing.read_bytes(), base.read_bytes())
                    # A changed model version reinstalls from the bundle even
                    # when the old files have the same names and lengths.
                    missing.write_bytes(b'x' * base.stat().st_size)
                    updated = {**manifest, 'base_model': {**manifest['base_model'], 'revision': 'updated'}}
                    restore_bundle(bundled, installed, updated)
                    self.assertEqual(missing.read_bytes(), base.read_bytes())
                # A cached bundle is already complete; only a new bundle checks
                # source models. Removing it simulates a fresh build machine.
                for cached in (root / 'tools/model-bundles').glob('*.tar.xz'):
                    cached.unlink()
                base.unlink()
                with self.assertRaisesRegex(ValueError, '缺少文件'):
                    package_release.build(root / 'output')
                base.write_bytes(b'complete base model')
                model.write_bytes(b"x" * len(data))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    package_release.build(root / "output")


if __name__ == "__main__":
    unittest.main()
