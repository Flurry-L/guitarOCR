import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import package_release


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
                "weights/distribution.json": json.dumps({"release": "v0.1.0", "files": {"auxiliary": []}}),
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



if __name__ == "__main__":
    unittest.main()
