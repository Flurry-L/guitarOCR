from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import package_release


class ReleasePackageTest(unittest.TestCase):
    def test_release_keeps_notices_and_excludes_local_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "pyproject.toml": '[project]\nversion = "0.1.0"\n',
                "weights/manifest.json": json.dumps({"models": []}),
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
            with patch.object(package_release, "ROOT", root):
                destination = package_release.build(root / "output")
            with ZipFile(destination) as archive:
                names = {
                    name.removeprefix("GuitarOCR-0.1.0/") for name in archive.namelist()
                }
                expected = {
                    name for name, data in payloads.items() if data != "PRIVATE"
                }
                self.assertEqual(names, expected | {"SHA256SUMS", "release.json"})
                self.assertEqual(json.loads(archive.read("GuitarOCR-0.1.0/release.json"))["version"], "0.1.0")
                self.assertEqual(
                    archive.read("GuitarOCR-0.1.0/start.bat"), b"@echo off\r\n"
                )
                for row in (
                    archive.read("GuitarOCR-0.1.0/SHA256SUMS").decode().splitlines()
                ):
                    digest, name = row.split("  ", 1)
                    self.assertEqual(
                        sha256(archive.read(f"GuitarOCR-0.1.0/{name}")).hexdigest(),
                        digest,
                    )

    def test_release_includes_verified_default_weights_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "weights/current/model.safetensors"
            model.parent.mkdir(parents=True)
            data = b"actual model bytes"
            model.write_bytes(data)
            (root / "pyproject.toml").write_text('[project]\nversion="0.1.0"\n')
            manifest = {"models": [{"path": "weights/current", "files": [{
                "name": model.name, "bytes": len(data), "sha256": sha256(data).hexdigest(),
            }]}]}
            (root / "weights/manifest.json").write_text(json.dumps(manifest))
            with patch.object(package_release, "ROOT", root):
                destination = package_release.build(root / "output")
                with ZipFile(destination) as archive:
                    self.assertEqual(archive.read("GuitarOCR-0.1.0/weights/current/model.safetensors"), data)
                model.write_bytes(b"x" * len(data))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    package_release.build(root / "output")


if __name__ == "__main__":
    unittest.main()
