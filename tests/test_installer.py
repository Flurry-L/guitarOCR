from hashlib import sha256
from argparse import Namespace
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from zipfile import ZipFile

from scripts import launcher
from scripts.launcher import (
    acquire_weights,
    choose_device,
    installation_lock,
    launch,
    torch_reinstall_args,
)
from shared.environment import verify_files


class InstallerTest(unittest.TestCase):
    def test_release_repairs_model_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = "a" * 40
            (root / "release.json").write_text(json.dumps({
                "repository": "Flurry-L/guitarOCR", "commit": commit,
            }))
            contents = {"adapter_config.json": b"{}\n", "adapter_model.safetensors": b"model bytes"}
            files = [{"name": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}
                     for name, data in contents.items()]
            urls = []

            def response(request, **kwargs):
                urls.append(request.full_url)
                return io.BytesIO(contents[request.full_url.rsplit("/", 1)[1]])

            with (
                patch.object(launcher, "ROOT", root),
                patch.object(launcher, "run", side_effect=AssertionError("Git must not run")),
                patch("scripts.launcher.urllib.request.urlopen", side_effect=response),
            ):
                acquire_weights({"models": [{"path": "weights/adapter", "files": files}]})
            self.assertEqual(verify_files(root / "weights/adapter", files), [])
            self.assertTrue(any("raw.githubusercontent.com" in u for u in urls))
            self.assertTrue(any("media.githubusercontent.com" in u for u in urls))

    @unittest.skipUnless(shutil.which("uv"), "uv executable required")
    def test_available_mirror_is_used_for_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror = root / "mirror"
            index = mirror / "guitarocr-installer-probe"
            index.mkdir(parents=True)
            wheel = mirror / "guitarocr_installer_probe-1.0-py3-none-any.whl"
            metadata = "guitarocr_installer_probe-1.0.dist-info"
            with ZipFile(wheel, "w") as archive:
                archive.writestr("guitarocr_installer_probe.py", "value = 'mirror'\n")
                archive.writestr(
                    f"{metadata}/METADATA",
                    "Metadata-Version: 2.1\nName: guitarocr-installer-probe\nVersion: 1.0\n",
                )
                archive.writestr(
                    f"{metadata}/WHEEL",
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                )
                archive.writestr(f"{metadata}/RECORD", "")
            (index / "index.html").write_text(
                f'<a href="../{wheel.name}">{wheel.name}</a>', encoding="utf-8"
            )
            environment = root / "environment"
            with (
                patch.object(launcher, "ROOT", root),
                patch.dict(
                    os.environ, {"UV_DEFAULT_INDEX": mirror.as_uri(), "UV_OFFLINE": "1"}
                ),
            ):
                launcher.run_uv(
                    shutil.which("uv"),
                    ["venv", "--python", sys.executable, environment],
                )
                python = launcher.environment_python(environment)
                launcher.run_uv(
                    shutil.which("uv"),
                    [
                        "pip",
                        "install",
                        "--python",
                        python,
                        "guitarocr-installer-probe==1.0",
                    ],
                )
            subprocess.run(
                [
                    python,
                    "-c",
                    "import guitarocr_installer_probe; assert guitarocr_installer_probe.value == 'mirror'",
                ],
                check=True,
            )

    @unittest.skipUnless(shutil.which("uv"), "uv executable required")
    def test_uv_retry_creates_environment_without_changing_user_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "user-uv.toml"
            config.write_text("unsupported-old-setting = true\n", encoding="utf-8")
            original = config.read_bytes()
            environment = root / "environment"
            with (
                patch.object(launcher, "ROOT", root),
                patch.dict(os.environ, {"UV_CONFIG_FILE": str(config)}),
            ):
                launcher.run_uv(
                    shutil.which("uv"),
                    ["venv", "--python", sys.executable, environment],
                )
                self.assertEqual(os.environ["UV_CONFIG_FILE"], str(config))
            self.assertEqual(config.read_bytes(), original)
            python = launcher.environment_python(environment)
            self.assertTrue(python.is_file())
            subprocess.run(
                [python, "-c", "import sys; assert sys.prefix != sys.base_prefix"],
                check=True,
            )

    @unittest.skipUnless(shutil.which("uv"), "uv executable required")
    def test_export_uses_shipped_lock_with_custom_uv_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("pyproject.toml", "uv.lock", "README.md"):
                shutil.copy2(launcher.ROOT / name, root / name)
            lock = (root / "uv.lock").read_bytes()
            (root / "uv.toml").write_text(
                'index-url = "https://packages.invalid/simple"\n', encoding="utf-8"
            )
            with (
                patch.object(launcher, "ROOT", root),
                patch.dict(os.environ, {"UV_OFFLINE": "1", "UV_PYTHON": "3.12"}),
            ):
                for device in ("cpu", "cuda"):
                    with self.subTest(device=device):
                        requirements = launcher.export_requirements(
                            shutil.which("uv"), Path(sys.executable), device
                        )
                        self.assertIn("transformers==5.8.0", requirements)
                        self.assertIn("peft==0.18.1", requirements)
                        backend = "cpu" if device == "cpu" else "cu130"
                        self.assertIn(f"torch==2.14.0+{backend}", requirements)
                        self.assertIn(f"torchvision==0.29.0+{backend}", requirements)
                        self.assertEqual((root / "uv.lock").read_bytes(), lock)
                        gpu_dependencies = [
                            line
                            for line in requirements.splitlines()
                            if line.startswith(("nvidia-", "cuda-", "triton=="))
                        ]
                        self.assertEqual(bool(gpu_dependencies), device == "cuda")

    def test_windows_checkout_preserves_model_config_checksum(self):
        config = Path("weights/measure_ocr/adapter_config.json")
        original = (launcher.ROOT / config).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gitattributes").write_bytes(
                (launcher.ROOT / ".gitattributes").read_bytes()
            )
            (root / config).parent.mkdir(parents=True)
            (root / config).write_bytes(original)
            for arguments in (["init", "--quiet"], ["add", "."]):
                subprocess.run(
                    ["git", "-c", "core.autocrlf=true", *arguments],
                    cwd=root,
                    check=True,
                    capture_output=True,
                )
            (root / config).unlink()
            subprocess.run(
                ["git", "-c", "core.autocrlf=true", "checkout-index", "--all"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            self.assertEqual((root / config).read_bytes(), original)

    def test_repair_downloads_regular_files_and_lfs_weights(self):
        commit = "a" * 40
        folder = "weights/adapter"
        config, weights = b'{"model": "test"}\n', b"complete model"
        contents = {"adapter_config.json": config, "adapter_model.safetensors": weights}
        files = [
            {"name": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in contents.items()
        ]
        git_output = {
            (
                "git",
                "remote",
                "get-url",
                "origin",
            ): "https://github.com/Flurry-L/guitarOCR.git",
            ("git", "rev-parse", "HEAD"): commit,
        }
        urls = {}
        for name, data in contents.items():
            path = f"{folder}/{name}"
            lfs = name.endswith(".safetensors")
            git_output[("git", "check-attr", "-z", "filter", "--", path)] = (
                f"{path}\0filter\0{'lfs' if lfs else 'unspecified'}\0"
            )
            host = (
                "media.githubusercontent.com/media"
                if lfs
                else "raw.githubusercontent.com"
            )
            urls[f"https://{host}/Flurry-L/guitarOCR/{commit}/{path}"] = data

        def response(request, **kwargs):
            if request.full_url not in urls:
                raise HTTPError(request.full_url, 404, "Not Found", {}, None)
            return io.BytesIO(urls[request.full_url])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / folder
            model.mkdir(parents=True)
            (model / "adapter_config.json").write_bytes(config.replace(b"\n", b"\r\n"))
            (model / "adapter_model.safetensors").write_bytes(
                b"version https://git-lfs.github.com/spec/v1\n"
            )
            with (
                patch.object(launcher, "ROOT", root),
                patch.object(
                    launcher,
                    "run",
                    side_effect=lambda command, **kw: git_output[tuple(command)],
                ),
                patch("scripts.launcher.urllib.request.urlopen", side_effect=response),
                patch("scripts.launcher.time.sleep"),
            ):
                acquire_weights({"models": [{"path": folder, "files": files}]})
            self.assertEqual(verify_files(model, files), [])

    def test_lfs_pointer_and_same_size_corruption_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.safetensors"
            content = b"complete model"
            files = [
                {
                    "name": path.name,
                    "bytes": len(content),
                    "sha256": sha256(content).hexdigest(),
                }
            ]
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            self.assertIn("LFS", verify_files(root, files)[0])
            path.write_bytes(b"x" * len(content))
            self.assertIn("SHA-256", verify_files(root, files)[0])
            path.write_bytes(content)
            self.assertEqual(verify_files(root, files), [])

    def test_auto_device_handles_old_drivers_and_no_gpu(self):
        with patch("scripts.launcher.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "580.95.05\n"
            self.assertEqual(choose_device("auto"), "cuda")
            run.return_value.stdout = "535.10\n"
            self.assertEqual(choose_device("auto"), "cpu")
            run.side_effect = FileNotFoundError
            self.assertEqual(choose_device("auto"), "cpu")
            self.assertEqual(choose_device("cuda"), "cuda")

    def test_installation_lock_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "interrupted"):
                with installation_lock(root):
                    raise ValueError("interrupted")
            with installation_lock(root):
                pass

    def test_switching_device_replaces_compatible_version_cpu_wheel(self):
        with patch("scripts.launcher.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "cpu\n"
            self.assertIn("torch", torch_reinstall_args(Path("python"), "cuda"))
            self.assertEqual(torch_reinstall_args(Path("python"), "cpu"), [])
            run.return_value.stdout = "13.0\n"
            self.assertEqual(torch_reinstall_args(Path("python"), "cuda"), [])
            self.assertIn("torch", torch_reinstall_args(Path("python"), "cpu"))

    @unittest.skipIf(os.name == "nt", "POSIX socket reuse semantics")
    def test_restart_after_connections_close_still_rejects_active_server(self):
        state = {"device": "cpu", "model": "model", "layout_python": "python"}
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            args = Namespace(
                port=server.getsockname()[1], no_browser=True, output="output"
            )
            with self.assertRaisesRegex(ValueError, "已被占用"):
                launch(args, Path("tools"), state)
            with socket.create_connection(server.getsockname(), timeout=5) as client:
                connection, _ = server.accept()
                with connection:
                    connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
        with patch("scripts.launcher.run") as run:
            launch(args, Path("tools"), state)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
