from hashlib import sha256
from argparse import Namespace
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from scripts.launcher import (
    choose_device,
    installation_lock,
    launch,
    torch_reinstall_args,
)
from shared.environment import verify_files


class InstallerTest(unittest.TestCase):
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
