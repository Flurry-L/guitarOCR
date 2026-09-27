from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.downloads import acquire_base_model, download_verified
from scripts import launcher


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = b"known model content" * 100000
        self.expected = {"name": "model.bin", "bytes": len(self.data),
                         "sha256": sha256(self.data).hexdigest()}
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                requested = self.headers.get("Range")
                previous = sum(path == self.path for path, _ in owner.requests)
                owner.requests.append((self.path, requested))
                body = owner.data
                if self.path.startswith("/missing"):
                    self.send_error(503)
                    return
                if self.path.startswith("/bad"):
                    body = b"x" * len(body)
                offset = int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
                if self.path.startswith("/ignore"):
                    offset = 0
                self.send_response(206 if offset else 200)
                self.send_header("Content-Length", str(len(body) - offset))
                if offset:
                    self.send_header("Content-Range", f"bytes {offset}-{len(body)-1}/{len(body)}")
                self.end_headers()
                if self.path.startswith(("/resume", "/ignore")) and not previous:
                    self.wfile.write(body[:1024 * 1024])
                else:
                    self.wfile.write(body[offset:])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def test_interrupted_transfer_resumes_and_handles_ignored_range(self):
        for mode in ("resume", "ignore"):
            with self.subTest(mode=mode), patch("scripts.downloads.time.sleep"):
                destination = self.root / mode
                download_verified(f"{self.url}/{mode}", destination, self.expected)
                self.assertEqual(destination.read_bytes(), self.data)
                self.assertIn((f"/{mode}", "bytes=1048576-"), self.requests)

    def test_corruption_never_replaces_a_valid_destination(self):
        destination = self.root / "model.bin"
        destination.write_bytes(self.data)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            download_verified(f"{self.url}/bad", destination, self.expected, attempts=1)
        self.assertEqual(destination.read_bytes(), self.data)
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_missing_or_corrupt_mirror_falls_back_and_cached_model_needs_no_network(self):
        for failure in ("bad", "missing"):
            with self.subTest(failure=failure), patch("scripts.downloads.time.sleep"), patch.dict(os.environ, {}, clear=True):
                base = {"repo_id": "unused/repo", "revision": "fixed", "path": failure,
                        "files": [self.expected], "download_sources": [f"{self.url}/{failure}", f"{self.url}/good"]}
                acquire_base_model(base, self.root)
                self.assertEqual((self.root / failure / "model.bin").read_bytes(), self.data)
                with patch("scripts.downloads.urlopen", side_effect=AssertionError("Cache should avoid network")):
                    acquire_base_model(base, self.root)

    def test_user_model_endpoint_precedes_default_sources(self):
        base = {"repo_id": "unused/repo", "revision": "fixed", "path": "base",
                "files": [self.expected], "download_sources": [f"{self.url}/missing"]}
        with patch.dict(os.environ, {"HF_ENDPOINT": f"{self.url}/good"}):
            acquire_base_model(base, self.root)
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(self.requests[0][0].startswith("/good/unused/repo/resolve/fixed/"))


class PackageSourceTest(unittest.TestCase):
    def test_torch_mirror_is_used_and_official_retry_restores_backend(self):
        arguments = ["pip", "install", "--torch-backend", "cpu", "torch==2.14.0"]
        with patch.object(launcher, "has_uv_config", return_value=False), patch.dict(os.environ, {"UV_TORCH_BACKEND": "cu130"}), patch.object(launcher, "run") as run:
            run.side_effect = [subprocess.CalledProcessError(1, "uv"), None]
            launcher.run_uv("uv", arguments)
        first, fallback = run.call_args_list
        self.assertNotIn("--torch-backend", first.args[0])
        self.assertIn(launcher.TORCH_MIRROR + "/cpu/", first.args[0])
        self.assertEqual(first.kwargs["env"]["UV_DEFAULT_INDEX"], launcher.PYPI_MIRROR)
        self.assertNotIn("UV_TORCH_BACKEND", first.kwargs["env"])
        self.assertIn("--torch-backend", fallback.args[0])
        self.assertIn("--no-config", fallback.args[0])
        self.assertNotIn("UV_DEFAULT_INDEX", fallback.kwargs["env"])

    def test_explicit_user_index_is_preserved(self):
        with patch.dict(os.environ, {"UV_DEFAULT_INDEX": "https://user.example/simple"}), patch.object(launcher, "run") as run:
            launcher.run_uv("uv", ["pip", "install", "example"])
        self.assertEqual(run.call_args.kwargs["env"]["UV_DEFAULT_INDEX"], "https://user.example/simple")



if __name__ == "__main__":
    unittest.main()
