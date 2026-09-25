import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import guitarpro
import pymupdf
from PIL import Image
from fastapi.testclient import TestClient

from layout.pages import expand_inputs
from layout.edit import save_layout
from shared.glm_backend import BackendPool
from webapp.app import create_app
from webapp.workflow import Workflow

TARGET = "M2 time=4/4 | V0{@0:q:s1f0 @960:q:s2f1 @1920:h:r}"


class VersionedClient(TestClient):
    """Normal clients send the revision they last read; tests can override it."""

    def __init__(self, app, **kwargs):
        super().__init__(app, base_url="http://127.0.0.1", **kwargs)

    def request(self, method, url, **kwargs):
        if method.upper() in {"PUT", "POST", "DELETE"} and str(url).startswith(
            "/api/sessions/"
        ):
            prefix = "/".join(str(url).split("/")[:4])
            current = super().request("GET", prefix)
            if current.status_code == 200 and "revision" in current.json():
                kwargs["headers"] = {
                    "If-Match": str(current.json()["revision"]),
                    **(kwargs.get("headers") or {}),
                }
        return super().request(method, url, **kwargs)


class WebWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.page = self.root / "input.png"
        Image.new("RGB", (300, 200), "white").save(self.page)
        self.workflow = Workflow(self.root / "sessions", device="cpu")
        self.client = self.enterContext(VersionedClient(create_app(self.workflow)))
        self.backend = self.enterContext(patch("shared.glm_backend.GlmBackend"))
        self.backend.return_value.generate.return_value = (TARGET, 30)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def wait(self, sid):
        for _ in range(150):
            state = self.client.get(f"/api/sessions/{sid}").json()
            if state.get("job", {}).get("status") not in {"queued", "running"}:
                self.assertNotEqual(state.get("job", {}).get("status"), "failed", state)
                return state
            time.sleep(0.01)
        self.fail("Background task did not complete")

    def upload(self):
        response = self.client.post(
            "/api/sessions",
            files=[("files", ("../unsafe.png", self.page.read_bytes(), "image/png"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        sid = response.json()["id"]
        return sid, self.wait(sid)

    def prepare(self):
        sid, state = self.upload()
        self.assertEqual(state["pages"][0]["width"], 300)
        result = self.client.put(
            f"/api/sessions/{sid}/boxes",
            json={
                "mode": "tab",
                "boxes": [
                    {"page": 1, "kind": "measure", "bbox": [20, 80, 130, 60]},
                    {"page": 1, "kind": "measure", "bbox": [160, 80, 120, 60]},
                ],
            },
        )
        self.assertEqual(result.status_code, 200, result.text)
        result = self.client.put(
            f"/api/sessions/{sid}/metadata",
            json={
                "title": "网页测试",
                "artist": "Someone",
                "tempo_quarter": 92,
                "capo": 2,
                "tuning_used": [64, 59, 55, 50, 45, 38],
            },
        )
        self.assertEqual(result.status_code, 200, result.text)
        self.client.post(f"/api/sessions/{sid}/recognize")
        return sid, self.wait(sid)

    def test_upload_edit_export_and_invalidation(self):
        sid, state = self.prepare()
        prefix = f"/api/sessions/{sid}"
        self.assertEqual(len(state["measures"]), 2)
        self.assertEqual(state["measures"][0]["parsed"]["tempo_quarter"], 92)
        edited = self.client.put(
            prefix + "/measures/1",
            json={"target": TARGET.replace("s1f0", "s1f7"), "reviewed": True},
        )
        self.assertEqual(edited.status_code, 200)
        state = self.client.post(prefix + "/export").json()
        data = self.client.get(state["gp5_url"]).content
        gp = self.root / "result.gp5"
        gp.write_bytes(data)
        song = guitarpro.parse(str(gp), encoding="cp936")
        self.assertEqual(song.title, "网页测试")
        self.assertEqual(
            song.tracks[0].measures[0].voices[0].beats[0].notes[0].value, 7
        )
        self.assertEqual(song.tracks[0].strings[-1].value, 38)
        self.assertEqual(song.tracks[0].offset, 2)
        restored = Workflow(self.workflow.root, device="cpu").public(sid)
        self.assertEqual(restored["gp5_url"], state["gp5_url"])
        # A bad edit must leave the usable project and export intact.
        response = self.client.put(
            prefix + "/measures/1", json={"target": "broken", "reviewed": True}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(prefix).json()["export"], state["export"])
        # Changing a valid upstream region invalidates all dependent artifacts.
        state["boxes"][0]["bbox"] = [10, 60, 130, 75]
        response = self.client.put(
            prefix + "/boxes", json={"boxes": state["boxes"], "mode": "tab"}
        )
        changed = response.json()
        self.assertIsNone(changed["info"])
        self.assertIsNone(changed["recognition"])
        self.assertIsNone(changed["export"])
        self.assertEqual(self.client.post(prefix + "/export").status_code, 400)
        layout = json.loads(Path(changed["layout"]).read_text())
        with Image.open(layout["records"][0]["image"]) as crop:
            self.assertGreater(crop.height, 75)

    def test_structured_edit_preserves_playback_and_effects(self):
        self.backend.return_value.generate.return_value = (
            "M2 time=4/4 | V0{@0:q:s1f3(vib)<dyn:70> @960:h.:r}",
            20,
        )
        sid, state = self.prepare()
        measure = state["measures"][0]["parsed"]
        measure["voices"][0]["events"][0]["notes"][0]["fret"] = 7
        response = self.client.put(
            f"/api/sessions/{sid}/measures/1",
            json={"measure": measure, "reviewed": True},
        )
        self.assertEqual(response.status_code, 200, response.text)
        target = response.json()["measures"][0]["target"]
        self.assertIn("s1f7(vib)", target)
        self.assertIn("dyn:70", target)
        self.assertEqual(
            self.client.put(
                f"/api/sessions/{sid}/measures/1", json={"measure": {"voices": [{}]}}
            ).status_code,
            400,
        )

    def test_fallback_requires_individual_review(self):
        self.backend.return_value.generate.return_value = ("broken", 5)
        sid, state = self.prepare()
        prefix = f"/api/sessions/{sid}"
        self.assertEqual(state["review_measures"], [1, 2])
        self.assertEqual(self.client.post(prefix + "/export").status_code, 400)
        for i in (1, 2):
            response = self.client.put(
                prefix + f"/measures/{i}", json={"target": TARGET, "reviewed": True}
            )
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.post(prefix + "/export").status_code, 200)

    def test_session_paths_and_cross_origin_mutation(self):
        sid, state = self.upload()
        other, _ = self.upload()
        root = self.workflow.directory(sid)
        (root / "leak.png").symlink_to(self.page)
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/files/leak.png").status_code, 404
        )
        self.assertEqual(
            self.client.get(
                f"/api/sessions/{other}/files/"
                + state["pages"][0]["url"].split("/files/")[1]
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(
                f"/api/sessions/{sid}", headers={"Origin": "https://evil.example"}
            ).status_code,
            403,
        )
        self.assertFalse((self.workflow.root / "unsafe.png").exists())
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").status_code, 404)

    def test_multipage_pdf_keeps_filename_and_all_pages(self):
        with pymupdf.open() as pdf:
            for i in range(4):
                page = pdf.new_page(width=160 + i * 10, height=120)
                page.insert_text((10, 30), f"Page {i + 1}")
            contents = pdf.tobytes()
        response = self.client.post(
            "/api/sessions",
            files=[("files", ("test.pdf", contents, "application/pdf"))],
        )
        self.assertEqual(response.status_code, 200)
        sid = response.json()["id"]
        state = self.wait(sid)
        self.assertEqual(state["input_names"], ["test.pdf"])
        self.assertEqual([p["pdf_page"] for p in state["pages"]], [1, 2, 3, 4])
        page_bytes = [self.client.get(p["url"]).content for p in state["pages"]]
        self.assertEqual(len(set(page_bytes)), 4)
        self.assertIn("test.pdf", self.client.get("/api/sessions").json()[0]["title"])
        response = self.client.put(
            f"/api/sessions/{sid}/boxes",
            json={
                "mode": "tab",
                "boxes": [
                    {"page": 1, "kind": "measure", "bbox": [10, 10, 100, 80]},
                    {"page": 4, "kind": "measure", "bbox": [20, 20, 120, 80]},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        layout = json.loads(Path(response.json()["layout"]).read_text())
        self.assertEqual([r["pdf_page"] for r in layout["records"]], [1, 4])
        self.assertEqual(len(Workflow(self.workflow.root).public(sid)["pages"]), 4)

    def test_host_validation_blocks_rebinding_and_preserves_local_access(self):
        for host in ("127.0.0.1:7860", "localhost:7860", "[::1]:7860"):
            with self.subTest(host=host):
                self.assertEqual(
                    self.client.get(
                        "/api/sessions", headers={"Host": host}
                    ).status_code,
                    200,
                )
        for host in ("attacker.example", "localhost.evil.example", "evil@localhost"):
            with self.subTest(host=host):
                headers = {"Host": host, "Origin": f"http://{host}"}
                self.assertEqual(
                    self.client.get("/api/sessions", headers=headers).status_code, 400
                )
                self.assertEqual(
                    self.client.post("/api/sessions", headers=headers).status_code, 400
                )
        self.assertEqual(
            self.client.post(
                "/api/sessions", headers={"Origin": "https://127.0.0.1"}
            ).status_code,
            403,
        )
        with TestClient(
            create_app(self.workflow, allowed_hosts=["scores.example"]),
            base_url="http://scores.example",
        ) as client:
            self.assertEqual(client.get("/api/config").status_code, 200)

    def test_busy_project_rejects_edits(self):
        sid, _ = self.upload()
        entered, release = threading.Event(), threading.Event()

        def detect(*args):
            entered.set()
            release.wait(5)

        with patch.object(self.workflow, "detect", side_effect=detect):
            self.client.post(f"/api/sessions/{sid}/detect", json={})
            self.assertTrue(entered.wait(2))
            response = self.client.put(
                f"/api/sessions/{sid}/boxes", json={"boxes": [], "mode": "tab"}
            )
            self.assertEqual(response.status_code, 409)
            release.set()
            self.wait(sid)

    def test_session_response_keeps_import_job_snapshot(self):
        entered, release = threading.Event(), threading.Event()
        create = self.workflow.create

        def paused_create(*args, **kwargs):
            entered.set()
            release.wait(5)
            return create(*args, **kwargs)

        session = next(
            route.endpoint
            for route in self.client.app.routes
            if route.path == "/api/sessions/{sid}" and "GET" in route.methods
        )
        with patch.object(self.workflow, "create", side_effect=paused_create):
            response = self.client.post(
                "/api/sessions", files={"files": ("input.png", self.page.read_bytes())}
            )
            self.assertEqual(response.status_code, 200)
            sid = response.json()["id"]
            try:
                self.assertTrue(entered.wait(2))
                # FastAPI serializes this value after the handler releases its lock.
                snapshot = session(sid)
                self.assertNotIn("pages", snapshot)
                self.assertEqual(snapshot["job"]["status"], "running")
            finally:
                release.set()
            completed = self.wait(sid)
        self.assertEqual(completed["job"]["status"], "complete")
        self.assertEqual(len(completed["pages"]), 1)
        self.assertEqual(snapshot["job"]["status"], "running")

    def test_invalid_boxes_never_change_project(self):
        sid, original = self.upload()
        for bbox in [[-1, 0, 100, 50], [0, 0, 400, 50], [10, 10, 0, 50]]:
            response = self.client.put(
                f"/api/sessions/{sid}/boxes",
                json={
                    "mode": "tab",
                    "boxes": [{"page": 1, "kind": "measure", "bbox": bbox}],
                },
            )
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.workflow.load(sid)["revision"], original["revision"])

    def test_stale_revision_cannot_replace_newer_boxes(self):
        sid, original = self.upload()
        url = f"/api/sessions/{sid}/boxes"
        headers = {"If-Match": str(original["revision"])}
        first = {
            "mode": "tab",
            "boxes": [{"page": 1, "kind": "measure", "bbox": [10, 10, 100, 60]}],
        }
        second = {"mode": "tab", "boxes": []}
        self.assertEqual(
            self.client.put(url, json=first, headers=headers).status_code, 200
        )
        self.assertEqual(
            self.client.put(url, json=second, headers=headers).status_code, 409
        )
        self.assertEqual(self.workflow.load(sid)["boxes"], first["boxes"])
        missing = TestClient.request(self.client, "PUT", url, json=second)
        self.assertEqual(missing.status_code, 428)

    def test_manual_boxes_clear_failed_job_across_restart(self):
        sid, _ = self.upload()
        with patch.object(
            self.workflow, "detect", side_effect=ValueError("bad detection")
        ):
            self.client.post(f"/api/sessions/{sid}/detect", json={})
            for _ in range(150):
                state = self.client.get(f"/api/sessions/{sid}").json()
                if state["job"]["status"] == "failed":
                    break
                time.sleep(0.01)
        self.assertEqual(state["job"]["status"], "failed")
        response = self.client.put(
            f"/api/sessions/{sid}/boxes",
            json={
                "boxes": [{"page": 1, "kind": "measure", "bbox": [10, 10, 100, 60]}],
                "mode": "tab",
            },
        )
        self.assertEqual(response.status_code, 200)
        with TestClient(
            create_app(Workflow(self.workflow.root)), base_url="http://127.0.0.1"
        ) as restored:
            self.assertIsNone(restored.get(f"/api/sessions/{sid}").json()["job"])

    def test_metadata_preserves_edits_and_updates_export(self):
        sid, state = self.prepare()
        prefix = f"/api/sessions/{sid}"
        self.client.put(
            prefix + "/measures/1",
            json={"target": TARGET.replace("s1f0", "s1f7"), "reviewed": True},
        )
        updated = self.client.put(
            prefix + "/metadata",
            json={
                "title": "新曲名",
                "artist": "新作者",
                "tempo_quarter": 108,
                "capo": 3,
                "tuning_used": state["metadata"]["tuning_used"],
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertIn("s1f7", updated.json()["measures"][0]["target"])
        exported = self.client.post(prefix + "/export").json()
        gp = self.root / "metadata.gp5"
        gp.write_bytes(self.client.get(exported["gp5_url"]).content)
        song = guitarpro.parse(str(gp), encoding="cp936")
        self.assertEqual(
            (song.title, song.artist, song.tempo), ("新曲名", "新作者", 108)
        )
        self.assertEqual(song.tracks[0].offset, 3)
        changed = self.client.put(
            prefix + "/metadata", json={"tuning_used": [64, 59, 55, 50, 45, 40]}
        ).json()
        self.assertIsNone(changed["recognition"])

    def test_retry_one_measure_preserves_other_manual_edits(self):
        sid, _ = self.prepare()
        prefix = f"/api/sessions/{sid}"
        self.client.put(
            prefix + "/measures/1",
            json={"target": TARGET.replace("s1f0", "s1f7"), "reviewed": True},
        )
        self.backend.return_value.generate.reset_mock()
        self.backend.return_value.generate.return_value = (
            TARGET.replace("s1f0", "s1f9"),
            20,
        )
        self.client.post(prefix + "/recognize", json={"measures": [2]})
        state = self.wait(sid)
        self.backend.return_value.generate.assert_called_once()
        self.assertIn("s1f7", state["measures"][0]["target"])
        self.assertTrue(state["measures"][0]["manually_edited"])
        self.assertIn("s1f9", state["measures"][1]["target"])

    def test_cancel_and_resume_skips_completed_measure(self):
        sid, _ = self.prepare()
        prefix = f"/api/sessions/{sid}"
        entered, release = threading.Event(), threading.Event()

        def generate(*args):
            entered.set()
            release.wait(5)
            return TARGET, 20

        with patch.object(self.backend.return_value, "generate", side_effect=generate):
            self.client.post(prefix + "/recognize")
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.client.post(prefix + "/cancel").status_code, 200)
            release.set()
            state = self.wait(sid)
        self.assertEqual(state["job"]["status"], "cancelled")
        self.assertTrue(state["ocr_task"])
        log = Path(state["ocr_task"]["output"]) / "recognition.jsonl"
        with log.open("ab") as handle:
            handle.write(b'{"measure_number": 2, "raw": "partial')
        self.backend.return_value.generate.reset_mock()
        self.client.post(prefix + "/recognize", json={"resume": True})
        state = self.wait(sid)
        self.backend.return_value.generate.assert_called_once()
        self.assertEqual(len(state["measures"]), 2)
        self.assertNotIn("ocr_task", state)

    def test_project_archive_moves_between_roots(self):
        sid, state = self.prepare()
        prefix = f"/api/sessions/{sid}"
        self.client.put(
            prefix + "/measures/1",
            json={"target": TARGET.replace("s1f0", "s1f7"), "reviewed": True},
        )
        self.client.post(prefix + "/export")
        archive = self.client.get(prefix + "/archive")
        self.assertEqual(
            archive.status_code, 200, archive.text if archive.status_code != 200 else ""
        )
        other = Workflow(self.root / "elsewhere")
        with VersionedClient(create_app(other)) as client:
            imported = client.post(
                "/api/projects/import",
                files={"file": ("project.zip", archive.content, "application/zip")},
            )
            self.assertEqual(imported.status_code, 200, imported.text)
            restored = imported.json()
            self.assertNotEqual(sid, restored["id"])
            self.assertIn("s1f7", restored["measures"][0]["target"])
            self.assertEqual(client.get(restored["gp5_url"]).status_code, 200)
            self.assertTrue(str(other.root) in restored["recognition"])
            self.assertEqual(
                client.post(f"/api/sessions/{restored['id']}/export").status_code, 200
            )

    def test_archive_rejects_escaping_member(self):
        from zipfile import ZipFile

        payload = io.BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("../escape.txt", "oops")
        response = self.client.post(
            "/api/projects/import", files={"file": ("bad.zip", payload.getvalue())}
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse((self.workflow.root.parent / "escape.txt").exists())

    def test_archive_rejects_malformed_manifest_without_leaving_a_project(self):
        from zipfile import ZipFile

        for manifest in (
            [],
            None,
            {"format": "guitarocr-project", "version": 1, "files": [], "session": []},
        ):
            with self.subTest(manifest=manifest):
                payload = io.BytesIO()
                with ZipFile(payload, "w") as archive:
                    archive.writestr("project.json", json.dumps(manifest))
                response = self.client.post(
                    "/api/projects/import",
                    files={"file": ("bad.zip", payload.getvalue())},
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(list(self.workflow.root.iterdir()), [])

    def test_archive_rejects_aliased_paths(self):
        from zipfile import ZipFile

        for names in (("./session.json",), ("a//b.json",), ("A.json", "a.json")):
            with self.subTest(names=names):
                payload = io.BytesIO()
                with ZipFile(payload, "w") as archive:
                    for name in names:
                        archive.writestr(name, "{}")
                response = self.client.post(
                    "/api/projects/import",
                    files={"file": ("bad.zip", payload.getvalue())},
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(list(self.workflow.root.iterdir()), [])


class InputPagesTest(unittest.TestCase):
    def test_renamed_unsupported_image_is_rejected(self):
        from PIL import UnidentifiedImageError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disguised = root / "input.png"
            Image.new("RGB", (10, 10)).save(disguised, format="GIF")
            with self.assertRaises(UnidentifiedImageError):
                expand_inputs([disguised], root / "render", root / "tmp", False)

    def test_pdf_types_input_order_exif_and_multiframe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bitmap = root / "z.png"
            Image.new("RGB", (120, 80), "white").save(bitmap)
            raster = root / "raster.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=120, height=80)
                page.insert_image(page.rect, filename=str(bitmap))
                pdf.save(raster)
            vector = root / "vector.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=120, height=80)
                for y in range(10, 70, 10):
                    page.draw_line((10, y), (110, y))
                pdf.save(vector)
            tiff = root / "multi.tiff"
            Image.new("RGB", (90, 70)).save(
                tiff, save_all=True, append_images=[Image.new("RGB", (80, 60))]
            )
            rotated = root / "rotated.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (30, 50)).save(rotated, exif=exif)
            pages = expand_inputs(
                [vector, raster, tiff, rotated, bitmap],
                root / "render",
                root / "tmp",
                False,
            )
            self.assertTrue(pages[0]["vector"])
            self.assertFalse(pages[1]["vector"])
            self.assertEqual(len(pages), 6)
            with Image.open(pages[4]["image"]) as image:
                self.assertEqual(image.size, (50, 30))
            # Explicit user order is also the saved reading order.
            result = save_layout(
                pages,
                [
                    {"page": 2, "kind": "measure", "bbox": [10, 10, 20, 20]},
                    {"page": 1, "kind": "measure", "bbox": [20, 10, 20, 20]},
                    {"page": 1, "kind": "measure", "bbox": [10, 10, 5, 20]},
                ],
                root / "manual",
                "tab",
            )
            records = json.loads(result.read_text())["records"]
            self.assertEqual([r["page"] for r in records], [1, 1, 2])
            self.assertEqual([r["bbox"][0] for r in records], [20, 10, 10])

    def test_pool_loads_one_base_and_restores_adapters(self):
        with patch("shared.glm_backend.GlmBackend") as backend:
            pool = BackendPool(Path("model"), "cpu")
            a, b = pool.adapter(Path("a")), pool.adapter(Path("b"))
            backend.assert_not_called()
            for handle in (a, b, a, b):
                handle.generate([], 1)
            backend.assert_called_once()
            model = backend.return_value.model
            model.load_adapter.assert_called_once_with(
                Path("b").resolve(), adapter_name="adapter_1"
            )
            self.assertEqual(
                [c.args[0] for c in model.set_adapter.call_args_list],
                ["default", "adapter_1", "default", "adapter_1"],
            )

    def test_pool_can_use_base_before_and_after_loading_an_adapter(self):
        peft_model = MagicMock()
        wrapped = peft_model.from_pretrained.return_value.eval.return_value
        with (
            patch("shared.glm_backend.GlmBackend") as backend,
            patch.dict("sys.modules", {"peft": SimpleNamespace(PeftModel=peft_model)}),
        ):
            pool = BackendPool(Path("model"), "cpu")
            base, adapter = pool.adapter(None), pool.adapter(Path("adapter"))
            original_model = backend.return_value.model
            base.generate([], 1)
            adapter.generate([], 1)
            base.generate([], 1)
            adapter.generate([], 1)
            backend.assert_called_once_with(Path("model").resolve(), None, "cpu")
            peft_model.from_pretrained.assert_called_once_with(
                original_model, Path("adapter").resolve(), adapter_name="adapter_0"
            )
            wrapped.disable_adapter.assert_called_once()
            wrapped.disable_adapter.return_value.__enter__.assert_called_once()
            self.assertEqual(wrapped.set_adapter.call_count, 2)


if __name__ == "__main__":
    unittest.main()
