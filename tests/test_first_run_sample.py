"""The shipped first-run preset works through the real local API without OCR."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import guitarpro

from pipeline.workspace import Workspace
from shared.artifacts import read_result
from webapp.app import create_app


class FirstRunSampleTest(unittest.TestCase):
    def test_bundled_sample_import_edit_review_export_without_models(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "shared.glm_backend.create_backend",
            side_effect=AssertionError("Preset must not load an OCR model"),
        ), patch.object(Workspace, "warmup", side_effect=AssertionError("No warmup")):
            workspace = Workspace(Path(directory), device="cpu")
            with TestClient(
                create_app(workspace, inference_enabled=False),
                base_url="http://127.0.0.1",
            ) as client:
                self.assertIn('id="openSample"', client.get("/").text)
                archive = client.get("/static/examples/Harbor-Light-synthetic-project.zip")
                self.assertEqual(archive.status_code, 200)
                response = client.post("/api/projects/import", files={
                    "file": ("Harbor-Light-synthetic-project.zip", archive.content, "application/zip"),
                })
                self.assertEqual(response.status_code, 200, response.text)
                state = response.json()
                sid = state["id"]
                self.assertEqual(len(state["pages"]), 1)
                self.assertEqual(len(state["metadata"]["parts"]), 2)
                self.assertEqual(len(state["measures"]), 16)
                self.assertEqual({row["bar_index"] for row in state["measures"]}, set(range(8)))
                recognition = read_result(Path(workspace.load(sid)["recognition"]), "measure_ocr")
                self.assertEqual(recognition["provenance"], {
                    "kind": "synthetic-preset", "ocr_executed": False, "license": "CC0-1.0",
                })
                for row in state["measures"]:
                    body = {"reviewed": True}
                    if row["measure_number"] == 1:
                        body["target"] = row["target"].replace("s1f0", "s1f2", 1)
                    response = client.put(
                        f"/api/sessions/{sid}/measures/{row['measure_number']}",
                        json=body, headers={"If-Match": str(state["revision"])},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    state = response.json()
                self.assertIn("s1f2", state["measures"][0]["target"])
                response = client.post(
                    f"/api/sessions/{sid}/export",
                    headers={"If-Match": str(state["revision"])},
                )
                self.assertEqual(response.status_code, 200, response.text)
                exported = response.json()
                score = guitarpro.parse(io.BytesIO(client.get(exported["gp5_url"]).content), encoding="cp936")
                self.assertEqual(len(score.tracks), 2)
                self.assertTrue(all(len(track.measures) == 8 for track in score.tracks))
                self.assertIn("<score-partwise", client.get(exported["musicxml_url"]).text)
                self.assertEqual(client.get(exported["score_document_url"]).status_code, 200)
                self.assertEqual(client.get(f"/api/sessions/{sid}/archive").status_code, 200)
                second = client.post("/api/projects/import", files={
                    "file": ("sample.zip", archive.content, "application/zip"),
                })
                self.assertEqual(second.status_code, 200, second.text)
                self.assertNotEqual(second.json()["id"], sid)
                self.assertIn("s1f0", second.json()["measures"][0]["target"])
                self.assertIn("s1f2", client.get(f"/api/sessions/{sid}").json()["measures"][0]["target"])


if __name__ == "__main__":
    unittest.main()
