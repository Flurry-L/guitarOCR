"""Service boundaries and recovery: users, leases, browser results and updates."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import subprocess
from threading import Event
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from server.app import create_app
from server.config import Config
from server.store import token_hash
from server.supervisor import Supervisor
from server.worker import make_workflow, run_once


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = Config(self.root, public_url="http://testserver")
        self.config.save()
        self.workflow = make_workflow(self.config)
        self.app = create_app(self.config, self.workflow)
        self.store = self.app.state.store
        self.client = self.enterContext(TestClient(self.app))
        result = self.client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "a long password"},
        )
        self.assertEqual(result.status_code, 200, result.text)
        self.auth = result.json()
        self.client.headers["X-CSRF-Token"] = self.auth["csrf"]
        image = io.BytesIO()
        Image.new("RGB", (300, 200), "white").save(image, "PNG")
        self.image = image.getvalue()

    def upload(self):
        response = self.client.post(
            "/api/sessions", files={"files": ("score.png", self.image, "image/png")}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def work(self):
        self.assertTrue(
            run_once(self.config, self.store, self.workflow, "gpu", Event())
        )

    def test_ownership_csrf_admin_and_revocation(self):
        sid = self.upload()["id"]
        self.work()
        other = self.enterContext(TestClient(self.app))
        result = other.post(
            "/api/auth/register",
            json={"username": "bob", "password": "a long password"},
        ).json()
        other.headers["X-CSRF-Token"] = result["csrf"]
        for path in [
            f"/api/sessions/{sid}",
            f"/api/sessions/{sid}/archive",
            f"/api/sessions/{sid}/files/session.json",
        ]:
            self.assertEqual(other.get(path).status_code, 404)
        self.assertEqual(other.post(f"/api/sessions/{sid}/cancel").status_code, 404)
        self.assertEqual(other.get("/api/admin/users").status_code, 403)
        self.assertEqual(
            self.client.post(
                f"/api/sessions/{sid}/cancel", headers={"X-CSRF-Token": "wrong"}
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                f"/api/sessions/{sid}/cancel", headers={"Origin": "https://bad.example"}
            ).status_code,
            403,
        )
        self.assertNotIn(
            str(self.root), json.dumps(self.client.get(f"/api/sessions/{sid}").json())
        )
        cookie = self.client.cookies.get("guitarocr-auth")
        self.client.post("/api/auth/logout")
        self.client.cookies.set("guitarocr-auth", cookie)
        self.assertEqual(self.client.get("/api/sessions").status_code, 401)

    def test_job_survives_api_restart_and_cancel_queue(self):
        first = self.upload()
        recreated = self.enterContext(TestClient(create_app(self.config)))
        recreated.cookies.update(self.client.cookies)
        self.assertEqual(
            recreated.get(f"/api/sessions/{first['id']}").json()["job"]["status"],
            "queued",
        )
        self.work()
        state = recreated.get(f"/api/sessions/{first['id']}").json()
        self.assertEqual(state["job"]["status"], "complete")
        self.assertEqual(len(state["pages"]), 1)
        second = self.upload()
        self.client.post(f"/api/sessions/{second['id']}/cancel")
        self.assertIsNone(self.store.claim("gpu"))
        self.assertEqual(self.store.job(second["id"])["status"], "cancelled")

    def test_claim_fairness_recovery_and_stale_completion(self):
        first = self.upload()
        self.upload()
        with ThreadPoolExecutor(2) as pool:
            jobs = list(pool.map(lambda _: self.store.claim("gpu"), range(2)))
        self.assertEqual(sum(j is not None for j in jobs), 1)
        old = next(j for j in jobs if j)
        self.assertEqual(old["project_id"], first["id"])
        self.store.execute(
            "UPDATE jobs SET heartbeat=? WHERE id=?", (time.time() - 100, old["id"])
        )
        self.store.recover(90)
        new = self.store.claim("gpu")
        self.assertEqual(new["id"], old["id"])
        self.assertNotEqual(new["lease"], old["lease"])
        self.assertFalse(self.store.finish(old, "complete"))
        self.assertEqual(self.store.job(first["id"])["status"], "running")
        self.assertTrue(self.store.finish(new, "complete"))
        self.assertIsNotNone(self.store.claim("gpu"))

    def test_project_revision_and_usage_survive_deletion(self):
        sid = self.upload()["id"]
        self.work()
        path = f"/api/sessions/{sid}"
        body = {
            "mode": "tab",
            "boxes": [{"kind": "measure", "page": 1, "bbox": [20, 30, 200, 100]}],
        }
        self.assertEqual(self.client.put(path + "/boxes", json=body).status_code, 428)
        self.assertEqual(
            self.client.put(
                path + "/boxes", json=body, headers={"If-Match": "99"}
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.put(
                path + "/boxes", json=body, headers={"If-Match": "0"}
            ).status_code,
            200,
        )
        usage = self.client.get("/api/usage").json()["engines"][0]
        self.assertEqual(self.client.delete(path).status_code, 200)
        self.assertEqual(self.client.get("/api/usage").json()["engines"][0], usage)
        self.assertFalse(self.workflow.directory(sid).exists())

    def test_browser_claim_requires_presence_and_result_token(self):
        item = self.upload()
        jid = item["job"]["id"]
        self.store.execute("UPDATE jobs SET engine='browser' WHERE id=?", (jid,))
        self.assertIsNone(self.store.claim("browser"))
        token = "a" * 40
        poll = self.client.post(f"/api/browser/jobs/{jid}/poll", json={"token": token})
        self.assertEqual(poll.status_code, 200)
        job = self.store.claim("browser")
        self.assertIsNotNone(job)
        other = self.client.post(
            f"/api/browser/jobs/{jid}/poll", json={"token": "b" * 40}
        )
        self.assertEqual(other.status_code, 409)
        self.store.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?)",
            (
                "call",
                jid,
                job["lease"],
                json.dumps({"kind": "measures", "max_new_tokens": 20}),
                None,
                time.time(),
            ),
        )
        self.assertEqual(
            self.client.post(
                f"/api/browser/jobs/{jid}/calls/call",
                json={"token": "b" * 40, "text": "data", "tokens": 1},
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.post(
                f"/api/browser/jobs/{jid}/calls/call",
                json={"token": token, "text": "data", "tokens": 21},
            ).status_code,
            400,
        )
        result = self.client.post(
            f"/api/browser/jobs/{jid}/calls/call",
            json={"token": token, "text": "data", "tokens": 1},
        )
        self.assertEqual(result.status_code, 200)
        self.store.finish(job, "queued")
        self.assertEqual(
            self.store.one("SELECT browser_token FROM jobs WHERE id=?", (jid,))[
                "browser_token"
            ],
            token_hash(token),
        )

    def test_upload_limits_and_pending_limit(self):
        self.config.max_upload_mb = 1
        self.assertEqual(
            self.client.post(
                "/api/sessions", files={"files": ("a.png", b"x" * (1024**2 + 1))}
            ).status_code,
            413,
        )
        self.assertEqual(self.store.one("SELECT count(*) AS n FROM projects")["n"], 0)
        for _ in range(3):
            self.upload()
        result = self.client.post(
            "/api/sessions", files={"files": ("a.png", self.image)}
        )
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.store.one("SELECT count(*) AS n FROM projects")["n"], 3)

    def test_update_failure_restores_database_and_old_release(self):
        self.upload()
        supervisor = Supervisor(self.config, self.config.data / "config.json")
        self.store.set(
            "prepared",
            {
                "source": str(self.root / "candidate"),
                "python": "python",
                "commit": "a" * 40,
            },
        )
        old = supervisor.current

        def candidate_migration(source, python):
            if source != old:
                self.store.execute("DELETE FROM projects")

        with (
            patch.object(
                supervisor, "start_children", side_effect=candidate_migration
            ) as start,
            patch.object(supervisor, "stop_children"),
            patch.object(supervisor, "healthy", return_value=False),
        ):
            supervisor.switch()
        self.assertEqual(start.call_count, 2)
        self.assertEqual(self.store.one("SELECT count(*) AS n FROM projects")["n"], 1)
        self.assertEqual(self.store.setting("update")["state"], "failed")
        self.assertFalse(self.store.setting("queue_paused"))
        self.assertFalse(self.store.setting("maintenance"))
        self.assertEqual(supervisor.current, old)

    def test_update_check_uses_configured_repository_and_caches_commit(self):
        source = self.root / "source"
        source.mkdir()

        def git(*args):
            return subprocess.check_output(
                ["git", *args], cwd=source, text=True, stderr=subprocess.DEVNULL
            ).strip()

        git("init", "-b", "main")
        git("config", "user.name", "Test")
        git("config", "user.email", "test@example.invalid")
        (source / "README").write_text("first")
        git("add", ".")
        git("commit", "-m", "First")
        self.config.source = str(source)
        self.config.repository = str(source)
        supervisor = Supervisor(self.config, self.config.data / "config.json")
        supervisor.check()
        self.assertEqual(self.store.setting("update")["state"], "current")
        current = git("rev-parse", "HEAD")
        (source / "README").write_text("updated")
        git("commit", "-am", "New behavior")
        newest = git("rev-parse", "HEAD")
        git("checkout", "--detach", current)
        supervisor.check()
        update = self.store.setting("update")
        self.assertEqual(update["state"], "available")
        self.assertEqual(update["latest"], newest)
        self.assertEqual(update["current"], current)


if __name__ == "__main__":
    unittest.main()
