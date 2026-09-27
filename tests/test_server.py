"""Service boundaries and recovery: users, leases, GPU jobs and updates."""

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
from server.store import Store
from server.supervisor import Supervisor
from server.projects import make_workflow
from server.worker import run_once
from shared.tasks import Cancelled


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
            run_once(self.config, self.store, self.workflow, Event())
        )

    def test_worker_model_snapshot_does_not_leak_to_the_next_job(self):
        first = self.upload()["id"]
        job = self.store.job(first)
        params = json.loads(job["params"])
        params["model_paths"] = {"info_adapter": str(self.root / "previous-release/info")}
        self.store.execute("UPDATE jobs SET params=? WHERE id=?", (json.dumps(params), job["id"]))
        self.work()
        self.assertEqual(self.workflow.info_adapter, self.root / "previous-release/info")
        second = self.upload()["id"]
        job = self.store.job(second)
        params = json.loads(job["params"])
        params.pop("model_paths", None)
        self.store.execute("UPDATE jobs SET params=? WHERE id=?", (json.dumps(params), job["id"]))
        self.work()
        self.assertEqual(self.workflow.info_adapter, Path(self.config.info_adapter))
        self.assertEqual(self.store.job(second)["status"], "complete")

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
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.job(second["id"])["status"], "cancelled")

    def test_claim_fairness_recovery_and_stale_completion(self):
        first = self.upload()
        self.upload()
        with ThreadPoolExecutor(2) as pool:
            jobs = list(pool.map(lambda _: self.store.claim(), range(2)))
        self.assertEqual(sum(j is not None for j in jobs), 1)
        old = next(j for j in jobs if j)
        self.assertEqual(old["project_id"], first["id"])
        self.store.execute(
            "UPDATE jobs SET heartbeat=? WHERE id=?", (time.time() - 100, old["id"])
        )
        self.store.recover(90)
        new = self.store.claim()
        self.assertEqual(new["id"], old["id"])
        self.assertNotEqual(new["lease"], old["lease"])
        self.assertFalse(self.store.finish(old, "complete"))
        self.assertEqual(self.store.job(first["id"])["status"], "running")
        self.assertTrue(self.store.finish(new, "complete"))
        self.assertIsNotNone(self.store.claim())

    def test_cancel_running_gpu_job_preserves_project_and_stops_worker(self):
        sid = self.upload()["id"]
        self.work()
        self.store.enqueue(
            self.store.one("SELECT * FROM projects WHERE id=?", (sid,)),
            "information",
            {},
        )
        entered, release = Event(), Event()

        def information(project, *, cancelled):
            entered.set()
            self.assertTrue(release.wait(5))
            self.assertTrue(cancelled())
            raise Cancelled("任务已停止")

        with patch.object(self.workflow, "information", side_effect=information):
            with ThreadPoolExecutor(1) as pool:
                worker = pool.submit(run_once, self.config, self.store, self.workflow, Event())
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(self.store.job(sid)["status"], "running")
                    response = self.client.post(f"/api/sessions/{sid}/cancel")
                    self.assertEqual(response.status_code, 200)
                finally:
                    release.set()
                self.assertTrue(worker.result(timeout=5))
        self.assertEqual(self.store.job(sid)["status"], "cancelled")
        self.assertTrue((self.workflow.directory(sid) / "session.json").exists())
        self.assertGreater(self.store.job(sid)["seconds"], 0)

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

    def test_home_is_public_but_upload_requires_login_and_gpu(self):
        anonymous = self.enterContext(TestClient(self.app))
        self.assertEqual(anonymous.get("/").status_code, 200)
        self.assertIsNone(anonymous.get("/api/auth/me").json()["user"])
        self.assertEqual(anonymous.post("/api/sessions", files={"files": ("a.png", self.image)}).status_code, 401)
        self.assertEqual(self.client.post("/api/sessions", files={"files": ("a.png", self.image)},
                                         data={"engine": "browser"}).status_code, 422)
        self.assertEqual(self.client.post("/api/auth/guest").status_code, 404)
        self.assertEqual(self.client.post("/api/browser/jobs/unused/poll").status_code, 404)
        self.assertNotIn("browser_ready", anonymous.get("/api/config").json())
        self.config.gpus = ""
        self.assertEqual(self.client.post("/api/sessions", files={"files": ("a.png", self.image)}).status_code, 409)

    def test_retired_browser_jobs_keep_results_and_require_explicit_gpu_retry(self):
        item = self.upload()
        self.work()
        sid = item["id"]
        job = self.store.enqueue(self.store.one("SELECT * FROM projects WHERE id=?", (sid,)), "detect", {})
        self.store.execute("UPDATE jobs SET engine='browser',status='running',lease='old',started=?,heartbeat=? WHERE id=?",
                           (time.time()-10, time.time()-1, job["id"]))
        self.store.execute("UPDATE projects SET engine='browser' WHERE id=?", (sid,))
        migrated = Store(self.config.database)
        self.assertEqual(migrated.job(sid)["status"], "cancelled")
        self.assertIsNone(migrated.claim())
        self.assertTrue((self.workflow.directory(sid)/"session.json").exists())
        self.assertEqual(migrated.one("SELECT engine FROM projects WHERE id=?", (sid,))["engine"], "gpu")
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/archive").status_code, 200)
        response = self.client.post(f"/api/sessions/{sid}/retry")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["job"]["engine"], "gpu")
        self.assertEqual(migrated.claim()["project_id"], sid)
        config_path = self.config.data / "config.json"
        values = json.loads(config_path.read_text())
        values["browser_workers"] = 2
        config_path.write_text(json.dumps(values))
        self.assertEqual(Config.load(config_path).gpus, "0")

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
