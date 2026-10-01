"""Independent workers claim durable jobs, heartbeat and save partial OCR results."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import signal
from threading import Event, Thread

from layout.pages import validate_inputs
from shared.artifacts import read_result
from shared.tasks import Cancelled
from shared.glm_backend import BackendPool
from server.projects import make_workflow, project_lock, sync_usage
from server.store import Store


@contextmanager
def heartbeat(store, job, interval):
    stop = Event()

    def run():
        while not stop.wait(interval):
            if not store.heartbeat(job):
                return

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def execute(config, store, workflow, job, stop):
    sid, params = job["project_id"], json.loads(job["params"])

    def cancelled():
        current = store.one(
            "SELECT status,cancel,lease FROM jobs WHERE id=?", (job["id"],)
        )
        return (
            stop.is_set()
            or not current
            or current["cancel"]
            or current["lease"] != job["lease"]
            or current["status"] != "running"
        )

    def check():
        if cancelled():
            raise Cancelled("任务已停止")

    def progress(done, total):
        store.execute(
            "UPDATE jobs SET done=?,total=?,message=? WHERE id=? AND lease=? AND cancel=0",
            (done, total, f"已识别 {done} / {total} 小节", job["id"], job["lease"]),
        )

    def stage(label):
        check()
        sync_usage(store, workflow, sid)
        used = store.one(
            """SELECT coalesce(sum(bytes),0) AS n FROM projects WHERE user_id=
            (SELECT user_id FROM projects WHERE id=?)""",
            (sid,),
        )["n"]
        if used >= config.storage_mb * 1024**2:
            raise ValueError("存储空间已达上限，请删除不需要的项目")
        store.execute(
            "UPDATE jobs SET message=? WHERE id=? AND lease=? AND cancel=0",
            (label, job["id"], job["lease"]),
        )

    action = job["action"]
    if (
        action in {"full", "import"}
        and not (workflow.directory(sid) / "session.json").exists()
    ):
        stage("正在展开页面")
        inputs = [workflow.directory(sid) / name for name in params["inputs"]]
        validate_inputs(inputs, config.max_pages)
        workflow.create(sid, inputs, params["names"])
    if action == "full":
        workflow.process(sid, mode=params.get("mode", "auto"), source=params.get("source", "auto"),
                         progress=progress, cancelled=cancelled, on_stage=stage,
                         measures=params.get("measures"))
    elif action == "detect":
        stage("正在检测小节和谱面类型")
        workflow.detect(sid, params.get("mode", "auto"), params.get("source", "auto"))
    elif action == "information":
        stage("正在识别谱面信息")
        workflow.information(sid, cancelled=cancelled)
    elif action == "recognize":
        stage("正在识别小节")
        workflow.recognize(sid, progress, resume=bool(workflow.load(sid).get("ocr_task")),
                           measures=params.get("measures"), cancelled=cancelled)
    check()
    if action == "full":
        state = workflow.load(sid)
        if state["recognition"] and not state["export"]:
            data = read_result(Path(state["recognition"]), "measure_ocr")
            if not data.get("review_measures"):
                stage("正在生成 GP5")
                workflow.export(sid)
    sync_usage(store, workflow, sid)


def run_once(config, store, workflow, stop):
    store.recover(config.lease_seconds)
    job = store.claim()
    if not job:
        return False
    try:
        # Heartbeat even while waiting for a previous lease holder to release the file lock.
        with heartbeat(store, job, max(1, config.lease_seconds // 3)):
            with project_lock(config, job["project_id"]):
                # Queued/resumable jobs keep their model paths across code updates.
                # Managed releases remain on disk, so old jobs can finish consistently.
                pinned = json.loads(job["params"]).get("model_paths", {})
                # Resolve every field for this job. A missing snapshot must never
                # inherit the preceding job's release or adapter selection.
                for key, default in config.model_paths().items():
                    setattr(workflow, key, Path(pinned.get(key, default)))
                if workflow.pool.model_path != workflow.model.resolve():
                    workflow.pool.close()
                    workflow.pool = BackendPool(workflow.model, workflow.device)
                from layout.persistent import LayoutBackend
                from shared.defaults import LAYOUT_MODEL, paddle_python

                detector = workflow.layout_detector
                model_path = Path(workflow.layout_model or LAYOUT_MODEL).resolve()
                python_path = Path(workflow.layout_python or paddle_python()).absolute()
                if isinstance(detector, LayoutBackend) and (detector.model != model_path or detector.python != python_path):
                    detector.close()
                    workflow.layout_detector = LayoutBackend(model_path, python_path, workflow.device)
                execute(config, store, workflow, job, stop)
    except Cancelled:
        # A graceful worker shutdown is a retry, an explicit user cancellation is final.
        store.finish(job, "queued" if stop.is_set() else "cancelled")
    except Exception:
        logging.exception("Job %s failed", job["id"])
        store.finish(
            job, "failed", "处理失败，可重试；若再次失败，请把任务编号提供给管理员。"
        )
    else:
        store.finish(job, "complete")
    finally:
        sync_usage(store, workflow, job["project_id"])
    return True


def run(config, device="cuda:0"):
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    store = Store(config.database)
    workflow = make_workflow(config, device)
    try:
        workflow.warmup()
        while not stop.is_set():
            if not run_once(config, store, workflow, stop):
                stop.wait(1)
    finally:
        workflow.close()
