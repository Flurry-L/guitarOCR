"""Authenticated HTTP API. The service never starts inference from a request."""

from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import shutil
import sqlite3
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat

from layout.pages import IMAGE_SUFFIXES
from server.config import Config
from server.locking import project_lock
from server.store import ACTIVE, Store, check_password, password_hash, token_hash
from server.worker import make_workflow, sync_usage
from webapp.app import Boxes, Correction, Detection, Metadata, Recognition
from webapp.projects import export_project


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(min_length=10, max_length=128)


class PasswordChange(BaseModel):
    current: str = Field(max_length=128)
    password: str = Field(min_length=10, max_length=128)


class AccountEdit(BaseModel):
    disabled: bool | None = None
    password: str | None = Field(default=None, min_length=10, max_length=128)


class BrowserLease(BaseModel):
    token: str = Field(min_length=32, max_length=128)


class DetectionBox(BaseModel):
    label: Literal["measure_tab", "measure_notation", "measure_both", "tempo_region"]
    score: float = Field(ge=0, le=1)
    coordinate: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


class BrowserResult(BrowserLease):
    text: str = Field(default="", max_length=50000)
    tokens: int = Field(default=0, ge=0, le=8192)
    boxes: list[DetectionBox] = Field(default_factory=list, max_length=5000)
    error: str | None = Field(default=None, max_length=500)


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = self.limit + 1
        if length > self.limit:
            return await JSONResponse({"detail": "上传内容过大"}, 413)(
                scope, receive, send
            )
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                raise HTTPException(413, "上传内容过大")
            return message

        await self.app(scope, limited_receive, send)


def create_app(config: Config, workflow=None):
    store = Store(config.database)
    workflow = workflow or make_workflow(config)
    app = FastAPI(title="GuitarOCR", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.workflow = store, workflow
    static = Path(__file__).parent / "static"
    editor = Path(__file__).parent.parent / "webapp/static"
    app.mount("/server-static", StaticFiles(directory=static), name="server-static")
    app.mount("/static", StaticFiles(directory=editor), name="static")
    dummy_hash = password_hash(secrets.token_urlsafe(24))

    @app.middleware("http")
    async def security(request, call_next):
        path = request.url.path
        mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
        if mutation and request.headers.get("origin") not in {None, config.public_url}:
            return JSONResponse({"detail": "不允许跨站请求"}, 403)
        request.state.user = None
        request.state.session = None
        token = request.cookies.get("guitarocr-auth", "")
        if token:
            session = store.one(
                """SELECT s.*,u.username,u.admin,u.disabled FROM sessions s
                JOIN users u ON s.user_id=u.id WHERE token=? AND expires>? AND u.disabled=0""",
                (token_hash(token), time.time()),
            )
            if session:
                request.state.session = session
                request.state.user = {
                    "id": session["user_id"],
                    "username": session["username"],
                    "admin": bool(session["admin"]),
                }
        public = path in {
            "/api/auth/login",
            "/api/auth/register",
            "/api/config",
            "/health",
        }
        protected = path.startswith("/api/") and not public
        if protected and not request.state.user:
            return JSONResponse({"detail": "请先登录"}, 401)
        if (
            protected
            and mutation
            and not secrets.compare_digest(
                request.headers.get("x-csrf-token", ""), request.state.session["csrf"]
            )
        ):
            return JSONResponse({"detail": "登录状态已更新，请刷新页面"}, 403)
        if path.startswith("/api/admin/") and not request.state.user["admin"]:
            return JSONResponse({"detail": "需要管理员权限"}, 403)
        if mutation and store.setting("maintenance", False):
            return JSONResponse({"detail": "服务正在更新，请稍后重试"}, 503)
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "same-origin",
                "X-Frame-Options": "DENY",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Cross-Origin-Resource-Policy": "same-origin",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; worker-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }
        )
        if path.startswith("/api/") and not path.startswith("/api/browser-models/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.add_middleware(BodyLimit, limit=(config.max_upload_mb + 2) * 1024**2)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, 400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({"detail": "项目或文件不存在"}, 404)

    def owner(sid, request):
        workflow.directory(sid)
        row = store.one(
            "SELECT * FROM projects WHERE id=? AND user_id=?",
            (sid, request.state.user["id"]),
        )
        if not row:
            raise HTTPException(404, "项目不存在")
        return row

    @contextmanager
    def edit(sid, request, revision=True):
        owner(sid, request)
        try:
            with project_lock(config, sid, blocking=False):
                if (store.job(sid) or {}).get("status") in ACTIVE:
                    raise HTTPException(409, "项目正在处理，请等待完成或取消任务")
                if revision:
                    expected = request.headers.get("if-match")
                    if expected is None:
                        raise HTTPException(428, "请先读取项目再保存")
                    if expected.strip('"') != str(workflow.load(sid)["revision"]):
                        raise HTTPException(409, "项目已更新，请刷新后再保存")
                    sync_usage(store, workflow, sid)
                    used = store.one(
                        "SELECT coalesce(sum(bytes),0) AS n FROM projects WHERE user_id=?",
                        (request.state.user["id"],),
                    )["n"]
                    if used >= config.storage_mb * 1024**2:
                        raise HTTPException(
                            409, "存储空间已达上限，请先删除不需要的项目"
                        )
                yield
                if (workflow.directory(sid) / "session.json").exists():
                    sync_usage(store, workflow, sid)
        except BlockingIOError:
            raise HTTPException(409, "项目正在处理，请稍后重试") from None

    def public(sid):
        if not (workflow.directory(sid) / "session.json").exists():
            return {"id": sid, "job": store.public_job(store.job(sid))}
        state = workflow.public(sid)
        # The editor only needs stage presence, display fields and authenticated URLs.
        hidden = {
            "inputs",
            "image",
            "source_pdf",
            "source_page",
            "predictions",
            "m2",
            "ocr_task",
            "output",
        }

        def clean(value):
            if isinstance(value, list):
                return [clean(x) for x in value]
            if isinstance(value, dict):
                return {
                    k: (
                        bool(v)
                        if k in {"layout", "info", "recognition", "export"}
                        else clean(v)
                    )
                    for k, v in value.items()
                    if k not in hidden
                }
            if isinstance(value, str) and value.startswith(str(workflow.root)):
                return None
            return value

        return {**clean(state), "job": store.public_job(store.job(sid))}

    def model_paths():
        return {
            key: getattr(config, key)
            for key in (
                "model",
                "layout_model",
                "layout_python",
                "info_adapter",
                "measure_adapter",
            )
        }

    def rate(request, suffix, count, interval):
        address = request.client.host if request.client else "unknown"
        if not store.rate_limit(token_hash(address + ":" + suffix), count, interval):
            raise HTTPException(429, "尝试过于频繁，请稍后重试")

    def login_response(user):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires = time.time() + config.session_days * 86400
        with store.connect(True) as db:
            db.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (token_hash(token), user["id"], csrf, expires),
            )
        response = JSONResponse(
            {"user": {k: user[k] for k in ("id", "username", "admin")}, "csrf": csrf}
        )
        response.set_cookie(
            "guitarocr-auth",
            token,
            httponly=True,
            secure=config.public_url.startswith("https:"),
            samesite="strict",
            max_age=config.session_days * 86400,
            path="/",
        )
        return response

    @app.get("/health")
    def health():
        store.one("SELECT 1")
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/workbench")
    def workbench():
        html = (editor / "index.html").read_text()
        html = html.replace(
            "</head>",
            '<link rel="stylesheet" href="/server-static/workbench.css"><script type="module" src="/server-static/workbench.js"></script></head>',
        )
        return HTMLResponse(html)

    @app.get("/api/config")
    def settings():
        manifest = config.browser_models / "manifest.json"
        browser = json.loads(manifest.read_text()) if manifest.exists() else None
        return {
            "server": True,
            "registration": store.setting("registration", True),
            "device": "服务器 GPU",
            "max_upload_mb": config.max_upload_mb,
            "max_pages": config.max_pages,
            "model_ready": Path(config.model, "config.json").exists(),
            "layout_ready": Path(config.layout_model).exists(),
            "gpu_available": bool(config.gpus),
            "browser_ready": bool(browser and config.browser_workers),
            "browser_download_bytes": browser["bytes"] if browser else 0,
            "browser_model_root": f"/api/browser-models/{browser['revision']}/"
            if browser
            else None,
            "queue_paused": store.setting("queue_paused", False),
        }

    @app.post("/api/auth/register")
    def register(body: Credentials, request: Request):
        rate(request, "register", 5, 3600)
        if not store.setting("registration", True):
            raise HTTPException(403, "管理员已关闭注册")
        try:
            uid = store.add_user(body.username, body.password)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "用户名已被使用") from None
        return login_response(store.one("SELECT * FROM users WHERE id=?", (uid,)))

    @app.post("/api/auth/login")
    def login(body: Credentials, request: Request):
        rate(request, "login", 20, 900)
        if not store.rate_limit(
            "account:" + token_hash(body.username.lower()), 20, 900
        ):
            raise HTTPException(429, "尝试过于频繁，请稍后重试")
        user = store.one("SELECT * FROM users WHERE username=?", (body.username,))
        valid = check_password(body.password, user["password"] if user else dummy_hash)
        if not valid or not user or user["disabled"]:
            raise HTTPException(401, "用户名或密码不正确")
        return login_response(user)

    @app.get("/api/auth/me")
    def me(request: Request):
        return {"user": request.state.user, "csrf": request.state.session["csrf"]}

    @app.post("/api/auth/logout")
    def logout(request: Request):
        store.execute(
            "DELETE FROM sessions WHERE token=?", (request.state.session["token"],)
        )
        response = JSONResponse({"ok": True})
        response.delete_cookie("guitarocr-auth", path="/")
        return response

    @app.put("/api/auth/password")
    def change_password(body: PasswordChange, request: Request):
        rate(request, "password", 10, 900)
        user = store.one("SELECT * FROM users WHERE id=?", (request.state.user["id"],))
        if not check_password(body.current, user["password"]):
            raise HTTPException(400, "当前密码不正确")
        with store.connect(True) as db:
            db.execute(
                "UPDATE users SET password=? WHERE id=?",
                (password_hash(body.password), user["id"]),
            )
            db.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
        return login_response(user)

    @app.get("/api/sessions")
    def projects(request: Request):
        rows = store.all(
            "SELECT * FROM projects WHERE user_id=? ORDER BY created DESC",
            (request.state.user["id"],),
        )
        for row in rows:
            row.pop("user_id")
            row["job"] = store.public_job(store.job(row["id"]))
            row["stage"] = "待处理"
            if (workflow.directory(row["id"]) / "session.json").exists():
                state = workflow.load(row["id"])
                row["stage"] = (
                    "已导出"
                    if state["export"]
                    else "待校对"
                    if state["recognition"]
                    else "编辑中"
                )
                row["pages"] = len(state["pages"])
        return rows

    @app.post("/api/sessions")
    async def upload(
        request: Request,
        files: list[UploadFile] = File(...),
        engine: Literal["gpu", "browser"] = Form("gpu"),
        action: Literal["full", "import"] = Form("import"),
    ):
        rate(request, "upload", 30, 3600)
        if not 1 <= len(files) <= config.max_pages:
            raise HTTPException(400, f"请选择 1 至 {config.max_pages} 个文件")
        if engine == "gpu" and not config.gpus:
            raise HTTPException(409, "管理员尚未启用 GPU")
        if engine == "browser" and not settings()["browser_ready"]:
            raise HTTPException(409, "管理员尚未安装浏览器模型")
        sid, uid = uuid4().hex, request.state.user["id"]
        reserved = config.max_upload_mb * 1024**2
        with store.connect(True) as db:
            row = db.execute(
                "SELECT count(*),coalesce(sum(bytes),0) FROM projects WHERE user_id=?",
                (uid,),
            ).fetchone()
            if (
                row[0] >= config.max_projects
                or row[1] + reserved > config.storage_mb * 1024**2
            ):
                raise HTTPException(409, "项目或存储空间已达上限，请先删除不需要的项目")
            db.execute(
                "INSERT INTO projects(id,user_id,title,engine,created,bytes) VALUES (?,?,?,?,?,?)",
                (
                    sid,
                    uid,
                    (files[0].filename or "未命名乐谱")[:200],
                    engine,
                    time.time(),
                    reserved,
                ),
            )
        root = workflow.directory(sid)
        (root / "uploads").mkdir(parents=True)
        names, inputs, total = [], [], 0
        try:
            for index, file in enumerate(files):
                suffix = Path(file.filename or "").suffix.lower()
                if suffix not in IMAGE_SUFFIXES | {".pdf"}:
                    raise ValueError("支持 PDF、PNG、JPEG、BMP 和 TIFF")
                path = root / "uploads" / f"{index:03d}{suffix}"
                with path.open("wb") as out:
                    while chunk := await file.read(1024**2):
                        total += len(chunk)
                        if total > reserved:
                            raise HTTPException(
                                413, f"上传总大小不能超过 {config.max_upload_mb} MB"
                            )
                        out.write(chunk)
                names.append(
                    (file.filename or path.name)
                    .replace("\\", "/")
                    .rsplit("/", 1)[-1][:200]
                )
                inputs.append(path.relative_to(root).as_posix())
            project = store.one("SELECT * FROM projects WHERE id=?", (sid,))
            params = {"inputs": inputs, "names": names, "model_paths": model_paths()}
            if engine == "browser":
                params["browser_bundle"] = json.loads(
                    (config.browser_models / "manifest.json").read_text()
                )["revision"]
            job = store.enqueue(project, action, params, config.max_pending)
            store.execute("UPDATE projects SET bytes=? WHERE id=?", (total, sid))
        except BaseException:
            store.execute("DELETE FROM projects WHERE id=?", (sid,))
            shutil.rmtree(root, ignore_errors=True)
            raise
        finally:
            for file in files:
                await file.close()
        return {"id": sid, "job": store.public_job(job)}

    @app.get("/api/sessions/{sid}")
    def project(sid: str, request: Request):
        row = owner(sid, request)
        return {**public(sid), "engine": row["engine"]}

    def enqueue(sid, request, action, params):
        with edit(sid, request):
            project = owner(sid, request)
            previous = store.job(sid)
            previous_params = json.loads(previous["params"]) if previous else {}
            params["model_paths"] = (
                previous_params.get("model_paths") if params.get("resume") else None
            ) or model_paths()
            if project["engine"] == "browser":
                bundle = (
                    json.loads(previous["params"]).get("browser_bundle")
                    if previous and params.get("resume")
                    else None
                )
                params["browser_bundle"] = (
                    bundle
                    or json.loads(
                        (config.browser_models / "manifest.json").read_text()
                    )["revision"]
                )
            if action == "recognize" and not params.get("resume"):
                state = workflow.load(sid)
                state.pop("ocr_task", None)
                workflow.store(state)
            job = store.enqueue(project, action, params, config.max_pending)
        return {"id": sid, "job": store.public_job(job)}

    @app.post("/api/sessions/{sid}/detect")
    def detect(sid: str, body: Detection, request: Request):
        return enqueue(sid, request, "detect", body.model_dump())

    @app.post("/api/sessions/{sid}/information")
    def information(sid: str, request: Request):
        return enqueue(sid, request, "information", {})

    @app.post("/api/sessions/{sid}/recognize")
    def recognize(sid: str, request: Request, body: Recognition = Recognition()):
        return enqueue(sid, request, "recognize", body.model_dump())

    @app.post("/api/sessions/{sid}/retry")
    def retry(sid: str, request: Request):
        with edit(sid, request, revision=False):
            project = owner(sid, request)
            previous = store.job(sid)
            if not previous or previous["status"] not in {"failed", "cancelled"}:
                raise HTTPException(409, "当前任务无需重试")
            params = json.loads(previous["params"])
            params["resume"] = True
            job = store.enqueue(project, previous["action"], params, config.max_pending)
        return {"id": sid, "job": store.public_job(job)}

    @app.post("/api/sessions/{sid}/cancel")
    def cancel(sid: str, request: Request):
        owner(sid, request)
        store.cancel(sid)
        return {"message": "已请求停止，当前步骤完成后生效"}

    @app.put("/api/sessions/{sid}/boxes")
    def boxes(sid: str, body: Boxes, request: Request):
        with edit(sid, request):
            workflow.boxes(sid, [b.model_dump() for b in body.boxes], body.mode)
            return public(sid)

    @app.put("/api/sessions/{sid}/metadata")
    def metadata(sid: str, body: Metadata, request: Request):
        with edit(sid, request):
            workflow.metadata(sid, body.model_dump())
            store.execute("UPDATE projects SET title=? WHERE id=?", (body.title, sid))
            return public(sid)

    @app.put("/api/sessions/{sid}/measures/{number}")
    def correct(sid: str, number: int, body: Correction, request: Request):
        with edit(sid, request):
            workflow.correct(sid, number, **body.model_dump())
            return public(sid)

    @app.post("/api/sessions/{sid}/export")
    def export(sid: str, request: Request):
        with edit(sid, request):
            workflow.export(sid)
            return public(sid)

    @app.get("/api/sessions/{sid}/archive")
    def archive(sid: str, request: Request):
        with edit(sid, request, revision=False):
            destination = workflow.directory(sid) / "download.zip"
            export_project(workflow, sid, destination)
        return FileResponse(destination, filename=f"guitarocr-{sid[:8]}.zip")

    @app.get("/api/sessions/{sid}/score.txt")
    def score(sid: str, request: Request):
        owner(sid, request)
        return Response(
            workflow.score_text(sid),
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="score.txt"'},
        )

    @app.get("/api/sessions/{sid}/files/{filename:path}")
    def asset(sid: str, filename: str, request: Request):
        owner(sid, request)
        root = workflow.directory(sid)
        path = (root / filename).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.suffix.lower() not in {".png", ".gp5", ".txt", ".json"}
        ):
            raise HTTPException(404, "文件不存在")
        if path.suffix == ".json" and "encoding" not in path.name:
            raise HTTPException(404, "文件不存在")
        return FileResponse(path, filename=path.name if path.suffix != ".png" else None)

    @app.delete("/api/sessions/{sid}")
    def delete(sid: str, request: Request):
        with edit(sid, request, revision=False):
            # Keep usage counts when a project is removed; only the project files/history disappear.
            totals = store.all(
                "SELECT engine,count(*) AS jobs,sum(seconds) AS seconds FROM jobs WHERE project_id=? GROUP BY engine",
                (sid,),
            )
            with store.connect(True) as db:
                for row in totals:
                    key = f"deleted_usage:{request.state.user['id']}:{row['engine']}"
                    old = db.execute(
                        "SELECT value FROM settings WHERE key=?", (key,)
                    ).fetchone()
                    value = json.loads(old[0]) if old else {"jobs": 0, "seconds": 0}
                    value["jobs"] += row["jobs"]
                    value["seconds"] += row["seconds"] or 0
                    db.execute(
                        "INSERT OR REPLACE INTO settings VALUES (?,?)",
                        (key, json.dumps(value)),
                    )
                db.execute("DELETE FROM projects WHERE id=?", (sid,))
            shutil.rmtree(workflow.directory(sid), ignore_errors=True)
        return {"deleted": True}

    @app.get("/api/usage")
    def usage(request: Request):
        uid = request.state.user["id"]
        result = []
        for engine in ("gpu", "browser"):
            row = store.one(
                "SELECT count(*) AS jobs,coalesce(sum(seconds),0) AS seconds FROM jobs WHERE user_id=? AND engine=?",
                (uid, engine),
            )
            old = store.setting(
                f"deleted_usage:{uid}:{engine}", {"jobs": 0, "seconds": 0}
            )
            result.append(
                {
                    "engine": engine,
                    "jobs": row["jobs"] + old["jobs"],
                    "seconds": row["seconds"] + old["seconds"],
                }
            )
        return {
            "engines": result,
            **store.one(
                "SELECT count(*) AS projects,coalesce(sum(pages),0) AS pages,coalesce(sum(bytes),0) AS bytes FROM projects WHERE user_id=?",
                (uid,),
            ),
        }

    def browser_job(jid, request):
        job = store.one(
            "SELECT * FROM jobs WHERE id=? AND user_id=? AND engine='browser'",
            (jid, request.state.user["id"]),
        )
        if not job:
            raise HTTPException(404, "任务不存在")
        return job

    @app.post("/api/browser/jobs/{jid}/poll")
    def browser_poll(jid: str, body: BrowserLease, request: Request):
        job = browser_job(jid, request)
        with store.connect(True) as db:
            job = dict(db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())
            if (
                job["browser_token"]
                and job["browser_token"] != token_hash(body.token)
                and (job["browser_seen"] or 0) > time.time() - 60
            ):
                raise HTTPException(409, "另一个网页正在运行此任务，请在原页面继续")
            db.execute(
                "UPDATE jobs SET browser_token=?,browser_seen=? WHERE id=?",
                (token_hash(body.token), time.time(), jid),
            )
            row = db.execute(
                "SELECT * FROM calls WHERE job_id=? AND lease=? AND result IS NULL ORDER BY created LIMIT 1",
                (jid, job["lease"]),
            ).fetchone()
        return {
            "job": store.public_job(job),
            "cancel": bool(job["cancel"]),
            "call": {"id": row["id"], **json.loads(row["payload"])} if row else None,
        }

    @app.post("/api/browser/jobs/{jid}/calls/{cid}")
    def browser_result(jid: str, cid: str, body: BrowserResult, request: Request):
        job = browser_job(jid, request)
        if (
            job["status"] != "running"
            or job["browser_token"] != token_hash(body.token)
            or job["cancel"]
        ):
            raise HTTPException(409, "任务已停止或由其他网页接管")
        call = store.one(
            "SELECT * FROM calls WHERE id=? AND job_id=? AND lease=?",
            (cid, jid, job["lease"]),
        )
        if not call:
            raise HTTPException(409, "识别步骤已更新，请重新连接")
        payload = json.loads(call["payload"])
        if payload["kind"] == "layout":
            # Bound coordinates before handing them to image post-processing.
            if any(
                any(abs(v) > 40000 for v in b.coordinate)
                or b.coordinate[2] <= b.coordinate[0]
                or b.coordinate[3] <= b.coordinate[1]
                for b in body.boxes
            ):
                raise ValueError("版面坐标无效")
        elif body.tokens > payload["max_new_tokens"]:
            raise ValueError("识别长度超出限制")
        store.execute(
            "UPDATE calls SET result=? WHERE id=? AND result IS NULL",
            (body.model_dump_json(exclude={"token"}), cid),
        )
        return {"ok": True}

    @app.get("/api/browser-models/{bundle}/{filename:path}")
    def browser_model(bundle: str, filename: str):
        if len(bundle) != 20 or any(c not in "0123456789abcdef" for c in bundle):
            raise HTTPException(404, "模型不存在")
        root = config.browser_models / bundle
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(404, "浏览器模型尚未安装")
        manifest = json.loads(manifest_path.read_text())
        path = (root / filename).resolve()
        if (
            not path.is_relative_to(root)
            or filename not in manifest["files"]
            or not path.is_file()
        ):
            raise HTTPException(404, "文件不存在")
        return FileResponse(
            path,
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "ETag": '"' + manifest["files"][filename]["sha256"] + '"',
            },
        )

    @app.get("/api/admin/users")
    def users():
        return store.all("""SELECT u.id,u.username,u.admin,u.disabled,u.created,
            (SELECT count(*) FROM projects p WHERE p.user_id=u.id) AS projects,
            (SELECT coalesce(sum(seconds),0) FROM jobs j WHERE j.user_id=u.id AND j.engine='gpu') AS gpu_seconds
            FROM users u ORDER BY u.created""")

    @app.patch("/api/admin/users/{uid}")
    def account(uid: str, body: AccountEdit, request: Request):
        user = store.one("SELECT * FROM users WHERE id=?", (uid,))
        if not user:
            raise HTTPException(404, "用户不存在")
        if user["admin"]:
            raise HTTPException(409, "管理员账号请通过命令行管理")
        with store.connect(True) as db:
            if body.disabled is not None:
                db.execute(
                    "UPDATE users SET disabled=? WHERE id=?", (int(body.disabled), uid)
                )
            if body.password:
                db.execute(
                    "UPDATE users SET password=? WHERE id=?",
                    (password_hash(body.password), uid),
                )
            db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            if body.disabled:
                db.execute(
                    "UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE user_id=? AND status IN ('queued','running')",
                    (uid,),
                )
        return {"ok": True}

    @app.get("/api/admin/status")
    def admin_status():
        return {
            "registration": store.setting("registration", True),
            "queue_paused": store.setting("queue_paused", False),
            "jobs": store.all("""SELECT j.id,j.project_id,j.engine,j.status,j.message,j.created,u.username FROM jobs j
                    JOIN users u ON u.id=j.user_id WHERE j.status IN ('queued','running') ORDER BY j.created LIMIT 200"""),
            "update": store.setting("update", {}),
            "supervisor_seen": store.setting("supervisor_seen", 0),
        }

    @app.put("/api/admin/registration")
    def registration(enabled: bool):
        store.set("registration", enabled)
        return {"enabled": enabled}

    @app.post("/api/admin/jobs/{jid}/cancel")
    def admin_cancel(jid: str):
        row = store.one(
            "SELECT project_id FROM jobs WHERE id=? AND status IN ('queued','running')",
            (jid,),
        )
        if row:
            store.cancel(row["project_id"])
        return {"ok": True}

    @app.post("/api/admin/updates/check")
    def check_update():
        if store.setting("supervisor_seen", 0) < time.time() - 30:
            raise HTTPException(409, "请使用 guitarocr-server serve 启动更新管理进程")
        if not store.rate_limit("update_check", 1, 60):
            raise HTTPException(429, "刚刚检查过，请稍后重试")
        store.set("check_update", True)
        return {"ok": True}

    @app.post("/api/admin/updates/apply")
    def apply_update():
        if store.setting("supervisor_seen", 0) < time.time() - 30:
            raise HTTPException(409, "更新管理进程未运行")
        with store.connect(True) as db:
            row = db.execute("SELECT value FROM settings WHERE key='update'").fetchone()
            update = json.loads(row[0]) if row else {}
            if update.get("state") != "available":
                raise HTTPException(409, "当前没有可安装的更新")
            update.update(state="requested", message="等待准备更新")
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('update',?)",
                (json.dumps(update),),
            )
        return {"ok": True}

    return app
