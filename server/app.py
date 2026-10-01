"""Account HTTP API. GPU inference runs in independent workers."""

from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import shutil
import sqlite3
import time
from typing import Literal
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from layout.pages import IMAGE_SUFFIXES
from server.config import Config
from server.store import ACTIVE, Store, check_password, password_hash, token_hash
from server.projects import make_workflow, project_lock, sync_usage
from webapp.views import project_view
from webapp.contracts import (
    Boxes,
    Correction,
    Detection,
    Metadata,
    Recognition,
    check_revision,
)
from pipeline.archive import MAX_BYTES, export_project, import_project


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(min_length=10, max_length=128)


class PasswordChange(BaseModel):
    current: str = Field(max_length=128)
    password: str = Field(min_length=10, max_length=128)


class AccountEdit(BaseModel):
    disabled: bool | None = None
    password: str | None = Field(default=None, min_length=10, max_length=128)


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
                """SELECT s.*,u.username,u.admin,u.disabled,u.guest FROM sessions s
                JOIN users u ON s.user_id=u.id WHERE token=? AND expires>? AND u.disabled=0""",
                (token_hash(token), time.time()),
            )
            if session:
                request.state.session = session
                request.state.user = {
                    "id": session["user_id"],
                    "username": session["username"],
                    "admin": bool(session["admin"]),
                    "guest": bool(session["guest"]),
                }
        public = path in {
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/me",
            "/api/config",
            "/health",
        }
        protected = path.startswith("/api/") and not public
        if protected and not request.state.user:
            return JSONResponse({"detail": "请先登录"}, 401)
        if (
            protected
            and mutation
            and request.state.session
            and not secrets.compare_digest(
                request.headers.get("x-csrf-token", ""), request.state.session["csrf"]
            )
        ):
            return JSONResponse({"detail": "登录状态已更新，请刷新页面"}, 403)
        if protected and mutation and request.state.user["guest"]:
            return JSONResponse({"detail": "请先登录账号"}, 403)
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
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }
        )
        if path.startswith("/api/"):
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
                    check_revision(
                        request.headers.get("if-match"), workflow.load(sid)["revision"]
                    )
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
        return {**project_view(workflow, sid), "job": store.public_job(store.job(sid))}

    def rate(request, suffix, count, interval):
        address = request.client.host if request.client else "unknown"
        if not store.rate_limit(token_hash(address + ":" + suffix), count, interval):
            raise HTTPException(429, "尝试过于频繁，请稍后重试")

    def identity(user):
        return {
            "id": user["id"],
            "username": None if user["guest"] else user["username"],
            "admin": bool(user["admin"]),
            "guest": bool(user["guest"]),
        }

    def login_response(user, request=None):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires = time.time() + config.session_days * 86400
        with store.connect(True) as db:
            if request and request.state.user and request.state.user["guest"]:
                guest_id = request.state.user["id"]
                # Keep access to projects created by an existing guest session.
                # Revoking the guest session prevents its old token accessing this account.
                db.execute(
                    "UPDATE projects SET user_id=? WHERE user_id=?",
                    (user["id"], guest_id),
                )
                db.execute(
                    "UPDATE jobs SET user_id=? WHERE user_id=?", (user["id"], guest_id)
                )
                db.execute("DELETE FROM sessions WHERE user_id=?", (guest_id,))
                db.execute(
                    "DELETE FROM settings WHERE key LIKE ?",
                    (f"deleted_usage:{guest_id}:%",),
                )
                db.execute("DELETE FROM users WHERE id=? AND guest=1", (guest_id,))
            db.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (token_hash(token), user["id"], csrf, expires),
            )
        response = JSONResponse(
            {"user": identity(user), "csrf": csrf, "expires": expires}
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
        return {
            "server": True,
            "registration": store.setting("registration", True),
            "device": "服务器 GPU",
            "max_upload_mb": config.max_upload_mb,
            "max_pages": config.max_pages,
            "session_days": config.session_days,
            "model_ready": Path(config.model, "config.json").exists(),
            "layout_ready": Path(config.layout_model).exists(),
            "gpu_available": bool(config.gpus),
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
        return login_response(
            store.one("SELECT * FROM users WHERE id=?", (uid,)), request
        )

    @app.post("/api/auth/login")
    def login(body: Credentials, request: Request):
        rate(request, "login", 20, 900)
        if not store.rate_limit(
            "account:" + token_hash(body.username.lower()), 20, 900
        ):
            raise HTTPException(429, "尝试过于频繁，请稍后重试")
        user = store.one("SELECT * FROM users WHERE username=?", (body.username,))
        valid = check_password(body.password, user["password"] if user else dummy_hash)
        if not valid or not user or user["disabled"] or user["guest"]:
            raise HTTPException(401, "用户名或密码不正确")
        return login_response(user, request)

    @app.get("/api/auth/me")
    def me(request: Request):
        if not request.state.user:
            return {"user": None, "csrf": None}
        return {
            "user": identity(request.state.user),
            "csrf": request.state.session["csrf"],
            "expires": request.state.session["expires"],
        }

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
        if request.state.user["guest"]:
            raise HTTPException(403, "请先创建账号")
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
                row["updated"] = (
                    (workflow.directory(row["id"]) / "session.json").stat().st_mtime
                )
                if state["info"]:
                    row["title"] = (
                        json.loads(Path(state["info"]).read_text(encoding="utf-8")).get(
                            "title"
                        )
                        or row["title"]
                    )
        return sorted(
            rows, key=lambda row: row.get("updated", row["created"]), reverse=True
        )

    @app.post("/api/sessions")
    async def upload(
        request: Request,
        files: list[UploadFile] = File(...),
        engine: Literal["gpu"] = Form("gpu"),
        action: Literal["full", "import"] = Form("import"),
        mode: Literal["auto", "tab", "notation", "both"] = Form("auto"),
    ):
        rate(request, "upload", 30, 3600)
        if not 1 <= len(files) <= config.max_pages:
            raise HTTPException(400, f"请选择 1 至 {config.max_pages} 个文件")
        if not config.gpus:
            raise HTTPException(409, "管理员尚未启用 GPU")
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
            params = {
                "inputs": inputs,
                "names": names,
                "mode": mode,
                "model_paths": config.model_paths(),
            }
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

    @app.post("/api/projects/import")
    def restore_project(request: Request, file: UploadFile):
        """Restore a private project without requiring a running GPU worker."""
        rate(request, "upload", 30, 3600)
        sid, uid = uuid4().hex, request.state.user["id"]
        with project_lock(config, sid):
            path = workflow.root / f"import-{sid}.zip"
            reserved = False
            try:
                total = 0
                with path.open("wb") as output:
                    while block := file.file.read(1024**2):
                        total += len(block)
                        if total > config.max_upload_mb * 1024**2:
                            raise HTTPException(413, f"项目 ZIP 最大 {config.max_upload_mb} MB")
                        output.write(block)
                try:
                    with ZipFile(path) as archive:
                        unpacked = sum(item.file_size for item in archive.infolist())
                except BadZipFile as error:
                    raise ValueError("请选择 GuitarOCR 导出的项目 ZIP") from error
                if unpacked > MAX_BYTES:
                    raise HTTPException(413, "项目包解压内容最大 2 GB")
                with store.connect(True) as db:
                    count, used = db.execute(
                        "SELECT count(*),coalesce(sum(bytes),0) FROM projects WHERE user_id=?", (uid,),
                    ).fetchone()
                    if count >= config.max_projects or used + unpacked > config.storage_mb * 1024**2:
                        raise HTTPException(409, "项目或存储空间已达上限，请先删除不需要的项目")
                    db.execute(
                        "INSERT INTO projects(id,user_id,title,engine,created,bytes) VALUES (?,?,?,?,?,?)",
                        (sid, uid, "正在恢复项目", "gpu", time.time(), unpacked),
                    )
                    reserved = True
                import_project(workflow, path, sid=sid, max_pages=config.max_pages,
                               max_bytes=min(MAX_BYTES, config.storage_mb * 1024**2))
                view = project_view(workflow, sid)
                size = sum(p.stat().st_size for p in workflow.directory(sid).rglob("*") if p.is_file())
                # Rewritten absolute paths can enlarge JSON; account for the actual
                # result atomically alongside other in-progress import reservations.
                with store.connect(True) as db:
                    used = db.execute(
                        "SELECT coalesce(sum(bytes),0) FROM projects WHERE user_id=? AND id<>?", (uid, sid),
                    ).fetchone()[0]
                    if used + size > config.storage_mb * 1024**2:
                        raise HTTPException(409, "恢复后的项目超过存储空间上限")
                    db.execute("UPDATE projects SET title=?,pages=?,bytes=? WHERE id=?",
                               ((view.get("metadata") or {}).get("title") or "恢复的项目",
                                len(view["pages"]), size, sid))
                return {**view, "engine": "gpu", "job": None}
            except BaseException:
                if reserved:
                    store.execute("DELETE FROM projects WHERE id=?", (sid,))
                    shutil.rmtree(workflow.directory(sid), ignore_errors=True)
                raise
            finally:
                file.file.close()
                path.unlink(missing_ok=True)

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
            ) or config.model_paths()
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
        return {"message": "已请求取消，已完成部分会保留"}

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
            or path.suffix.lower() not in {".png", ".gp5", ".txt", ".json", ".musicxml"}
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
        for engine in ("gpu",):
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

    @app.get("/api/admin/users")
    def users():
        return store.all("""SELECT u.id,u.username,u.admin,u.disabled,u.created,
            (SELECT count(*) FROM projects p WHERE p.user_id=u.id) AS projects,
            (SELECT coalesce(sum(seconds),0) FROM jobs j WHERE j.user_id=u.id AND j.engine='gpu') AS gpu_seconds
            FROM users u WHERE u.guest=0 ORDER BY u.created""")

    @app.patch("/api/admin/users/{uid}")
    def account(uid: str, body: AccountEdit, request: Request):
        user = store.one("SELECT * FROM users WHERE id=?", (uid,))
        if not user or user["guest"]:
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
            "jobs": store.all("""SELECT j.id,j.project_id,j.engine,j.status,j.message,j.created,
                    CASE WHEN u.guest=1 THEN '访客' ELSE u.username END AS username FROM jobs j
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
