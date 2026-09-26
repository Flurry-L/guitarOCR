"""FastAPI transport. Run one worker so the model and job queue are shared."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from shared.defaults import MODEL, LAYOUT_MODEL
from threading import RLock, Event
from typing import Literal, Annotated
from uuid import uuid4
import logging
import json
import shutil
from urllib.parse import urlparse

import pymupdf
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat, StrictInt

from webapp.workflow import Workflow
from webapp.projects import export_project, import_project
from layout.pages import IMAGE_FORMATS, IMAGE_SUFFIXES
from shared.artifacts import write_json
from shared.tasks import Cancelled
from shared.score_text import display_error


class Detection(BaseModel):
    mode: str = Field(default="auto", pattern="^(auto|tab|notation|both)$")
    source: str = Field(default="auto", pattern="^(auto|image|geometry)$")


class Region(BaseModel):
    page: int = Field(ge=1, le=100)
    kind: Literal["measure", "header", "tempo"]
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    mode: Literal["tab", "notation", "both"] | None = None


class Metadata(BaseModel):
    title: str = Field(default="未命名乐谱", max_length=500)
    artist: str = Field(default="", max_length=500)
    tuning_used: list[Annotated[StrictInt, Field(ge=0, le=127)]] = Field(
        default_factory=lambda: [64, 59, 55, 50, 45, 40], min_length=1, max_length=7
    )
    capo: StrictInt = Field(default=0, ge=0, le=24)
    tempo_quarter: StrictInt = Field(default=120, ge=20, le=400)


class Boxes(BaseModel):
    boxes: list[Region] = Field(max_length=5000)
    mode: str = Field(default="auto", pattern="^(auto|tab|notation|both)$")


class Correction(BaseModel):
    target: str | None = None
    measure: dict | None = None
    reviewed: bool = False


class Recognition(BaseModel):
    resume: bool = False
    measures: list[Annotated[StrictInt, Field(ge=1)]] | None = Field(
        default=None, min_length=1, max_length=5000
    )


def create_app(
    workflow: Workflow | None = None, *, allowed_hosts: list[str] | None = None
):
    workflow = workflow or Workflow(Path("output/webui"))
    hosts = {"127.0.0.1", "localhost", "::1"}
    hosts.update(host.casefold() for host in allowed_hosts or [])
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guitarocr")
    lock, jobs, cancellations = RLock(), {}, {}
    for path in workflow.root.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in {"queued", "running"}:
                job.update(
                    status="interrupted",
                    message="上次处理已中断，可继续识别或重新执行该步骤",
                    error=None,
                )
                write_json(path, job)
            jobs[path.parent.name] = job
        except (ValueError, KeyError):
            logging.warning("Ignoring invalid job file: %s", path)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for event in list(cancellations.values()):
            event.set()
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="GuitarOCR", lifespan=lifespan)
    app.state.workflow = workflow
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # Check Host before Origin: a rebound domain otherwise passes both sides
        # of the same-origin comparison while reaching this local server.
        host = request.headers.get("host", "")
        try:
            authority = urlparse("//" + host)
            valid_host = (
                authority.hostname in hosts
                and authority.username is None
                and authority.password is None
                and not (authority.path or authority.query or authority.fragment)
            )
        except ValueError:
            valid_host = False
        if not valid_host:
            return JSONResponse(
                {"detail": "不允许使用此主机名访问工作台"}, status_code=400
            )
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            try:
                parsed_origin = urlparse(origin)
                same_origin = (
                    parsed_origin.scheme == request.url.scheme
                    and parsed_origin.netloc.casefold() == host.casefold()
                )
            except ValueError:
                same_origin = False
            if not same_origin:
                return JSONResponse({"detail": "不允许跨站修改项目"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": display_error(str(error))}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({"detail": "项目或文件不存在"}, status_code=404)

    def idle(sid):
        workflow.directory(sid)
        if jobs.get(sid, {}).get("status") in {"queued", "running"}:
            raise HTTPException(409, "这个项目正在处理，请等待当前步骤结束")

    def check_revision(sid, request):
        expected = request.headers.get("if-match")
        if expected is None:
            raise HTTPException(428, "请先读取项目，并在 If-Match 中提供 revision")
        if expected.strip('"') != str(workflow.load(sid)["revision"]):
            raise HTTPException(
                409,
                "其他页面已经修改了这个项目。本页修改尚未保存，请保留需要的内容后重新载入项目。",
            )

    def save_job(sid, job):
        write_json(workflow.directory(sid) / "job.json", job)

    def edited(sid):
        jobs.pop(sid, None)
        (workflow.directory(sid) / "job.json").unlink(missing_ok=True)
        return workflow.public(sid)

    def submit(sid, label, operation, request=None, cancellable=False):
        with lock:
            idle(sid)
            if request is not None:
                check_revision(sid, request)
            job = {
                "status": "queued",
                "message": label,
                "done": 0,
                "total": 0,
                "error": None,
                "cancellable": cancellable,
            }
            jobs[sid] = job
            cancel = cancellations[sid] = Event()
            save_job(sid, job)

            def progress(done, total):
                with lock:
                    job.update(
                        done=done,
                        total=total,
                        message=f"正在识别第 {done} / {total} 小节",
                    )
                    save_job(sid, job)

            def run():
                with lock:
                    job["status"] = "running"
                    save_job(sid, job)
                try:
                    if cancel.is_set():
                        raise Cancelled("任务已取消")
                    if cancellable:
                        operation(progress, cancel.is_set)
                    else:
                        operation(progress)
                except Cancelled as error:
                    outcome = dict(status="cancelled", error=None, message=str(error))
                except Exception as error:
                    logging.exception("Session %s failed", sid)
                    outcome = dict(
                        status="failed", error=display_error(f"{type(error).__name__}: {error}")
                    )
                else:
                    outcome = dict(status="complete", message="处理完成")
                with lock:
                    job.update(**outcome, cancellable=False)
                    save_job(sid, job)
                    cancellations.pop(sid, None)

            executor.submit(run)
            return {"id": sid, "job": job.copy()}

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "device": workflow.device,
            "max_upload_mb": 200,
            "max_pages": 100,
            "model_ready": (workflow.model / "config.json").exists(),
            "layout_ready": (workflow.layout_model or LAYOUT_MODEL).exists(),
        }

    @app.get("/api/sessions")
    def recent():
        results = []
        for path in sorted(
            workflow.root.glob("*/session.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:30]:
            try:
                state = workflow.load(path.parent.name)
                title = "、".join(
                    state.get("input_names") or [Path(p).name for p in state["inputs"]]
                )
                if state["info"]:
                    title = json.loads(
                        Path(state["info"]).read_text(encoding="utf-8")
                    ).get("title", title)
                if state["export"]:
                    stage = "已导出"
                elif state["recognition"]:
                    stage = "待校对"
                else:
                    stage = "编辑中"
                results.append(
                    {
                        "id": state["id"],
                        "title": title,
                        "pages": len(state["pages"]),
                        "stage": stage,
                    }
                )
            except (ValueError, OSError, KeyError):
                continue
        return results

    @app.post("/api/sessions")
    async def upload(files: list[UploadFile] = File(...)):
        if not 1 <= len(files) <= 100:
            raise HTTPException(400, "每个项目支持 1–100 个文件")
        sid = uuid4().hex
        directory = workflow.directory(sid) / "uploads"
        directory.mkdir(parents=True)
        inputs, input_names, total = [], [], 0
        try:
            for index, upload in enumerate(files):
                suffix = Path(upload.filename or "").suffix.lower()
                if suffix not in IMAGE_SUFFIXES | {".pdf"}:
                    raise ValueError("支持 PDF、PNG、JPEG、BMP 和 TIFF")
                destination = directory / f"{index + 1:03d}{suffix}"
                with destination.open("wb") as output:
                    while chunk := await upload.read(1024 * 1024):
                        total += len(chunk)
                        if total > 200 * 1024 * 1024:
                            raise HTTPException(413, "总上传大小不能超过 200 MB")
                        output.write(chunk)
                inputs.append(destination)
                input_names.append(
                    (upload.filename or destination.name)
                    .replace("\\", "/")
                    .rsplit("/", 1)[-1]
                )
        except Exception:
            shutil.rmtree(directory.parent)
            raise
        finally:
            for upload in files:
                await upload.close()

        def import_files(progress):
            page_count = 0
            for path in inputs:
                if path.suffix == ".pdf":
                    with pymupdf.open(path) as document:
                        if document.needs_pass:
                            raise ValueError("请先解除 PDF 密码保护")
                        page_count += len(document)
                        for page in document:
                            if page.rect.width * page.rect.height * 6.25 > 40_000_000:
                                raise ValueError("PDF 单页尺寸过大")
                else:
                    with Image.open(path, formats=IMAGE_FORMATS) as image:
                        page_count += getattr(image, "n_frames", 1)
                        for frame in range(getattr(image, "n_frames", 1)):
                            image.seek(frame)
                            if image.width * image.height > 40_000_000:
                                raise ValueError("单张图片不能超过 4000 万像素")
                if page_count > 100:
                    raise ValueError("每个项目最多支持 100 页")
            workflow.create(sid, inputs, input_names)

        return submit(sid, "正在导入页面", import_files)

    @app.post("/api/projects/import")
    def restore_project(file: UploadFile):
        path = workflow.root / f"import-{uuid4().hex}.zip"
        try:
            total = 0
            with path.open("wb") as output:
                while block := file.file.read(1024 * 1024):
                    total += len(block)
                    if total > 250 * 1024**2:
                        raise HTTPException(413, "项目 ZIP 最大 250 MB")
                    output.write(block)
            sid = import_project(workflow, path)
            return workflow.public(sid)
        finally:
            file.file.close()
            path.unlink(missing_ok=True)

    @app.get("/api/sessions/{sid}/archive")
    def archive(sid: str):
        with lock:
            idle(sid)
            state = workflow.load(sid)
            destination = workflow.root / ".archives" / f"{sid}-{state['revision']}.zip"
            export_project(workflow, sid, destination)
        return FileResponse(destination, filename=f"guitarocr-project-{sid[:8]}.zip")

    @app.get("/api/sessions/{sid}")
    def session(sid: str):
        with lock:
            directory = workflow.directory(sid)
            if not directory.exists():
                raise HTTPException(404, "项目不存在")
            state = (
                workflow.public(sid)
                if (directory / "session.json").exists()
                else {"id": sid}
            )
            # Serialization happens after this lock is released.
            job = jobs.get(sid)
            public_job = job.copy() if job is not None else None
            if public_job and public_job.get("error"):
                public_job["error"] = display_error(public_job["error"])
            return {**state, "job": public_job}

    @app.post("/api/sessions/{sid}/detect")
    def detect(sid: str, body: Detection, request: Request):
        workflow.load(sid)
        return submit(
            sid,
            "正在检测页面区域",
            lambda _: workflow.detect(sid, body.mode, body.source),
            request,
        )

    @app.put("/api/sessions/{sid}/boxes")
    def boxes(sid: str, body: Boxes, request: Request):
        with lock:
            idle(sid)
            check_revision(sid, request)
            workflow.boxes(sid, [box.model_dump() for box in body.boxes], body.mode)
            return edited(sid)

    @app.post("/api/sessions/{sid}/information")
    def information(sid: str, request: Request):
        workflow.load(sid)
        return submit(
            sid,
            "正在识别标题、调弦和速度",
            lambda _: workflow.information(sid),
            request,
        )

    @app.put("/api/sessions/{sid}/metadata")
    def metadata(sid: str, body: Metadata, request: Request):
        with lock:
            idle(sid)
            check_revision(sid, request)
            workflow.metadata(sid, body.model_dump())
            return edited(sid)

    @app.post("/api/sessions/{sid}/recognize")
    def recognize(sid: str, request: Request, body: Recognition = Recognition()):
        workflow.load(sid)
        return submit(
            sid,
            "正在加载小节识别模型",
            lambda progress, cancelled: workflow.recognize(
                sid, progress, **body.model_dump(), cancelled=cancelled
            ),
            request,
            cancellable=True,
        )

    @app.post("/api/sessions/{sid}/cancel")
    def cancel(sid: str, request: Request):
        with lock:
            check_revision(sid, request)
            event = cancellations.get(sid)
            if event is None or not jobs[sid].get("cancellable"):
                raise HTTPException(409, "当前没有可以停止的识别任务")
            event.set()
            return {"message": "正在停止，当前小节处理结束后生效"}

    @app.put("/api/sessions/{sid}/measures/{number}")
    def correct(sid: str, number: int, body: Correction, request: Request):
        with lock:
            idle(sid)
            check_revision(sid, request)
            workflow.correct(sid, number, **body.model_dump())
            return edited(sid)

    @app.post("/api/sessions/{sid}/export")
    def export(sid: str, request: Request):
        with lock:
            idle(sid)
            check_revision(sid, request)
            workflow.export(sid)
            return edited(sid)

    @app.delete("/api/sessions/{sid}")
    def delete(sid: str, request: Request):
        with lock:
            idle(sid)
            check_revision(sid, request)
            shutil.rmtree(workflow.directory(sid))
            for archive in (workflow.root / ".archives").glob(f"{sid}-*.zip"):
                archive.unlink(missing_ok=True)
            jobs.pop(sid, None)
            return {"deleted": True}

    @app.get("/api/sessions/{sid}/score.txt")
    def score_text(sid: str):
        return Response(
            workflow.score_text(sid),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="score.txt"', "Cache-Control": "no-store"},
        )

    @app.get("/api/sessions/{sid}/files/{filename:path}")
    def asset(sid: str, filename: str):
        root = workflow.directory(sid)
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {
            ".png",
            ".gp5",
            ".m2",
            ".txt",
            ".json",
        }:
            raise HTTPException(404, "文件不存在")
        if not path.is_file():
            raise HTTPException(404, "文件不存在")
        return FileResponse(
            path,
            filename=path.name if path.suffix != ".png" else None,
            headers={"Cache-Control": "no-store"},
        )

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Launch the GuitarOCR web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Additional exact hostname allowed to access the workbench (repeatable)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output", type=Path, default=Path("output/webui"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout-model", type=Path)
    parser.add_argument("--layout-python", type=Path)
    args = parser.parse_args()
    workflow = Workflow(
        args.output,
        model=args.model,
        device=args.device,
        layout_model=args.layout_model,
        layout_python=args.layout_python,
    )
    allowed_hosts = args.allow_host
    if args.host not in {"0.0.0.0", "::"}:
        allowed_hosts.append(args.host)
    uvicorn.run(
        create_app(workflow, allowed_hosts=allowed_hosts),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
