"""Portable project ZIPs: relative references on disk, validated paths on import."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

from shared.artifacts import write_json

PATH_FIELDS = {
    "image",
    "source_pdf",
    "inputs",
    "layout",
    "info",
    "recognition",
    "export",
    "m2",
    "recognition_log",
    "predictions",
    "gp5",
    "encoding_report",
}
MAX_BYTES = 2 * 1024**3


def transform(value, convert, key=""):
    if isinstance(value, dict):
        return {k: transform(v, convert, k) for k, v in value.items()}
    if isinstance(value, list):
        return [transform(v, convert, key) for v in value]
    return convert(value, key) if isinstance(value, str) else value


def safe_relative(name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("项目包包含无效文件路径")
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.as_posix() != name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("项目包包含无效文件路径")
    return path


def transform_file(contents, suffix, convert):
    """Apply the same path conversion to JSON and JSONL in both directions."""
    if suffix not in {".json", ".jsonl"}:
        return contents
    records = (
        [contents]
        if suffix == ".json"
        else [line for line in contents.decode("utf-8").splitlines() if line]
    )
    return "".join(
        json.dumps(transform(json.loads(record), convert), ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")


def export_project(workflow, sid, destination):
    root = workflow.directory(sid)
    state = workflow.load(sid)
    state.pop(
        "ocr_task", None
    )  # Model checkpoint signatures belong to the originating machine.

    def encode(value, key):
        try:
            relative = Path(value).relative_to(root)
        except ValueError:
            return value
        return "project://" + relative.as_posix()

    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if (
                path.is_symlink()
                or not path.is_file()
                or path.name in {"job.json", "session.json"}
            ):
                continue
            if "tmp" in path.relative_to(root).parts or path.suffix in {".tmp", ".zip"}:
                continue
            contents = transform_file(path.read_bytes(), path.suffix, encode)
            archive.writestr(relative, contents)
            files.append({"path": relative, "sha256": sha256(contents).hexdigest()})
        archive.writestr(
            "project.json",
            json.dumps(
                {
                    "format": "guitarocr-project",
                    "version": 1,
                    "session": transform(state, encode),
                    "files": files,
                },
                ensure_ascii=False,
            ),
        )
    return destination


def import_project(workflow, source):
    sid = uuid4().hex
    root = workflow.directory(sid)

    def decode(value, key):
        if value.startswith("project://"):
            relative = safe_relative(value.removeprefix("project://"))
            return str(root / relative)
        if key in PATH_FIELDS and value:
            raise ValueError(f"项目引用必须在压缩包内：{key}")
        return value

    try:
        with ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > 25000 or sum(m.file_size for m in members) > MAX_BYTES:
                raise ValueError("项目包过大：最多 2 GB 解压内容、25000 个文件")
            names = [m.filename for m in members]
            if len(names) != len({name.casefold() for name in names}):
                raise ValueError("项目包包含重复文件")
            for member in members:
                safe_relative(member.filename)
                if member.filename.casefold() in {"session.json", "job.json"}:
                    raise ValueError("项目包包含保留文件名")
                if stat.S_ISLNK(member.external_attr >> 16) or member.is_dir():
                    raise ValueError("项目包不能包含链接或目录条目")
            if archive.getinfo("project.json").file_size > 20 * 1024**2:
                raise ValueError("项目清单过大")
            package = json.loads(archive.read("project.json"))
            if (
                not isinstance(package, dict)
                or package.get("format") != "guitarocr-project"
                or package.get("version") != 1
            ):
                raise ValueError("请选择 GuitarOCR 导出的项目 ZIP")
            if not isinstance(package.get("files"), list) or not isinstance(
                package.get("session"), dict
            ):
                raise ValueError("项目包缺少有效的文件清单或会话信息")
            listed = [r["path"] for r in package["files"]]
            if len(listed) != len(set(listed)) or set(names) != {
                "project.json",
                *listed,
            }:
                raise ValueError("项目包文件与清单不一致")
            for item in package["files"]:
                contents = archive.read(item["path"])
                if sha256(contents).hexdigest() != item["sha256"]:
                    raise ValueError(f"项目文件校验失败：{item['path']}")
                path = root / safe_relative(item["path"])
                contents = transform_file(contents, path.suffix, decode)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            state = transform(package["session"], decode)
            state.update(id=sid, revision=0)
            state.pop("ocr_task", None)
            if not 1 <= len(state["pages"]) <= 100:
                raise ValueError("项目必须包含 1–100 页")
            write_json(root / "session.json", state)
            workflow.public(
                sid
            )  # Resolve every displayed artifact before accepting the import.
            return sid
    except (BadZipFile, KeyError, TypeError, UnicodeError) as error:
        shutil.rmtree(root, ignore_errors=True)
        raise ValueError(f"无法读取项目包：{error}") from error
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
