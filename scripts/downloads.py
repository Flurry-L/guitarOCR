"""Verified model downloads with source fallback and resumable transfers."""

from hashlib import sha256
import http.client
import os
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from shared.environment import verify_files


def download_verified(url, destination, expected, *, attempts=3):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Different sources keep separate partial files; all must match the same hash.
    suffix = sha256(url.encode()).hexdigest()[:12]
    partial = destination.with_name(f"{destination.name}.{suffix}.part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= expected["bytes"]:
                if not verify_files(partial.parent, [{**expected, "name": partial.name}]):
                    partial.replace(destination)
                    return
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "GuitarOCR-installer", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"下载 {destination.name}：{urlsplit(url).hostname}（{attempt + 1}/{attempts}）", flush=True)
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                if response.status == 206:
                    content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not content_range or int(content_range[1]) != offset or int(content_range[3]) != expected["bytes"]:
                        raise ValueError("下载源返回的续传范围不正确")
                elif response.status == 200:
                    offset = 0  # Servers may ignore Range; replace instead of appending.
                else:
                    raise OSError(f"下载源返回 HTTP {response.status}")
                received = offset
                last_progress = time.monotonic()
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if received > expected["bytes"]:
                            raise ValueError("下载内容超过预期大小")
                        if time.monotonic() - last_progress >= 5:
                            print(f"  {received / 1024**2:.1f} / {expected['bytes'] / 1024**2:.1f} MiB", flush=True)
                            last_progress = time.monotonic()
            if partial.stat().st_size < expected["bytes"]:
                raise OSError("下载中断，将从已下载位置继续")
            errors = verify_files(partial.parent, [{**expected, "name": partial.name}])
            if errors:
                raise ValueError("；".join(errors))
            partial.replace(destination)
            return
        except (OSError, ValueError, http.client.HTTPException) as error:
            if isinstance(error, ValueError):
                partial.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(1)


def acquire_base_model(base, root):
    directory = Path(root) / base["path"]
    official = f"https://huggingface.co/{base['repo_id']}/resolve/{base['revision']}"
    sources = list(base.get("download_sources", []))
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        sources.insert(0, f"{endpoint.rstrip('/')}/{base['repo_id']}/resolve/{base['revision']}")
    sources.append(official)
    sources = list(dict.fromkeys(sources))
    for item in base["files"]:
        if not verify_files(directory, [item]):
            continue
        for index, source in enumerate(sources):
            try:
                download_verified(f"{source.rstrip('/')}/{quote(item['name'])}", directory / item["name"], item)
                break
            except (OSError, ValueError, http.client.HTTPException):
                if index + 1 == len(sources):
                    raise RuntimeError(f"无法下载 {item['name']}，请检查网络后重跑启动脚本。已下载部分会保留。") from None
                print(f"{urlsplit(source).hostname} 下载或校验失败，正在尝试备用源。", flush=True)
    print("GLM-OCR 基座校验通过。", flush=True)
