"""Prepare updates beside the active release, drain jobs, then switch processes."""

import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
from threading import Event, Thread
import time
import urllib.request

from server.store import Store


def command(args, cwd=None, timeout=300, log=None):
    result = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if log:
        with Path(log).open("a") as handle:
            handle.write(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({args[0]}): {result.stdout[-1500:]}")
    return result.stdout.strip()


def revision(path):
    try:
        return command(["git", "rev-parse", "HEAD"], cwd=path, timeout=10)
    except (OSError, RuntimeError):
        return "unknown"


class Supervisor:
    def __init__(self, config, config_path):
        self.config, self.config_path = config, Path(config_path).resolve()
        self.store = Store(config.database)
        self.stop = Event()
        self.children = []
        self.logs = []
        self.releases = config.data / "releases"
        self.releases.mkdir(parents=True, exist_ok=True)
        self.repo = self.releases / "repository.git"
        self.current = Path(self.store.setting("active_source", config.source))
        self.python = self.store.setting("active_python", sys.executable)
        self.update_thread = None
        self.restart_supervisor = False

    def report(self, state, message, protect_request=False, **values):
        with self.store.connect(True) as db:
            row = db.execute("SELECT value FROM settings WHERE key='update'").fetchone()
            update = json.loads(row[0]) if row else {}
            if protect_request and update.get("state") in {
                "requested",
                "preparing",
                "draining",
                "applying",
            }:
                return
            update.update(state=state, message=message, **values)
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('update',?)",
                (json.dumps(update),),
            )

    def check(self):
        previous = self.store.setting("update", {})
        if previous.get("state") in {"requested", "preparing", "draining", "applying"}:
            return
        try:
            if not self.repo.exists():
                command(["git", "init", "--bare", str(self.repo)])
            command(
                [
                    "git",
                    "--git-dir",
                    str(self.repo),
                    "config",
                    "remote.origin.url",
                    self.config.repository,
                ]
            )
            command(
                [
                    "git",
                    "--git-dir",
                    str(self.repo),
                    "fetch",
                    "--depth=1",
                    "origin",
                    self.config.branch,
                ],
                timeout=120,
            )
            latest = command(
                ["git", "--git-dir", str(self.repo), "rev-parse", "FETCH_HEAD"]
            )
            current = revision(self.current)
            message = command(
                ["git", "--git-dir", str(self.repo), "log", "-1", "--format=%s", latest]
            )
            self.report(
                "available" if latest != current else "current",
                "发现新版本" if latest != current else "已是当前版本",
                current=current,
                latest=latest,
                subject=message,
                checked=time.time(),
                error=None,
                protect_request=True,
            )
        except Exception as error:
            logging.exception("Update check failed")
            self.report(
                "check_failed",
                "检查更新失败，可稍后重试",
                error=str(error)[-1500:],
                checked=time.time(),
                protect_request=True,
            )

    def prepare(self):
        try:
            update = self.store.setting("update")
            target = update["latest"]
            if len(target) != 40 or any(c not in "0123456789abcdef" for c in target):
                raise ValueError("Invalid update commit")
            candidate = self.releases / target
            self.report("preparing", "正在下载并安装更新，当前服务继续运行")
            log = self.config.data / "update.log"
            if not candidate.exists():
                command(
                    [
                        "git",
                        "--git-dir",
                        str(self.repo),
                        "worktree",
                        "add",
                        "--detach",
                        str(candidate),
                        target,
                    ],
                    timeout=300,
                    log=log,
                )
            command(["git", "lfs", "pull"], cwd=candidate, timeout=600, log=log)
            if not (candidate / "server/cli.py").is_file():
                raise ValueError("此提交不包含服务端入口")
            uv = shutil.which("uv")
            if not uv:
                raise ValueError("更新需要 uv，请先安装 uv 并加入服务的 PATH")
            python = candidate / ".venv/bin/python"
            if not python.exists():
                command(
                    [uv, "venv", "--python", sys.executable, str(candidate / ".venv")],
                    timeout=120,
                    log=log,
                )
            command(
                [
                    uv,
                    "sync",
                    "--frozen",
                    "--extra",
                    "webui",
                    "--extra",
                    "glm-ocr",
                    "--no-dev",
                ],
                cwd=candidate,
                timeout=1800,
                log=log,
            )
            command(
                [
                    str(python),
                    "-m",
                    "server.cli",
                    "validate",
                    "--config",
                    str(self.runtime_config(candidate)),
                ],
                cwd=candidate,
                timeout=120,
                log=log,
            )
            self.store.set(
                "prepared",
                {"source": str(candidate), "python": str(python), "commit": target},
            )
            self.store.set("queue_paused", True)
            self.report("draining", "更新已准备好，等待当前任务结束；新任务继续排队")
        except Exception as error:
            logging.exception("Update preparation failed")
            self.store.set("queue_paused", False)
            self.report(
                "failed", "更新准备失败，当前服务继续运行", error=str(error)[-1500:]
            )

    def runtime_config(self, source):
        values = json.loads(self.config_path.read_text())
        values["source"] = str(source)
        # Explicit model locations stay fixed. Default adapters follow managed releases.
        for key, relative in [
            ("layout_model", "weights/layout"),
            ("info_adapter", "weights/document_info"),
            ("measure_adapter", "weights/measure_ocr"),
        ]:
            original = Path(self.config.source) / relative
            if Path(values[key]) == original:
                values[key] = str(source / relative)
        path = self.config.data / ("runtime-" + source.name + ".json")
        path.write_text(json.dumps(values, indent=2))
        path.chmod(0o600)
        return path

    def start_children(self, source, python):
        config_path = self.runtime_config(source)
        commands = [[python, "-m", "server.cli", "api", "--config", str(config_path)]]
        for gpu in [x.strip() for x in self.config.gpus.split(",") if x.strip()]:
            commands.append(
                [
                    python,
                    "-m",
                    "server.cli",
                    "worker",
                    "--config",
                    str(config_path),
                    "--engine",
                    "gpu",
                    "--gpu",
                    gpu,
                ]
            )
        for _ in range(self.config.browser_workers):
            commands.append(
                [
                    python,
                    "-m",
                    "server.cli",
                    "worker",
                    "--config",
                    str(config_path),
                    "--engine",
                    "browser",
                ]
            )
        logroot = self.config.data / "logs"
        logroot.mkdir(exist_ok=True)
        self.children = []
        for index, args in enumerate(commands):
            handle = (logroot / f"process-{index}.log").open("a")
            self.logs.append(handle)
            process = subprocess.Popen(
                args,
                cwd=source,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.children.append((process, args, handle))

    def stop_children(self, timeout=30):
        for process, _, _ in self.children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        for process, _, _ in self.children:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self.children = []
        for handle in self.logs:
            handle.close()
        self.logs = []

    def healthy(self, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.stop.is_set():
            if any(p.poll() is not None for p, _, _ in self.children):
                return False
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.config.port}/health", timeout=2
                ) as response:
                    if json.load(response).get("ok"):
                        return True
            except (OSError, ValueError):
                pass
            self.stop.wait(1)
        return False

    def switch(self):
        candidate = self.store.setting("prepared")
        previous_source, previous_python = self.current, self.python
        self.report("applying", "正在重启服务，任务和结果会保留")
        self.store.set("maintenance", True)
        self.stop_children()
        backup = self.config.data / "before-update.sqlite3"
        with self.store.connect() as source, sqlite3.connect(backup) as destination:
            source.backup(destination)
        self.store.set(
            "rollback",
            {
                "source": str(previous_source),
                "python": previous_python,
                "backup": str(backup),
            },
        )
        try:
            self.start_children(Path(candidate["source"]), candidate["python"])
            if not self.healthy():
                raise RuntimeError("新进程未通过健康检查")
            self.current, self.python = Path(candidate["source"]), candidate["python"]
            self.store.set("active_source", str(self.current))
            self.store.set("active_python", self.python)
            self.report("current", "更新完成", current=candidate["commit"], error=None)
            self.store.set("rollback", None)
            # The next supervisor imports its own code from the new release too.
            # Claims stay paused until it starts, so the handoff cannot interrupt OCR.
            self.restart_supervisor = True
        except Exception as error:
            logging.exception("Update startup failed")
            self.stop_children()
            with sqlite3.connect(backup) as source, self.store.connect() as destination:
                source.backup(destination)
            self.store.set("queue_paused", False)
            self.store.set("maintenance", False)
            self.store.set("rollback", None)
            self.report(
                "failed", "新版本启动失败，已恢复原版本", error=str(error)[-1500:]
            )
            self.start_children(previous_source, previous_python)

    def recover_update(self):
        rollback = self.store.setting("rollback")
        if rollback:
            with (
                sqlite3.connect(rollback["backup"]) as source,
                self.store.connect() as destination,
            ):
                source.backup(destination)
            self.current, self.python = Path(rollback["source"]), rollback["python"]
            self.store.set("active_source", str(self.current))
            self.store.set("active_python", self.python)
            self.store.set("rollback", None)
            self.report("failed", "更新过程中服务中断，已恢复原版本")
        elif self.store.setting("update", {}).get("state") in {
            "requested",
            "preparing",
            "draining",
            "applying",
        }:
            self.report("failed", "更新被中断，可重新检查后安装")
        self.store.set("queue_paused", False)
        self.store.set("maintenance", False)

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        self.recover_update()
        self.start_children(self.current, self.python)
        last_check = 0
        try:
            while not self.stop.wait(1):
                self.store.set("supervisor_seen", time.time())
                busy = self.update_thread is not None and self.update_thread.is_alive()
                state = self.store.setting("update", {}).get("state")
                if not busy and state == "requested":
                    self.update_thread = Thread(target=self.prepare, daemon=True)
                    self.update_thread.start()
                elif state == "draining":
                    self.store.recover(self.config.lease_seconds)
                    if not self.store.one(
                        "SELECT id FROM jobs WHERE status='running' LIMIT 1"
                    ):
                        self.switch()
                        if self.restart_supervisor:
                            self.stop_children()
                            os.chdir(self.current)
                            os.execv(
                                self.python,
                                [
                                    self.python,
                                    "-m",
                                    "server.cli",
                                    "serve",
                                    "--config",
                                    str(self.config_path),
                                ],
                            )
                elif not busy and (
                    self.store.setting("check_update", False)
                    or time.time() - last_check > self.config.update_interval
                ):
                    self.store.set("check_update", False)
                    last_check = time.time()
                    self.update_thread = Thread(target=self.check, daemon=True)
                    self.update_thread.start()
                for index, (process, args, handle) in enumerate(self.children):
                    if process.poll() is not None:
                        logging.warning("Restarting exited child %s", index)
                        replacement = subprocess.Popen(
                            args,
                            cwd=self.current,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        self.children[index] = (replacement, args, handle)
        finally:
            self.stop_children()
            self.store.set("supervisor_seen", 0)
