"""Project storage shared by the API and GPU workers; accounts stay in Store."""

import fcntl
from contextlib import contextmanager
from pathlib import Path

from pipeline.workspace import Workspace


@contextmanager
def project_lock(config, sid, blocking=True):
    """Fence an API edit or a worker whose database lease has expired."""
    root = config.data / "locks"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{sid}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def make_workflow(config, device="cpu"):
    return Workspace(
        config.projects,
        device=device,
        **{key: Path(value) for key, value in config.model_paths().items()},
    )


def sync_usage(store, workflow, sid):
    root = workflow.directory(sid)
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    pages = len(workflow.load(sid)["pages"]) if (root / "session.json").exists() else 0
    store.execute("UPDATE projects SET pages=?,bytes=? WHERE id=?", (pages, size, sid))
    return size
