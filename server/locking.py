"""A project lock also fences a worker whose database lease expired."""

from contextlib import contextmanager
import fcntl


@contextmanager
def project_lock(config, sid, blocking=True):
    root = config.data / "locks"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{sid}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
