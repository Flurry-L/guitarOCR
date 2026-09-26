"""SQLite is the queue authority; model work never runs inside a transaction."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
from uuid import uuid4


ACTIVE = ("queued", "running")


def password_hash(password, salt=None):
    if not 10 <= len(password) <= 128:
        raise ValueError("密码长度须为 10 至 128 个字符")
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1
    ).hex()
    return f"{salt}:{digest}"


def check_password(password, encoded):
    try:
        return secrets.compare_digest(
            password_hash(password, encoded.split(":")[0]), encoded
        )
    except (ValueError, TypeError):
        return False


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, username TEXT UNIQUE COLLATE NOCASE NOT NULL,
                password TEXT NOT NULL, admin INTEGER NOT NULL DEFAULT 0,
                disabled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                csrf TEXT NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                title TEXT NOT NULL, engine TEXT NOT NULL, created REAL NOT NULL,
                pages INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id), engine TEXT NOT NULL,
                action TEXT NOT NULL, params TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, started REAL, finished REAL,
                lease TEXT, heartbeat REAL, attempts INTEGER NOT NULL DEFAULT 0,
                cancel INTEGER NOT NULL DEFAULT 0, done INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '', error TEXT,
                seconds REAL NOT NULL DEFAULT 0, browser_seen REAL, browser_token TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_project_job ON jobs(project_id)
                WHERE status IN ('queued','running');
            CREATE INDEX IF NOT EXISTS queue_order ON jobs(engine,status,created);
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                lease TEXT NOT NULL, payload TEXT NOT NULL, result TEXT, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rate_limits (key TEXT PRIMARY KEY, count INTEGER NOT NULL, until REAL NOT NULL);
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=20000")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def one(self, sql, args=()):
        with self.connect() as db:
            row = db.execute(sql, args).fetchone()
            return dict(row) if row else None

    def all(self, sql, args=()):
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, args)]

    def execute(self, sql, args=()):
        with self.connect(True) as db:
            return db.execute(sql, args).rowcount

    def setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.execute(
            "INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value))
        )

    def add_user(self, username, password, admin=False):
        uid = uuid4().hex
        self.execute(
            "INSERT INTO users(id,username,password,admin,created) VALUES (?,?,?,?,?)",
            (uid, username, password_hash(password), int(admin), time.time()),
        )
        return uid

    def rate_limit(self, key, count, interval):
        now = time.time()
        with self.connect(True) as db:
            db.execute("DELETE FROM rate_limits WHERE until<?", (now,))
            row = db.execute("SELECT * FROM rate_limits WHERE key=?", (key,)).fetchone()
            if row and row["count"] >= count:
                return False
            db.execute(
                "INSERT INTO rate_limits VALUES (?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1",
                (key, now + interval),
            )
            return True

    def enqueue(self, project, action, params, max_pending=3):
        jid, now = uuid4().hex, time.time()
        with self.connect(True) as db:
            if (
                db.execute(
                    "SELECT count(*) FROM jobs WHERE user_id=? AND status IN ('queued','running')",
                    (project["user_id"],),
                ).fetchone()[0]
                >= max_pending
            ):
                raise ValueError("待处理任务已达上限，请等待完成或取消任务")
            if db.execute(
                "SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued','running')",
                (project["id"],),
            ).fetchone():
                raise ValueError("这个项目正在处理")
            db.execute(
                """INSERT INTO jobs(id,project_id,user_id,engine,action,params,status,created,message)
                VALUES (?,?,?,?,?,?,'queued',?,'等待处理')""",
                (
                    jid,
                    project["id"],
                    project["user_id"],
                    project["engine"],
                    action,
                    json.dumps(params),
                    now,
                ),
            )
        return self.one("SELECT * FROM jobs WHERE id=?", (jid,))

    def recover(self, lease_seconds):
        now = time.time()
        with self.connect(True) as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE status='running' AND heartbeat<?",
                (now - lease_seconds,),
            ).fetchall()
            for job in rows:
                status = (
                    "cancelled"
                    if job["cancel"]
                    else ("failed" if job["attempts"] >= 3 else "queued")
                )
                db.execute(
                    """UPDATE jobs SET status=?,lease=NULL,heartbeat=NULL,finished=?,
                    seconds=seconds+max(0,coalesce(heartbeat,started)-started),message=?,error=? WHERE id=?""",
                    (
                        status,
                        None if status == "queued" else now,
                        "工作进程重启，等待继续"
                        if status == "queued"
                        else "任务已停止",
                        "工作进程多次中断，请联系管理员"
                        if status == "failed"
                        else None,
                        job["id"],
                    ),
                )
                db.execute("DELETE FROM calls WHERE job_id=?", (job["id"],))

    def claim(self, engine):
        now, lease = time.time(), secrets.token_hex(24)
        with self.connect(True) as db:
            paused = db.execute(
                "SELECT value FROM settings WHERE key='queue_paused'"
            ).fetchone()
            if paused and json.loads(paused[0]):
                return None
            row = db.execute(
                """SELECT j.* FROM jobs j JOIN users u ON u.id=j.user_id
                WHERE j.status='queued' AND j.engine=? AND j.cancel=0 AND u.disabled=0
                AND (? != 'browser' OR j.browser_seen>?)
                AND NOT EXISTS (SELECT 1 FROM jobs r WHERE r.user_id=j.user_id AND r.engine=j.engine AND r.status='running')
                ORDER BY j.created,j.id LIMIT 1""",
                (engine, engine, now - 30),
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET status='running',lease=?,heartbeat=?,started=?,attempts=attempts+1,message='正在处理' WHERE id=?",
                (lease, now, now, row["id"]),
            )
            return dict(
                db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            )

    def heartbeat(self, job):
        return (
            self.execute(
                "UPDATE jobs SET heartbeat=? WHERE id=? AND lease=? AND status='running'",
                (time.time(), job["id"], job["lease"]),
            )
            == 1
        )

    def finish(self, job, status, error=None):
        now = time.time()
        message = {
            "complete": "处理完成",
            "failed": "处理失败",
            "cancelled": "已取消",
            "queued": "等待浏览器继续",
        }[status]
        with self.connect(True) as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE id=? AND lease=? AND status='running'",
                (job["id"], job["lease"]),
            ).fetchone()
            if not row:
                return False
            if row["cancel"]:
                status, message = "cancelled", "已取消"
            db.execute(
                """UPDATE jobs SET status=?,error=?,message=?,finished=?,lease=NULL,
                seconds=seconds+max(0,?-started), attempts=CASE WHEN ?='queued' THEN max(0,attempts-1) ELSE attempts END
                WHERE id=?""",
                (
                    status,
                    error,
                    message,
                    None if status == "queued" else now,
                    now,
                    status,
                    job["id"],
                ),
            )
            db.execute("DELETE FROM calls WHERE job_id=?", (job["id"],))
        return True

    def cancel(self, project_id):
        self.execute(
            """UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
            finished=CASE WHEN status='queued' THEN ? ELSE finished END,message='正在停止'
            WHERE project_id=? AND status IN ('queued','running')""",
            (time.time(), project_id),
        )

    def job(self, sid):
        return self.one(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created DESC,id DESC LIMIT 1",
            (sid,),
        )

    def public_job(self, row):
        if not row:
            return None
        result = {
            k: row[k]
            for k in (
                "id",
                "status",
                "engine",
                "action",
                "created",
                "started",
                "finished",
                "done",
                "total",
                "message",
                "error",
                "seconds",
            )
        }
        result["cancellable"] = row["status"] in ACTIVE and not row["cancel"]
        result["position"] = 0
        if row["status"] == "queued":
            result["position"] = self.one(
                "SELECT count(*) AS n FROM jobs WHERE status='queued' AND engine=? AND created<=?",
                (row["engine"], row["created"]),
            )["n"]
        return result
