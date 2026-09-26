from dataclasses import asdict, dataclass
import json
from pathlib import Path
from urllib.parse import urlsplit

from shared.defaults import (
    MODEL,
    LAYOUT_MODEL,
    INFO_ADAPTER,
    MEASURE_ADAPTER,
    paddle_python,
)


@dataclass
class Config:
    data: Path
    public_url: str = "http://localhost:8080"
    host: str = "127.0.0.1"
    port: int = 8080
    gpus: str = "0"
    browser_workers: int = 2
    model: str = str(MODEL.resolve())
    layout_model: str = str(LAYOUT_MODEL.resolve())
    layout_python: str = str(paddle_python().absolute())
    info_adapter: str = str(INFO_ADAPTER.resolve())
    measure_adapter: str = str(MEASURE_ADAPTER.resolve())
    max_upload_mb: int = 100
    max_pages: int = 50
    max_projects: int = 30
    max_pending: int = 3
    storage_mb: int = 2048
    session_days: int = 7
    lease_seconds: int = 90
    repository: str = "https://github.com/Flurry-L/guitarOCR.git"
    branch: str = "main"
    update_interval: int = 900
    source: str = str(Path(__file__).resolve().parent.parent)

    def __post_init__(self):
        self.data = Path(self.data).resolve()
        parsed = urlsplit(self.public_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            raise ValueError(
                "public_url 必须是完整站点地址，例如 https://ocr.example.com"
            )
        self.public_url = self.public_url.rstrip("/")
        for field in (
            "max_upload_mb",
            "max_pages",
            "max_projects",
            "max_pending",
            "storage_mb",
            "lease_seconds",
            "session_days",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} 必须大于零")
        if self.browser_workers < 0:
            raise ValueError("browser_workers 不能小于零")
        if any(
            not value.strip().isdigit()
            for value in self.gpus.split(",")
            if value.strip()
        ):
            raise ValueError("gpus 使用逗号分隔的 GPU 编号，或空字符串")

    @property
    def database(self):
        return self.data / "service.sqlite3"

    @property
    def projects(self):
        return self.data / "projects"

    @property
    def browser_models(self):
        return self.data / "browser-models"

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self):
        self.data.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.data / "config.json"
        path.write_text(json.dumps(asdict(self), default=str, indent=2) + "\n")
        path.chmod(0o600)
        return path
