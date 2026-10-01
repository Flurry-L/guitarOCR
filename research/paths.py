import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configured_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name)
    return Path(value).expanduser() if value else default


DATABASE_ROOT = configured_path("GUITAROCR_DATABASE_ROOT", PROJECT_ROOT / "database")
