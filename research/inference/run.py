"""Recognize documents through the same native backend used by the application."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import httpx


class NativeClient:
    def __init__(self, endpoint="http://127.0.0.1:7860"):
        self.http = httpx.Client(base_url=endpoint.rstrip("/"), timeout=120)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.http.close()

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"{method} {path}: {response.status_code} {response.text}")
        return response.json()

    def wait(self, sid):
        try:
            while True:
                project = self.request("GET", f"/api/sessions/{sid}")
                job = project.get("job") or {}
                if job.get("status") not in {"queued", "running"}:
                    if job.get("status") in {"failed", "cancelled"}:
                        raise RuntimeError(job.get("error") or job.get("message") or job["status"])
                    return project
                time.sleep(.5)
        except KeyboardInterrupt:
            self.request("POST", f"/api/sessions/{sid}/cancel")
            raise

    def recognize(self, inputs, output, *, mode="auto", session=None, action="process"):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        detect_after_import = session is None and action == "detect"
        if session:
            project = self.request("GET", f"/api/sessions/{session}")
            self.request("POST", f"/api/sessions/{session}/{action}", json={"mode": mode},
                         headers={"If-Match": str(project["revision"])})
        else:
            with ExitStack() as files:
                upload = [("files", (Path(path).name, files.enter_context(Path(path).open("rb"))))
                          for path in inputs]
                session = self.request("POST", "/api/sessions", files=upload,
                                       data={"action": "import" if detect_after_import else "full",
                                             "mode": mode})["id"]
        # The backend owns the project and its resume journal. Retain its ID even
        # when recognition is interrupted so the app or CLI can continue it.
        (output / "project.json").write_text(json.dumps({
            "endpoint": str(self.http.base_url), "id": session,
        }, indent=2) + "\n")
        project = self.wait(session)
        if detect_after_import:
            self.request("POST", f"/api/sessions/{session}/detect", json={"mode": mode},
                         headers={"If-Match": str(project["revision"])})
            project = self.wait(session)
        project["seconds"] = time.monotonic() - started
        for key, name in [("score_text_url", "score.txt"),
                          ("score_document_url", "score.json"),
                          ("musicxml_url", "score.musicxml")]:
            if project.get(key):
                response = self.http.get(project[key])
                response.raise_for_status()
                (output / name).write_bytes(response.content)
        (output / "result.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n")
        return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:7860")
    parser.add_argument("--mode", choices=("auto", "tab", "notation", "both"), default="auto")
    parser.add_argument("--session", help="Continue an existing native project")
    parser.add_argument("--action", choices=("process", "detect"), default="process")
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.session):
        parser.error("Supply input files or --session")
    with NativeClient(args.endpoint) as client:
        project = client.recognize(args.inputs, args.output, mode=args.mode, session=args.session,
                                   action=args.action)
    print(json.dumps({key: project.get(key) for key in (
        "id", "mode", "review_measures", "seconds",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
