"""Persist model calls for a browser's Web Worker; no GPU fallback is allowed."""

from copy import deepcopy
import json
from pathlib import Path
import time
from uuid import uuid4

from PIL import Image
from layout.postprocess import order_measure_boxes, refine_measure_boxes
from shared.layout_labels import mode_vote
from shared.tasks import Cancelled
from server.worker import BrowserDisconnected


class BrowserBridge:
    def __init__(self, config, store, workflow, job, stop):
        self.config, self.store, self.workflow, self.job, self.stop = (
            config,
            store,
            workflow,
            job,
            stop,
        )

    def url(self, path):
        sid = self.job["project_id"]
        relative = Path(path).resolve().relative_to(self.workflow.directory(sid))
        return f"/api/sessions/{sid}/files/{relative.as_posix()}"

    def call(self, payload):
        cid = uuid4().hex
        bundle = json.loads(self.job["params"]).get("browser_bundle")
        if bundle:
            payload["model_root"] = f"/api/browser-models/{bundle}/"
        self.store.execute(
            "INSERT INTO calls(id,job_id,lease,payload,created) VALUES (?,?,?,?,?)",
            (cid, self.job["id"], self.job["lease"], json.dumps(payload), time.time()),
        )
        try:
            while not self.stop.wait(0.5):
                job = self.store.one("SELECT * FROM jobs WHERE id=?", (self.job["id"],))
                if not job or job["cancel"] or job["lease"] != self.job["lease"]:
                    raise Cancelled("任务已取消")
                if not job["browser_seen"] or job["browser_seen"] < time.time() - 60:
                    raise BrowserDisconnected()
                row = self.store.one("SELECT result FROM calls WHERE id=?", (cid,))
                if row and row["result"] is not None:
                    result = json.loads(row["result"])
                    if result.get("error"):
                        raise ValueError(
                            "Browser inference failed: " + result["error"][:500]
                        )
                    return result
            raise BrowserDisconnected()
        finally:
            self.store.execute("DELETE FROM calls WHERE id=?", (cid,))

    def adapter(self, path):
        return BrowserOCR(
            self,
            "information" if Path(path) == self.workflow.info_adapter else "measures",
        )

    def detect(self, paths):
        results = []
        for path in paths:
            result = self.call({"kind": "layout", "image": self.url(path)})
            boxes = result["boxes"]
            ordered = order_measure_boxes(boxes, 0.25)
            with Image.open(path) as image:
                measures = refine_measure_boxes(image, ordered)
            results.append(
                {
                    "measures": measures,
                    "tempo_regions": [
                        b
                        for b in boxes
                        if b["label"] == "tempo_region" and b["score"] >= 0.25
                    ],
                    **mode_vote(measures),
                }
            )
        return results


class BrowserOCR:
    def __init__(self, bridge, kind):
        self.bridge, self.kind = bridge, kind

    def generate(self, messages, max_new_tokens, *, skip_special_tokens=True):
        messages = deepcopy(messages)
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    content["url"] = self.bridge.url(content["url"])
        result = self.bridge.call(
            {
                "kind": self.kind,
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": skip_special_tokens,
            }
        )
        return result["text"], result["tokens"]
