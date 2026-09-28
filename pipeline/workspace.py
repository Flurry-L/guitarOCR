"""Persistent editing sessions; stages remain independently runnable."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from PIL import Image

from document_info import run as info_stage
from document_info.edit import edited_information
from gp5_export import run as export_stage
from layout import run as layout_stage
from layout.edit import save_layout
from layout.pages import expand_inputs
from measure_ocr import run as measure_stage
from measure_ocr.result import correct_measure, save_recognition, update_information
from shared.artifacts import read_result, write_json, write_result
from shared.defaults import INFO_ADAPTER, MEASURE_ADAPTER, MODEL
from shared.glm_backend import BackendPool
from shared.score_text import display_score_text


class Workspace:
    def __init__(
        self,
        root: Path,
        *,
        model=MODEL,
        device="cuda",
        layout_model=None,
        layout_python=None,
        info_adapter=INFO_ADAPTER,
        measure_adapter=MEASURE_ADAPTER,
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.device = model, device
        self.layout_model, self.layout_python = layout_model, layout_python
        self.pool = BackendPool(model, device)
        self._state_lock = Lock()
        self.info_adapter = Path(info_adapter)
        self.measure_adapter = Path(measure_adapter)
        from layout.persistent import LayoutBackend

        self.layout_detector = LayoutBackend(layout_model, layout_python, device)

    def close(self):
        self.pool.close()
        self.layout_detector.close()

    def warmup(self):
        from concurrent.futures import ThreadPoolExecutor

        def prepare_state_reader():
            path = self.measure_adapter / 'capabilities.json'
            capabilities = json.loads(path.read_text()) if path.exists() else {}
            if capabilities.get('state_reader'):
                from measure_ocr.state_reader import cached_reader

                weights = (path.parent / capabilities['state_reader']).resolve()
                cached_reader(str(weights), self.device, weights.stat().st_mtime_ns)

        with ThreadPoolExecutor(max_workers=3) as loaders:
            jobs = [loaders.submit(self.pool.prepare, [self.info_adapter, self.measure_adapter]),
                    loaders.submit(self.layout_detector, []),
                    loaders.submit(prepare_state_reader)]
            errors = []
            for job in jobs:
                try:
                    job.result()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]

    def directory(self, sid):
        if len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
            raise ValueError("无效的项目编号")
        return self.root / sid

    def load(self, sid):
        with self._state_lock:
            return json.loads(
                (self.directory(sid) / "session.json").read_text(encoding="utf-8")
            )

    def store(self, state):
        # Windows cannot reliably open a file during its atomic replacement.
        with self._state_lock:
            write_json(self.directory(state["id"]) / "session.json", state)
        return state

    def output(self, sid, stage):
        return self.directory(sid) / f"{stage}_{uuid4().hex[:12]}"

    def create(self, sid, inputs, input_names=None):
        root = self.directory(sid)
        pages = expand_inputs(inputs, root / "pages", root / "tmp", False)
        if not pages:
            raise ValueError("上传内容没有可读取的页面")
        for p in pages:
            p["image"] = str(p["image"].resolve())
            if p.get("source_pdf"):
                p["source_pdf"] = str(p["source_pdf"].resolve())
            with Image.open(p["image"]) as image:
                p["width"], p["height"] = image.size
        return self.store(
            {
                "id": sid,
                "pages": pages,
                "inputs": [str(p) for p in inputs],
                "input_names": input_names or [p.name for p in inputs],
                "mode": "auto",
                "mode_setting": "auto",
                "boxes": [],
                "layout": None,
                "info": None,
                "recognition": None,
                "export": None,
                "revision": 0,
            }
        )

    def invalidate(self, state, stage):
        state.pop("ocr_task", None)
        order = ["layout", "info", "recognition", "export"]
        for key in order[order.index(stage) + 1 :]:
            state[key] = None
        state["revision"] += 1

    @staticmethod
    def _layout_boxes(data):
        return [
            {
                "kind": "measure",
                "page": r["page"],
                "bbox": r["bbox"],
                "mode": r.get("mode") or data["mode"],
            }
            for r in data["records"]
        ] + [
            {"kind": r["kind"], "page": r["page"], "bbox": r["bbox"]}
            for r in data["regions"]
        ]

    def detect(self, sid, mode="auto", source="auto"):
        state = self.load(sid)
        result = layout_stage.run(
            [Path(p) for p in state["inputs"]],
            self.output(sid, "layout"),
            mode=mode,
            layout_source=source,
            pages=state["pages"],
            layout_model_dir=self.layout_model,
            layout_python=self.layout_python,
            allow_empty=True,
            detector=self.layout_detector,
        )
        data = read_result(result, "layout")
        state.update(
            layout=str(result),
            mode=data["mode"],
            mode_setting=mode,
            pages=data.get("pages", state["pages"]),
            boxes=self._layout_boxes(data),
        )
        self.invalidate(state, "layout")
        return self.store(state)

    def boxes(self, sid, boxes, mode):
        state = self.load(sid)
        result = save_layout(state["pages"], boxes, self.output(sid, "layout"), mode)
        data = read_result(result, "layout")
        state.update(
            layout=str(result),
            boxes=self._layout_boxes(data),
            pages=data["pages"],
            mode=data["mode"],
            mode_setting=mode,
        )
        self.invalidate(state, "layout")
        return self.store(state)

    def information(self, sid, cancelled=None):
        state = self.load(sid)
        if not state["layout"]:
            raise ValueError("请先保存小节框")
        result = info_stage.run(
            Path(state["layout"]),
            self.output(sid, "info"),
            model=self.model,
            device=self.device,
            adapter=self.info_adapter,
            backend=self.pool.adapter(self.info_adapter),
            cancelled=cancelled,
        )
        self._information_changed(state, result)
        return self.store(state)

    def _information_changed(self, state, result):
        information = read_result(result, "document_info")
        source = (
            read_result(Path(state["recognition"]), "measure_ocr")
            if state["recognition"]
            else None
        )
        state["info"] = str(result)
        state.pop("ocr_task", None)
        if source is not None:
            source = update_information(source, information)
        if source is not None:
            source["info"] = str(result)
            state["recognition"] = str(
                save_recognition(self.output(state["id"], "metadata"), source)
            )
            self.invalidate(state, "recognition")
        else:
            self.invalidate(state, "info")

    def metadata(self, sid, values):
        state = self.load(sid)
        if not state["layout"]:
            raise ValueError("请先保存小节框")
        previous = (
            read_result(Path(state["info"]), "document_info") if state["info"] else {}
        )
        layout = read_result(Path(state["layout"]), "layout")
        result = write_result(
            self.output(sid, "info"),
            "document_info",
            layout=state["layout"],
            **edited_information(layout, previous, values),
        )
        self._information_changed(state, result)
        return self.store(state)

    def recognize(
        self, sid, progress=None, *, resume=False, measures=None, cancelled=None
    ):
        state = self.load(sid)
        if not state["layout"] or not state["info"]:
            raise ValueError("请先保存小节框和谱面信息")
        if not read_result(Path(state["layout"]), "layout")["records"]:
            raise ValueError("请至少添加一个小节框")
        if resume and not state.get("ocr_task"):
            raise ValueError("没有可以继续的识别任务")
        task = state.get("ocr_task") if resume else None
        if not task:
            if measures and not state["recognition"]:
                raise ValueError("请先完成一次识别，再重试指定小节")
            task = {
                "output": str(self.output(sid, "ocr")),
                "measures": measures,
                "source": state["recognition"] if measures else None,
            }
            state["ocr_task"] = task
            self.store(state)
        seeds = (
            read_result(Path(task["source"]), "measure_ocr")["records"]
            if task["source"]
            else None
        )
        result = measure_stage.run(
            Path(state["layout"]),
            Path(state["info"]),
            Path(task["output"]),
            model=self.model,
            device=self.device,
            adapter=self.measure_adapter,
            backend=self.pool.adapter(self.measure_adapter),
            progress=progress,
            resume=resume,
            initial_records=seeds,
            retry_measures=task["measures"],
            cancelled=cancelled,
        )
        state["recognition"] = str(result)
        state.pop("ocr_task", None)
        self.invalidate(state, "recognition")
        return self.store(state)

    def correct(self, sid, number, target=None, measure=None, reviewed=False):
        state = self.load(sid)
        if not state["recognition"]:
            raise ValueError("请先识别小节")
        source = read_result(Path(state["recognition"]), "measure_ocr")
        correct_measure(
            source, number, target=target, measure=measure, reviewed=reviewed
        )
        state["recognition"] = str(
            save_recognition(self.output(sid, "correction"), source)
        )
        self.invalidate(state, "recognition")
        return self.store(state)

    def export(self, sid):
        state = self.load(sid)
        if not state["recognition"]:
            raise ValueError("请先识别并校对小节")
        exported = export_stage.run(
            Path(state["recognition"]), self.output(sid, "gp5"),
            fallback_name=(state.get("input_names") or state["inputs"])[0],
        )
        state["export"] = str(exported) if read_result(exported, 'gp5_export').get('gp5') else None
        state["revision"] += 1
        return self.store(state)

    def score_text(self, sid):
        state = self.load(sid)
        if not state["recognition"]:
            raise ValueError("请先识别小节")
        source = read_result(Path(state["recognition"]), "measure_ocr")
        return display_score_text(
            "\n".join(row["target"] for row in source["records"]) + "\n"
        )
