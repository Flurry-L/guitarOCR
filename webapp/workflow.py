"""Persistent editing sessions; stages remain independently runnable."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from shared.defaults import MODEL, MEASURE_ADAPTER, INFO_ADAPTER
from uuid import uuid4
import json

from PIL import Image
from layout.pages import expand_inputs
from layout.edit import save_layout
from layout import run as layout_stage
from document_info import run as info_stage
from measure_ocr import run as measure_stage
from gp5_export import run as export_stage
from shared.artifacts import read_result, write_json, write_result
from shared.constraints import validate_measure_target
from shared.glm_backend import BackendPool
from shared.m2 import format_measure_target, parse_measure_target
from shared.tuning import DEFAULT_TUNING


class Workflow:
    def __init__(
        self,
        root: Path,
        *,
        model=MODEL,
        device="cuda",
        layout_model=None,
        layout_python=None,
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.device = model, device
        self.layout_model, self.layout_python = layout_model, layout_python
        self.pool = BackendPool(model, device)
        self.info_adapter = INFO_ADAPTER
        self.measure_adapter = MEASURE_ADAPTER

    def directory(self, sid):
        if len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
            raise ValueError("无效的项目编号")
        return self.root / sid

    def load(self, sid):
        return json.loads(
            (self.directory(sid) / "session.json").read_text(encoding="utf-8")
        )

    def store(self, state):
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
                "mode": "tab",
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

    def detect(self, sid, mode="tab", source="auto"):
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
        )
        data = read_result(result, "layout")
        state.update(
            layout=str(result),
            mode=data["mode"],
            boxes=[
                {"kind": "measure", "page": r["page"], "bbox": r["bbox"]}
                for r in data["records"]
            ]
            + [
                {"kind": r["kind"], "page": r["page"], "bbox": r["bbox"]}
                for r in data["regions"]
            ],
        )
        self.invalidate(state, "layout")
        return self.store(state)

    def boxes(self, sid, boxes, mode):
        state = self.load(sid)
        result = save_layout(state["pages"], boxes, self.output(sid, "layout"), mode)
        state.update(layout=str(result), boxes=boxes, mode=mode)
        self.invalidate(state, "layout")
        return self.store(state)

    def information(self, sid):
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
        )
        self._information_changed(state, result)
        return self.store(state)

    def _save_recognition(self, sid, source, stage="correction"):
        out = self.output(sid, stage)
        out.mkdir(parents=True)
        m2 = out / "prediction.m2"
        m2.write_text(
            "\n".join(r["target"] for r in source["records"]) + "\n", encoding="utf-8"
        )
        source["m2"] = str(m2)
        return str(write_json(out / "manifest.json", source))

    def _information_changed(self, state, result):
        information = read_result(result, "document_info")
        source = (
            read_result(Path(state["recognition"]), "measure_ocr")
            if state["recognition"]
            else None
        )
        state["info"] = str(result)
        state.pop("ocr_task", None)
        if source and source["tuning_used"] == information["tuning_used"]:
            previous_tempo = source["document_metadata"].get("tempo_quarter")
            for key in ("title", "artist", "tuning_used", "capo", "document_metadata"):
                source[key] = information[key]
            source["info"] = str(result)
            tempo = information["document_metadata"].get("tempo_quarter")
            if source["records"] and tempo and tempo != previous_tempo:
                first = parse_measure_target(source["records"][0]["target"])
                first["tempo_quarter"] = int(tempo)
                source["records"][0]["target"] = format_measure_target(
                    first, source["mode"], preserve_playback=True
                )
            state["recognition"] = self._save_recognition(
                state["id"], source, "metadata"
            )
            self.invalidate(state, "recognition")
        else:
            self.invalidate(state, "info")

    def metadata(self, sid, values):
        state = self.load(sid)
        if not state["layout"]:
            raise ValueError("请先保存小节框")
        tuning = values.get("tuning_used", list(DEFAULT_TUNING))
        if not 1 <= len(tuning) <= 7 or any(
            type(p) is not int or not 0 <= p <= 127 for p in tuning
        ):
            raise ValueError("GP5 支持 1–7 根弦，调弦使用 0–127 的 MIDI 音高")
        capo, tempo = values.get("capo", 0), values.get("tempo_quarter", 120)
        if (
            type(capo) is not int
            or not 0 <= capo <= 24
            or type(tempo) is not int
            or not 20 <= tempo <= 400
        ):
            raise ValueError("变调夹范围 0–24，速度范围 20–400")
        metadata = (
            read_result(Path(state["info"]), "document_info")["document_metadata"]
            if state["info"]
            else {}
        )
        metadata.update(tempo_quarter=tempo)
        result = write_result(
            self.output(sid, "info"),
            "document_info",
            layout=state["layout"],
            document_metadata=metadata,
            predictions=None,
            title=str(values.get("title", "Untitled"))[:500],
            artist=str(values.get("artist", ""))[:500],
            tuning_used=tuning,
            capo=capo,
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
        if not 1 <= number <= len(source["records"]):
            raise ValueError("无效的小节编号")
        if measure is not None:
            try:
                target = format_measure_target(
                    measure, source["mode"], preserve_playback=True
                )
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"小节结构无效：{error}") from error
        if not isinstance(target, str) or len(target) > 50000 or "\n" in target:
            raise ValueError("请输入单个小节的内容")
        parsed, errors = validate_measure_target(
            target, source["mode"], tuning=source["tuning_used"]
        )
        if errors:
            raise ValueError("小节内容无效：" + "; ".join(errors))
        row = source["records"][number - 1]
        row.update(target=target, manually_edited=True)
        if reviewed:
            row["needs_review"] = False
        source["review_measures"] = [
            r["measure_number"] for r in source["records"] if r.get("needs_review")
        ]
        source["status"] = "needs_review" if source["review_measures"] else "complete"
        state["recognition"] = self._save_recognition(sid, source)
        self.invalidate(state, "recognition")
        return self.store(state)

    def export(self, sid):
        state = self.load(sid)
        if not state["recognition"]:
            raise ValueError("请先识别并校对小节")
        state["export"] = str(
            export_stage.run(Path(state["recognition"]), self.output(sid, "gp5"))
        )
        state["revision"] += 1
        return self.store(state)

    def public(self, sid):
        state = deepcopy(self.load(sid))

        def asset(path):
            relative = Path(path).resolve().relative_to(self.directory(sid))
            return f"/api/sessions/{sid}/files/{relative.as_posix()}"

        for page in state["pages"]:
            page["url"] = asset(page["image"])
        state["metadata"] = (
            read_result(Path(state["info"]), "document_info") if state["info"] else None
        )
        state["measures"] = []
        state["review_measures"] = []
        if state["recognition"]:
            data = read_result(Path(state["recognition"]), "measure_ocr")
            state["review_measures"] = data.get("review_measures", [])
            state["m2_url"] = asset(data["m2"])
            for row in data["records"]:
                state["measures"].append(
                    {
                        **row,
                        "url": asset(row["image"]),
                        "parsed": parse_measure_target(row["target"]),
                    }
                )
        if state["export"]:
            exported = read_result(Path(state["export"]), "gp5_export")
            state["gp5_url"] = asset(exported["gp5"])
            state["encoding_url"] = asset(exported["encoding_report"])
        return state
