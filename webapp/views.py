"""Project responses for both HTTP applications; storage paths stay on disk."""

from pathlib import Path
from urllib.parse import quote

from shared.artifacts import read_result
from shared.m2 import parse_measure_target
from shared.score_text import display_error, display_score_text


def project_view(workspace, sid):
    saved = workspace.load(sid)
    state = {
        key: saved[key]
        for key in (
            "id",
            "revision",
            "mode",
            "boxes",
            "input_names",
        )
        if key in saved
    }
    state["input_names"] = saved.get("input_names") or [
        Path(p).name for p in saved.get("inputs", [])
    ]
    state["mode_setting"] = saved.get("mode_setting", saved["mode"])
    state.update(
        {
            key: bool(saved.get(key))
            for key in ("layout", "info", "recognition", "export")
        }
    )
    if saved.get("ocr_task"):
        state["ocr_task"] = True

    def asset(path):
        relative = Path(path).resolve().relative_to(workspace.directory(sid))
        return f"/api/sessions/{sid}/files/{quote(relative.as_posix())}"

    state["pages"] = [
        {
            **{k: v for k, v in page.items() if k not in {"image", "source_pdf"}},
            "url": asset(page["image"]),
        }
        for page in saved["pages"]
    ]
    state["metadata"] = None
    if saved["info"]:
        info = read_result(Path(saved["info"]), "document_info")
        state["metadata"] = {
            k: info[k]
            for k in (
                "title",
                "artist",
                "instrument",
                "midi_program",
                "tuning_used",
                "capo",
                "transpose",
                "document_metadata",
            )
            if k in info
        }
        metadata = dict(state["metadata"]["document_metadata"])
        if "pitch_instructions" in metadata:
            metadata["pitch_instructions"] = [
                {
                    k: instruction[k]
                    for k in ("kind", "parsed", "page", "bbox")
                    if k in instruction
                }
                for instruction in metadata["pitch_instructions"]
            ]
        state["metadata"]["document_metadata"] = metadata
    state["measures"], state["review_measures"] = [], []
    if saved["recognition"]:
        data = read_result(Path(saved["recognition"]), "measure_ocr")
        state["review_measures"] = data.get("review_measures", [])
        state["score_text_url"] = f"/api/sessions/{sid}/score.txt"
        for row in data["records"]:
            state["measures"].append(
                {
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in {"image", "source_page", "source_pdf"}
                    },
                    "score_text": display_score_text(row["target"]),
                    "context_text": display_score_text(
                        row.get("previous_context") or ""
                    ),
                    "fallback_reason": [
                        display_error(reason)
                        for reason in (row.get("fallback_reason") or [])
                    ],
                    "url": asset(row["image"]),
                    "parsed": parse_measure_target(row["target"]),
                }
            )
    if saved["export"]:
        exported = read_result(Path(saved["export"]), "gp5_export")
        state["gp5_url"] = asset(exported["gp5"])
        state["gp5_name"] = Path(exported["gp5"]).name
        state["encoding_url"] = asset(exported["encoding_report"])
    return state
