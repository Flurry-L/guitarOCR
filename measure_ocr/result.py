"""Persist a recognized or edited score and its downloadable projections.

Records in manifest.json are authoritative. Text and score.json are generated
from those records here, regardless of whether they came from OCR or an edit.
"""

from pathlib import Path

from shared.artifacts import write_json, write_result
from shared.constraints import gp5_timing_errors, validate_measure_target
from shared.m2 import format_measure_target, parse_measure_target
from shared.score_document import score_document
from shared.score_text import display_score_text, model_score_text
from shared.validation_messages import correction_message


def save_recognition(output: Path, result: dict) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    records = result["records"]
    result["measures"] = len(records)
    result["review_measures"] = [
        r["measure_number"] for r in records if r.get("needs_review")
    ]
    result["status"] = "needs_review" if result["review_measures"] else "complete"
    text = "\n".join(row["target"] for row in records) + "\n"
    for key, name, contents in (
        ("m2", "prediction.m2", text),
        ("score_text", "score.txt", display_score_text(text)),
    ):
        path = output / name
        path.write_text(contents, encoding="utf-8")
        result[key] = str(path)
    result["score_document"] = str(
        write_json(output / "score.json", score_document(result))
    )
    result.pop("stage", None)
    result.pop("schema_version", None)
    return write_result(output, "measure_ocr", **result)


def correct_measure(source, number, target=None, measure=None, reviewed=False):
    if not 1 <= number <= len(source["records"]):
        raise ValueError("无效的小节编号")
    row = source["records"][number - 1]
    mode = row.get("mode") or source["mode"]
    if reviewed and target is None and measure is None:
        # Acknowledging an unchanged result must not require serializing it
        # through the editor, which may not expose every notation field.
        target = row["target"]
    if measure is not None:
        try:
            target = format_measure_target(measure, mode, preserve_playback=True)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"小节结构无效：{error}") from error
    if not isinstance(target, str) or len(target) > 50000 or "\n" in target:
        raise ValueError("请输入单个小节的内容")
    target = model_score_text(target)
    parsed, errors = validate_measure_target(target, mode, tuning=source["tuning_used"])
    if errors:
        raise ValueError(correction_message(errors))
    if timing_errors := gp5_timing_errors(target):
        raise ValueError("；".join(timing_errors) + "，请调整起点或时值")
    changed = target != row["target"]
    row.update(
        target=target,
        manually_edited=changed or row.get("manually_edited", False),
        timing_errors=[],
    )
    if reviewed:
        row["needs_review"] = False
        row["reviewed"] = True
        row["pitch_needs_review"] = False
    elif changed:
        row["reviewed"] = False


def update_information(source, information):
    """Keep edits when their pitch interpretation still holds; otherwise return None."""
    if not (
        source["tuning_used"] == information["tuning_used"]
        and source.get("instrument", "guitar")
        == information.get("instrument", "guitar")
        and source.get("transpose") == information.get("transpose")
        and source.get("measure_pitch_contexts", [])
        == information.get("measure_pitch_contexts", [])
    ):
        return None

    previous_tempo = source["document_metadata"].get("tempo_quarter")
    if source.get("capo", 0) != information.get("capo", 0) and information.get(
        "instrument", "guitar"
    ) in {"guitar", "bass"}:
        for row in source["records"]:
            if (row.get("mode") or source["mode"]) != "tab":
                row.update(pitch_needs_review=True, needs_review=True)
    for key in ("title", "artist", "tuning_used", "capo", "document_metadata"):
        source[key] = information[key]
    source["instrument"] = information.get("instrument", "guitar")
    source["midi_program"] = information.get("midi_program", 25)
    tempo = information["document_metadata"].get("tempo_quarter")
    if source["records"] and tempo and tempo != previous_tempo:
        first = parse_measure_target(source["records"][0]["target"])
        first["tempo_quarter"] = int(tempo)
        source["records"][0]["target"] = format_measure_target(
            first,
            source["records"][0].get("mode") or source["mode"],
            preserve_playback=True,
        )
    return source
