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
    document = score_document(result)
    result["score_document"] = str(write_json(output / "score.json", document))
    from shared.musicxml import write_musicxml

    result.pop('musicxml', None)
    result.pop('musicxml_error', None)
    try:
        result['musicxml'] = str(write_musicxml(document, output / 'score.musicxml'))
    except ValueError as error:
        # The score IR remains usable even when a projection cannot represent
        # incomplete metadata. Never lose OCR records to an optional export.
        result['musicxml_error'] = str(error)
        (output / 'score.musicxml').unlink(missing_ok=True)
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
    tuning = row.get('tuning', source['tuning_used'])
    instrument = row.get('instrument', source.get('instrument', 'guitar'))
    parsed, errors = validate_measure_target(target, mode, tuning=tuning)
    if errors:
        raise ValueError(correction_message(errors))
    if timing_errors := gp5_timing_errors(target):
        raise ValueError("；".join(timing_errors) + "，请调整起点或时值")
    if mode == 'notation' and instrument in {'guitar', 'bass'} and row.get('tuning_explicit'):
        from gp5_export.fingering import notation_fingering_errors

        if fingering_errors := notation_fingering_errors(parsed, tuning):
            raise ValueError('音高或和弦超出当前调弦的可演奏范围，请调整音符或调弦：' + '; '.join(fingering_errors[:3]))
    changed = target != row["target"]
    row.update(
        target=target,
        manually_edited=changed or row.get("manually_edited", False),
        timing_errors=[],
        fingering_errors=[],
        export_errors=[],
    )
    if reviewed:
        row["needs_review"] = False
        row["reviewed"] = True
        row["pitch_needs_review"] = False
    elif changed:
        row["reviewed"] = False


def update_information(source, information):
    """Keep edits when their pitch interpretation still holds; otherwise return None."""
    def part_contexts(value):
        return {p['id']: {k: p.get(k) for k in ('instrument', 'tuning_used', 'transpose')}
                for p in value.get('parts', [])}

    def pitch_contexts(value):
        return {r['measure_number']: r for r in value.get('measure_pitch_contexts', [])}

    if part_contexts(source) != part_contexts(information):
        return None
    if not (
        source["tuning_used"] == information["tuning_used"]
        and source.get("instrument", "guitar")
        == information.get("instrument", "guitar")
        and source.get("transpose") == information.get("transpose")
        and pitch_contexts(source) == pitch_contexts(information)
    ):
        return None

    previous_tempo = source["document_metadata"].get("tempo_quarter")
    if not information.get('measure_profiles') and source.get("capo", 0) != information.get("capo", 0) and information.get(
        "instrument", "guitar"
    ) in {"guitar", "bass"}:
        for row in source["records"]:
            if (row.get("mode") or source["mode"]) != "tab":
                row.update(pitch_needs_review=True, needs_review=True)
    for key in ("title", "artist", "tuning_used", "capo", "document_metadata"):
        source[key] = information[key]
    source["instrument"] = information.get("instrument", "guitar")
    source["midi_program"] = information.get("midi_program", 25)
    source['parts'] = information.get('parts', [])
    profiles = {p['measure_number']: p for p in information.get('measure_profiles', [])}
    for row in source['records']:
        profile = profiles.get(row['measure_number'])
        if profile is None:
            continue
        if row.get('capo', 0) != profile['capo'] and row.get('mode') != 'tab':
            row.update(pitch_needs_review=True, needs_review=True)
        for key in ('part_name', 'instrument', 'midi_program', 'tuning', 'capo', 'tuning_explicit', 'fingering_tunings'):
            if key in profile:
                row[key] = profile[key]
        if row.get('pitch_context'):
            row['pitch_context']['capo'] = profile['capo']
    tempo = information["document_metadata"].get("tempo_quarter")
    if source["records"] and tempo and tempo != previous_tempo:
        # A document tempo applies to every staff at the opening bar. Updating
        # only one part lets the unchanged parts outvote it in the shared IR.
        opening = source['records'][0].get('bar_index', 0)
        for index, row in enumerate(source['records']):
            if row.get('bar_index', index) != opening:
                continue
            first = parse_measure_target(row['target'])
            first['tempo_quarter'] = int(tempo)
            row['target'] = format_measure_target(first, row.get('mode') or source['mode'], preserve_playback=True)
    return source
