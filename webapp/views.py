"""Project responses for both HTTP applications; storage paths stay on disk."""

from pathlib import Path
from urllib.parse import quote

from shared.artifacts import read_result
from shared.m2 import parse_measure_target
from shared.score_text import display_error, display_score_text


def public_annotations(annotations):
    return [{k: a[k] for k in ('kind', 'candidate_kind', 'parsed', 'page', 'bbox', 'part_id') if k in a}
            for a in annotations]


def public_metadata(metadata):
    return {key: public_annotations(value) if key in {'pitch_instructions', 'score_annotations'} else value
            for key, value in metadata.items()}


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
        state["metadata"]["document_metadata"] = public_metadata(state["metadata"]["document_metadata"])
        if info.get('parts'):
            state['metadata']['parts'] = [{k: public_metadata(p[k]) if k == 'document_metadata' else p[k] for k in (
                'id', 'name', 'instrument', 'midi_program', 'tuning_used', 'capo', 'transpose', 'document_metadata'
            ) if k in p} for p in info['parts']]
    state["measures"], state["review_measures"] = [], []
    if saved["recognition"]:
        data = read_result(Path(saved["recognition"]), "measure_ocr")
        if state['metadata'] is not None:
            state['metadata']['tuning_used'] = data['tuning_used']
            state['metadata']['document_metadata'] = public_metadata(data['document_metadata'])
        state["review_measures"] = data.get("review_measures", [])
        state["score_text_url"] = f"/api/sessions/{sid}/score.txt"
        state['score_document_url'] = asset(data['score_document'])
        if data.get('musicxml'):
            state['musicxml_url'] = asset(data['musicxml'])
        if (state.get('metadata') or {}).get('parts'):
            for part in state['metadata']['parts']:
                resolved = next((p for p in data.get('parts', []) if p['id'] == part['id']), None)
                if resolved is not None:
                    part['document_metadata'] = public_metadata(resolved['document_metadata'])
                row = next((r for r in data['records'] if r.get('part_id') == part['id']), None)
                if row is not None:
                    part['tuning_used'] = row.get('tuning', part['tuning_used'])
        for row in data["records"]:
            reasons = row.get('fallback_reason') or []
            annotation_review = bool(reasons) and all(
                reason.startswith(('Uncertain chord symbol', 'Annotation review failed'))
                for reason in reasons
            ) and not any(row.get(key) for key in (
                'timing_errors', 'fingering_errors', 'export_errors', 'pitch_needs_review', 'state_needs_review',
            ))
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
                        for reason in reasons
                    ],
                    "annotation_review": annotation_review,
                    "url": asset(row["image"]),
                    "parsed": parse_measure_target(row["target"]),
                    "chord_annotations": public_annotations(row.get('chord_annotations', [])),
                }
            )
    if saved["export"]:
        exported = read_result(Path(saved["export"]), "gp5_export")
        if exported.get('gp5'):
            state["gp5_url"] = asset(exported["gp5"])
            state["gp5_name"] = Path(exported["gp5"]).name
            state["encoding_url"] = asset(exported["encoding_report"])
        if exported.get('musicxml'):
            state['musicxml_url'] = asset(exported['musicxml'])
    return state
