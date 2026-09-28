"""Resolve manually edited document information using the current layout."""

from shared.instruments import DEFAULT_PROGRAMS, INSTRUMENTS
from shared.pitch_context import apply_pitch_regions
from shared.tuning import DEFAULT_TUNING


def _edited_single(layout, previous, values):
    instrument = values.get("instrument", "guitar")
    if instrument not in INSTRUMENTS:
        raise ValueError("请选择乐器类型")
    fretted = instrument in {"guitar", "bass"}
    tuning = values.get("tuning_used", list(DEFAULT_TUNING)) if fretted else []
    if (fretted and not 1 <= len(tuning) <= 12) or any(
        type(p) is not int or not 0 <= p <= 127 for p in tuning
    ):
        raise ValueError("请填写 1–12 根弦，调弦使用 0–127 的 MIDI 音高")
    capo, tempo = values.get("capo", 0), values.get("tempo_quarter", 120)
    if (
        type(capo) is not int
        or not 0 <= capo <= 24
        or type(tempo) is not int
        or not 20 <= tempo <= 400
    ):
        raise ValueError("变调夹范围 0–24，速度范围 20–400")
    metadata = dict(previous.get("document_metadata", {}))
    metadata.update(tempo_quarter=tempo)
    transpose = values.get("transpose")
    if transpose is not None and (
        type(transpose) is not int or not -36 <= transpose <= 36
    ):
        raise ValueError("记谱移调须为 -36 至 36 的整数，留空则自动读取")
    from copy import deepcopy

    pitch_contexts = []
    if previous.get('instrument', 'guitar') == instrument and previous.get('transpose') == transpose:
        pitch_contexts = deepcopy(previous.get('measure_pitch_contexts', []))
    elif previous.get("measure_pitch_contexts") or transpose is not None:
        records = apply_pitch_regions(
            [{**r, 'instrument': instrument} for r in layout["records"]],
            metadata.get("pitch_instructions", []),
            instrument=instrument,
            transpose=transpose,
        )
        pitch_contexts = [
            {
                k: r[k]
                for k in (
                    "measure_number",
                    "pitch_context",
                    "pitch_reference",
                    "pitch_needs_review",
                )
                if k in r
            }
            for r in records
        ]
    program = values.get("midi_program")
    if program is None and previous.get("instrument") == instrument:
        program = previous.get("midi_program")
    program = DEFAULT_PROGRAMS[instrument] if program is None else program
    if type(program) is not int or not 0 <= program <= 127:
        raise ValueError("MIDI 乐器编号须在 0 至 127 之间")
    return dict(
        document_metadata=metadata,
        predictions=None,
        title=str(values.get("title", "Untitled"))[:500],
        artist=str(values.get("artist", ""))[:500],
        tuning_used=tuning,
        capo=capo,
        instrument=instrument,
        midi_program=program,
        transpose=transpose,
        measure_pitch_contexts=pitch_contexts,
    )


def edited_information(layout, previous, values):
    from copy import deepcopy

    if not previous.get('parts'):
        return _edited_single(layout, previous, values)
    result = deepcopy(previous)
    part_id = values.get('part_id') or result['parts'][0]['id']
    part = next((p for p in result['parts'] if p['id'] == part_id), None)
    if part is None:
        raise ValueError('无效的音轨')
    profiles = {p['measure_number']: p for p in result['measure_profiles'] if p['part_id'] == part_id}
    local = {**layout, 'records': [{**r, **profiles[r['measure_number']]} for r in layout['records'] if r['measure_number'] in profiles]}
    edited = _edited_single(local, part, values)
    part.update(edited)
    if values.get('part_name') is not None:
        name = str(values['part_name']).strip()
        if not name or len(name) > 160:
            raise ValueError('请填写音轨名称，最多 160 个字符')
        part['name'] = name
    for profile in profiles.values():
        profile.update(instrument=edited['instrument'], tuning=edited['tuning_used'], capo=edited['capo'],
                       midi_program=edited['midi_program'], tuning_explicit=True,
                       fingering_tunings=[edited['tuning_used']], part_name=part['name'])
    result['measure_pitch_contexts'] = [r for p in result['parts'] for r in p.get('measure_pitch_contexts', [])]
    result.update(title=edited['title'], artist=edited['artist'])
    if part_id == result['parts'][0]['id']:
        result.update({k: v for k, v in edited.items() if k not in {'measure_pitch_contexts'}})
    result['document_metadata']['tempo_quarter'] = edited['document_metadata']['tempo_quarter']
    for other in result['parts']:
        other.update(title=edited['title'], artist=edited['artist'])
        other['document_metadata']['tempo_quarter'] = edited['document_metadata']['tempo_quarter']
    for key in ('stage', 'schema_version', 'layout'):
        result.pop(key, None)
    return result
