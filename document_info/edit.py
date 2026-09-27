"""Resolve manually edited document information using the current layout."""

from shared.instruments import DEFAULT_PROGRAMS, INSTRUMENTS
from shared.pitch_context import apply_pitch_regions
from shared.tuning import DEFAULT_TUNING


def edited_information(layout, previous, values):
    instrument = values.get("instrument", "guitar")
    if instrument not in INSTRUMENTS:
        raise ValueError("请选择乐器类型")
    fretted = instrument in {"guitar", "bass"}
    tuning = values.get("tuning_used", list(DEFAULT_TUNING)) if fretted else []
    if (fretted and not 1 <= len(tuning) <= 7) or any(
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
    metadata = dict(previous.get("document_metadata", {}))
    metadata.update(tempo_quarter=tempo)
    transpose = values.get("transpose")
    if transpose is not None and (
        type(transpose) is not int or not -36 <= transpose <= 36
    ):
        raise ValueError("记谱移调须为 -36 至 36 的整数，留空则自动读取")
    pitch_contexts = []
    if previous.get("measure_pitch_contexts") or transpose is not None:
        records = apply_pitch_regions(
            layout["records"],
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
