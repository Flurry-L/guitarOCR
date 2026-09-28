"""Group recognized measures by part and staff without changing model tokens."""

from collections import Counter, OrderedDict, defaultdict

from shared.instruments import DEFAULT_PROGRAMS, DEFAULT_TUNINGS, INSTRUMENTS, pitch_reference
from shared.m2 import parse_measure_target
from shared.score_text import model_score_text


def score_document(result: dict) -> dict:
    parts = OrderedDict()
    signatures, tempos = defaultdict(Counter), defaultdict(Counter)
    indices = set()
    for record in result["records"]:
        part_id = str(record.get("part_id", "part-1"))
        staff_id = str(record.get("staff_id", "staff-1"))
        instrument = record.get("instrument") or result.get("instrument", "guitar")
        if instrument not in INSTRUMENTS:
            raise ValueError(f"Unsupported instrument: {instrument}")
        same_instrument = instrument == result.get("instrument", "guitar")
        tuning = list(
            record.get(
                "tuning",
                result.get("tuning_used", DEFAULT_TUNINGS[instrument])
                if same_instrument
                else DEFAULT_TUNINGS[instrument],
            )
        )
        if instrument in {"pitched", "drums"}:
            tuning = []
        part = parts.setdefault(
            part_id,
            {
                "id": part_id,
                "name": record.get("part_name", instrument),
                "instrument": instrument,
                "pitch_reference": pitch_reference(instrument),
                "midi_program": record.get(
                    "midi_program",
                    result.get("midi_program", DEFAULT_PROGRAMS[instrument])
                    if same_instrument
                    else DEFAULT_PROGRAMS[instrument],
                ),
                "tuning": tuning,
                "capo": int(record.get("capo", result.get("capo", 0) if same_instrument else 0)),
                "staves": OrderedDict(),
            },
        )
        if part["instrument"] != instrument or part["tuning"] != tuning:
            raise ValueError(f"Conflicting instrument or tuning in part {part_id}")
        staff = part["staves"].setdefault(staff_id, {"id": staff_id, "measures": []})
        index = int(record.get("bar_index", len(staff["measures"])))
        if any(m['index'] == index for m in staff['measures']):
            raise ValueError(f"Duplicate bar in {part_id}/{staff_id}: {index}")
        parsed = parse_measure_target(model_score_text(record["target"]))
        written = None
        if record.get('written_target') and not record.get('manually_edited'):
            try:
                written = parse_measure_target(record['written_target'])
            except ValueError:
                pass
        state = record.get('score_state') or {}
        indices.add(index)
        if signature := parsed.get('time_signature') or state.get('time'):
            signatures[index][signature] += 1
        if parsed.get('tempo_quarter'):
            tempos[index][parsed['tempo_quarter']] += 1
        staff["measures"].append(
            {
                **parsed,
                "index": index,
                "source_record": record['measure_number'],
                "mode": record.get("mode") or result["mode"],
                "needs_review": bool(record.get("needs_review")),
                "pitch_reference": pitch_reference(instrument),
                **({"pitch_context": record["pitch_context"]} if record.get("pitch_context") else {}),
                **({'written': written} if written else {}),
            }
        )
    for part in parts.values():
        part["staves"] = list(part["staves"].values())
        for staff in part['staves']:
            staff['measures'].sort(key=lambda m: m['index'])
    timeline, signature = [], '4/4'
    for index in range(max(indices, default=-1) + 1):
        if signatures[index]:
            signature = signatures[index].most_common(1)[0][0]
        timing = {'index': index, 'time_signature': signature}
        if tempos[index]:
            timing['tempo_quarter'] = tempos[index].most_common(1)[0][0]
        timeline.append(timing)
    return {
        "schema": "guitarocr.score/2",
        "ticks_per_quarter": 960,
        "title": result.get("title", ""),
        "artist": result.get("artist", ""),
        "timeline": timeline,
        "parts": list(parts.values()),
    }
