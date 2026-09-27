"""Group recognized measures by part and staff without changing model tokens."""

from collections import OrderedDict

from shared.instruments import DEFAULT_PROGRAMS, DEFAULT_TUNINGS, INSTRUMENTS, pitch_reference
from shared.m2 import parse_measure_target
from shared.score_text import model_score_text


def score_document(result: dict) -> dict:
    parts = OrderedDict()
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
        if index != len(staff["measures"]):
            raise ValueError(
                f"Missing or duplicate bar in {part_id}/{staff_id}: {index}"
            )
        staff["measures"].append(
            {
                **parse_measure_target(model_score_text(record["target"])),
                "index": index,
                "mode": record.get("mode") or result["mode"],
                "needs_review": bool(record.get("needs_review")),
                "pitch_reference": pitch_reference(instrument),
                **({"pitch_context": record["pitch_context"]} if record.get("pitch_context") else {}),
            }
        )
    for part in parts.values():
        part["staves"] = list(part["staves"].values())
    return {
        "title": result.get("title", ""),
        "artist": result.get("artist", ""),
        "parts": list(parts.values()),
    }
