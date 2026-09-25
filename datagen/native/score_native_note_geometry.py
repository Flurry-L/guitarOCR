from __future__ import annotations

import math
import re
from decimal import Decimal
from collections import defaultdict
from collections.abc import Mapping


FIELDS = {
    "kind",
    "staff_index",
    "measure_index",
    "voice_index",
    "event_index",
    "note_index",
    "native_string_index",
    "page",
    "system_index",
    "visibility_code",
    "text",
    "bounds_source",
    "bbox_mm",
    "layout_bbox_mm",
    "render_origin_mm",
}
OWNER_FIELDS = (
    "staff_index",
    "measure_index",
    "voice_index",
    "event_index",
    "note_index",
)

_FRET_TEXT = re.compile(
    r"(?P<paren>\()?(?:<(?P<hf>[0-9]+(?:\.[0-9]+)?)>|(?P<fret>[0-9]+)|(?P<dead>[Xx]))(?(paren)\))"
)


def validate_printed_note_values(note: Mapping, glyphs: list[dict]) -> None:
    actual = set()
    for glyph in glyphs:
        match = _FRET_TEXT.fullmatch(glyph["text"])
        if match is None:
            raise ValueError(f"unaudited native fret text: {glyph['text']!r}")
        if match["hf"] is not None:
            actual.add(("printed_harmonic_fret", Decimal(match["hf"])))
        elif match["fret"] is not None:
            actual.add(("fret_token", Decimal(match["fret"])))
        else:
            actual.add(("fret_token", "x"))
    expected = {
        (key, "x" if value == "x" else Decimal(str(value)))
        for key, value in note.items()
        if key in ("fret_token", "printed_harmonic_fret")
    }
    if actual != expected:
        raise ValueError(
            f"visible core digits disagree with the actual native fret/harmonic glyphs: core={dict(note)}, drawn={[glyph['text'] for glyph in glyphs]}"
        )


def _numbers(value, length, label):
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
    ):
        raise ValueError(f"{label} needs {length} finite numbers")
    return value


def validate_note_geometry(
    layout: Mapping, *, require_complete: bool = False
) -> list[dict]:
    payload = layout.get("note_geometry")
    if "note_geometry" not in layout:
        if require_complete:
            raise ValueError("native note geometry is unavailable in this asset")
        return []
    if not isinstance(payload, Mapping):
        raise ValueError("native note geometry must be an object when present")
    if (
        set(payload) != {"schema", "status", "errors", "glyphs"}
        or payload["schema"] != "gpomr.note-glyph-geometry"
    ):
        raise ValueError("invalid native note geometry schema")
    if payload["status"] not in ("complete", "unresolved"):
        raise ValueError("invalid note geometry status")
    errors = payload["errors"]
    if not isinstance(errors, list) or any(
        not isinstance(e, str) or not e for e in errors
    ):
        raise ValueError("invalid note geometry error list")
    if (payload["status"] == "complete") != (len(errors) == 0):
        raise ValueError("note geometry status disagrees with errors")
    if require_complete and errors:
        raise ValueError("native note geometry is unresolved: " + errors[0])
    if not isinstance(payload["glyphs"], list):
        raise ValueError("native glyphs must be a list")
    pages = {page["index"]: page for page in layout["pages"]}
    systems = {system["system_index"]: system for system in layout["systems"]}
    for glyph in payload["glyphs"]:
        if not isinstance(glyph, dict) or set(glyph) != FIELDS:
            raise ValueError("invalid native glyph fields")
        for field in (*OWNER_FIELDS[:-1], "page", "system_index", "visibility_code"):
            if type(glyph[field]) is not int or glyph[field] < 0:
                raise ValueError(f"invalid glyph {field}")
        if glyph["visibility_code"] != 0:
            raise ValueError("spatial labels require actually visible glyphs")
        if glyph["kind"] == "fret":
            if any(
                type(glyph[k]) is not int or glyph[k] < 0
                for k in ("note_index", "native_string_index")
            ):
                raise ValueError("fret glyph needs a native note and string owner")
            if not isinstance(glyph["text"], str) or not glyph["text"]:
                raise ValueError("fret glyph needs actual drawn text")
            if glyph["bounds_source"] != "native_draw_text_bounds_0":
                raise ValueError("fret geometry must use draw-time text metrics")
        elif glyph["kind"] == "rest":
            if any(
                glyph[k] is not None
                for k in ("note_index", "native_string_index", "text")
            ):
                raise ValueError("rest glyph cannot invent note ownership or text")
            if glyph["bounds_source"] != "rest_view_layout":
                raise ValueError("unexpected rest bounds provenance")
        else:
            raise ValueError("unsupported native glyph kind")
        page, system = pages.get(glyph["page"]), systems.get(glyph["system_index"])
        if (
            page is None
            or system is None
            or system["page"] != glyph["page"]
            or not system["first_measure_index"]
            <= glyph["measure_index"]
            < system["last_measure_index"]
        ):
            raise ValueError("glyph has no matching page/system/measure owner")
        _, _, width, height = page["bbox_mm"]
        for field in ("bbox_mm", "layout_bbox_mm"):
            x, y, w, h = _numbers(glyph[field], 4, field)
            if (
                w <= 0
                or h <= 0
                or min(x, y) < -1e-5
                or x + w > width + 1e-5
                or y + h > height + 1e-5
            ):
                raise ValueError("glyph bounds exceed the page")
        _numbers(glyph["render_origin_mm"], 2, "render origin")
    return payload["glyphs"]


def index_official_owners(score: Mapping) -> tuple[dict, dict]:
    events, notes = {}, {}
    for track in score["tracks"]:
        for staff in track["staves"]:
            for measure in staff["measures"]:
                for voice in measure["voices"]:
                    for event in voice["events"]:
                        owner = (
                            staff["staff_index"],
                            measure["measure_index"],
                            voice["voice_index"],
                            event["event_index"],
                        )
                        if owner in events:
                            raise ValueError("multiple official events share an owner")
                        events[owner] = event
                        for note in event["notes"]:
                            key = (*owner, note["note_index"])
                            if key in notes:
                                raise ValueError(
                                    "multiple official notes share an owner"
                                )
                            notes[key] = note
    return events, notes


def group_glyphs_by_owner(
    score: Mapping, layout: Mapping, *, require_complete: bool = True
) -> dict[tuple, list[dict]]:
    if score.get("source_track_index") != layout.get("source_track_index"):
        raise ValueError("native geometry and score select different tracks")
    glyphs = validate_note_geometry(layout, require_complete=require_complete)
    events, notes = index_official_owners(score)
    grouped = defaultdict(list)
    for glyph in glyphs:
        key = tuple(glyph[field] for field in OWNER_FIELDS)
        event = events.get(key[:-1])
        if event is None:
            raise ValueError("native glyph refers to an absent official event")
        if glyph["kind"] == "rest":
            if not event["rest"]:
                raise ValueError("native rest glyph refers to a non-rest event")
        else:
            note = notes.get(key)
            if (
                note is None
                or note["native_string_index"] != glyph["native_string_index"]
            ):
                raise ValueError(
                    "native fret glyph refers to an absent or different official note"
                )
        grouped[key].append(glyph)
    return dict(grouped)


def quantize_page_box(box_mm: list[float], crop_mm: list[float]) -> list[int]:
    x, y, w, h = _numbers(box_mm, 4, "page box")
    cx, cy, cw, ch = _numbers(crop_mm, 4, "crop")
    if min(w, h, cw, ch) <= 0:
        raise ValueError("boxes must have positive dimensions")
    raw = [(x - cx) / cw, (y - cy) / ch, (x + w - cx) / cw, (y + h - cy) / ch]
    if min(raw) < -1e-7 or max(raw) > 1 + 1e-7:
        raise ValueError("spatial crop would clip supervised ink")
    result = [min(1000, max(0, round(value * 1000))) for value in raw]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("glyph collapses at the spatial quantization resolution")
    return result


def page_box_from_quantized(box: list[int], crop_mm: list[float]) -> list[float]:
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(x) is not int or not 0 <= x <= 1000 for x in box)
        or box[0] >= box[2]
        or box[1] >= box[3]
    ):
        raise ValueError("invalid quantized spatial box")
    x, y, w, h = _numbers(crop_mm, 4, "crop")
    if min(w, h) <= 0:
        raise ValueError("crop must have positive dimensions")
    return [
        x + box[0] * w / 1000,
        y + box[1] * h / 1000,
        (box[2] - box[0]) * w / 1000,
        (box[3] - box[1]) * h / 1000,
    ]
