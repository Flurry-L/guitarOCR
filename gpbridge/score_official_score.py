from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_SCORE_SCHEMA = "gpomr.official-score"


ROOT_FIELDS = frozenset(
    {"schema", "pdf_ok", "source_track_index", "document", "tracks"}
)
DOCUMENT_FIELDS = frozenset(
    {
        "master_measures",
        "metadata",
        "diagram_style",
        "tempo",
        "tempo_automations",
    }
)
METADATA_FIELDS = frozenset(
    {
        "properties",
        "first_page_header",
        "even_page_header",
        "odd_page_header",
        "first_page_footer",
        "even_page_footer",
        "odd_page_footer",
    }
)
METADATA_PROPERTY_FIELDS = frozenset(
    {
        "title",
        "subtitle",
        "artist",
        "album",
        "words",
        "music",
        "words_and_music",
        "copyright",
        "tabber",
        "instructions",
        "notice",
    }
)
HEADER_PROPERTY_FIELDS = frozenset(
    {
        "title",
        "subtitle",
        "artist",
        "album",
        "words",
        "music",
        "words_and_music",
        "tabber",
    }
)
STYLED_TEXT_FIELDS = frozenset({"visible", "formatted_text"})
PAGE_HEADER_FIELDS = frozenset({"visible", "field"})
FOOTER_FIELDS = frozenset({"visible", "fields"})
FOOTER_PROPERTY_FIELDS = frozenset({"copyright", "copyright2", "page_number"})
DIAGRAM_STYLE_FIELDS = frozenset(
    {
        "header_visibility_code",
        "header_visible",
        "header_chord_name_display_code",
        "header_fingering_display_code",
        "score_visibility_code",
        "score_visible",
        "score_chord_name_display_code",
        "score_fingering_display_code",
    }
)
TEMPO_FIELDS = frozenset({"value", "visible", "label", "unit_code", "unit_name"})
TEMPO_AUTOMATION_FIELDS = frozenset(
    {
        "bar_index",
        "position",
        "value",
        "unit_code",
        "unit_name",
        "visible",
        "linear",
        "text",
    }
)
TRACK_FIELDS = frozenset(
    {
        "semantic_id",
        "track_index",
        "name",
        "short_name",
        "let_ring_throughout",
        "staves",
    }
)
STAFF_FIELDS = frozenset(
    {
        "semantic_id",
        "staff_index",
        "string_count",
        "capo_present",
        "capo_fret",
        "partial_capo_present",
        "partial_capo",
        "tuning_pitches_low_to_high",
        "open_string_frets_low_to_high",
        "total_capo_frets_low_to_high",
        "tuning_displayed_label",
        "tuning_label_visible",
        "tuning_role_code",
        "tuning_is_flat",
        "short_drone_string",
        "diagram_collection",
        "measures",
    }
)
MEASURE_FIELDS = frozenset(
    {
        "semantic_id",
        "measure_index",
        "staff_index",
        "master_measure_index",
        "drawing_key_signature",
        "simile_present",
        "simile_code",
        "simile_name",
        "ottavia_present",
        "ottavia_code",
        "ottavia_name",
        "voices",
    }
)
MASTER_MEASURE_FIELDS = frozenset(
    {
        "master_measure_index",
        "time_signature",
        "repeat",
        "double_bar",
        "free_time",
        "anacrusis",
        "fermata_present",
        "fermatas",
        "extended_alternate_endings_present",
        "extended_alternate_endings",
        "key_accidental_count",
        "section_present",
        "section_letter",
        "section_text",
        "triplet_feel_present",
        "triplet_feel_code",
        "triplet_feel_name",
        "directions_present",
        "directions",
    }
)
TIME_SIGNATURE_FIELDS = frozenset({"numerator", "denominator"})
DRAWING_KEY_SIGNATURE_FIELDS = frozenset({"accidental_count"})
REPEAT_FIELDS = frozenset(
    {"start", "end", "count", "alternate_ending", "alternate_ending_mask"}
)
FERMATA_FIELDS = frozenset({"type_code", "offset", "length"})
PARTIAL_CAPO_FIELDS = frozenset({"native_string_index", "fret"})
VOICE_FIELDS = frozenset({"semantic_id", "voice_index", "events"})
EVENT_FIELDS = frozenset(
    {
        "semantic_id",
        "event_index",
        "placeholder",
        "rest",
        "sustain_pedal",
        "dead_slapped",
        "grace",
        "grace_type_code",
        "grace_type_name",
        "grace_beat_count",
        "ottavia_present",
        "ottavia_code",
        "ottavia_name",
        "offset",
        "rhythm",
        "barre_present",
        "barre_string",
        "barre_fret",
        "fade_present",
        "fade_code",
        "fade_name",
        "hairpin_present",
        "hairpin_code",
        "hairpin_name",
        "dynamic",
        "golpe_present",
        "golpe_code",
        "golpe_name",
        "wah_present",
        "wah_code",
        "wah_name",
        "slashed",
        "timer_present",
        "timer",
        "free_text_present",
        "free_text",
        "chord_present",
        "chord",
        "chord_diagram",
        "lyrics",
        "is_brushed",
        "brush_code",
        "brush_name",
        "has_arpeggio",
        "arpeggio_code",
        "arpeggio_name",
        "pick_stroke_present",
        "pick_stroke_code",
        "pick_stroke_name",
        "slapped",
        "popped",
        "rasgueado_present",
        "rasgueado_code",
        "rasgueado_name",
        "tremolo",
        "whammy",
        "whammy_extend",
        "whammy_beat_count",
        "whammy_begin_ref",
        "whammy_end_ref",
        "tremolo_bar_vibrato_present",
        "tremolo_bar_vibrato_code",
        "tremolo_bar_vibrato_name",
        "legato_origin",
        "legato_destination",
        "legato_origin_ref",
        "legato_destination_ref",
        "notes",
    }
)
RHYTHM_FIELDS = frozenset({"note_value_code", "augmentation_dots", "tuplets"})
TUPLET_FIELDS = frozenset({"primary", "secondary"})
WHAMMY_FIELDS = frozenset(
    {
        "type_code",
        "type",
        "origin",
        "middle",
        "destination",
    }
)
CHORD_DIAGRAM_FIELDS = frozenset(
    {
        "name",
        "string_count",
        "base_fret",
        "fret_count",
        "span_limit",
        "show_diagram",
        "show_name",
        "show_fingering",
        "barres",
        "strings_low_to_high",
    }
)
CHORD_DIAGRAM_BARRE_FIELDS = frozenset(
    {
        "relative_fret",
        "start_native_string_index",
        "end_native_string_index",
    }
)
CHORD_DIAGRAM_STRING_FIELDS = frozenset(
    {"native_string_index", "relative_fret", "finger_code"}
)
LYRIC_FIELDS = frozenset(
    {
        "line_index",
        "text",
        "displayed_text",
        "extend_code",
        "syllabic_code",
        "h_alignment_code",
        "x_offset",
    }
)
NOTE_FIELDS = frozenset(
    {
        "semantic_id",
        "note_index",
        "native_string_index",
        "fret",
        "printable_in_tab",
        "dead",
        "palm_muted",
        "let_ring",
        "accent_flags",
        "anti_accent",
        "anti_accent_code",
        "anti_accent_name",
        "tie_origin",
        "tie_destination",
        "hopo_origin",
        "hopo_destination",
        "slide_flags",
        "slides",
        "bend",
        "harmonic",
        "trill_present",
        "trill_fret",
        "trill_valid",
        "trill_midi",
        "vibrato_present",
        "vibrato_code",
        "vibrato_name",
        "tapped",
        "left_hand_tapped",
        "pizzicato",
        "show_string_number",
        "sustain_pedal",
        "ornament_present",
        "ornament_code",
        "ornament_name",
        "left_fingering_code",
        "left_fingering_name",
        "right_fingering_code",
        "right_fingering_name",
        "tie_origin_ref",
        "tie_destination_ref",
        "hopo_origin_ref",
        "hopo_destination_ref",
        "slide_begin_ref",
        "slide_end_ref",
    }
)
SLIDE_FIELDS = frozenset(
    {
        "in_above",
        "in_below",
        "legato",
        "shift",
        "out_down",
        "out_up",
        "pick_scrape_down",
        "pick_scrape_up",
    }
)
BEND_FIELDS = frozenset({"type_code", "type", "origin", "middle", "destination"})
HARMONIC_FIELDS = frozenset(
    {
        "type_code",
        "type",
        "fret_code",
        "touch_offset",
    }
)
NAMED_ENUM_FIELDS = frozenset({"code", "name"})


CHILDREN = {
    "staff": ("measures", "measure", "measure_index"),
    "measure": ("voices", "voice", "voice_index"),
    "voice": ("events", "event", "event_index"),
    "event": ("notes", "note", "note_index"),
}


@dataclass(frozen=True, slots=True)
class OfficialNode:
    node_id: str
    kind: str
    order: int
    value: Mapping[str, Any]


@dataclass
class _ValidationState:
    semantic_ids: set[str]


def load_official_score(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    validate_official_score(value)
    return deepcopy(value)


def validate_official_score(score: Mapping[str, Any]) -> None:
    root = _expect_object(score, "official score", ROOT_FIELDS)
    if _expect_string(root["schema"], "official score.schema") != OFFICIAL_SCORE_SCHEMA:
        raise ValueError(f"official score.schema must be {OFFICIAL_SCORE_SCHEMA!r}")
    if not _expect_bool(root["pdf_ok"], "official score.pdf_ok"):
        raise ValueError("official score.pdf_ok must be true")
    _expect_nonnegative_int(
        root["source_track_index"], "official score.source_track_index"
    )

    document_measure_count = _validate_document(root["document"])
    tracks = _indexed_children(
        root["tracks"],
        index_field="track_index",
        label="official score track",
    )
    if len(tracks) != 1:
        raise ValueError("official score must contain exactly one selected track")
    if tracks[0][0] != 0:
        raise ValueError("the selected output track must have track_index 0")

    state = _ValidationState(set())
    _validate_track(
        tracks[0][1],
        document_measure_count=document_measure_count,
        state=state,
    )


def walk_official_nodes(
    score: Mapping[str, Any],
    *,
    track_index: int = 0,
    staff_index: int = 0,
) -> Iterator[OfficialNode]:
    validate_official_score(score)
    track = _indexed(score["tracks"], track_index, "track", "track_index")
    staff = _indexed(track["staves"], staff_index, "staff", "staff_index")
    track_id = _expect_string(track["semantic_id"], "track.semantic_id")
    yield OfficialNode(track_id, "track", track_index, track)

    def visit(
        value: Mapping[str, Any],
        kind: str,
        order: int,
    ) -> Iterator[OfficialNode]:
        node_id = _expect_string(value["semantic_id"], f"{kind}.semantic_id")
        yield OfficialNode(node_id, kind, order, value)
        child_spec = CHILDREN.get(kind)
        if child_spec is None:
            return
        child_field, child_kind, index_field = child_spec
        for child_order, child in _indexed_children(
            value[child_field],
            index_field=index_field,
            label=f"{kind} {node_id} {child_kind}",
        ):
            yield from visit(child, child_kind, child_order)

    yield from visit(staff, "staff", staff_index)


def selected_staff(
    score: Mapping[str, Any],
    *,
    track_index: int = 0,
    staff_index: int = 0,
) -> Mapping[str, Any]:
    validate_official_score(score)
    track = _indexed(score["tracks"], track_index, "track", "track_index")
    return _indexed(track["staves"], staff_index, "staff", "staff_index")


def _validate_document(value: Any) -> int:
    document = _expect_object(value, "official score.document", DOCUMENT_FIELDS)
    master_measures = _indexed_children(
        document["master_measures"],
        index_field="master_measure_index",
        label="official score master measure",
        contiguous=True,
    )
    for measure_index, master_measure in master_measures:
        _validate_master_measure_fields(master_measure, measure_index)
    master_measure_count = len(master_measures)
    _validate_metadata(document["metadata"])
    _validate_diagram_style(document["diagram_style"])
    tempo = _expect_object(
        document["tempo"], "official score.document.tempo", TEMPO_FIELDS
    )
    _expect_number(tempo["value"], "official score.document.tempo.value")
    _expect_bool(tempo["visible"], "official score.document.tempo.visible")
    _expect_string(tempo["label"], "official score.document.tempo.label")
    _expect_int(tempo["unit_code"], "official score.document.tempo.unit_code")
    _expect_optional_string(
        tempo["unit_name"], "official score.document.tempo.unit_name"
    )
    automation_locations: set[tuple[int, int | float]] = set()
    for position, raw_automation in enumerate(
        _expect_list(
            document["tempo_automations"],
            "official score.document.tempo_automations",
        )
    ):
        label = f"official score.document.tempo_automations[{position}]"
        automation = _expect_object(raw_automation, label, TEMPO_AUTOMATION_FIELDS)
        bar_index = _expect_nonnegative_int(
            automation["bar_index"], f"{label}.bar_index"
        )
        if bar_index >= master_measure_count:
            raise ValueError("official tempo automation is outside the score")
        automation_position = _expect_number(
            automation["position"], f"{label}.position"
        )
        if not 0 <= automation_position <= 1:
            raise ValueError("official tempo automation position must be in 0..1")
        location = (bar_index, automation_position)
        if location in automation_locations:
            raise ValueError("official tempo automation locations must be unique")
        automation_locations.add(location)
        _expect_number(automation["value"], f"{label}.value")
        _expect_int(automation["unit_code"], f"{label}.unit_code")
        _expect_optional_string(automation["unit_name"], f"{label}.unit_name")
        _expect_bool(automation["visible"], f"{label}.visible")
        _expect_bool(automation["linear"], f"{label}.linear")
        _expect_string(automation["text"], f"{label}.text")
    return master_measure_count


def _validate_diagram_style(value: Any) -> None:
    style = _expect_object(
        value,
        "official score.document.diagram_style",
        DIAGRAM_STYLE_FIELDS,
    )
    for field in ("header_visibility_code", "score_visibility_code"):
        code = _expect_nonnegative_int(
            style[field], f"official score.document.diagram_style.{field}"
        )
        if code not in {0, 1, 2}:
            raise ValueError(
                f"official score.document.diagram_style.{field} "
                "must be an official visibility code"
            )
    for field in (
        "header_chord_name_display_code",
        "header_fingering_display_code",
        "score_chord_name_display_code",
        "score_fingering_display_code",
    ):
        code = _expect_nonnegative_int(
            style[field], f"official score.document.diagram_style.{field}"
        )
        if code not in {0, 1, 2}:
            raise ValueError(
                f"official score.document.diagram_style.{field} "
                "must be Visible, Hidden, or Auto"
            )
    for field in ("header_visible", "score_visible"):
        _expect_bool(style[field], f"official score.document.diagram_style.{field}")


def _validate_metadata(value: Any) -> None:
    metadata = _expect_object(
        value, "official score.document.metadata", METADATA_FIELDS
    )
    properties = _expect_object(
        metadata["properties"],
        "official score.document.metadata.properties",
        METADATA_PROPERTY_FIELDS,
    )
    for field in METADATA_PROPERTY_FIELDS:
        _expect_string(
            properties[field], f"official score.document.metadata.properties.{field}"
        )

    header = _expect_object(
        metadata["first_page_header"],
        "official score.document.metadata.first_page_header",
        HEADER_PROPERTY_FIELDS,
    )
    for field in HEADER_PROPERTY_FIELDS:
        _validate_styled_text(
            header[field],
            f"official score.document.metadata.first_page_header.{field}",
        )
    for field in ("even_page_header", "odd_page_header"):
        page_header = _expect_object(
            metadata[field],
            f"official score.document.metadata.{field}",
            PAGE_HEADER_FIELDS,
        )
        _expect_bool(
            page_header["visible"],
            f"official score.document.metadata.{field}.visible",
        )
        _validate_styled_text(
            page_header["field"],
            f"official score.document.metadata.{field}.field",
        )
    for field in ("first_page_footer", "even_page_footer", "odd_page_footer"):
        _validate_footer(metadata[field], f"official score.document.metadata.{field}")


def _validate_styled_text(value: Any, label: str) -> None:
    item = _expect_object(value, label, STYLED_TEXT_FIELDS)
    _expect_bool(item["visible"], f"{label}.visible")
    _expect_string(item["formatted_text"], f"{label}.formatted_text")


def _validate_footer(value: Any, label: str) -> None:
    footer = _expect_object(value, label, FOOTER_FIELDS)
    _expect_bool(footer["visible"], f"{label}.visible")
    fields = _expect_object(footer["fields"], f"{label}.fields", FOOTER_PROPERTY_FIELDS)
    for field in FOOTER_PROPERTY_FIELDS:
        _validate_styled_text(fields[field], f"{label}.fields.{field}")


def _validate_track(
    value: Mapping[str, Any],
    *,
    document_measure_count: int,
    state: _ValidationState,
) -> None:
    track = _expect_object(value, "track", TRACK_FIELDS)
    track_index = _expect_nonnegative_int(track["track_index"], "track.track_index")
    if track_index != 0:
        raise ValueError("the selected output track must have track_index 0")
    track_id = f"track:{track_index}"
    _expect_semantic_id(track, track_id, "track", state)
    _expect_string(track["name"], "track.name")
    _expect_string(track["short_name"], "track.short_name")
    _expect_bool(track["let_ring_throughout"], "track.let_ring_throughout")

    staves = _indexed_children(
        track["staves"],
        index_field="staff_index",
        label=f"{track_id} staff",
        contiguous=True,
    )
    if not staves:
        raise ValueError("the selected TAB track must contain at least one staff")
    for _, staff in staves:
        _validate_staff(
            staff,
            track_id=track_id,
            measure_count=document_measure_count,
            state=state,
        )


def _validate_staff(
    value: Mapping[str, Any],
    *,
    track_id: str,
    measure_count: int,
    state: _ValidationState,
) -> None:
    staff = _expect_object(value, "staff", STAFF_FIELDS)
    staff_index = _expect_nonnegative_int(staff["staff_index"], "staff.staff_index")
    staff_id = f"{track_id}/staff:{staff_index}"
    _expect_semantic_id(staff, staff_id, "staff", state)
    string_count = _expect_nonnegative_int(staff["string_count"], "staff.string_count")
    if string_count == 0:
        raise ValueError("the selected TAB staff must contain strings")
    _expect_bool(staff["capo_present"], "staff.capo_present")
    capo_fret = _expect_nonnegative_int(staff["capo_fret"], "staff.capo_fret")
    if bool(capo_fret) != staff["capo_present"]:
        raise ValueError("staff.capo_present must agree with staff.capo_fret")
    _expect_bool(staff["partial_capo_present"], "staff.partial_capo_present")
    partial_capo = _expect_list(staff["partial_capo"], "staff.partial_capo")
    partial_strings: set[int] = set()
    for position, raw_item in enumerate(partial_capo):
        label = f"staff.partial_capo[{position}]"
        item = _expect_object(raw_item, label, PARTIAL_CAPO_FIELDS)
        native_string_index = _expect_nonnegative_int(
            item["native_string_index"], f"{label}.native_string_index"
        )
        if native_string_index >= string_count:
            raise ValueError("staff partial capo string is outside the staff tuning")
        if native_string_index in partial_strings:
            raise ValueError("staff.partial_capo repeats a native_string_index")
        partial_strings.add(native_string_index)
        _expect_nonnegative_int(item["fret"], f"{label}.fret")
    if bool(partial_capo) != staff["partial_capo_present"]:
        raise ValueError(
            "staff.partial_capo_present must agree with staff.partial_capo"
        )
    tuning_pitches = _expect_list(
        staff["tuning_pitches_low_to_high"],
        "staff.tuning_pitches_low_to_high",
    )
    if len(tuning_pitches) != string_count:
        raise ValueError("staff tuning pitch count must equal staff.string_count")
    for index, pitch in enumerate(tuning_pitches):
        _expect_int(pitch, f"staff.tuning_pitches_low_to_high[{index}]")
    open_string_frets = _expect_list(
        staff["open_string_frets_low_to_high"],
        "staff.open_string_frets_low_to_high",
    )
    if len(open_string_frets) != string_count:
        raise ValueError("staff open string fret count must equal staff.string_count")
    for index, fret in enumerate(open_string_frets):
        _expect_nonnegative_int(fret, f"staff.open_string_frets_low_to_high[{index}]")
    total_capo_frets = _expect_list(
        staff["total_capo_frets_low_to_high"],
        "staff.total_capo_frets_low_to_high",
    )
    if len(total_capo_frets) != string_count:
        raise ValueError("staff total capo fret count must equal staff.string_count")
    for index, fret in enumerate(total_capo_frets):
        _expect_nonnegative_int(fret, f"staff.total_capo_frets_low_to_high[{index}]")
    _expect_string(staff["tuning_displayed_label"], "staff.tuning_displayed_label")
    _expect_bool(staff["tuning_label_visible"], "staff.tuning_label_visible")
    _expect_nonnegative_int(staff["tuning_role_code"], "staff.tuning_role_code")
    _expect_bool(staff["tuning_is_flat"], "staff.tuning_is_flat")
    _expect_bool(staff["short_drone_string"], "staff.short_drone_string")
    diagram_collection = staff["diagram_collection"]
    if diagram_collection is not None:
        for position, diagram in enumerate(
            _expect_list(diagram_collection, "staff.diagram_collection")
        ):
            _validate_chord_diagram(
                diagram, label=f"staff.diagram_collection[{position}]"
            )

    measures = _indexed_children(
        staff["measures"],
        index_field="measure_index",
        label=f"{staff_id} measure",
        contiguous=True,
    )
    if len(measures) != measure_count:
        raise ValueError("staff measure count must equal the master measure count")
    for _, measure in measures:
        _validate_measure(
            measure,
            staff_id=staff_id,
            staff_index=staff_index,
            string_count=string_count,
            state=state,
        )


def _validate_measure(
    value: Mapping[str, Any],
    *,
    staff_id: str,
    staff_index: int,
    string_count: int,
    state: _ValidationState,
) -> None:
    measure = _expect_object(value, "measure", MEASURE_FIELDS)
    measure_index = _expect_nonnegative_int(
        measure["measure_index"], "measure.measure_index"
    )
    measure_id = f"{staff_id}/measure:{measure_index}"
    _expect_semantic_id(measure, measure_id, "measure", state)
    if (
        _expect_nonnegative_int(measure["staff_index"], "measure.staff_index")
        != staff_index
    ):
        raise ValueError("measure.staff_index must equal its containing staff index")
    _expect_bool(measure["simile_present"], "measure.simile_present")
    _expect_bool(measure["ottavia_present"], "measure.ottavia_present")
    _validate_optional_named_fields(measure, "simile", "measure")
    _validate_optional_named_fields(measure, "ottavia", "measure")
    master_measure_index = _expect_nonnegative_int(
        measure["master_measure_index"], "measure.master_measure_index"
    )
    if master_measure_index != measure_index:
        raise ValueError(
            "measure.master_measure_index must equal measure.measure_index"
        )
    drawing_key_signature = _expect_object(
        measure["drawing_key_signature"],
        "measure.drawing_key_signature",
        DRAWING_KEY_SIGNATURE_FIELDS,
    )
    _expect_int(
        drawing_key_signature["accidental_count"],
        "measure.drawing_key_signature.accidental_count",
    )

    voices = _indexed_children(
        measure["voices"], index_field="voice_index", label=f"{measure_id} voice"
    )
    for _, voice in voices:
        _validate_voice(
            voice,
            measure_id=measure_id,
            string_count=string_count,
            state=state,
        )


def _validate_master_measure_fields(
    value: Mapping[str, Any], measure_index: int
) -> None:
    measure = _expect_object(value, "master measure", MASTER_MEASURE_FIELDS)
    master_index = _expect_nonnegative_int(
        measure["master_measure_index"], "master measure.master_measure_index"
    )
    if master_index != measure_index:
        raise ValueError("master measure index must equal its array position")
    signature = _expect_object(
        measure["time_signature"], "measure.time_signature", TIME_SIGNATURE_FIELDS
    )
    numerator = _expect_nonnegative_int(
        signature["numerator"], "measure.time_signature.numerator"
    )
    if numerator == 0:
        raise ValueError("measure.time_signature.numerator must be positive")
    denominator = _expect_nonnegative_int(
        signature["denominator"], "measure.time_signature.denominator"
    )
    if denominator == 0:
        raise ValueError("measure.time_signature.denominator must be positive")
    repeat = _expect_object(measure["repeat"], "measure.repeat", REPEAT_FIELDS)
    _expect_bool(repeat["start"], "measure.repeat.start")
    _expect_bool(repeat["end"], "measure.repeat.end")
    _expect_nonnegative_int(repeat["count"], "measure.repeat.count")
    _expect_bool(repeat["alternate_ending"], "measure.repeat.alternate_ending")
    alternate_ending_mask = _expect_nonnegative_int(
        repeat["alternate_ending_mask"], "measure.repeat.alternate_ending_mask"
    )
    if bool(alternate_ending_mask) != repeat["alternate_ending"]:
        raise ValueError(
            "measure.repeat.alternate_ending must agree with alternate_ending_mask"
        )
    for field in ("double_bar", "free_time", "anacrusis", "fermata_present"):
        _expect_bool(measure[field], f"measure.{field}")
    fermatas = _expect_list(measure["fermatas"], "measure.fermatas")
    for position, raw_fermata in enumerate(fermatas):
        label = f"measure.fermatas[{position}]"
        fermata = _expect_object(raw_fermata, label, FERMATA_FIELDS)
        _expect_int(fermata["type_code"], f"{label}.type_code")
        _validate_rational(fermata["offset"], f"{label}.offset")
        length = _expect_number(fermata["length"], f"{label}.length")
        if length < 0:
            raise ValueError(f"{label}.length must be nonnegative")
    if measure["fermata_present"] != bool(fermatas):
        raise ValueError("measure.fermata_present must agree with measure.fermatas")
    _expect_bool(
        measure["extended_alternate_endings_present"],
        "measure.extended_alternate_endings_present",
    )
    extended_endings = _expect_list(
        measure["extended_alternate_endings"],
        "measure.extended_alternate_endings",
    )
    ending_indexes: list[int] = []
    for position, ending in enumerate(extended_endings):
        ending_index = _expect_nonnegative_int(
            ending, f"measure.extended_alternate_endings[{position}]"
        )
        if not 1 <= ending_index <= 32:
            raise ValueError("extended alternate ending index must be between 1 and 32")
        ending_indexes.append(ending_index)
    if ending_indexes != sorted(set(ending_indexes)):
        raise ValueError(
            "measure.extended_alternate_endings must be unique and ordered"
        )
    if measure["extended_alternate_endings_present"] != bool(ending_indexes):
        raise ValueError(
            "measure.extended_alternate_endings_present must agree with its values"
        )
    _expect_int(measure["key_accidental_count"], "measure.key_accidental_count")
    for field in (
        "section_present",
        "triplet_feel_present",
        "directions_present",
    ):
        _expect_bool(measure[field], f"measure.{field}")
    for field in ("section_letter", "section_text"):
        section_value = _expect_optional_string(measure[field], f"measure.{field}")
        if measure["section_present"] and section_value is None:
            raise ValueError(
                f"measure.{field} must be a string when section is present"
            )
        if not measure["section_present"] and section_value is not None:
            raise ValueError(f"measure.{field} must be null when section is absent")
    _validate_optional_named_fields(measure, "triplet_feel", "measure")
    directions = _expect_list(measure["directions"], "measure.directions")
    for position, direction in enumerate(directions):
        _validate_named_enum(direction, f"measure.directions[{position}]")
    if bool(directions) != measure["directions_present"]:
        raise ValueError(
            "measure.directions_present must agree with measure.directions"
        )


def _validate_voice(
    value: Mapping[str, Any],
    *,
    measure_id: str,
    string_count: int,
    state: _ValidationState,
) -> None:
    voice = _expect_object(value, "voice", VOICE_FIELDS)
    voice_index = _expect_nonnegative_int(voice["voice_index"], "voice.voice_index")
    voice_id = f"{measure_id}/voice:{voice_index}"
    _expect_semantic_id(voice, voice_id, "voice", state)
    events = _indexed_children(
        voice["events"], index_field="event_index", label=f"{voice_id} event"
    )
    for _, event in events:
        _validate_event(
            event,
            voice_id=voice_id,
            string_count=string_count,
            state=state,
        )


def _validate_event(
    value: Mapping[str, Any],
    *,
    voice_id: str,
    string_count: int,
    state: _ValidationState,
) -> None:
    event = _expect_object(value, "event", EVENT_FIELDS)
    event_index = _expect_nonnegative_int(event["event_index"], "event.event_index")
    event_id = f"{voice_id}/event:{event_index}"
    _expect_semantic_id(event, event_id, "event", state)

    for field in (
        "placeholder",
        "rest",
        "sustain_pedal",
        "dead_slapped",
        "grace",
        "ottavia_present",
        "slashed",
        "barre_present",
        "fade_present",
        "hairpin_present",
        "golpe_present",
        "wah_present",
        "timer_present",
        "free_text_present",
        "chord_present",
        "is_brushed",
        "has_arpeggio",
        "pick_stroke_present",
        "slapped",
        "popped",
        "rasgueado_present",
        "tremolo_bar_vibrato_present",
        "legato_origin",
        "legato_destination",
    ):
        _expect_bool(event[field], f"event.{field}")
    _expect_int(event["grace_type_code"], "event.grace_type_code")
    grace_name = _expect_optional_string(
        event["grace_type_name"], "event.grace_type_name"
    )
    if not event["grace"] and grace_name is not None:
        raise ValueError("event.grace_type_name must be null when grace is false")
    _expect_nonnegative_int(event["grace_beat_count"], "event.grace_beat_count")
    for prefix in (
        "ottavia",
        "fade",
        "hairpin",
        "golpe",
        "wah",
        "pick_stroke",
        "rasgueado",
        "tremolo_bar_vibrato",
    ):
        _validate_optional_named_fields(event, prefix, "event")
    _validate_rational(event["offset"], "event.offset")
    _validate_rhythm(event["rhythm"])

    for field in ("barre_string", "barre_fret"):
        value = event[field]
        if value is not None:
            _expect_nonnegative_int(value, f"event.{field}")
        if event["barre_present"] != (value is not None):
            raise ValueError(
                f"event.{field} presence must agree with event.barre_present"
            )

    timer = event["timer"]
    if timer is not None:
        _expect_nonnegative_int(timer, "event.timer")
    if event["timer_present"] != (timer is not None):
        raise ValueError("event.timer presence must agree with event.timer_present")
    free_text = _expect_optional_string(event["free_text"], "event.free_text")
    if event["free_text_present"] != (free_text is not None):
        raise ValueError(
            "event.free_text presence must agree with event.free_text_present"
        )
    chord = _expect_optional_string(event["chord"], "event.chord")
    if not event["chord_present"] and chord is not None:
        raise ValueError("event.chord must be null when chord_present is false")
    chord_diagram = event["chord_diagram"]
    if chord_diagram is not None:
        _validate_chord_diagram(chord_diagram, label="event.chord_diagram")
    if not event["chord_present"] and chord_diagram is not None:
        raise ValueError("event.chord_diagram must be null when chord_present is false")
    lyrics = _expect_list(event["lyrics"], "event.lyrics")
    lyric_lines: list[int] = []
    for position, raw_lyric in enumerate(lyrics):
        label = f"event.lyrics[{position}]"
        lyric = _expect_object(raw_lyric, label, LYRIC_FIELDS)
        line_index = _expect_nonnegative_int(lyric["line_index"], f"{label}.line_index")
        if line_index >= 5:
            raise ValueError(f"{label}.line_index must be below 5")
        lyric_lines.append(line_index)
        text = _expect_string(lyric["text"], f"{label}.text")
        displayed_text = _expect_string(
            lyric["displayed_text"], f"{label}.displayed_text"
        )
        extend_code = _expect_int(lyric["extend_code"], f"{label}.extend_code")
        syllabic_code = _expect_int(lyric["syllabic_code"], f"{label}.syllabic_code")
        if extend_code not in range(5):
            raise ValueError(f"{label}.extend_code is not an official continuation")
        if syllabic_code not in range(5):
            raise ValueError(f"{label}.syllabic_code is not an official continuation")
        if not text and not displayed_text and extend_code == 0 and syllabic_code < 2:
            raise ValueError(f"{label} contains no visible lyric content")
        _expect_int(lyric["h_alignment_code"], f"{label}.h_alignment_code")
        _expect_number(lyric["x_offset"], f"{label}.x_offset")
    if lyric_lines != sorted(set(lyric_lines)):
        raise ValueError("event.lyrics must have unique ordered line indexes")

    _validate_named_enum(event["dynamic"], "event.dynamic")
    for code_field, name_field in (
        ("brush_code", "brush_name"),
        ("arpeggio_code", "arpeggio_name"),
    ):
        _expect_int(event[code_field], f"event.{code_field}")
        _expect_optional_string(event[name_field], f"event.{name_field}")
    for gate_field, name_field in (
        ("is_brushed", "brush_name"),
        ("has_arpeggio", "arpeggio_name"),
    ):
        if not event[gate_field] and event[name_field] is not None:
            raise ValueError(
                f"event.{name_field} must be null when {gate_field} is false"
            )
    if event["tremolo"] is not None:
        _validate_rational(event["tremolo"], "event.tremolo")
    if event["whammy"] is not None:
        _validate_whammy(event["whammy"])
    _expect_bool(event["whammy_extend"], "event.whammy_extend")
    _expect_nonnegative_int(event["whammy_beat_count"], "event.whammy_beat_count")
    for field in ("whammy_begin_ref", "whammy_end_ref"):
        _expect_optional_string(event[field], f"event.{field}")
    for field in ("legato_origin_ref", "legato_destination_ref"):
        _expect_optional_string(event[field], f"event.{field}")

    notes = _indexed_children(
        event["notes"], index_field="note_index", label=f"{event_id} note"
    )
    for _, note in notes:
        _validate_note(
            note,
            event_id=event_id,
            string_count=string_count,
            state=state,
        )


def _validate_rhythm(value: Any) -> None:
    rhythm = _expect_object(value, "event.rhythm", RHYTHM_FIELDS)
    note_value_code = _expect_int(
        rhythm["note_value_code"], "event.rhythm.note_value_code"
    )
    if note_value_code not in range(11):
        raise ValueError("event.rhythm.note_value_code is not an official note value")
    _expect_nonnegative_int(
        rhythm["augmentation_dots"], "event.rhythm.augmentation_dots"
    )
    tuplets = _expect_object(rhythm["tuplets"], "event.rhythm.tuplets", TUPLET_FIELDS)
    for field in ("primary", "secondary"):
        if tuplets[field] is not None:
            ratio = _validate_rational(tuplets[field], f"event.rhythm.tuplets.{field}")
            if ratio[0] <= 0 or ratio[1] <= 0:
                raise ValueError(f"event.rhythm.tuplets.{field} must be positive")


def _validate_whammy(value: Any) -> None:
    whammy = _expect_object(value, "event.whammy", WHAMMY_FIELDS)
    _validate_curve(whammy, "event.whammy")


def _validate_chord_diagram(value: Any, *, label: str) -> None:
    diagram = _expect_object(
        value,
        label,
        CHORD_DIAGRAM_FIELDS,
    )
    _expect_string(diagram["name"], f"{label}.name")
    string_count = _expect_nonnegative_int(
        diagram["string_count"], f"{label}.string_count"
    )
    if string_count == 0:
        raise ValueError(f"{label}.string_count must be positive")
    _expect_nonnegative_int(diagram["base_fret"], f"{label}.base_fret")
    fret_count = _expect_nonnegative_int(diagram["fret_count"], f"{label}.fret_count")
    if fret_count == 0:
        raise ValueError(f"{label}.fret_count must be positive")
    _expect_nonnegative_int(diagram["span_limit"], f"{label}.span_limit")
    for field in ("show_diagram", "show_name", "show_fingering"):
        _expect_bool(diagram[field], f"{label}.{field}")
    barres = _expect_list(diagram["barres"], f"{label}.barres")
    for position, raw_barre in enumerate(barres):
        barre_label = f"{label}.barres[{position}]"
        barre = _expect_object(
            raw_barre,
            barre_label,
            CHORD_DIAGRAM_BARRE_FIELDS,
        )
        relative_fret = _expect_nonnegative_int(
            barre["relative_fret"], f"{barre_label}.relative_fret"
        )
        if relative_fret == 0 or relative_fret > fret_count:
            raise ValueError(f"{barre_label}.relative_fret is outside the grid")
        start = _expect_nonnegative_int(
            barre["start_native_string_index"],
            f"{barre_label}.start_native_string_index",
        )
        end = _expect_nonnegative_int(
            barre["end_native_string_index"],
            f"{barre_label}.end_native_string_index",
        )
        if start >= end or end >= string_count:
            raise ValueError(f"{barre_label} has invalid string endpoints")
    strings = _expect_list(
        diagram["strings_low_to_high"],
        f"{label}.strings_low_to_high",
    )
    if len(strings) != string_count:
        raise ValueError(f"{label} string count must match strings_low_to_high")
    for position, raw_string in enumerate(strings):
        string_label = f"{label}.strings_low_to_high[{position}]"
        string = _expect_object(
            raw_string,
            string_label,
            CHORD_DIAGRAM_STRING_FIELDS,
        )
        native_index = _expect_nonnegative_int(
            string["native_string_index"],
            f"{string_label}.native_string_index",
        )
        if native_index != position:
            raise ValueError(f"{label} strings must be ordered from native string zero")
        relative_fret = string["relative_fret"]
        if relative_fret is not None:
            relative_fret = _expect_nonnegative_int(
                relative_fret, f"{string_label}.relative_fret"
            )
            if relative_fret > fret_count:
                raise ValueError(f"{string_label}.relative_fret is outside the grid")
        finger_code = string["finger_code"]
        if finger_code is not None:
            finger_code = _expect_int(finger_code, f"{string_label}.finger_code")
            if finger_code not in range(-1, 5):
                raise ValueError(f"{string_label}.finger_code is not official")
            if relative_fret in {None, 0} and finger_code != -1:
                raise ValueError(
                    f"{string_label}.finger_code requires a fretted string"
                )


def _validate_note(
    value: Mapping[str, Any],
    *,
    event_id: str,
    string_count: int,
    state: _ValidationState,
) -> None:
    note = _expect_object(value, "note", NOTE_FIELDS)
    note_index = _expect_nonnegative_int(note["note_index"], "note.note_index")
    note_id = f"{event_id}/note:{note_index}"
    _expect_semantic_id(note, note_id, "note", state)
    native_string_index = _expect_nonnegative_int(
        note["native_string_index"], "note.native_string_index"
    )
    if native_string_index >= string_count:
        raise ValueError("note.native_string_index is outside the staff tuning")
    for field in ("fret", "accent_flags", "slide_flags"):
        _expect_nonnegative_int(note[field], f"note.{field}")
    for field in (
        "printable_in_tab",
        "dead",
        "palm_muted",
        "let_ring",
        "anti_accent",
        "tie_origin",
        "tie_destination",
        "hopo_origin",
        "hopo_destination",
        "trill_present",
        "trill_valid",
        "vibrato_present",
        "tapped",
        "left_hand_tapped",
        "pizzicato",
        "show_string_number",
        "sustain_pedal",
        "ornament_present",
    ):
        _expect_bool(note[field], f"note.{field}")
    for field in (
        "tie_origin_ref",
        "tie_destination_ref",
        "hopo_origin_ref",
        "hopo_destination_ref",
        "slide_begin_ref",
        "slide_end_ref",
    ):
        _expect_optional_string(note[field], f"note.{field}")
    _expect_int(note["anti_accent_code"], "note.anti_accent_code")
    anti_accent_name = _expect_optional_string(
        note["anti_accent_name"], "note.anti_accent_name"
    )
    if not note["anti_accent"] and anti_accent_name is not None:
        raise ValueError("note.anti_accent_name must be null when anti_accent is false")

    slides = _expect_object(note["slides"], "note.slides", SLIDE_FIELDS)
    for field in SLIDE_FIELDS:
        _expect_bool(slides[field], f"note.slides.{field}")
    if note["bend"] is not None:
        bend = _expect_object(note["bend"], "note.bend", BEND_FIELDS)
        _validate_curve(bend, "note.bend")
    if note["harmonic"] is not None:
        _validate_harmonic(note["harmonic"])

    trill_fret = note["trill_fret"]
    if trill_fret is not None:
        _expect_int(trill_fret, "note.trill_fret")
    if note["trill_present"] != (trill_fret is not None):
        raise ValueError("note.trill_fret presence must agree with note.trill_present")
    trill_midi = note["trill_midi"]
    if trill_midi is not None:
        _expect_nonnegative_int(trill_midi, "note.trill_midi")
    if note["trill_valid"] != (trill_midi is not None):
        raise ValueError("note.trill_midi presence must agree with note.trill_valid")
    if note["trill_valid"] and not note["trill_present"]:
        raise ValueError("note.trill_valid requires note.trill_present")
    _validate_optional_named_fields(note, "vibrato", "note")
    _validate_optional_named_fields(note, "ornament", "note")

    for code_field, name_field in (
        ("left_fingering_code", "left_fingering_name"),
        ("right_fingering_code", "right_fingering_name"),
    ):
        _expect_int(note[code_field], f"note.{code_field}")
        _expect_optional_string(note[name_field], f"note.{name_field}")


def _validate_harmonic(value: Any) -> None:
    harmonic = _expect_object(value, "note.harmonic", HARMONIC_FIELDS)
    _expect_int(harmonic["type_code"], "note.harmonic.type_code")
    _expect_optional_string(harmonic["type"], "note.harmonic.type")
    _expect_int(harmonic["fret_code"], "note.harmonic.fret_code")
    _expect_number(harmonic["touch_offset"], "note.harmonic.touch_offset")


def _validate_curve(value: Mapping[str, Any], label: str) -> None:
    _expect_int(value["type_code"], f"{label}.type_code")
    _expect_optional_string(value["type"], f"{label}.type")
    _validate_numeric_array(value["origin"], 2, f"{label}.origin")
    _validate_numeric_array(value["middle"], 3, f"{label}.middle")
    _validate_numeric_array(value["destination"], 2, f"{label}.destination")


def _validate_named_enum(value: Any, label: str) -> None:
    enum_value = _expect_object(value, label, NAMED_ENUM_FIELDS)
    _expect_int(enum_value["code"], f"{label}.code")
    _expect_optional_string(enum_value["name"], f"{label}.name")


def _validate_optional_named_fields(
    value: Mapping[str, Any], prefix: str, label: str
) -> None:
    present_field = f"{prefix}_present"
    code_field = f"{prefix}_code"
    name_field = f"{prefix}_name"
    present = _expect_bool(value[present_field], f"{label}.{present_field}")
    _expect_int(value[code_field], f"{label}.{code_field}")
    name = _expect_optional_string(value[name_field], f"{label}.{name_field}")
    if not present and name is not None:
        raise ValueError(
            f"{label}.{name_field} must be null when {present_field} is false"
        )


def _expect_semantic_id(
    value: Mapping[str, Any],
    expected: str,
    label: str,
    state: _ValidationState,
) -> None:
    actual = _expect_string(value["semantic_id"], f"{label}.semantic_id")
    if actual != expected:
        raise ValueError(
            f"{label}.semantic_id must be derived from its explicit indexes: {expected!r}"
        )
    if actual in state.semantic_ids:
        raise ValueError(f"duplicate official semantic_id {actual!r}")
    state.semantic_ids.add(actual)


def _indexed_children(
    values: Any,
    *,
    index_field: str,
    label: str,
    contiguous: bool = False,
) -> list[tuple[int, Mapping[str, Any]]]:
    children = _expect_list(values, label)
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    seen: set[int] = set()
    for position, raw_child in enumerate(children):
        child = _expect_mapping(raw_child, f"{label}[{position}]")
        if index_field not in child:
            raise ValueError(f"{label}[{position}] is missing {index_field!r}")
        index = _expect_nonnegative_int(
            child[index_field], f"{label}[{position}].{index_field}"
        )
        if index in seen:
            raise ValueError(f"{label} repeats {index_field} {index}")
        seen.add(index)
        indexed.append((index, child))
    if contiguous and seen != set(range(len(children))):
        raise ValueError(f"{label} {index_field} values must be contiguous from zero")
    indexed.sort(key=lambda item: item[0])
    return indexed


def _indexed(
    values: Any,
    index: int,
    label: str,
    index_field: str,
) -> Mapping[str, Any]:
    target = _expect_nonnegative_int(index, f"requested {label} index")
    for child_index, child in _indexed_children(
        values, index_field=index_field, label=label
    ):
        if child_index == target:
            return child
    raise IndexError(f"{label} index {target} is out of range")


def _expect_object(
    value: Any, label: str, exact_fields: frozenset[str]
) -> Mapping[str, Any]:
    result = _expect_mapping(value, label)
    actual_fields = set(result)
    if actual_fields != exact_fields:
        missing = sorted(exact_fields - actual_fields)
        unknown = sorted(actual_fields - exact_fields, key=str)
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing!r}")
        if unknown:
            details.append(f"unknown fields {unknown!r}")
        raise ValueError(f"{label} has " + " and ".join(details))
    return result


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _expect_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _expect_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _expect_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _expect_nonnegative_int(value: Any, label: str) -> int:
    result = _expect_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _expect_number(value: Any, label: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite number")
    return value


def _expect_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, label)


def _validate_rational(value: Any, label: str) -> tuple[int, int]:
    pair = _expect_list(value, label)
    if len(pair) != 2:
        raise ValueError(f"{label} must contain exactly two integers")
    numerator = _expect_int(pair[0], f"{label}[0]")
    denominator = _expect_int(pair[1], f"{label}[1]")
    if denominator == 0:
        raise ValueError(f"{label} denominator must be nonzero")
    return numerator, denominator


def _validate_numeric_array(value: Any, size: int, label: str) -> None:
    items = _expect_list(value, label)
    if len(items) != size:
        raise ValueError(f"{label} must contain exactly {size} numbers")
    for index, item in enumerate(items):
        _expect_number(item, f"{label}[{index}]")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


__all__ = [
    "OFFICIAL_SCORE_SCHEMA",
    "OfficialNode",
    "load_official_score",
    "selected_staff",
    "validate_official_score",
    "walk_official_nodes",
]
