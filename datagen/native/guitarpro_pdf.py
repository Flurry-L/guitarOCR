from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .score_official_score import OFFICIAL_SCORE_SCHEMA
from .native_session import NativeExportSession


RENDER_LAYOUT_SCHEMA = "gpomr.render-layout"
OFFICIAL_PRIMARY_LOAD_REJECTED = "official_primary_load_rejected"

_ROOT_FIELDS = frozenset(
    {
        "schema",
        "pdf_ok",
        "source_track_index",
        "tab_only",
        "tab_style",
        "tuning_style",
        "pages",
        "systems",
        "rest_views",
        "lyric_events",
        "tempo_indications",
    }
)
_TUNING_STYLE_FIELDS = frozenset(
    {"position_code", "mode_code", "column_count", "boxed"}
)
_TAB_STYLE_FIELDS = frozenset(
    {
        "force_rhythmic_band",
        "extend_rhythmic_in_tablature",
        "hide_useless_rests",
        "show_quarter_rest_as_dash",
        "always_show_tie_notes",
        "hide_ties_between_bar",
        "display_fret_relative_to_capo",
        "fret_visibility_code",
        "trill_fret_visibility_code",
        "artificial_harmonic_fret_visibility_code",
    }
)
_PAGE_FIELDS = frozenset({"index", "bbox_mm"})
_SYSTEM_FIELDS = frozenset(
    {
        "system_index",
        "page",
        "first_measure_index",
        "last_measure_index",
        "bbox_mm",
        "measure_boxes",
        "track_label",
        "time_signatures",
        "key_signatures",
    }
)
_MEASURE_BOX_FIELDS = frozenset({"measure_index", "bbox_mm"})
_TIME_SIGNATURE_FIELDS = frozenset(
    {"staff_index", "measure_index", "numerator", "denominator"}
)
_KEY_SIGNATURE_FIELDS = frozenset({"staff_index", "measure_index", "accidental_count"})
_LYRIC_EVENT_FIELDS = frozenset(
    {
        "staff_index",
        "measure_index",
        "voice_index",
        "event_index",
    }
)
_TEMPO_INDICATION_FIELDS = frozenset(
    {
        "page",
        "system_index",
        "master_measure_index",
        "position",
        "bbox_mm",
        "source",
    }
)
_REST_VIEW_FIELDS = frozenset(
    {
        "staff_index",
        "measure_index",
        "voice_index",
        "event_index",
        "visibility_code",
    }
)


@dataclass(frozen=True, slots=True)
class PdfExportJob:
    source: Path
    pdf: Path
    layout: Path
    official_score: Path
    track_index: int = 0


@dataclass(frozen=True, slots=True)
class SourceTrack:
    source_track_index: int
    name: str
    instrument_kind: str


class OfficialPrimaryLoadRejected(RuntimeError):
    """The bundled official importer rejected the source before loading a score."""


def _raise_native_failure(result: Mapping[str, Any], fallback: str) -> None:
    message = str(result.get("error") or fallback)
    error_code = result.get("error_code")
    if error_code is None:
        raise RuntimeError(message)
    if error_code == OFFICIAL_PRIMARY_LOAD_REJECTED:
        raise OfficialPrimaryLoadRejected(message)
    if not isinstance(error_code, str) or not error_code:
        raise ValueError("native failure error_code must be a nonempty string")
    raise RuntimeError(f"native failure {error_code}: {message}")


class GuitarProPdfExporter:
    """Export PDF, official score data, and the raw native render layout."""

    def __init__(self, runtime_dir: str | Path, *, ready_timeout: int = 120) -> None:
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.ready_timeout = int(ready_timeout)
        self._session: NativeExportSession | None = None

    def __enter__(self) -> GuitarProPdfExporter:
        session = NativeExportSession(
            runtime_dir=self.runtime_dir,
            ready_timeout=self.ready_timeout,
        )
        self._session = session.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.__exit__(*args)

    def export(self, job: PdfExportJob) -> None:
        if self._session is None:
            raise RuntimeError("Guitar Pro exporter is not open")
        track_index = _integer(
            job.track_index,
            "PDF export job track_index",
            nonnegative=True,
        )
        source = Path(job.source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        outputs = (job.pdf, job.layout, job.official_score)
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(output)

        try:
            result = self._session.export(
                source,
                job.pdf,
                layout_output=job.layout,
                official_score_output=job.official_score,
                track_index=track_index,
            )
            if result.get("ok") is not True:
                _raise_native_failure(result, "Guitar Pro export failed")
            missing = [str(path) for path in outputs if not path.is_file()]
            if missing:
                raise RuntimeError(
                    "Guitar Pro export did not create: " + ", ".join(missing)
                )
            layout = load_render_layout(job.layout)
            official_score = _load_official_export_identity(job.official_score)
            if layout["source_track_index"] != track_index:
                raise ValueError(
                    "render layout source_track_index differs from the requested track"
                )
            if official_score["source_track_index"] != track_index:
                raise ValueError(
                    "official score source_track_index differs from the requested track"
                )
            _validate_export_measure_coverage(official_score, layout)
            validate_pdf_layout(job.pdf, layout)
        except BaseException:
            for output in outputs:
                try:
                    output.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def list_tracks(self, source: str | Path) -> tuple[SourceTrack, ...]:
        if self._session is None:
            raise RuntimeError("Guitar Pro exporter is not open")
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        result = self._session.list_tracks(source_path)
        if result.get("ok") is not True:
            _raise_native_failure(result, "Guitar Pro track enumeration failed")
        if set(result) != {"ok", "tracks"}:
            raise ValueError("track enumeration response has unexpected fields")
        raw_tracks = result["tracks"]
        if not isinstance(raw_tracks, list):
            raise TypeError("track enumeration response tracks must be an array")

        tracks: list[SourceTrack] = []
        indexes: set[int] = set()
        for position, value in enumerate(raw_tracks):
            label = f"track enumeration response track {position}"
            track = _mapping(
                value,
                label,
                frozenset({"source_track_index", "name", "instrument_kind"}),
            )
            source_track_index = _integer(
                track["source_track_index"],
                f"{label}.source_track_index",
                nonnegative=True,
            )
            if source_track_index in indexes:
                raise ValueError(
                    "track enumeration response repeats source_track_index "
                    f"{source_track_index}"
                )
            indexes.add(source_track_index)
            name = track["name"]
            if not isinstance(name, str):
                raise TypeError(f"{label}.name must be a string")
            instrument_kind = track["instrument_kind"]
            if instrument_kind not in {"guitar", "bass"}:
                raise ValueError(f"{label}.instrument_kind must be guitar or bass")
            tracks.append(
                SourceTrack(
                    source_track_index=source_track_index,
                    name=name,
                    instrument_kind=instrument_kind,
                )
            )
        tracks.sort(key=lambda track: track.source_track_index)
        return tuple(tracks)

    def release_source(self) -> None:
        if self._session is None:
            raise RuntimeError("Guitar Pro exporter is not open")
        result = self._session.release_source()
        if result.get("ok") is not True:
            _raise_native_failure(result, "Guitar Pro source release failed")
        if set(result) != {"ok"}:
            raise ValueError("source release response has unexpected fields")


def load_render_layout(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json_number,
    )
    validate_render_layout(value)
    return value


def _load_official_export_identity(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json_number,
    )
    if not isinstance(value, Mapping):
        raise TypeError("official export root must be an object")
    if value.get("schema") != OFFICIAL_SCORE_SCHEMA:
        raise ValueError("official export has an unsupported schema")
    if value.get("pdf_ok") is not True:
        raise ValueError("official export does not describe a successful PDF")
    _integer(
        value.get("source_track_index"),
        "official export source_track_index",
        nonnegative=True,
    )
    tracks = value.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != 1:
        raise ValueError("official export must contain one selected track")
    track = tracks[0]
    if not isinstance(track, Mapping) or track.get("track_index") != 0:
        raise ValueError("official export selected track must have track_index zero")
    return value


def _validate_export_measure_coverage(
    official_score: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> None:
    document = official_score.get("document")
    if not isinstance(document, Mapping):
        raise TypeError("official export document must be an object")
    master_measures = document.get("master_measures")
    if not isinstance(master_measures, list):
        raise TypeError("official export master_measures must be an array")
    for position, measure in enumerate(master_measures):
        if (
            not isinstance(measure, Mapping)
            or measure.get("master_measure_index") != position
        ):
            raise ValueError(
                "official export master measure indexes must be consecutive from zero"
            )
    measure_count = len(master_measures)

    track = official_score["tracks"][0]
    staves = track.get("staves")
    if not isinstance(staves, list) or not staves:
        raise ValueError("official export selected track must contain TAB staves")
    for staff_position, staff in enumerate(staves):
        if not isinstance(staff, Mapping):
            raise TypeError("official export staff must be an object")
        measures = staff.get("measures")
        if not isinstance(measures, list) or len(measures) != measure_count:
            raise ValueError(
                "official export staff measure coverage differs from its master measures"
            )
        for measure_position, measure in enumerate(measures):
            if (
                not isinstance(measure, Mapping)
                or measure.get("measure_index") != measure_position
                or measure.get("master_measure_index") != measure_position
            ):
                raise ValueError(
                    "official export staff measure indexes must match master measures"
                )
        if staff.get("staff_index") != staff_position:
            raise ValueError(
                "official export staff indexes must be consecutive from zero"
            )

    systems = layout["systems"]
    layout_measure_count = int(systems[-1]["last_measure_index"]) if systems else 0
    if layout_measure_count != measure_count:
        raise ValueError(
            "official export and render layout cover different measure counts"
        )


def validate_pdf_layout(pdf_path: str | Path, layout: Mapping[str, Any]) -> None:
    """Require the PDF and native page table to describe the same pages."""

    validate_render_layout(layout)
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    pages = layout["pages"]
    mm_to_points = 72.0 / 25.4
    with fitz.open(source) as document:
        if len(document) != len(pages):
            raise ValueError("PDF page count does not match the native layout")
        for page_index, (pdf_page, layout_page) in enumerate(
            zip(document, pages, strict=True), start=1
        ):
            layout_bbox = _bbox(
                layout_page["bbox_mm"],
                f"render layout page {page_index} bbox",
                nonempty=True,
            )
            pdf_width_points = float(pdf_page.rect.width)
            pdf_height_points = float(pdf_page.rect.height)
            if not _matches_pdf_page_dimension(
                layout_bbox[2], pdf_width_points, mm_to_points
            ) or not _matches_pdf_page_dimension(
                layout_bbox[3], pdf_height_points, mm_to_points
            ):
                raise ValueError(
                    f"PDF page {page_index} dimensions do not match the native layout"
                )


def _matches_pdf_page_dimension(
    layout_mm: float, pdf_points: float, mm_to_points: float
) -> bool:
    expected_points = layout_mm * mm_to_points
    return math.isclose(pdf_points, expected_points, rel_tol=0.0, abs_tol=1e-4) or (
        math.isclose(
            pdf_points,
            float(round(expected_points)),
            rel_tol=0.0,
            abs_tol=1e-4,
        )
    )


def validate_render_layout(layout: Any) -> None:
    """Validate the page, system, and explicit display gates from the exporter."""

    from .score_native_note_geometry import validate_note_geometry

    has_geometry = isinstance(layout, Mapping) and "note_geometry" in layout
    root = _mapping(
        {key: value for key, value in layout.items() if key != "note_geometry"}
        if has_geometry
        else layout,
        "render layout root",
        _ROOT_FIELDS,
    )
    if root["schema"] != RENDER_LAYOUT_SCHEMA:
        raise ValueError("render layout has an unsupported schema")
    if root["pdf_ok"] is not True:
        raise ValueError("render layout does not describe a successful export")
    _integer(
        root["source_track_index"],
        "render layout source_track_index",
        nonnegative=True,
    )
    if has_geometry:
        validate_note_geometry(layout)
    if root["tab_only"] is not True:
        raise ValueError("render layout must describe a TAB-only export")
    tab_style = _mapping(
        root["tab_style"],
        "render layout tab_style",
        _TAB_STYLE_FIELDS,
    )
    for field in (
        "force_rhythmic_band",
        "extend_rhythmic_in_tablature",
        "hide_useless_rests",
        "show_quarter_rest_as_dash",
        "always_show_tie_notes",
        "hide_ties_between_bar",
        "display_fret_relative_to_capo",
    ):
        if not isinstance(tab_style[field], bool):
            raise TypeError(f"render layout tab_style.{field} must be a boolean")
    for field in (
        "fret_visibility_code",
        "trill_fret_visibility_code",
        "artificial_harmonic_fret_visibility_code",
    ):
        code = _integer(
            tab_style[field],
            f"render layout tab_style.{field}",
            nonnegative=True,
        )
        if code not in {0, 1, 2}:
            raise ValueError(
                f"render layout tab_style.{field} must be an official visibility code"
            )

    tuning_style = _mapping(
        root["tuning_style"],
        "render layout tuning_style",
        _TUNING_STYLE_FIELDS,
    )
    position_code = _integer(
        tuning_style["position_code"],
        "render layout tuning_style.position_code",
        nonnegative=True,
    )
    if position_code not in {0, 1, 2}:
        raise ValueError(
            "render layout tuning_style.position_code must be an official position code"
        )
    mode_code = _integer(
        tuning_style["mode_code"],
        "render layout tuning_style.mode_code",
        nonnegative=True,
    )
    if mode_code not in {0, 1, 2, 3}:
        raise ValueError(
            "render layout tuning_style.mode_code must be an official mode code"
        )
    _integer(
        tuning_style["column_count"],
        "render layout tuning_style.column_count",
    )
    if not isinstance(tuning_style["boxed"], bool):
        raise TypeError("render layout tuning_style.boxed must be a boolean")
    pages = root["pages"]
    if not isinstance(pages, list) or not pages:
        raise ValueError("render layout must contain pages")

    pages_by_index: dict[int, tuple[float, float, float, float]] = {}
    for position, value in enumerate(pages):
        page = _mapping(value, f"render layout page {position}", _PAGE_FIELDS)
        page_index = _integer(page["index"], "page.index", positive=True)
        if page_index != position + 1:
            raise ValueError("render layout page indexes must be consecutive from 1")
        pages_by_index[page_index] = _bbox(page["bbox_mm"], "page bbox", nonempty=True)

    systems = root["systems"]
    if not isinstance(systems, list):
        raise TypeError("render layout systems must be an array")
    next_measure_index = 0
    previous_page = 1
    for position, value in enumerate(systems):
        label = f"render layout system {position}"
        system = _mapping(value, label, _SYSTEM_FIELDS)
        system_index = _integer(
            system["system_index"], f"{label}.system_index", nonnegative=True
        )
        if system_index != position:
            raise ValueError("render layout system indexes must be consecutive from 0")

        page = _integer(system["page"], f"{label}.page", positive=True)
        if page not in pages_by_index:
            raise ValueError(f"{label} has an unknown page")
        if page < previous_page:
            raise ValueError("render layout systems must be in page order")
        previous_page = page

        first = _integer(
            system["first_measure_index"],
            f"{label}.first_measure_index",
            nonnegative=True,
        )
        last = _integer(
            system["last_measure_index"],
            f"{label}.last_measure_index",
            nonnegative=True,
        )
        if first != next_measure_index or last <= first:
            raise ValueError(
                "render layout system measure ranges must be nonempty and sequential"
            )
        next_measure_index = last

        system_bbox = _bbox(system["bbox_mm"], f"{label} bbox", nonempty=True)
        if not _bbox_is_inside(system_bbox, pages_by_index[page]):
            raise ValueError(f"{label} bbox lies outside its declared page")

        measure_boxes = system["measure_boxes"]
        if not isinstance(measure_boxes, list):
            raise TypeError(f"{label}.measure_boxes must be an array")
        if len(measure_boxes) != last - first:
            raise ValueError(
                f"{label}.measure_boxes must cover the system range exactly once"
            )
        previous_measure_x: float | None = None
        for measure_position, raw_measure_box in enumerate(measure_boxes):
            measure_label = f"{label}.measure_boxes[{measure_position}]"
            measure_box = _mapping(
                raw_measure_box,
                measure_label,
                _MEASURE_BOX_FIELDS,
            )
            measure_index = _integer(
                measure_box["measure_index"],
                f"{measure_label}.measure_index",
                nonnegative=True,
            )
            if measure_index != first + measure_position:
                raise ValueError(
                    f"{label}.measure_boxes must be ordered by consecutive measure index"
                )
            bbox = _bbox(
                measure_box["bbox_mm"],
                f"{measure_label}.bbox_mm",
                nonempty=True,
            )
            if not _bbox_is_inside(bbox, pages_by_index[page]):
                raise ValueError(f"{measure_label} lies outside its declared page")
            if previous_measure_x is not None and bbox[0] < previous_measure_x:
                raise ValueError(f"{label}.measure_boxes must be in visual order")
            previous_measure_x = bbox[0]

        if system["track_label"] not in {None, "name", "short_name"}:
            raise ValueError(f"{label}.track_label is unsupported")
        for field, locator_fields in (
            ("time_signatures", _TIME_SIGNATURE_FIELDS),
            ("key_signatures", _KEY_SIGNATURE_FIELDS),
        ):
            locators = system[field]
            if not isinstance(locators, list):
                raise TypeError(f"{label}.{field} must be an array")
            locations: set[tuple[int, int]] = set()
            for locator_position, raw_locator in enumerate(locators):
                locator_label = f"{label}.{field}[{locator_position}]"
                locator = _mapping(
                    raw_locator,
                    locator_label,
                    locator_fields,
                )
                location = tuple(
                    _integer(
                        locator[locator_field],
                        f"{locator_label}.{locator_field}",
                        nonnegative=True,
                    )
                    for locator_field in ("staff_index", "measure_index")
                )
                if location in locations:
                    raise ValueError(f"{label}.{field} must be unique")
                locations.add(location)
                if location[1] < first or location[1] >= last:
                    raise ValueError(f"{locator_label} lies outside the system range")
                if field == "time_signatures":
                    _integer(
                        locator["numerator"],
                        f"{locator_label}.numerator",
                        positive=True,
                    )
                    _integer(
                        locator["denominator"],
                        f"{locator_label}.denominator",
                        positive=True,
                    )
                else:
                    _integer(
                        locator["accidental_count"],
                        f"{locator_label}.accidental_count",
                    )

    _validate_layout_views(
        root["rest_views"],
        label="render layout rest_views",
        fields=_REST_VIEW_FIELDS,
        location_fields=(
            "staff_index",
            "measure_index",
            "voice_index",
            "event_index",
        ),
        measure_count=next_measure_index,
        require_text=False,
    )
    _validate_event_locators(
        root["lyric_events"],
        label="render layout lyric_events",
        fields=_LYRIC_EVENT_FIELDS,
        measure_count=next_measure_index,
    )
    _validate_tempo_indications(
        root["tempo_indications"],
        pages_by_index=pages_by_index,
        systems=systems,
        measure_count=next_measure_index,
    )


def _validate_tempo_indications(
    value: Any,
    *,
    pages_by_index: Mapping[int, tuple[float, float, float, float]],
    systems: Sequence[Mapping[str, Any]],
    measure_count: int,
) -> None:
    label = "render layout tempo_indications"
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    systems_by_index = {int(system["system_index"]): system for system in systems}
    identities: set[tuple[str, int, int | float]] = set()
    sort_keys: list[tuple[int, float, str]] = []
    initial_count = 0
    for record_position, raw_indication in enumerate(value):
        indication_label = f"{label}[{record_position}]"
        indication = _mapping(
            raw_indication,
            indication_label,
            _TEMPO_INDICATION_FIELDS,
        )
        source = indication["source"]
        if source not in {"initial", "automation"}:
            raise ValueError(f"{indication_label}.source is unsupported")
        page = _integer(indication["page"], f"{indication_label}.page", positive=True)
        if page not in pages_by_index:
            raise ValueError(f"{indication_label} has an unknown page")
        measure = _integer(
            indication["master_measure_index"],
            f"{indication_label}.master_measure_index",
            nonnegative=True,
        )
        if measure >= measure_count:
            raise ValueError(f"{indication_label} lies outside the score")
        position = indication["position"]
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(float(position))
            or not 0 <= position <= 1
        ):
            raise ValueError(
                f"{indication_label}.position must be a finite number in 0..1"
            )
        bbox = _bbox(
            indication["bbox_mm"],
            f"{indication_label}.bbox_mm",
            nonempty=True,
        )
        if not _bbox_is_inside(bbox, pages_by_index[page]):
            raise ValueError(f"{indication_label} bbox lies outside its page")

        system_index = indication["system_index"]
        if source == "initial":
            initial_count += 1
            if (
                initial_count > 1
                or page != 1
                or system_index is not None
                or measure != 0
                or position != 0
            ):
                raise ValueError(
                    "initial tempo indication must be the unique first-page "
                    "measure-zero position-zero page element"
                )
        else:
            system_index = _integer(
                system_index,
                f"{indication_label}.system_index",
                nonnegative=True,
            )
            system = systems_by_index.get(system_index)
            if system is None:
                raise ValueError(f"{indication_label} has an unknown system")
            if (
                int(system["page"]) != page
                or measure < int(system["first_measure_index"])
                or measure >= int(system["last_measure_index"])
            ):
                raise ValueError(
                    f"{indication_label} does not belong to its declared system"
                )

        identity = (source, measure, position)
        if identity in identities:
            raise ValueError(f"{label} repeats an official owner")
        identities.add(identity)
        sort_keys.append((measure, float(position), source))
    if sort_keys != sorted(sort_keys):
        raise ValueError(f"{label} is not in canonical owner order")


def _validate_event_locators(
    value: Any,
    *,
    label: str,
    fields: frozenset[str],
    measure_count: int,
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array or null")
    locations: set[tuple[int, int, int, int]] = set()
    for position, raw_locator in enumerate(value):
        locator_label = f"{label}[{position}]"
        locator = _mapping(raw_locator, locator_label, fields)
        location = tuple(
            _integer(
                locator[field],
                f"{locator_label}.{field}",
                nonnegative=True,
            )
            for field in (
                "staff_index",
                "measure_index",
                "voice_index",
                "event_index",
            )
        )
        if location in locations:
            raise ValueError(f"{label} must contain unique owner locators")
        locations.add(location)
        if location[1] >= measure_count:
            raise ValueError(f"{locator_label} lies outside the system ranges")


def _validate_layout_views(
    value: Any,
    *,
    label: str,
    fields: frozenset[str],
    location_fields: tuple[str, ...],
    measure_count: int,
    require_text: bool,
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array or null")
    locations: set[tuple[int, ...]] = set()
    for position, raw_view in enumerate(value):
        view_label = f"{label}[{position}]"
        view = _mapping(raw_view, view_label, fields)
        location = tuple(
            _integer(
                view[field],
                f"{view_label}.{field}",
                nonnegative=True,
            )
            for field in location_fields
        )
        if location in locations:
            raise ValueError(f"{label} must contain unique owner locators")
        locations.add(location)
        if location[1] >= measure_count:
            raise ValueError(f"{view_label} lies outside the system ranges")
        visibility_code = _integer(
            view["visibility_code"],
            f"{view_label}.visibility_code",
            nonnegative=True,
        )
        if visibility_code not in {0, 1, 2}:
            raise ValueError(
                f"{view_label}.visibility_code must be an official visibility code"
            )
        if require_text and (not isinstance(view["text"], str) or not view["text"]):
            raise ValueError(f"{view_label}.text must be a nonempty string")


def _integer(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _mapping(value: Any, label: str, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    actual = set(value)
    missing = fields.difference(actual)
    unknown = actual.difference(fields)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(
            f"{label} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    return value


def _bbox_is_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    tolerance = 1e-4
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[0] + inner[2] <= outer[0] + outer[2] + tolerance
        and inner[1] + inner[3] <= outer[1] + outer[3] + tolerance
    )


def _bbox(
    value: Any,
    label: str,
    *,
    nonempty: bool = False,
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise ValueError(f"{label} must contain four numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise TypeError(f"{label} must contain only numbers")
    result = tuple(float(item) for item in value)
    if (
        not all(math.isfinite(item) for item in result)
        or result[2] < 0
        or result[3] < 0
    ):
        raise ValueError(f"{label} is invalid")
    if nonempty and (result[2] <= 0 or result[3] <= 0):
        raise ValueError(f"{label} must be nonempty")
    return result


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"render layout contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise ValueError(f"render layout contains non-finite JSON number {value}")


__all__ = [
    "GuitarProPdfExporter",
    "PdfExportJob",
    "RENDER_LAYOUT_SCHEMA",
    "load_render_layout",
    "validate_pdf_layout",
    "validate_render_layout",
]
