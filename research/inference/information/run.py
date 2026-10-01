"""Read document metadata and resolve the information used by OCR and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.defaults import MODEL
from research.defaults import INFO_ADAPTER

from research.inference.information.image_ocr import recognize_document_info
from research.inference.information.image_ocr import metadata_from_predictions
from research.inference.information.pdf_metadata import extract_pdf_vector_metadata
from research.common.artifacts import read_result
from research.common.artifacts import write_json
from research.common.artifacts import write_result
from scorelib.instruments import INSTRUMENTS
from scorelib.instruments import DEFAULT_PROGRAMS
from scorelib.instruments import standard_tuning
from scorelib.instruments import program_from_visible_name
from scorelib.tuning import tuning_from_name
from scorelib.pitch_context import apply_pitch_regions
from scorelib.pitch_context import conventional_octave
from scorelib.pitch_context import explicit_transposition
from scorelib.pitch_context import _distance


def staff_region(source: dict, output: Path) -> dict | None:
    """Include the clef and printed instrument name beside the first measure."""
    from PIL import Image

    if not source.get("records"):
        return None
    row = source["records"][0]
    if not row.get("source_page"):
        return None
    row_id = row.get('row_index', row['system_index'])
    staff = [r for r in source['records'] if r['page'] == row['page']
             and r.get('row_index', r['system_index']) == row_id]
    y = min(r['bbox'][1] for r in staff)
    bottom = max(r['bbox'][1] + r['bbox'][3] for r in staff)
    path = output / "staff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(row["source_page"]) as page:
        from research.inference.information.staff_image import focus_staff

        focus_staff(page.crop((0, max(0, int(y) - 12), page.width, min(page.height, int(bottom) + 13)))).save(path)
    return {"kind": "staff", "image": str(path.resolve())}


def _run_single(
    layout: Path, output: Path, *, model: Path = MODEL,
    adapter: Path = INFO_ADAPTER,
    device: str = "cuda", title: str | None = None, artist: str | None = None,
    tuning: list[int] | None = None, capo: int | None = None, backend=None,
    instrument: str | None = None, midi_program: int | None = None,
    transpose: int | None = None, cancelled=None, source=None, profile=None, precomputed=None,
) -> Path:
    source = source if source is not None else read_result(layout, 'layout')
    predictions = []
    metadata = {}
    capabilities = adapter / "capabilities.json"
    trained_capabilities = json.loads(capabilities.read_text()) if capabilities.is_file() else {}
    supports_pitch = getattr(backend, "supports_pitch_context", trained_capabilities.get("pitch_context", False))
    if transpose is not None and (type(transpose) is not int or not -36 <= transpose <= 36):
        raise ValueError("Transpose must be an integer in -36..36 semitones")
    if source["info_source"] == "image":
        if precomputed is not None:
            predictions = precomputed
            metadata = metadata_from_predictions(predictions)
        else:
            metadata, predictions = recognize_document_info(
                [r for r in source["regions"] if supports_pitch or r["kind"] not in {"clef", "transposition", "annotation"}],
                model.resolve(), adapter.resolve(), device, backend=backend, cancelled=cancelled
            )
    else:
        pdfs = {row.get("source_pdf") for row in source["records"]}
        if len(pdfs) == 1 and None not in pdfs:
            metadata = extract_pdf_vector_metadata(Path(next(iter(pdfs))))
    supports_profile = getattr(backend, "supports_staff_profile", None)
    if supports_profile is None:
        supports_profile = trained_capabilities.get("staff_profile", False)
    if instrument is None and supports_profile:
        region = staff_region(source, output.resolve())
        if region:
            detected_profile, rows = recognize_document_info([region], model.resolve(), adapter.resolve(), device, backend=backend, cancelled=cancelled)
            metadata.update({k: v for k, v in detected_profile.items() if k != "source"})
            predictions.extend(rows)
    if profile:
        metadata.update(instrument=profile['instrument'], string_count=profile.get('strings'))
    named_programs = {
        program for p in predictions
        if p["kind"] == "transposition" and p.get("parsed", {}).get("kind") == "instrument"
        and (program := program_from_visible_name(p["parsed"].get("text"))) is not None
    }
    has_tab = any((row.get("mode") or source.get("mode")) in {"tab", "both"}
                  for row in source["records"])
    if instrument is None and not named_programs and has_tab and metadata.get("instrument") in {"pitched", "drums"}:
        # The staff classifier sees an independent crop and can contradict the
        # layout model. TAB needs a fretted instrument; use the existing guitar
        # default while retaining the raw classification in predictions.json.
        metadata["instrument"] = None
        metadata.setdefault("warnings", []).append("谱面包含 TAB，已根据弦数选择乐器，请核对调弦。")
    if (instrument is None and has_tab and metadata.get('instrument') is None
            and metadata.get('string_count') in {4, 5}):
        # Use a fretted default only when no instrument identity is visible.
        metadata['instrument'] = 'bass'
    instrument = instrument or ("pitched" if len(named_programs) == 1 else metadata.get("instrument")) or "guitar"
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unsupported instrument: {instrument}")
    if instrument in {"pitched", "drums"}:
        tuning_used = []
    elif tuning is not None:
        tuning_used = tuning
    elif detected_tuning := tuning_from_name(metadata.get("tuning_name"), instrument, metadata.get("string_count")):
        tuning_used = detected_tuning
        metadata["tuning_midi_high_to_low"] = detected_tuning
    elif profile and profile.get('tuning'):
        tuning_used = profile['tuning']
    elif metadata.get("tuning_midi_high_to_low") and instrument == "guitar" and metadata.get("string_count") in {None, 6}:
        tuning_used = metadata["tuning_midi_high_to_low"]
    else:
        tuning_used = standard_tuning(instrument, metadata.get("string_count"))
        if metadata.get("tuning_name") and not metadata.get("warnings"):
            metadata.setdefault("warnings", []).append(
                f"无法确定这件乐器的“{metadata['tuning_name']}”调弦，请核对各弦音高。"
            )
    tuning_candidates = [tuning_used]
    named_tuning = tuning_from_name(metadata.get('tuning_name'), instrument, metadata.get('string_count'))
    tuning_source = ('manual' if tuning is not None else 'printed' if named_tuning or
                     (metadata.get('tuning_midi_high_to_low') and instrument == 'guitar'
                      and metadata.get('string_count') in {None, 6}) else 'default')
    if (not has_tab and tuning is None and metadata.get('string_count') is None
            and instrument in {'guitar', 'bass'}):
        # Notation does not expose the number of strings. Choose a playable
        # standard fingering after reading all pitches, without changing notes.
        for count in range(len(tuning_used) + 1, 8):
            candidate = tuning_from_name(metadata.get('tuning_name') or 'Standard', instrument, count)
            if candidate and candidate not in tuning_candidates:
                tuning_candidates.append(candidate)
    if midi_program is None:
        midi_program = (next(iter(named_programs)) if instrument == "pitched" and len(named_programs) == 1
                        else DEFAULT_PROGRAMS[instrument])
    if type(midi_program) is not int or not 0 <= midi_program <= 127:
        raise ValueError("MIDI program must be in 0..127")
    if capo is None:
        printed_capos = {p['parsed']['capo'] for p in predictions
                         if p.get('parsed', {}).get('kind') == 'capo'
                         and p['parsed'].get('capo') is not None}
        capo = next(iter(printed_capos)) if len(printed_capos) == 1 else 0
        if len(printed_capos) > 1:
            metadata.setdefault('warnings', []).append('谱面出现不同的变调夹位置，请核对设置。')
    if type(capo) is not int or not 0 <= capo <= 24:
        raise ValueError('Capo must be an integer in 0..24')
    default_transpose = conventional_octave(profile.get('name')) if profile else None
    if transpose is None and profile:
        name = profile.get('name')
        if default_transpose is None:
            transpose = explicit_transposition(name)
    from research.inference.octave_lines import link_octave_continuations

    predictions = link_octave_continuations(source['records'], predictions)
    pitch_records = apply_pitch_regions([{**r, 'instrument': instrument} for r in source["records"]], predictions,
                                       instrument=instrument, transpose=transpose, default_transpose=default_transpose)
    if (instrument == 'guitar' and len(tuning_used) in {4, 5} and tuning_source == 'default'
            and all((r.get('mode') or source.get('mode')) == 'tab' for r in source['records'])):
        # With TAB alone, the fret numbers cannot reveal an unusual track's
        # open pitches. Keep its fingering and require a tuning confirmation.
        for row in pitch_records:
            row['pitch_needs_review'] = True
        metadata.setdefault('warnings', []).append('已识别弦品；该弦数没有唯一的吉他标准定弦，请核对各弦空弦音。')
    if (instrument == 'pitched' and transpose is None and default_transpose is None and profile
            and (program_from_visible_name(profile.get('name')) in {56, 60, 64, 65, 66, 67, 69, 71}
                 or str(profile.get('name', '')).casefold().rstrip('.') in {'sax', 'saxophone'})
            and not any(p.get('parsed', {}).get('kind') == 'instrument'
                        and p['parsed'].get('semitones') is not None for p in predictions)):
        # "tpt." or "sax." names the family but leaves the instrument key
        # unknown. Preserve written notes instead of inventing a global shift.
        for row in pitch_records:
            row['pitch_needs_review'] = True
        metadata.setdefault('warnings', []).append('已保留书面音高；乐器标签未注明移调音程，请核对实音设置。')
    pitch_contexts = [
        {key: row[key] for key in ("measure_number", "pitch_context", "pitch_reference", "pitch_needs_review") if key in row}
        for row in pitch_records
    ] if supports_pitch or transpose is not None else []
    metadata["pitch_instructions"] = [p for p in predictions if p["kind"] in {"clef", "transposition"}]
    metadata['score_annotations'] = [p for p in predictions if p['kind'] == 'annotation']
    if any(row.get("pitch_needs_review") for row in pitch_records):
        metadata.setdefault("warnings", []).append("部分音高信息需要确认，请核对谱号、移调和变调夹设置。")
    predictions_path = write_json(output.resolve() / "predictions.json", predictions)
    return write_result(
        output, 'document_info', layout=str(layout.resolve()),
        document_metadata=metadata, predictions=str(predictions_path),
        title=title or metadata.get("title") or Path(source["inputs"][0]).stem,
        artist=artist if artist is not None else metadata.get("artist") or "",
        tuning_used=tuning_used, instrument=instrument, midi_program=midi_program,
        tuning_candidates=tuning_candidates,
        tuning_source=tuning_source,
        capo=capo,
        transpose=transpose, measure_pitch_contexts=pitch_contexts,
    )


def run(layout: Path, output: Path, **kwargs) -> Path:
    """Resolve each part separately; document fields remain backwards compatible."""
    from copy import deepcopy
    from collections import Counter
    from research.inference.layout.structure import read_structure
    from research.inference.backends.glm_backend import create_backend

    adapter = kwargs.get('adapter', INFO_ADAPTER)
    capability_path = Path(adapter) / 'capabilities.json'
    capabilities = json.loads(capability_path.read_text()) if capability_path.is_file() else {}
    if not capabilities.get('score_structure'):
        return _run_single(layout, output, **kwargs)
    source = read_result(layout, 'layout')
    backend = kwargs.get('backend') or create_backend(kwargs.get('model', MODEL), adapter, kwargs.get('device', 'cuda'))
    kwargs['backend'] = backend
    parts, structure = read_structure(source, output, backend, kwargs.get('cancelled'),
                                      compact=capabilities.get('compact_structure', False))
    predictions = None
    if source['info_source'] == 'image':
        from research.inference.information.region_recovery import opening_annotations

        source['regions'].extend(opening_annotations(source, output))
        regions = [r for r in source['regions'] if capabilities.get('pitch_context') or r['kind'] not in {'clef', 'transposition', 'annotation'}]
        if capabilities.get('staff_profile'):
            for part in parts:
                records = [r for r in source['records'] if r['part_id'] == part['id']]
                region = staff_region({'records': records}, output / part['id'])
                if region:
                    regions.append({**region, 'part_id': part['id']})
        _metadata, predictions = recognize_document_info(regions, kwargs.get('model', MODEL), adapter,
                                                        kwargs.get('device', 'cuda'), backend=backend,
                                                        cancelled=kwargs.get('cancelled'))
        staff_predictions = {p['part_id']: p for p in predictions if p['kind'] == 'staff' and p.get('part_id')}
        for part in parts:
            prediction = staff_predictions.get(part['id'], {})
            staff = prediction.get('parsed', {})
            instrument = staff.get('instrument')
            printed_name = staff.get('name')
            printed_program = program_from_visible_name(printed_name)
            modes = [r['mode'] for r in source['records'] if r['part_id'] == part['id']]
            has_tab = any(mode in {'tab', 'both'} for mode in modes)
            if not has_tab and printed_program is not None:
                instrument = 'pitched'
                part.update(name=printed_name, program=printed_program)
            if (not has_tab and part['name'] not in {'Piano', 'Guitar', 'Bass', 'Drums'}
                    and program_from_visible_name(part['name']) is not None):
                # A bass clef alone does not make a piano, cello or bassoon a
                # bass guitar. Keep a specific printed identity. Generic
                # structure defaults must not override the enlarged staff label.
                instrument = 'pitched'
            visible_count = None
            if has_tab:
                from research.inference.layout.classifier import part_tab_strings

                visible_count = part_tab_strings([r for r in source['records'] if r['part_id'] == part['id']])
            if has_tab and instrument not in {'guitar', 'bass'}:
                # A contradictory profile must not give a TAB staff an empty
                # tuning. Prefer the structure prediction, then line count.
                instrument = part['instrument'] if part['instrument'] in {'guitar', 'bass'} else (
                    'bass' if (staff.get('string_count') or part.get('strings')) in {4, 5} else 'guitar')
            if instrument is None:
                continue
            if instrument != part['instrument']:
                part['program'] = (program_from_visible_name(part['name']) if instrument == 'pitched' else None)
                if part['program'] is None:
                    part['program'] = DEFAULT_PROGRAMS[instrument]
                if part['name'] in {'Guitar', 'Bass', 'Piano', 'Drums'}:
                    part['name'] = {'guitar': 'Guitar', 'bass': 'Bass', 'pitched': 'Piano', 'drums': 'Drums'}[instrument]
                part['instrument'] = instrument
            part['strings'] = (visible_count or staff.get('string_count') or part.get('strings')) if has_tab else None
            for row in source['records']:
                if row['part_id'] == part['id']:
                    row['part_name'] = part['name']
    values, profiles, contexts = [], [], []
    assignments = {}
    for region in source['regions']:
        if region['kind'] == 'header':
            continue
        x, y, w, h = region['bbox']
        candidates = [r for r in source['records'] if r['page'] == region['page']]
        if candidates:
            anchor = min(candidates, key=lambda r: (
                _distance(r['bbox'], region['bbox']),
                abs(r['bbox'][0] - x)))
            assignments[region['image']] = anchor['part_id']
    if predictions is not None:
        for prediction in predictions:
            if part_id := assignments.get(prediction['image']):
                prediction['part_id'] = part_id
    for index, part in enumerate(parts):
        local = deepcopy(source)
        local['records'] = [r for r in local['records'] if r['part_id'] == part['id']]
        local['regions'] = [r for r in local['regions'] if r['kind'] == 'header' or assignments.get(r['image']) == part['id']]
        local['mode'] = Counter(r['mode'] for r in local['records']).most_common(1)[0][0]
        options = dict(kwargs)
        if predictions is not None:
            images = {r['image'] for r in local['regions']}
            options['precomputed'] = [p for p in predictions if p['image'] in images or p.get('part_id') == part['id']]
        if index or options.get('instrument') is None:
            options['instrument'] = part['instrument']
        if index or options.get('midi_program') is None:
            options['midi_program'] = part['program'] if part['program'] or part['instrument'] != 'pitched' else None
        if len(parts) > 1 and index:
            for field in ('tuning', 'capo', 'transpose'):
                options.pop(field, None)
        manifest = _run_single(layout, output / part['id'], source=local, profile=part, **options)
        value = read_result(manifest, 'document_info')
        values.append({**value, 'id': part['id'], 'name': part['name'], 'mode': local['mode']})
        contexts.extend(value.get('measure_pitch_contexts', []))
        for row in local['records']:
            profiles.append({
                **{k: row[k] for k in ('measure_number', 'part_id', 'part_name', 'staff_id', 'bar_index', 'row_index', 'system_index')},
                'instrument': value['instrument'], 'midi_program': value['midi_program'],
                'tuning': value['tuning_used'], 'capo': value['capo'],
                'tuning_explicit': value.get('tuning_source') in {'manual', 'printed'},
                'tuning_source': value.get('tuning_source', 'default'),
                'fingering_tunings': value.get('tuning_candidates'),
            })
    if not values:
        return _run_single(layout, output, **kwargs)
    base = {k: v for k, v in values[0].items() if k not in {'schema_version', 'stage', 'id', 'name', 'mode'}}
    base.update(parts=values, measure_profiles=profiles, measure_pitch_contexts=contexts,
                resolved_records=source['records'],
                structure_predictions=str(write_json(output / 'structure.json', structure)))
    return write_result(output, 'document_info', **base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read metadata from a layout stage result.")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=INFO_ADAPTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--title")
    parser.add_argument("--artist")
    parser.add_argument("--tuning", help="Comma-separated MIDI pitches from highest to lowest string")
    parser.add_argument("--capo", type=int, help='Override a printed capo position; 0 means no capo')
    parser.add_argument("--instrument", choices=INSTRUMENTS)
    parser.add_argument("--midi-program", type=int)
    parser.add_argument("--transpose", type=int, help="Sounding minus written pitch in semitones; e.g. -2 for B-flat trumpet")
    args = parser.parse_args()
    if args.tuning is not None:
        args.tuning = [int(value) for value in args.tuning.split(",") if value.strip()]
        if not args.tuning:
            parser.error("--tuning must contain at least one MIDI pitch")
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
