"""Read document metadata and resolve the information used by OCR and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.defaults import MODEL, INFO_ADAPTER

from document_info.image_ocr import recognize_document_info
from document_info.pdf_metadata import extract_pdf_vector_metadata
from shared.artifacts import read_result, write_json, write_result
from shared.instruments import INSTRUMENTS, DEFAULT_PROGRAMS, standard_tuning, program_from_visible_name
from shared.tuning import tuning_from_name
from shared.pitch_context import apply_pitch_regions


def staff_region(source: dict, output: Path) -> dict | None:
    """Include the clef and printed instrument name beside the first measure."""
    from PIL import Image

    if not source.get("records"):
        return None
    row = source["records"][0]
    if not row.get("source_page"):
        return None
    _x, y, _w, h = row["bbox"]
    path = output / "staff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(row["source_page"]) as page:
        page.crop((0, max(0, int(y) - 12), page.width, min(page.height, int(y + h) + 13))).convert("RGB").save(path)
    return {"kind": "staff", "image": str(path.resolve())}


def run(
    layout: Path, output: Path, *, model: Path = MODEL,
    adapter: Path = INFO_ADAPTER,
    device: str = "cuda", title: str | None = None, artist: str | None = None,
    tuning: list[int] | None = None, capo: int | None = None, backend=None,
    instrument: str | None = None, midi_program: int | None = None,
    transpose: int | None = None, cancelled=None,
) -> Path:
    source = read_result(layout, "layout")
    predictions = []
    metadata = {}
    capabilities = adapter / "capabilities.json"
    trained_capabilities = json.loads(capabilities.read_text()) if capabilities.is_file() else {}
    supports_pitch = getattr(backend, "supports_pitch_context", trained_capabilities.get("pitch_context", False))
    if transpose is not None and (type(transpose) is not int or not -36 <= transpose <= 36):
        raise ValueError("Transpose must be an integer in -36..36 semitones")
    if source["info_source"] == "image":
        metadata, predictions = recognize_document_info(
            [r for r in source["regions"] if supports_pitch or r["kind"] not in {"clef", "transposition"}],
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
            profile, rows = recognize_document_info([region], model.resolve(), adapter.resolve(), device, backend=backend, cancelled=cancelled)
            metadata.update({k: v for k, v in profile.items() if k != "source"})
            predictions.extend(rows)
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
    if (instrument is None and has_tab and metadata.get('instrument') in {None, 'guitar'}
            and metadata.get('string_count') in {4, 5}):
        # Native scores sometimes retain a guitar track name on a bass staff.
        # The visible four/five-line TAB determines the supported tuning family.
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
    elif metadata.get("tuning_midi_high_to_low") and instrument == "guitar" and metadata.get("string_count") in {None, 6}:
        tuning_used = metadata["tuning_midi_high_to_low"]
    else:
        tuning_used = standard_tuning(instrument, metadata.get("string_count"))
        if metadata.get("tuning_name") and not metadata.get("warnings"):
            metadata.setdefault("warnings", []).append(
                f"无法确定这件乐器的“{metadata['tuning_name']}”调弦，请核对各弦音高。"
            )
    tuning_candidates = [tuning_used]
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
    pitch_records = apply_pitch_regions(source["records"], predictions, instrument=instrument, transpose=transpose)
    pitch_contexts = [
        {key: row[key] for key in ("measure_number", "pitch_context", "pitch_reference", "pitch_needs_review") if key in row}
        for row in pitch_records
    ] if supports_pitch or transpose is not None else []
    metadata["pitch_instructions"] = [p for p in predictions if p["kind"] in {"clef", "transposition"}]
    if any(row.get("pitch_needs_review") for row in pitch_records):
        metadata.setdefault("warnings", []).append("部分音高信息需要确认，请核对谱号、移调和变调夹设置。")
    predictions_path = write_json(output.resolve() / "predictions.json", predictions)
    return write_result(
        output, "document_info", layout=str(layout.resolve()),
        document_metadata=metadata, predictions=str(predictions_path),
        title=title or metadata.get("title") or Path(source["inputs"][0]).stem,
        artist=artist if artist is not None else metadata.get("artist") or "",
        tuning_used=tuning_used, instrument=instrument, midi_program=midi_program,
        tuning_candidates=tuning_candidates,
        capo=capo,
        transpose=transpose, measure_pitch_contexts=pitch_contexts,
    )


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
