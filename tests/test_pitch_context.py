import json
from pathlib import Path
import tempfile
import unittest

import guitarpro

from datagen.gp_sources import analyze_source
from datagen.build_pitch_data import visible_clef
from datagen.native_alignment import note_differences
from document_info.image_ocr import parse_info_response
from document_info.run import run as read_information
from gp5_export.writer import write_targets_gp5
from layout.postprocess import deduplicate_pitch_boxes
from shared.constraints import validate_measure_target
from shared.m2 import parse_measure_target
from shared.pitch_context import apply_pitch_regions, prompt_pitch_context
from shared.pitch_context import convert_pitch_target
from measure_ocr.recognizer import recognize_crops
from measure_ocr.prompts import recognition_prompt
from shared.artifacts import read_result, write_result
from shared.score_document import score_document


class PitchContextTest(unittest.TestCase):
    def test_written_pitch_conversion_preserves_octave_marks_and_ornaments(self):
        context = {"instrument_transpose": -2, "clef_octave": 0}
        written = "M2 key=DMajor | V0{@0:w:p60(grace:p62:32:none:0:0)<ottava:12>}"
        sounding = parse_measure_target(convert_pitch_target(written, context))
        self.assertEqual(sounding["key_signature"], "CMajor")
        event = sounding["voices"][0]["events"][0]
        self.assertEqual(event["notes"][0]["pitch"], 70)
        self.assertIn("grace:p72:32:none:0:0", event["notes"][0]["effects"])
        self.assertEqual(event["effects"], ["ottava:12"])
        restored = convert_pitch_target(convert_pitch_target(written, context), context, to_written=True)
        self.assertEqual(parse_measure_target(restored), parse_measure_target(written))
        prompt = recognition_prompt("notation", "C2 key=CMajor | V0{@0:w:p70<ottava:12>}",
                                    "pitched", context, written_pitch=True)
        self.assertIn("Previous measure context: C2 key=DMajor | V0{@0:w:p60<ottava:12>}", prompt)
        capo = convert_pitch_target("M2 | V0{@0:w:p79}", {"instrument_transpose": -12, "capo": 3})
        self.assertEqual(parse_measure_target(capo)["voices"][0]["events"][0]["notes"][0]["pitch"], 64)
        with self.assertRaisesRegex(ValueError, "MIDI range"):
            convert_pitch_target("M2 | V0{@0:w:p120<ottava:24>}", context)

    def test_written_adapter_output_is_converted_before_validation_and_resume(self):
        class Backend:
            supports_pitch_context = True
            written_pitch = True

            def generate(self, messages, _limit):
                self.prompt = messages[0]["content"][1]["text"]
                return "M2 key=DMajor | V0{@0:w:p60<ottava:12>}", 24

        row = dict(measure_number=1, image="unused.png", mode="notation",
                   pitch_context={"instrument_transpose": -2, "clef_octave": 0})
        with tempfile.TemporaryDirectory() as folder:
            backend = Backend()
            records = [dict(row)]
            targets = recognize_crops(records, "notation", Path("unused"), None, "cpu", 128, 256,
                                      [], 1, Path(folder) / "log.jsonl", False, backend=backend, instrument="pitched")
            self.assertIn("Return written MIDI pitches", backend.prompt)
            self.assertEqual(parse_measure_target(targets[0])["voices"][0]["events"][0]["notes"][0]["pitch"], 70)
            resumed = recognize_crops([dict(row)], "notation", Path("unused"), None, "cpu", 128, 256,
                                      [], 1, Path(folder) / "log.jsonl", True, backend=backend, instrument="pitched")
            self.assertEqual(resumed, targets)

    def test_tab_symbol_does_not_replace_notation_clef(self):
        class Page:
            def text(self, _box):
                return "\ue06d"

        parsed = visible_clef(Page(), {"bbox_mm": [0, 0, 4, 10]}, {"clef": "F4"})
        self.assertEqual(parsed, {"clef": "tab", "clef_octave": 0})
        self.assertEqual(parse_info_response(json.dumps(parsed), "clef"), parsed)
        records = [dict(measure_number=1, page=1, bbox=[0, 40, 120, 100])]
        marks = [dict(kind="clef", page=1, bbox=[0, 40, 12, 20],
                      parsed={"clef": "F4", "clef_octave": 0}),
                 dict(kind="clef", page=1, bbox=[0, 100, 12, 30], parsed=parsed)]
        result = apply_pitch_regions(records, marks, instrument="bass")
        self.assertEqual(result[0]["pitch_context"]["clef"], "F4")
        self.assertFalse(result[0].get("pitch_needs_review"))

    def test_capo_is_separate_from_stored_note_pitch(self):
        target = "M2 | V0{@0:w:p64}"
        document = score_document({"mode": "notation", "instrument": "guitar", "capo": 3,
                                   "records": [{"measure_number": 1, "target": target}]})
        part = document["parts"][0]
        self.assertEqual((part["pitch_reference"], part["capo"]), ("before_capo", 3))
        label = {"track": {"instrument": "guitar", "capo": 3},
                 "measures": [{"targets": {"notation": target}}]}
        staff = {"tuning_pitches_low_to_high": [64], "total_capo_frets_low_to_high": [3],
                 "measures": [{"measure_index": 0, "voices": [{"voice_index": 0, "events": [
                     {"offset": [0, 1], "grace": False, "rest": False, "placeholder": False,
                      "notes": [{"dead": False, "native_string_index": 0, "fret": 0}]}]}]}]}
        native = {"tracks": [{"staves": [staff]}]}
        self.assertEqual(note_differences(label, native), [])
        staff["total_capo_frets_low_to_high"] = [2]
        self.assertEqual(note_differences(label, native)[0]["native"], [66])
        with tempfile.TemporaryDirectory() as folder:
            path = write_targets_gp5([target], Path(folder) / "capo.gp5", mode="notation", capo=3)
            track = guitarpro.parse(str(path)).tracks[0]
            self.assertEqual(track.offset, 3)
            self.assertEqual(track.measures[0].voices[0].beats[0].notes[0].realValue, 64)

    def test_visible_transposing_instrument_reaches_gp5_without_shifting_notes(self):
        class Backend:
            supports_pitch_context = True
            supports_staff_profile = False

            def generate(self, messages, limit):
                return '{"kind":"instrument","semitones":-2,"capo":null,"text":"Trumpet in Bb"}', 24

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            layout = write_result(root / "layout", "layout", info_source="image", inputs=["score.png"],
                                  records=[dict(measure_number=1, page=1, bbox=[0, 40, 120, 40])],
                                  regions=[dict(kind="transposition", page=1, bbox=[0, 10, 120, 20], image="label.png")])
            info = read_result(read_information(layout, root / "info", adapter=root / "info-adapter", backend=Backend()), "document_info")
            self.assertEqual(info["instrument"], "pitched")
            self.assertEqual(info["midi_program"], 56)
            self.assertEqual(info["measure_pitch_contexts"][0]["pitch_context"]["instrument_transpose"], -2)
            gp5 = write_targets_gp5(["M2 | V0{@0:w:p58}"], root / "score.gp5", mode="notation",
                                    instrument="pitched", midi_program=info["midi_program"])
            track = guitarpro.parse(str(gp5)).tracks[0]
            self.assertEqual(track.name, "Trumpet")
            self.assertEqual(track.channel.instrument, 56)
            self.assertEqual(track.measures[0].voices[0].beats[0].notes[0].realValue, 58)

    def test_duplicate_octave_boxes_do_not_duplicate_context_spans(self):
        boxes = [dict(label="transposition_region", score=score, coordinate=box)
                 for score, box in ((0.9, [10, 10, 110, 30]), (0.8, [11, 10, 110, 30]),
                                    (0.7, [100, 10, 200, 30]))]
        boxes.append(dict(label="clef_region", score=0.95, coordinate=[10, 10, 110, 30]))
        kept = deduplicate_pitch_boxes(boxes)
        self.assertEqual(len(kept), 3)
        self.assertEqual([b["score"] for b in kept if b["label"] == "transposition_region"], [0.9, 0.7])

    def test_octave_engraving_does_not_shift_sounding_notes_on_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            targets = [f"M2 | V0{{@0:w:p60<ottava:{shift}>}}" for shift in (12, -12, 24, -24)]
            path = write_targets_gp5(targets, Path(folder) / "octaves.gp5", mode="notation", instrument="pitched")
            beats = [m.voices[0].beats[0] for m in guitarpro.parse(str(path)).tracks[0].measures]
            self.assertEqual([b.notes[0].realValue for b in beats], [60] * 4)
            self.assertEqual([b.octave for b in beats], [guitarpro.Octave.ottava, guitarpro.Octave.ottavaBassa,
                                                      guitarpro.Octave.quindicesima, guitarpro.Octave.quindicesimaBassa])
            labels = analyze_source(path)
            for target, bar in zip(targets, labels["measures"]):
                self.assertEqual(parse_measure_target(target)["voices"], parse_measure_target(bar["targets"]["notation"])["voices"])
            self.assertTrue(validate_measure_target("M2 | V0{@0:w:p60<ottava:12,ottava:-12>}", "notation")[1])

    def test_marks_keep_staff_and_horizontal_scope_in_edited_layouts(self):
        # Edited boxes have a shared system index even on different rows.
        records = [dict(measure_number=i + 1, page=1, system_index=0, bbox=box)
                   for i, box in enumerate(([0, 50, 100, 40], [100, 50, 100, 40],
                                            [0, 150, 100, 40], [100, 150, 100, 40]))]
        marks = [dict(kind="transposition", page=1, bbox=[0, 30, 80, 15],
                      parsed={"kind": "instrument", "semitones": -2}),
                 dict(kind="clef", page=1, bbox=[0, 50, 12, 35], parsed={"clef": "G2", "clef_octave": 0}),
                 dict(kind="transposition", page=1, bbox=[150, 92, 45, 10], image="private-path.png",
                      parsed={"kind": "ottava", "semitones": 12})]
        result = apply_pitch_regions(records, marks, instrument="pitched")
        self.assertEqual([r["pitch_context"]["instrument_transpose"] for r in result], [-2] * 4)
        self.assertEqual([len(r["pitch_context"]["octave_spans"]) for r in result], [0, 1, 0, 0])
        span = result[1]["pitch_context"]["octave_spans"][0]
        self.assertEqual((span["start"], span["end"]), (0.5, 0.95))
        self.assertNotIn("private-path", json.dumps(prompt_pitch_context(result[1]["pitch_context"])))
        self.assertNotIn("pitch_context", records[0])
        overridden = apply_pitch_regions(records, marks, instrument="pitched", transpose=0)
        self.assertEqual(overridden[1]["pitch_context"]["instrument_transpose"], 0)
        midrow = apply_pitch_regions(records, [dict(kind="transposition", page=1, bbox=[120, 30, 70, 15],
                                                   parsed={"kind": "instrument", "semitones": -7})], instrument="pitched")
        self.assertEqual([r["pitch_context"]["instrument_transpose"] for r in midrow], [0, -7, -7, -7])

    def test_unreadable_instruction_is_not_assigned_a_numeric_shift(self):
        self.assertIsNone(parse_info_response('{"kind":[]}', "transposition")["kind"])
        self.assertIsNone(parse_info_response('{"clef":{}}', "clef")["clef"])
        explicit = parse_info_response('{"kind":"instrument","semitones":12,"text":"Written to sounding: -2 semitones"}', "transposition")
        self.assertEqual(explicit["semitones"], -2)
        explicit = parse_info_response('{"kind":"instrument","semitones":null,"text":"Sounds 7 semitones lower"}', "transposition")
        self.assertEqual(explicit["semitones"], -7)
        for text, shift in (("E-flat Alto Saxophone", -9), ("Tenor Sax in Bb", -14),
                            ("Baritone Saxophone in E♭", -21), ("F Horn", -7),
                            ("B-flat Trumpet", -2), ("Cor anglais in F", -7)):
            parsed = parse_info_response(json.dumps({"kind": "instrument", "semitones": None, "text": text}), "transposition")
            self.assertEqual(parsed["semitones"], shift)
        ambiguous = parse_info_response('{"kind":"instrument","semitones":null,"text":"Alto Saxophone"}', "transposition")
        self.assertIsNone(ambiguous["semitones"])
        parsed = parse_info_response('{"kind":"instrument","semitones":true,"text":"Bb"}', "transposition")
        self.assertIsNone(parsed["semitones"])
        parsed = parse_info_response('{"kind":"ottava","semitones":7}', "transposition")
        self.assertIsNone(parsed["semitones"])
        result = apply_pitch_regions([dict(measure_number=1, page=1, bbox=[0, 30, 100, 50])],
                                     [dict(kind="transposition", page=1, bbox=[0, 15, 60, 10], parsed=parsed)])
        # Unsubstantiated model numbers are rejected as non-pitch annotations.
        self.assertIsNone(parsed["kind"])
        self.assertFalse(result[0].get("pitch_needs_review"))
        self.assertFalse(result[0]["pitch_context"].get("octave_spans"))
        # A located, identified octave span with unreadable magnitude still
        # requires review and must never invent a numeric shift.
        uncertain = apply_pitch_regions([dict(measure_number=1, page=1, bbox=[0, 30, 100, 50])],
            [dict(kind="transposition", page=1, bbox=[0, 15, 60, 10],
                  parsed={"kind": "ottava", "semitones": None})])
        self.assertTrue(uncertain[0]["pitch_needs_review"])

    def test_unreadable_static_transpose_requires_review_until_resolved(self):
        records = [dict(measure_number=i + 1, page=1, system_index=i, bbox=[0, 50 + i * 100, 100, 40])
                   for i in range(3)]
        marks = [dict(kind="transposition", page=1, bbox=[0, 30, 80, 15],
                      parsed={"kind": "instrument", "semitones": None}),
                 dict(kind="transposition", page=1, bbox=[0, 230, 80, 15],
                      parsed={"kind": "instrument", "semitones": -2})]
        result = apply_pitch_regions(records, marks, instrument="pitched")
        self.assertEqual([bool(r.get("pitch_needs_review")) for r in result], [True, True, False])
        self.assertEqual(result[2]["pitch_context"]["instrument_transpose"], -2)
        overridden = apply_pitch_regions(records, marks, instrument="pitched", transpose=-7)
        self.assertFalse(any(r.get("pitch_needs_review") for r in overridden))

    def test_octave_clef_replaces_implicit_guitar_octave_default(self):
        records = [dict(measure_number=i + 1, page=1, system_index=i, bbox=[0, 50 + i * 100, 100, 40])
                   for i in range(2)]
        marks = [dict(kind="clef", page=1, bbox=[0, 50, 12, 35], parsed={"clef": "G2", "clef_octave": -12}),
                 dict(kind="clef", page=1, bbox=[0, 150, 12, 35], parsed={"clef": "G2", "clef_octave": 0})]
        result = apply_pitch_regions(records, marks, instrument="guitar")
        self.assertEqual([r["pitch_context"]["instrument_transpose"] for r in result], [0, -12])


if __name__ == "__main__":
    unittest.main()
