from fractions import Fraction
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import guitarpro

from datagen.gp_sources import analyze_source
from gp5_export.writer import write_targets_gp5, targets_to_song
from gp5_export.timing import gp5_timing_errors
from shared.constraints import validate_measure_target
from shared.m2 import duration_ticks, parse_duration_token, parse_measure_target, format_measure_target
from shared.musicxml import duration_ticks as musicxml_duration_ticks
from shared.score_document import score_document


class M2SemanticsTest(unittest.TestCase):
    def test_python_and_native_share_the_same_ir_goldens(self):
        # Both runtimes consume these files; neither may silently redefine the
        # expected semantics by updating only its own implementation or tests.
        fixtures = Path(__file__).resolve().parents[1] / 'desktop/native-core/tests/fixtures'
        golden = json.loads((fixtures / 'score-golden.json').read_text(encoding='utf-8'))
        for case in golden['cases']:
            with self.subTest(target=case['target']):
                parsed, errors = validate_measure_target(case['target'], case['mode'],
                                                         tuning=case['tuning'], string_count=case['string_count'])
                self.assertEqual(parsed, case['parsed'])
                self.assertEqual(errors, case['errors'])
                for output in case['formats']:
                    self.assertEqual(format_measure_target(parsed, output['mode'],
                                                           preserve_playback=output['preserve_playback']), output['target'])
        for target in golden['invalid']:
            with self.subTest(invalid=target), self.assertRaises(ValueError):
                parse_measure_target(target)
        for case in golden['durations']:
            self.assertEqual(duration_ticks(parse_duration_token(case['token'])),
                             Fraction(case['numerator'], case['denominator']))
        for case in json.loads((fixtures / 'score-document-golden.json').read_text(encoding='utf-8')):
            source = deepcopy(case['result'])
            self.assertEqual(score_document(source), case['score'])
            self.assertEqual(source, case['result'])

    def test_duration_math_stays_exact_until_export_projection(self):
        for token, expected in (
            ("q", Fraction(960)),
            ("q.", Fraction(1440)),
            ("h..", Fraction(3360)),
            ("q[3:2]", Fraction(640)),
            ("q[7:4]", Fraction(3840, 7)),
            ("f..[7:4]", Fraction(60)),
        ):
            with self.subTest(duration=token):
                duration = parse_duration_token(token)
                self.assertEqual(duration_ticks(duration), expected)
                self.assertIsInstance(duration_ticks(duration), Fraction)
                self.assertEqual(musicxml_duration_ticks(duration), int(expected))
                self.assertIsInstance(musicxml_duration_ticks(duration), int)
        self.assertEqual(duration_ticks({"value": 4}), Fraction(960))

    def test_overfull_legacy_measure_remains_legal_and_exportable(self):
        target = "M2 time=4/4 | V0{@0:w:s1f0 @3840:w:s1f3}"
        self.assertEqual(validate_measure_target(target, "tab")[1], [])
        self.assertEqual(gp5_timing_errors(target), [])
        with tempfile.TemporaryDirectory() as directory:
            path = write_targets_gp5([target], Path(directory) / "overfull.gp5", mode="tab")
            song = guitarpro.parse(str(path), encoding="cp936")
            measure = song.tracks[0].measures[0]
            beats = measure.voices[0].beats
            self.assertEqual([beat.start - measure.header.start for beat in beats], [0, 3840])
            self.assertEqual([beat.notes[0].value for beat in beats], [0, 3])

    def test_mixed_note_without_pitch_reports_error_with_tuning(self):
        _, errors = validate_measure_target(
            "M2 | V0{@0:w:s1f3}", "both", tuning=[64, 59, 55, 50, 45, 40],
        )
        self.assertEqual(errors, ["both_note_fields:V0:E0:N0"])

    def test_gp5_preserves_explicit_gaps_and_rejects_overlapping_events(self):
        overlap = "M2 | V0{@0:h:s1f3 @960:q:s1f5}"
        self.assertEqual(validate_measure_target(overlap, "tab")[1], [])
        self.assertTrue(gp5_timing_errors(overlap))
        self.assertEqual(gp5_timing_errors("M2 | V0{@960:q:s1f3 @2880:q:s1f5}"), [])
        self.assertEqual(gp5_timing_errors("M2 | V0{@0:w:p60} || V1{@0:w:p64}"), [])
        with tempfile.TemporaryDirectory() as directory:
            path = write_targets_gp5(["M2 time=4/4 | V0{@960:q:s1f3 @2880:q:s1f5}"], Path(directory) / "gaps.gp5", mode="tab")
            song = guitarpro.parse(str(path), encoding="cp936")
            beats = song.tracks[0].measures[0].voices[0].beats
            self.assertEqual([b.start - song.measureHeaders[0].start for b in beats], [0, 960, 1920, 2880])
            self.assertEqual([b.status.name for b in beats], ["rest", "normal", "rest", "normal"])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            targets_to_song(["M2 | V0{@0:h:s1f3 @960:q:s1f5}"], mode="tab")
        with self.assertRaisesRegex(ValueError, "cannot be played"):
            targets_to_song(["M2 | V0{@0:w:p12}"], mode="notation")

    def test_rejects_metadata_that_would_be_silently_lost(self):
        for target in ("M20 | V0{@0:w:r}", "M2 title=x | V0{@0:w:r}", "M2 time=4/4 time=3/4 | V0{@0:w:r}"):
            with self.assertRaises(ValueError):
                parse_measure_target(target)

    def test_score_ir_voices_are_separate_from_single_track_gp5_limits(self):
        # The score IR keeps V0..V15; the single-track writer must not discard
        # voices that need the whole-score export's synchronized track groups.
        for voice_id in range(16):
            with self.subTest(voice=voice_id):
                target = f"M2 | V{voice_id}" + "{@0:w:s1f3}"
                measure, errors = validate_measure_target(target, "tab")
                self.assertEqual(errors, [])
                self.assertEqual(measure["voices"][0]["voice"], voice_id)
                self.assertEqual(format_measure_target(measure, "tab"), target)
                if voice_id > 1:
                    with self.assertRaisesRegex(ValueError, "only V0 and V1"):
                        targets_to_song([target], mode="tab")
        for voice_id in (16, 99):
            with self.subTest(invalid_voice=voice_id):
                target = f"M2 | V{voice_id}" + "{@0:w:s1f3}"
                self.assertEqual(validate_measure_target(target, "tab")[1],
                                 [f"invalid_voice:V{voice_id}"])

    def test_ornaments_survive_gp5_and_notation_hides_frets(self):
        target = "M2 time=4/4 | V0{@0:q:s1f7p71(grace:f5p69:32:hammer:0:0) @960:q:s1f7p71(trill:f9p73:32) @1920:h:s1f7p71(trem:32)}"
        parsed = parse_measure_target(target)
        for mode in ("tab", "notation", "both"):
            text = format_measure_target(parsed, mode, preserve_playback=True)
            self.assertEqual(validate_measure_target(text, mode)[1], [])
            if mode == "notation":
                self.assertIn("grace:p69", text)
                self.assertNotIn("f5", text)
            with tempfile.TemporaryDirectory() as directory:
                path = write_targets_gp5([text], Path(directory) / "ornaments.gp5", mode=mode)
                restored = analyze_source(path)["measures"][0]["targets"][mode]
                self.assertIn("grace:", restored)
                self.assertIn(":hammer:", restored)
                self.assertIn("trem:32", restored)
                song = guitarpro.parse(str(path), encoding="cp936")
                notes = [b.notes[0] for b in song.tracks[0].measures[0].voices[0].beats]
                tuning = {s.number:s.value for s in song.tracks[0].strings}
                self.assertEqual(notes[0].effect.grace.fret + tuning[notes[0].string], 69)
                self.assertEqual(notes[1].effect.trill.fret + tuning[notes[1].string], 73)
                self.assertEqual(notes[1].effect.trill.duration.value, 32)
                self.assertEqual(notes[2].effect.tremoloPicking.duration.value, 32)

    def test_visible_ornaments_omit_unprinted_playback_settings(self):
        target = "M2 | V0{@0:h:s1f7p71(grace:f5p69:64:hammer:1:0) @1920:h:s1f7p71(trill:f9p73:64)}"
        parsed = parse_measure_target(target)
        tab = format_measure_target(parsed, "tab")
        notation = format_measure_target(parsed, "notation")
        self.assertIn("grace:f5:hammer:-:0", tab)
        self.assertIn("trill:f9", tab)
        self.assertNotIn(":64", tab)
        self.assertIn("grace:p69:hammer:1:0", notation)
        self.assertIn("p71(trill)", notation)
        for mode, text in (("tab", tab), ("notation", notation)):
            self.assertEqual(validate_measure_target(text, mode)[1], [])
            song = targets_to_song([text], mode=mode)
            self.assertEqual(song.tracks[0].measures[0].voices[0].beats[0].notes[0].effect.grace.duration, 32)
        dead = "M2 | V0{@0:w:s1f7(grace:x:hammer:-:1)}"
        self.assertEqual(validate_measure_target(dead, "tab")[1], [])
        self.assertTrue(targets_to_song([dead], mode="tab").tracks[0].measures[0].voices[0].beats[0].notes[0].effect.grace.isDead)

    def test_bend_release_contours_and_legacy_flags(self):
        song = targets_to_song(["M2 | V0{@0:h:s1f5(bend:prebendRelease:100) @1920:h:s1f7(grace,trill,trem)}"], mode="tab")
        first, second = song.tracks[0].measures[0].voices[0].beats
        self.assertEqual([p.value for p in first.notes[0].effect.bend.points], [4, 4, 0])
        self.assertIsNotNone(second.notes[0].effect.grace)
        self.assertIsNotNone(second.notes[0].effect.trill)
        self.assertIsNotNone(second.notes[0].effect.tremoloPicking)
        # Enum lookup is case-insensitive; the selected contour must be too.
        alternate = targets_to_song(["M2 | V0{@0:w:s1f5(bend:PREBENDRELEASE:100)}"], mode="tab")
        bend = alternate.tracks[0].measures[0].voices[0].beats[0].notes[0].effect.bend
        self.assertEqual(bend.type.name, "prebendRelease")
        self.assertEqual([point.value for point in bend.points], [4, 4, 0])


if __name__ == "__main__":
    unittest.main()
