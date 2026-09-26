from pathlib import Path
import tempfile
import unittest

import guitarpro

from datagen.gp_sources import analyze_source
from gp5_export.writer import write_targets_gp5, targets_to_song
from shared.constraints import validate_measure_target
from shared.m2 import parse_measure_target, format_measure_target


class M2SemanticsTest(unittest.TestCase):
    def test_mixed_note_without_pitch_reports_error_with_tuning(self):
        _, errors = validate_measure_target(
            "M2 | V0{@0:w:s1f3}", "both", tuning=[64, 59, 55, 50, 45, 40],
        )
        self.assertEqual(errors, ["both_note_fields:V0:E0:N0"])

    def test_gp5_preserves_explicit_gaps_and_rejects_overlapping_events(self):
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
        target = "M2 | V2{@0:w:r}"
        self.assertIn("unsupported_gp5_voice:V2", validate_measure_target(target, "tab")[1])
        with self.assertRaisesRegex(ValueError, "only V0 and V1"):
            targets_to_song([target])

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
