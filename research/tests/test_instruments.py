from pathlib import Path
import tempfile
import unittest

import guitarpro

from research.inference.information.image_ocr import parse_info_response
from research.inference.information.run import run as read_information
from research.data.build_staff_data import visible_instrument
from scorelib.gp5.writer import targets_to_song
from scorelib.gp5.writer import write_targets_gp5
from research.inference.measures.recognizer import recognize_crops
from scorelib.constraints import validate_measure_target
from scorelib.instruments import standard_tuning
from scorelib.percussion import visible_drum_key
from scorelib.tuning import tuning_from_name
from scorelib.score_document import score_document
from research.common.artifacts import read_result
from research.common.artifacts import write_result


class InstrumentTest(unittest.TestCase):
    def test_tab_layout_rejects_incompatible_automatic_staff_profile(self):
        from PIL import Image

        class Backend:
            supports_staff_profile = True
            supports_pitch_context = True

            def generate(self, _messages, _limit):
                return '{"instrument":"pitched","string_count":6}', 20

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "tab.png"
            Image.new("RGB", (100, 100), "white").save(page)
            layout = write_result(root / 'layout', 'layout', info_source="image", inputs=[str(page)],
                                  mode="tab", regions=[], records=[dict(measure_number=1, page=1,
                                  mode="tab", system_index=0, bbox=[0, 0, 100, 100], source_page=str(page))])
            info = read_result(read_information(layout, root / "info", adapter=root / "info-adapter", backend=Backend()), 'document_info')
            self.assertEqual(info["instrument"], "guitar")
            self.assertEqual(info["tuning_used"], standard_tuning("guitar", 6))
            self.assertTrue(info["document_metadata"]["warnings"])

    def test_piano_source_offset_becomes_sounding_pitches_once(self):
        from research.data.gp_sources import analyze_source
        from scorelib.m2 import parse_measure_target

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "piano.gp5"
            song = targets_to_song(["M2 | V0{@0:w:p60(grace:p58:32:none:0:0,trill:p62:32)}"],
                                   mode="notation", instrument="pitched")
            song.tracks[0].offset = 12
            guitarpro.write(song, str(path))
            label = analyze_source(path)
            self.assertEqual((label["track"]["capo"], label["track"]["source_capo"]), (0, 12))
            target = label["measures"][0]["targets"]["notation"]
            note = parse_measure_target(target)["voices"][0]["events"][0]["notes"][0]
            self.assertEqual(note["pitch"], 72)
            source_note = label["measures"][0]["voices"][0]["events"][0]["notes"][0]
            self.assertTrue(any(e.startswith("grace:") and "p70:" in e for e in source_note["effects"]))
            self.assertTrue(any(e.startswith("trill:") and "p74:" in e for e in source_note["effects"]))
            restored = targets_to_song([target], mode="notation", instrument="pitched")
            self.assertEqual(restored.tracks[0].offset, 0)
            self.assertEqual(restored.tracks[0].measures[0].voices[0].beats[0].notes[0].realValue, 72)
            self.assertEqual(guitarpro.parse(str(path)).tracks[0].offset, 12)
            with self.assertRaisesRegex(ValueError, "实际音高"):
                targets_to_song([target], mode="notation", instrument="pitched", capo=12)

    def test_bass_tuning_and_percussion_channel_survive_gp5(self):
        self.assertEqual(tuning_from_name("Dropped C", "bass", 4), [41, 36, 31, 24])
        self.assertEqual(
            tuning_from_name("Tune down 1 step", "guitar", 7),
            [62, 57, 53, 48, 43, 38, 33],
        )
        with tempfile.TemporaryDirectory() as directory:
            bass = write_targets_gp5(
                ["M2 | V0{@0:w:s5f0}"],
                Path(directory) / "bass.gp5",
                mode="tab",
                instrument="bass",
                tuning=standard_tuning("bass", 5),
            )
            track = guitarpro.parse(str(bass)).tracks[0]
            self.assertEqual(track.channel.instrument, 33)
            self.assertEqual(
                track.measures[0].voices[0].beats[0].notes[0].realValue, 23
            )
            drums = write_targets_gp5(
                ["M2 | V0{@0:w:p36,p42}"],
                Path(directory) / "drums.gp5",
                mode="notation",
                instrument="drums",
            )
            track = guitarpro.parse(str(drums)).tracks[0]
            self.assertTrue(track.isPercussionTrack)
            self.assertEqual(track.channel.channel, 9)
            self.assertEqual(
                sorted(n.realValue for n in track.measures[0].voices[0].beats[0].notes),
                [36, 42],
            )

    def test_piano_keeps_pitches_inside_native_gp5_offset_limit(self):
        # PyGuitarPro accepts larger melodic fret bytes, but GP8 silently
        # imports them as zero. Every pitched storage offset must remain <=30.
        targets = [
            "M2 | V0{@0:h:p21 @1920:h:p108}",
            "M2 | V0{@0:h:p49,p56,p59,p64,p68 @1920:h:p49(tie),p56(tie),p59(tie),p64(tie),p68(tie)}",
            "M2 | V0{@0:w:p40,p52,p59,p64,p67,p71}",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = write_targets_gp5(
                targets,
                Path(directory) / "piano.gp5",
                mode="notation",
                instrument="pitched",
            )
            song = guitarpro.parse(str(path))
            notes = [
                n
                for m in song.tracks[0].measures
                for v in m.voices
                for b in v.beats
                for n in b.notes
            ]
            self.assertTrue(all(0 <= n.value <= 30 for n in notes))
            self.assertEqual([n.realValue for n in notes[:2]], [21, 108])
            self.assertEqual(
                sorted(n.realValue for n in notes[-11:-6]), [49, 56, 59, 64, 68]
            )
            self.assertEqual(sorted(n.realValue for n in notes[-6:]), [40, 52, 59, 64, 67, 71])
            self.assertEqual(song.tracks[0].channel.instrument, 0)
            ornament = write_targets_gp5(
                ["M2 | V0{@0:w:p60(grace:p58:32:none:0:0,trill:p62:32)}"],
                Path(directory) / "ornament.gp5",
                mode="notation",
                instrument="pitched",
            )
            track = guitarpro.parse(str(ornament)).tracks[0]
            note = track.measures[0].voices[0].beats[0].notes[0]
            base = track.strings[note.string - 1].value
            self.assertEqual(note.realValue, 60)
            self.assertEqual(base + note.effect.grace.fret, 58)
            self.assertEqual(base + note.effect.trill.fret, 62)
            with self.assertRaisesRegex(ValueError, "native importer range"):
                targets_to_song(["M2 | V0{@0:w:s1f31}"], mode="tab")

    def test_information_does_not_treat_staff_lines_as_strings(self):
        self.assertEqual(visible_instrument("Bass Guitar", "both", False), "bass")
        self.assertEqual(visible_instrument("el.bs.", "tab", False), "bass")
        self.assertEqual(visible_instrument("n.guit.", "both", False), "guitar")
        self.assertEqual(visible_instrument("Track 1", "notation", False), "pitched")
        self.assertIsNone(visible_instrument("Track 1", "tab", False))
        self.assertEqual(visible_instrument("David", "notation", True), "drums")
        self.assertEqual(
            parse_info_response('{"instrument":"drums","string_count":null}', "staff"),
            {"instrument": "drums", "string_count": None, "name": None},
        )
        self.assertEqual(
            parse_info_response(
                '{"instrument":"unknown","string_count":true}', "staff"
            ),
            {"instrument": None, "string_count": None, "name": None},
        )
        self.assertEqual(
            validate_measure_target("M2 time=42/8 | V0{@0:w:r}", "notation")[1], []
        )
        self.assertEqual(visible_drum_key(40), 38)
        with self.assertRaises(ValueError):
            visible_drum_key(0)
        song = targets_to_song(
            ["M2 | V0{@0:w:s8f0}"], mode="tab", tuning=standard_tuning("guitar", 8)
        )
        self.assertEqual(sum(len(beat.notes) for track in song.tracks
                             for measure in track.measures for voice in measure.voices
                             for beat in voice.beats), 1)

    def test_recognition_context_stays_with_its_part(self):
        class Backend:
            def __init__(self):
                self.prompts = []

            def generate(self, messages, _limit):
                self.prompts.append(messages[0]["content"][1]["text"])
                return "M2 time=3/4 | V0{@0:h:p60 @1920:q:p62}", 30

        records = [
            {
                "measure_number": i + 1,
                "image": "unused.png",
                "part_id": part,
                "instrument": instrument,
                "mode": "notation",
            }
            for i, (part, instrument) in enumerate(
                [("piano", "pitched"), ("bass", "bass"), ("piano", "pitched")]
            )
        ]
        backend = Backend()
        with tempfile.TemporaryDirectory() as directory:
            recognize_crops(
                records,
                "notation",
                Path("unused"),
                None,
                "cpu",
                128,
                256,
                [],
                1,
                Path(directory) / "recognition.jsonl",
                False,
                backend=backend,
                instrument="pitched",
            )
        self.assertIn("context: START", backend.prompts[1])
        self.assertIn("Bass notation", backend.prompts[1])
        self.assertNotIn("context: START", backend.prompts[2])
        document = score_document(
            {
                "records": records,
                "mode": "notation",
                "instrument": "pitched",
                "tuning_used": [],
            }
        )
        self.assertEqual(
            [len(p["staves"][0]["measures"]) for p in document["parts"]], [2, 1]
        )
        self.assertEqual(document["parts"][1]["midi_program"], 33)


if __name__ == "__main__":
    unittest.main()
