import unittest

from document_info.image_ocr import parse_info_response, tuning_from_name


class DocumentInfoTest(unittest.TestCase):
    def test_header_and_tempo_json(self):
        self.assertEqual(
            parse_info_response('{"title":" 夏霞 ","artist":null,"tuning_name":"Standard tuning"}', "header"),
            {"title": "夏霞", "artist": None, "tuning_name": "Standard tuning"},
        )
        self.assertEqual(parse_info_response('{"tempo_quarter":101}', "tempo"), {"tempo_quarter": 101})
        self.assertEqual(parse_info_response('{"tempo_quarter":"fast"}', "tempo"), {})

    def test_visible_tuning_mapping(self):
        self.assertEqual(tuning_from_name("Standard tuning"), [64, 59, 55, 50, 45, 40])
        self.assertIsNone(tuning_from_name("Custom"))


if __name__ == "__main__":
    unittest.main()
