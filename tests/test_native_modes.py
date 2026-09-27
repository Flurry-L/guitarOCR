import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from datagen.export_scores import (
    _load_manifest, _manifest_jsonl, _validate_document_source, _validate_document_mode,
)
from datagen.native.native_client import NativeExportClient


class NativeDisplayModeTest(unittest.TestCase):
    def test_native_reuse_and_export_check_requested_display_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory)
            path = document / "tracks/track-000/layout.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"display_mode":"both"}')
            _validate_document_mode(document, "both")
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                _validate_document_mode(document, "notation")
            path.write_text("{}")
            _validate_document_mode(document, "tab")
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                _validate_document_mode(document, "both")

    def test_native_reuse_rejects_changed_source_or_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.gp5"
            document = root / "document"
            document.mkdir()
            source.write_bytes(b"GP5")
            (document / "source.gp5").write_bytes(b"GP5")
            _validate_document_source(document, source)
            sidecar = root / "input.gp5.metadata.json"
            archived = document / "source.gp5.metadata.json"
            sidecar.write_text('{"title":"标题"}')
            with self.assertRaisesRegex(ValueError, "input changed"):
                _validate_document_source(document, source)
            archived.write_bytes(sidecar.read_bytes())
            _validate_document_source(document, source)
            sidecar.write_text('{"title":"新标题"}')
            with self.assertRaisesRegex(ValueError, "input changed"):
                _validate_document_source(document, source)
            sidecar.unlink()
            with self.assertRaisesRegex(ValueError, "input changed"):
                _validate_document_source(document, source)
            archived.unlink()
            source.write_bytes(b"changed GP5")
            with self.assertRaisesRegex(ValueError, "input changed"):
                _validate_document_source(document, source)

    def test_protocol_carries_explicit_display_mode(self):
        client = NativeExportClient.__new__(NativeExportClient)
        client._handle = 1
        client._request = Mock(return_value={"ok": True})
        for mode in ("tab", "notation", "both"):
            with self.subTest(mode=mode):
                client.export(
                    "score.gp5",
                    "score.pdf",
                    layout_output="layout.json",
                    official_score_output="score.json",
                    display_mode=mode,
                )
                request = client._request.call_args.args[0]
                self.assertEqual(request["display_mode"], mode)
                self.assertIs(request["tab_only"], mode == "tab")
        with self.assertRaisesRegex(ValueError, "display mode"):
            client.export(
                "score.gp5",
                "score.pdf",
                layout_output="layout.json",
                official_score_output="score.json",
                display_mode="unknown",
            )

    def test_manifest_defaults_to_tab_and_preserves_new_modes(self):
        rows = [
            {
                "document_id": f"{mode}-a",
                "source_family_id": "a",
                "split": "train",
                "instrument_kind": "guitar",
                "source": f"{mode}/a.gp5",
                **({"display_mode": mode} if mode != "tab" else {}),
            }
            for mode in ("tab", "notation", "both")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            entries = _load_manifest(path)
            self.assertEqual(
                [entry.display_mode for entry in entries], ["tab", "notation", "both"]
            )
            path.write_text(_manifest_jsonl(entries))
            self.assertEqual(_load_manifest(path), entries)
            rows[1]["split"] = "test"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                _load_manifest(path)


if __name__ == "__main__":
    unittest.main()
