"""Small build-input fixtures, never execute archives or download model weights."""
import base64
import io
import tarfile
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from scripts.collect_native_assets import collect_archive, safe_name


class NativeAssetsTests(unittest.TestCase):
    def test_verified_wheel_keeps_native_layout_and_all_notices_without_python(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / 'pypdfium2-5.13.0-py3-none-win_amd64.whl'
            info = 'pypdfium2-5.13.0.dist-info'
            files = {
                'pypdfium2_raw/pdfium.dll': b'native fixture',
                'pypdfium2_raw/bindings.py': b'not client code',
                f'{info}/METADATA': b'Name: pypdfium2\r\nVersion: 5.13.0\r\n',
                f'{info}/licenses/a/LICENSE': b'first notice',
                f'{info}/licenses/b/LICENSE': b'second notice',
            }
            record = ''.join(f'{name},sha256={base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()},{len(data)}\n' for name, data in files.items())
            with ZipFile(wheel, 'w') as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
                archive.writestr(f'{info}/RECORD', record)
            upstream = {'urls': [{'filename': wheel.name, 'url': 'https://files.pythonhosted.org/verified.whl',
                                  'digests': {'sha256': sha256(wheel.read_bytes()).hexdigest()}}]}
            with patch('scripts.collect_native_assets.upstream_json', return_value=upstream):
                selected, provenance = collect_archive(wheel, 'pypdfium2', '5.13.0', root / 'out', 'pdfium')
                self.assertEqual(len(selected), 3)
                self.assertFalse(any(f['path'].endswith('.py') for f in selected))
                self.assertEqual(sum(f['role'] == 'license' for f in selected), 2)
                self.assertFalse(provenance['pythonFilesIncluded'])
                wheel.write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError, 'official PyPI digest'):
                    collect_archive(wheel, 'pypdfium2', '5.13.0', root / 'bad', 'pdfium')

    def test_official_ort_tar_copies_regular_target_not_soname_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'onnxruntime-osx-arm64-1.23.2.tgz'
            with tarfile.open(source, 'w:gz') as archive:
                for name, data in [('lib/libonnxruntime.1.23.2.dylib', b'native'), ('LICENSE', b'license'), ('ThirdPartyNotices.txt', b'notices')]:
                    item = tarfile.TarInfo('onnxruntime/' + name)
                    item.size = len(data)
                    archive.addfile(item, io.BytesIO(data))
                item = tarfile.TarInfo('onnxruntime/lib/libonnxruntime.dylib')
                item.type = tarfile.SYMTYPE
                item.linkname = 'libonnxruntime.1.23.2.dylib'
                archive.addfile(item)
            upstream = {'assets': [{'name': source.name, 'digest': 'sha256:' + sha256(source.read_bytes()).hexdigest(), 'browser_download_url': 'https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/verified.tgz'}]}
            with patch('scripts.collect_native_assets.upstream_json', return_value=upstream):
                files, _ = collect_archive(source, 'onnxruntime', '1.23.2', root / 'out', 'onnxruntime')
            self.assertEqual(sum(f['role'] == 'library' for f in files), 1)
            self.assertEqual(len(files), 3)
            upstream['assets'][0]['digest'] = None
            with patch('scripts.collect_native_assets.upstream_json', return_value=upstream):
                with self.assertRaisesRegex(ValueError, 'official GitHub release digest'):
                    collect_archive(source, 'onnxruntime', '1.23.2', root / 'bad', 'onnxruntime')

    def test_native_archive_paths_reject_cross_platform_traversal(self):
        for name in ('../x', 'a/../b', '/x', 'C:/x', 'a\\b', 'a//b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)


if __name__ == '__main__':
    unittest.main()
