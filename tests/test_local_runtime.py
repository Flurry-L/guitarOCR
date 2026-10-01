"""Local runtime selection, trust boundary and offline cache tests; no model downloads."""
import argparse
from hashlib import sha256
import json
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from zipfile import ZipFile, ZipInfo

from scripts import launcher, llamacpp_runtime as runtime
from scripts.distribution import environment
from shared.llamacpp_backend import LlamaCppBackend


class LocalRuntimeTests(unittest.TestCase):
    def test_auto_prefers_lightweight_engine(self):
        with patch.object(launcher, 'choose_device', return_value='cpu'):
            self.assertEqual(launcher.resolve_profile(argparse.Namespace(device='auto', engine='auto')),
                             ('llamacpp', 'cpu', 'llamacpp-cpu'))

    def test_apple_silicon_selects_metal_without_nvidia_probe(self):
        with patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='arm64'), patch('subprocess.run') as run:
            self.assertEqual(launcher.choose_device('auto'), 'metal')
            run.assert_not_called()

    def test_metal_enables_gpu_offload(self):
        env = environment(dict(engine='llamacpp', device='metal', models='/models/generation'))
        self.assertEqual(env['GUITAROCR_LLAMA_GPU_LAYERS'], '99')

    def test_explicit_native_metal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Metal'):
            launcher.resolve_profile(argparse.Namespace(device='metal', engine='transformers'))

    def test_missing_release_is_honest_and_does_not_download(self):
        with patch.object(runtime, 'catalog', return_value={'artifacts': {}}), patch.object(runtime, 'download_verified') as download:
            with self.assertRaisesRegex(ValueError, '尚未提供已验证'):
                runtime.acquire(Path('/unused'), 'cpu')
            download.assert_not_called()

    def test_cpu_fallback_only_when_auto_and_cpu_is_published(self):
        with patch.object(runtime, 'catalog', return_value={'artifacts': {'linux-x64-cpu': {}}}), patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'):
            self.assertEqual(runtime.select_device('cuda', allow_fallback=True), 'cpu')
            with self.assertRaises(ValueError):
                runtime.select_device('cuda')

    def test_probe_rejects_wrong_revision(self):
        with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'version 999 (deadbeef)', '')):
            with self.assertRaisesRegex(ValueError, '固定提交'):
                runtime.probe('/tmp/llama-server')

    def test_probe_accepts_pinned_revision(self):
        with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, f'version 999 ({runtime.COMMIT[:9]})', '')):
            self.assertEqual(runtime.probe('/tmp/llama-server'), '/tmp/llama-server')

    def test_manifest_rejects_untrusted_url_and_missing_checksum(self):
        entry = dict(url='https://evil.example/program.zip', bytes=1, sha256='a'*64, files=[], executable='llama-server')
        with self.assertRaisesRegex(ValueError, '官方'):
            runtime.validate_entry(entry)
        entry['url'] = 'https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/test.zip'
        entry.pop('sha256')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            runtime.validate_entry(entry)

    def bundle(self, folder):
        contents = {'llama-server': b'fake binary for test', 'LICENSE': b'MIT test fixture', 'libggml.so': b'library'}
        files = [dict(name=name, bytes=len(data), sha256=sha256(data).hexdigest()) for name, data in contents.items()]
        archive = folder / 'fixture.zip'
        with ZipFile(archive, 'w') as output:
            for name, data in contents.items():
                output.writestr(name, data)
        entry = dict(url='https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/test.zip',
                     bytes=archive.stat().st_size, sha256=sha256(archive.read_bytes()).hexdigest(),
                     executable='llama-server', files=files)
        return archive, entry

    def test_download_verify_extract_then_offline_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root)
            key = runtime.target('cpu')
            def download(url, destination, expected):
                destination.write_bytes(archive.read_bytes())
            with patch.object(runtime, 'catalog', return_value={'artifacts': {key: entry}}), patch.object(runtime, 'probe', side_effect=lambda path: str(path)), patch.object(runtime, 'download_verified', side_effect=download) as fetch:
                binary = runtime.acquire(root / 'tools', 'cpu')
                self.assertTrue(Path(binary).is_file())
                self.assertTrue((Path(binary).parent / 'LICENSE').is_file())
                self.assertEqual(runtime.acquire(root / 'tools', 'cpu'), binary)
                self.assertEqual(fetch.call_count, 1)
                # A damaged cached binary is repaired from the already verified ZIP offline.
                Path(binary).write_bytes(b'corrupt')
                runtime.acquire(root / 'tools', 'cpu')
                self.assertEqual(fetch.call_count, 1)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, entry = self.bundle(Path(tmp))
            entry['files'].append(dict(name='../escape', bytes=1, sha256='a'*64))
            with self.assertRaisesRegex(ValueError, '不安全'):
                runtime.validate_entry(entry)

    def embedded_catalog(self, root, archive, entry):
        key = 'linux-x64-cpu'
        bundle = root / runtime.BUNDLED_DIRECTORY
        bundle.mkdir(parents=True)
        data = dict(source='https://github.com/ggml-org/llama.cpp', commit=runtime.COMMIT, license='MIT', artifacts={})
        (root / 'scripts/llamacpp-runtime.json').write_text(json.dumps(data))
        data['artifacts'][key] = entry
        (bundle / 'manifest.json').write_text(json.dumps(data))
        (bundle / f'{key}.zip').write_bytes(archive.read_bytes())
        return bundle

    def test_embedded_runtime_installs_and_repairs_without_unpublished_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root)
            self.embedded_catalog(root, archive, entry)
            with patch.object(runtime, 'ROOT', root), patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), patch.object(runtime, 'probe', side_effect=str), patch.object(runtime, 'download_verified') as download:
                self.assertEqual(runtime.select_device('cuda', allow_fallback=True), 'cpu')
                binary = Path(runtime.acquire(root / 'tools', 'cpu'))
                self.assertEqual(binary.read_bytes(), b'fake binary for test')
                binary.write_bytes(b'corrupt')
                self.assertEqual(runtime.acquire(root / 'tools', 'cpu'), str(binary))
                self.assertEqual(binary.read_bytes(), b'fake binary for test')
                download.assert_not_called()
            self.assertEqual(json.loads((root / 'scripts/llamacpp-runtime.json').read_text())['artifacts'], {})

    def test_embedded_corrupt_archive_never_falls_back_to_url_or_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root)
            bundle = self.embedded_catalog(root, archive, entry)
            (bundle / 'linux-x64-cpu.zip').write_bytes(b'corrupt')
            with patch.object(runtime, 'ROOT', root), patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), patch.object(runtime, 'probe') as probe, patch.object(runtime, 'download_verified') as download:
                with self.assertRaises(ValueError):
                    runtime.acquire(root / 'tools', 'cpu')
                probe.assert_not_called()
                download.assert_not_called()

    def test_embedded_manifest_rejects_wrong_commit_and_traversal_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root)
            bundle = self.embedded_catalog(root, archive, entry)
            manifest = bundle / 'manifest.json'
            data = json.loads(manifest.read_text())
            data['commit'] = '0' * 40
            manifest.write_text(json.dumps(data))
            with patch.object(runtime, 'ROOT', root), self.assertRaisesRegex(ValueError, '版本'):
                runtime.catalog()
            data['commit'] = runtime.COMMIT
            data['artifacts'] = {'../linux-x64-cpu': entry}
            manifest.write_text(json.dumps(data))
            with patch.object(runtime, 'ROOT', root), self.assertRaisesRegex(ValueError, '目标平台'):
                runtime.catalog()

    def test_extract_checks_archive_file_hashes_license_and_symlinks_before_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root)
            entry['files'][0]['sha256'] = '0' * 64
            with patch.object(runtime, 'probe') as probe:
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    runtime.extract_archive(archive, entry, root / 'extract')
                probe.assert_not_called()
            entry['files'] = [item for item in entry['files'] if item['name'] != 'LICENSE']
            with self.assertRaisesRegex(ValueError, '许可证'):
                runtime.validate_entry(entry)
            archive, entry = self.bundle(root)
            with ZipFile(archive, 'w') as output:
                for item in entry['files']:
                    info = ZipInfo(item['name'])
                    info.create_system = 3
                    info.external_attr = (0o120777 << 16)
                    output.writestr(info, b'x' * item['bytes'])
            entry.update(bytes=archive.stat().st_size, sha256=sha256(archive.read_bytes()).hexdigest())
            with patch.object(runtime, 'probe') as probe:
                with self.assertRaisesRegex(ValueError, '类型'):
                    runtime.extract_archive(archive, entry, root / 'links')
                probe.assert_not_called()

    def test_release_url_cannot_escape_official_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, entry = self.bundle(Path(tmp))
            for suffix in ('../../elsewhere.zip', '%2e%2e/elsewhere.zip', 'tag/asset.zip?redirect=other', 'tag%2fother/asset.zip'):
                with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, '官方'):
                    runtime.validate_entry({**entry, 'url': 'https://github.com/Flurry-L/guitarOCR/releases/download/' + suffix})

    def test_local_defaults_and_request_timeout(self):
        with patch('shared.llamacpp_backend.load_policy', return_value={}), patch.dict('os.environ', {}, clear=True):
            backend = LlamaCppBackend('/model/merged', endpoint='http://127.0.0.1:10000')
            self.assertEqual(backend.parallel, 1)
            self.assertEqual(backend.request_timeout, 1800)
            backend.close()

    def test_spawn_uses_bounded_context_and_metal_offload(self):
        with tempfile.TemporaryDirectory() as tmp, patch('shared.paths.PROJECT_ROOT', Path(tmp)), patch('shared.llamacpp_backend.load_policy', return_value={}), patch.dict('os.environ', {}, clear=True), patch('shared.llamacpp_backend.subprocess.Popen') as spawn, patch('shared.llamacpp_backend.urlopen') as health:
            spawn.return_value.poll.return_value = None
            response = MagicMock()
            response.status = 200
            health.return_value.__enter__.return_value = response
            backend = LlamaCppBackend('/models/score_ocr/merged', device='metal')
            command = spawn.call_args.args[0]
            self.assertEqual(command[command.index('-np') + 1], '1')
            self.assertEqual(command[command.index('-c') + 1], '8192')
            self.assertEqual(command[command.index('-ngl') + 1], '99')
            self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
            backend.close()
            spawn.return_value.terminate.assert_called_once()
            self.assertTrue(backend.log.closed)
            # Avoid a second terminate from the registered atexit handler.
            spawn.return_value.poll.return_value = 0

    def test_real_inference_config_does_not_import_vllm_memory_profile(self):
        from shared.glm_backend import create_backend
        folder = runtime.ROOT / 'weights/score_ocr'
        settings = json.loads((folder / 'inference.json').read_text())
        self.assertEqual(settings['options']['max_model_len'], 16384)
        with patch.dict('os.environ', {'GUITAROCR_BACKEND': 'llamacpp', 'GUITAROCR_LLAMA_SCORE_URL': 'http://127.0.0.1:10000'}, clear=True), patch('shared.llamacpp_backend.load_policy', return_value={}):
            backend = create_backend(folder / 'merged', folder, 'cpu')
            self.assertEqual(backend.context, 8192)
            self.assertEqual(backend.parallel, 1)
            self.assertLessEqual(backend.threads, 4)

    def test_explicit_environment_overrides_only_llamacpp_options(self):
        env = {'GUITAROCR_LLAMA_CONTEXT': '4096', 'GUITAROCR_LLAMA_THREADS': '2', 'GUITAROCR_LLAMA_PARALLEL': '1'}
        with patch.dict('os.environ', env, clear=True), patch('shared.llamacpp_backend.load_policy', return_value={}):
            backend = LlamaCppBackend('/models/merged', endpoint='http://127.0.0.1:10000',
                                      options={'context': 16384, 'threads': 8, 'parallel': 4})
            self.assertEqual((backend.context, backend.threads, backend.parallel), (4096, 2, 1))

    def test_llamacpp_specific_config_reaches_backend(self):
        from shared.glm_backend import create_backend
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'inference.json').write_text(json.dumps({'options': {'max_model_len': 65536, 'parallel': 4},
                'llamacpp_options': {'context': 4096, 'threads': 2, 'parallel': 1}}))
            with patch.dict('os.environ', {'GUITAROCR_BACKEND': 'llamacpp', 'GUITAROCR_LLAMA_SCORE_URL': 'http://127.0.0.1:10000'}, clear=True), patch('shared.llamacpp_backend.load_policy', return_value={}):
                backend = create_backend(folder, None, 'cpu')
                self.assertEqual((backend.context, backend.threads, backend.parallel), (4096, 2, 1))

    def test_distribution_reuse_rejects_same_size_corruption(self):
        from scripts.distribution import reuse
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.onnx'
            source.write_bytes(b'evil')
            item = {'bytes': 4, 'sha256': sha256(b'good').hexdigest()}
            self.assertFalse(reuse(source, root / 'cache/model.onnx', item))
            self.assertFalse((root / 'cache/model.onnx').exists())
            source.write_bytes(b'good')
            self.assertTrue(reuse(source, root / 'cache/model.onnx', item))

    def test_distribution_acquire_repairs_same_size_corrupt_cache(self):
        from scripts import distribution
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / 'auxiliary/layout.onnx'
            cached.parent.mkdir()
            cached.write_bytes(b'evil')
            item = {'path': 'auxiliary/layout.onnx', 'asset': 'layout.onnx', 'bytes': 4,
                    'sha256': sha256(b'good').hexdigest()}
            data = {'repository': 'Flurry-L/guitarOCR', 'release': 'v0.1.0', 'files': {'auxiliary': [item]}}
            def fetch(url, destination, expected, **kwargs):
                self.assertEqual(expected['sha256'], sha256(b'good').hexdigest())
                destination.write_bytes(b'good')
            with patch.object(distribution, 'model_root', return_value=root), patch.object(distribution, 'catalog', return_value=data), patch.object(distribution, 'selected_manifest', return_value={'models': []}), patch('scripts.launcher.acquire_weights'), patch.object(distribution, 'reuse', return_value=False), patch.object(distribution, 'download_verified', side_effect=fetch) as download:
                distribution.acquire('transformers')
                download.assert_called_once()
                self.assertEqual(cached.read_bytes(), b'good')

    def test_managed_runtime_authenticates_health_and_completions(self):
        fixture_key = 'public-unit-test-placeholder-not-a-secret'
        with tempfile.TemporaryDirectory() as tmp, patch('shared.paths.PROJECT_ROOT', Path(tmp)), patch('shared.llamacpp_backend.load_policy', return_value={}), patch('shared.llamacpp_backend.normalize_messages', side_effect=lambda messages, policy: messages), patch.dict('os.environ', {'LLAMA_API_KEY': 'ignored-inherited-test-key', 'LLAMA_ARG_API_KEY_FILE': '/unused/key', 'LLAMA_ARG_LOG_VERBOSITY': '5'}, clear=True), patch('shared.llamacpp_backend.secrets.token_urlsafe', return_value=fixture_key), patch('shared.llamacpp_backend.subprocess.Popen') as spawn, patch('shared.llamacpp_backend.urlopen') as transport:
            spawn.return_value.poll.return_value = None
            health = MagicMock()
            health.status = 200
            transport.return_value.__enter__.return_value = health
            backend = LlamaCppBackend('/models/score_ocr/merged')
            command = spawn.call_args.args[0]
            self.assertEqual(command[command.index('--api-key') + 1], fixture_key)
            self.assertIn('--no-webui', command)
            self.assertIn('--no-cors-credentials', command)
            self.assertEqual(command[command.index('--cors-origins') + 1], backend.endpoint)
            self.assertEqual(command[command.index('--log-verbosity') + 1], '3')
            self.assertNotIn('LLAMA_API_KEY', spawn.call_args.kwargs['env'])
            self.assertNotIn('LLAMA_ARG_API_KEY_FILE', spawn.call_args.kwargs['env'])
            self.assertNotIn('LLAMA_ARG_LOG_VERBOSITY', spawn.call_args.kwargs['env'])
            health_request = transport.call_args.args[0]
            self.assertEqual(health_request.get_header('Authorization'), 'Bearer ' + fixture_key)
            transport.return_value.__enter__.return_value = io.BytesIO(json.dumps({
                'choices': [{'message': {'content': 'ok'}}], 'usage': {'completion_tokens': 1}}).encode())
            self.assertEqual(backend.generate([{'role': 'user', 'content': 'test'}], 128), ('ok', 1))
            completion_request = transport.call_args.args[0]
            self.assertEqual(completion_request.get_header('Authorization'), 'Bearer ' + fixture_key)
            self.assertEqual(json.loads(completion_request.data)['max_tokens'], 128)
            backend.close()
            self.assertIsNone(backend._api_key)
            self.assertTrue(backend.log.closed)
            self.assertNotIn(fixture_key, backend.log_path.read_text())
            spawn.return_value.terminate.assert_called_once()
            spawn.return_value.poll.return_value = 0

    def test_explicit_endpoint_does_not_receive_managed_credentials(self):
        with patch('shared.llamacpp_backend.load_policy', return_value={}), patch('shared.llamacpp_backend.normalize_messages', side_effect=lambda messages, policy: messages), patch('shared.llamacpp_backend.secrets.token_urlsafe') as secret, patch('shared.llamacpp_backend.subprocess.Popen') as spawn, patch('shared.llamacpp_backend.urlopen') as transport:
            backend = LlamaCppBackend('/models/merged', endpoint='http://127.0.0.1:10000')
            transport.return_value.__enter__.return_value = io.BytesIO(json.dumps({
                'choices': [{'message': {'content': 'ok'}}], 'usage': {'completion_tokens': 1}}).encode())
            backend.generate([{'role': 'user', 'content': 'test'}], 128)
            self.assertIsNone(transport.call_args.args[0].get_header('Authorization'))
            secret.assert_not_called()
            spawn.assert_not_called()
            backend.close()

    def test_parallel_validation(self):
        with self.assertRaisesRegex(ValueError, 'parallel'):
            LlamaCppBackend('/model/merged', endpoint='http://localhost', options={'parallel': 0})

    def test_binary_fail_fast_precedes_model_download(self):
        with patch.object(launcher, 'resolve_profile', return_value=('llamacpp', 'cpu', 'llamacpp-cpu')), patch.object(runtime, 'select_device', side_effect=ValueError('unpublished')), patch('scripts.distribution.acquire') as models:
            with self.assertRaisesRegex(ValueError, 'unpublished'):
                launcher.install(argparse.Namespace(device='auto', llama_server=None), '/uv', Path('/unused'))
            models.assert_not_called()

    def test_catalog_has_no_invented_artifacts(self):
        self.assertEqual(runtime.catalog()['commit'], runtime.COMMIT)
        self.assertEqual(json.loads((runtime.ROOT / 'scripts/llamacpp-runtime.json').read_text())['license'], 'MIT')


if __name__ == '__main__':
    unittest.main()
