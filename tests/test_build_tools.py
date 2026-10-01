"""Maintainer build commands must preserve platform and source trust boundaries."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_linux_desktop as desktop
from scripts import build_llamacpp_runtime as runtime


class RuntimeBuildTests(unittest.TestCase):
    def test_mismatched_target_stops_before_configuring(self):
        with patch.object(runtime, 'target', return_value='linux-x64-cpu'):
            with self.assertRaisesRegex(ValueError, 'native OS'):
                runtime.configure(Path('/source'), Path('/build'), 'windows-x64-cpu')

    def test_generic_cpu_does_not_inherit_runner_isa_or_shared_library_chain(self):
        with patch.object(runtime, 'target', return_value='linux-x64-cpu'):
            command = runtime.configure(Path('/source'), Path('/build'), 'linux-x64-cpu')
        for setting in ('GGML_NATIVE', 'GGML_SSE42', 'GGML_AVX2', 'GGML_FMA',
                        'GGML_OPENMP', 'BUILD_SHARED_LIBS', 'GGML_BACKEND_DL', 'LLAMA_OPENSSL'):
            self.assertIn(f'-D{setting}=OFF', command)

    def test_windows_does_not_require_a_separately_installed_msvc_runtime(self):
        with patch.object(runtime, 'target', return_value='windows-x64-cpu'):
            command = runtime.configure(Path('/source'), Path('/build'), 'windows-x64-cpu')
        self.assertIn('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded', command)

    def test_metal_is_embedded_and_macos_minimum_is_preserved(self):
        with patch.object(runtime, 'target', return_value='macos-arm64-metal'):
            command = runtime.configure(Path('/source'), Path('/build'), 'macos-arm64-metal')
        self.assertIn('-DGGML_METAL=ON', command)
        self.assertIn('-DGGML_METAL_EMBED_LIBRARY=ON', command)
        self.assertIn('-DCMAKE_OSX_DEPLOYMENT_TARGET=12.3', command)
        self.assertIn('-DGGML_ACCELERATE=OFF', command)

    def test_wrong_source_revision_is_rejected_before_build(self):
        with patch.object(runtime.subprocess, 'check_output', return_value='bad-revision\n'), \
                patch.object(runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'official llama.cpp revision'):
                runtime.build_runtime(Path('/source'), Path('/build'), Path('/output'),
                                      'linux-x64-cpu', 'v0.1.0', 2)
        run.assert_not_called()


class RootlessDesktopTests(unittest.TestCase):
    def test_user_directory_is_used_for_pkg_config_and_linker_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pc = root / 'usr/lib/x86_64-linux-gnu/pkgconfig/webkit2gtk-4.1.pc'
            pc.parent.mkdir(parents=True)
            pc.touch()
            with patch.object(desktop.subprocess, 'check_output', return_value='x86_64-linux-gnu\n'), \
                    patch.dict(desktop.os.environ, {'PATH': '/usr/bin', 'LIBRARY_PATH': '/existing'}, clear=True):
                env = desktop.build_environment(root)
            self.assertEqual(env['PKG_CONFIG_SYSROOT_DIR'], str(root))
            self.assertTrue(env['PKG_CONFIG_LIBDIR'].startswith(str(root)))
            self.assertTrue(env['LIBRARY_PATH'].endswith(':/existing'))
            self.assertEqual(env['LD_GTK_LIBRARY_PATH'], str(pc.parent.parent))
            self.assertEqual(env['CARGO_BUILD_JOBS'], '2')

    def test_non_debian_host_is_rejected_without_downloads(self):
        with patch.object(desktop.platform, 'freedesktop_os_release', return_value={'ID': 'ubuntu'}), \
                patch.object(desktop.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'matching Debian'):
                desktop.prepare(Path('/unused'))
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
