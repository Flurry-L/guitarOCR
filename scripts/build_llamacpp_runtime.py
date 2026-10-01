"""Build and package a pinned native CPU/Metal runtime without publishing it.

Run on the target OS/architecture (the workflow supplies Linux, Windows and macOS
builders). Packaging verifies the executable revision and every archive member.
The version smoke test is not a substitute for model inference or device testing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.llamacpp_runtime import COMMIT, probe, target  # noqa: E402
from scripts.package_llamacpp_runtime import package  # noqa: E402


def configure(source, build, key):
    system, arch, backend = key.split('-')
    if backend not in {'cpu', 'metal'} or key != target(backend):
        raise ValueError('Build on the selected native OS/architecture; CPU or Metal only')
    if backend == 'metal' and (system, arch) != ('macos', 'arm64'):
        raise ValueError('Metal release builds target Apple Silicon')
    options = {
        'CMAKE_BUILD_TYPE': 'Release',
        'BUILD_SHARED_LIBS': 'OFF',
        'GGML_BACKEND_DL': 'OFF',
        'GGML_NATIVE': 'OFF',
        'GGML_OPENMP': 'OFF',
        'GGML_CUDA': 'OFF',
        'GGML_METAL': 'ON' if backend == 'metal' else 'OFF',
        'GGML_METAL_EMBED_LIBRARY': 'ON',
        'LLAMA_OPENSSL': 'OFF',
        'LLAMA_BUILD_TESTS': 'OFF',
        'LLAMA_BUILD_EXAMPLES': 'OFF',
        'LLAMA_BUILD_SERVER': 'ON',
        'LLAMA_BUILD_TOOLS': 'ON',
        'LLAMA_BUILD_UI': 'OFF',
        'LLAMA_USE_PREBUILT_UI': 'OFF',
    }
    if arch == 'x64':
        # A generic CPU package must not inherit the hosted runner's ISA. Keep
        # the baseline runnable on x86-64 machines without AVX2/FMA support.
        for feature in ('SSE42', 'AVX', 'AVX2', 'AVX512', 'AVX_VNNI', 'FMA', 'F16C', 'BMI2'):
            options[f'GGML_{feature}'] = 'OFF'
    if system == 'windows':
        options['CMAKE_MSVC_RUNTIME_LIBRARY'] = 'MultiThreaded'
    if system == 'macos':
        options['CMAKE_OSX_DEPLOYMENT_TARGET'] = '12.3'
        options['GGML_ACCELERATE'] = 'OFF'  # New LAPACK symbols require macOS 13.3.
    return ['cmake', '-S', str(source), '-B', str(build),
            *[f'-D{name}={value}' for name, value in options.items()]]


def build_runtime(source, build, output, key, release_tag, jobs):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?', release_tag):
        raise ValueError('A versioned project release tag is required')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != COMMIT:
        raise ValueError(f'Check out the official llama.cpp revision {COMMIT}')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=source, text=True).strip():
        raise ValueError('The pinned source has tracked modifications')
    command = configure(source, build, key)
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(build), '--config', 'Release',
                    '--target', 'llama-server', '--parallel', str(jobs)], check=True)
    executable = 'llama-server.exe' if key.startswith('windows-') else 'llama-server'
    candidates = [build / 'bin/Release' / executable, build / 'bin' / executable]
    binary = next((path for path in candidates if path.is_file()), None)
    if binary is None:
        raise ValueError('The build did not produce llama-server')
    output.mkdir(parents=True, exist_ok=True)
    # Stage in a fresh directory, outside the source/build trees. Static ggml
    # and llama libraries avoid omitted DLL/dylib chains in the release ZIP.
    with tempfile.TemporaryDirectory(prefix='llamacpp-stage-', dir=output) as temp:
        staging = Path(temp)
        shutil.copy2(binary, staging / executable)
        shutil.copy2(source / 'LICENSE', staging / 'LICENSE')
        for folder in ('licenses', 'vendor', 'ggml'):
            for path in (source / folder).rglob('*'):
                if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
                    dest = staging / 'notices' / path.relative_to(source)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
        probe(staging / executable)
        (staging / 'build.json').write_text(json.dumps({
            'source': 'https://github.com/ggml-org/llama.cpp', 'commit': COMMIT,
            'target': key, 'platform': platform.platform(), 'configure': command,
            'version_smoke_test': 'passed', 'model_inference_test': 'not_run',
            'device_acceptance': 'not_run',
        }, indent=2) + '\n', encoding='utf-8')
        filename = f'llamacpp-{COMMIT[:12]}-{key}.zip'
        url = f'https://github.com/Flurry-L/guitarOCR/releases/download/{release_tag}/{filename}'
        return package(source, staging, output, key, url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--release-tag', default='v0.1.0')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    print(build_runtime(args.source.resolve(), args.build.resolve(), args.output.resolve(),
                        args.target, args.release_tag, args.jobs))


if __name__ == '__main__':
    main()
