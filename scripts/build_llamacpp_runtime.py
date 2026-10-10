"""Build and package a native CPU/CUDA/Metal runtime on its target platform."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
from zipfile import ZIP_DEFLATED, ZipFile


def target(device):
    system = {'Darwin': 'macos', 'Windows': 'windows', 'Linux': 'linux'}.get(platform.system())
    arch = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}.get(platform.machine().lower())
    if not system or not arch:
        raise ValueError('此系统暂未提供本机 runtime，请使用远程识别或仅校对与导出')
    return f'{system}-{arch}-{device}'


def probe(executable):
    try:
        result = subprocess.run([str(executable), '--version'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f'llama-server 无法运行：{error}') from error
    version = result.stdout + result.stderr
    if result.returncode:
        raise ValueError(f'llama-server 无法运行；退出码：{result.returncode}；检测结果：{version[-800:]}')
    return str(Path(executable).absolute())


def package(source, staging, output, key, url):
    if not re.fullmatch(r'(linux|windows|macos)-(x64|arm64)-(cpu|cuda|metal)', key):
        raise ValueError('Invalid runtime target')
    if output.resolve().is_relative_to(staging.resolve()):
        raise ValueError('Output must be outside the staging directory')
    if (staging / 'LICENSE').read_bytes() != (source / 'LICENSE').read_bytes():
        raise ValueError('Include the exact upstream MIT LICENSE')
    executable = 'llama-server.exe' if key.startswith('windows-') else 'llama-server'
    probe(staging / executable)
    files = []
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'llamacpp-{key}.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
        for path in sorted(staging.rglob('*')):
            if path.is_symlink():
                raise ValueError('Stage actual library files instead of symbolic links')
            if not path.is_file():
                continue
            name = path.relative_to(staging).as_posix()
            files.append(dict(name=name, bytes=path.stat().st_size))
            bundle.write(path, name)
    entry = dict(url=url, bytes=archive.stat().st_size, sha256=sha256(archive.read_bytes()).hexdigest(),
                 executable=executable, files=files)
    if key == 'windows-x64-cuda':
        entry['cuda_driver'] = 12080
    manifest = output / f'{key}.json'
    manifest.write_text(json.dumps({key: entry}, indent=2) + '\n', encoding='utf-8')
    return manifest


def configure(source, build, key):
    system, arch, backend = key.split('-')
    if backend not in {'cpu', 'metal', 'cuda'} or key != target(backend):
        raise ValueError('Build on the selected native OS/architecture; CPU, CUDA or Metal')
    if backend == 'cuda' and system == 'macos':
        raise ValueError('CUDA builds target Linux or Windows')
    if backend == 'metal' and (system, arch) != ('macos', 'arm64'):
        raise ValueError('Metal release builds target Apple Silicon')
    cpu_dispatch = arch == 'x64' and backend == 'cpu'
    options = {
        'CMAKE_BUILD_TYPE': 'Release',
        'BUILD_SHARED_LIBS': 'ON' if cpu_dispatch else 'OFF',
        'GGML_BACKEND_DL': 'ON' if cpu_dispatch else 'OFF',
        'GGML_CPU_ALL_VARIANTS': 'ON' if cpu_dispatch else 'OFF',
        'GGML_NATIVE': 'OFF',
        'GGML_OPENMP': 'OFF',
        'GGML_CUDA': 'ON' if backend == 'cuda' else 'OFF',
        'GGML_STATIC': 'ON' if backend == 'cuda' else 'OFF',
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
    if backend == 'cuda':
        options['CMAKE_CUDA_ARCHITECTURES'] = os.environ.get('GUITAROCR_CUDA_ARCHS', '75-virtual;80-virtual;86-real;89-real;90-virtual')
    if arch == 'x64':
        # Keep the baseline portable. ALL_VARIANTS applies explicit ISA flags
        # only to its individual CPU modules, selected by upstream at runtime.
        for feature in ('SSE42', 'AVX', 'AVX2', 'AVX512', 'AVX_VNNI', 'FMA', 'F16C', 'BMI2',
                        'AVX512_VBMI', 'AVX512_VNNI', 'AVX512_BF16', 'AMX_TILE', 'AMX_INT8', 'AMX_BF16'):
            options[f'GGML_{feature}'] = 'OFF'
    if cpu_dispatch:
        # Modules are discovered beside the executable, never in a build tree.
        options['GGML_BLAS'] = 'OFF'
        options['GGML_BACKEND_DIR'] = ''
        options['CMAKE_BUILD_WITH_INSTALL_RPATH'] = 'ON'
        options['CMAKE_INSTALL_RPATH_USE_LINK_PATH'] = 'OFF'
        if system == 'linux':
            options['CMAKE_BUILD_RPATH_USE_ORIGIN'] = 'ON'
            options['CMAKE_INSTALL_RPATH'] = '$ORIGIN'
        elif system == 'macos':
            options['CMAKE_INSTALL_RPATH'] = '@loader_path'
            options['CMAKE_INSTALL_NAME_DIR'] = '@rpath'
            options['CMAKE_BUILD_WITH_INSTALL_NAME_DIR'] = 'ON'
    if system == 'windows':
        options['CMAKE_MSVC_RUNTIME_LIBRARY'] = 'MultiThreaded'
        if backend == 'cuda':
            # Avoid a load-time nvcuda.dll dependency on CPU-only build hosts.
            options['GGML_CUDA_NO_VMM'] = 'ON'
    if system == 'macos':
        options['CMAKE_OSX_DEPLOYMENT_TARGET'] = '12.3'
        options['GGML_ACCELERATE'] = 'OFF'  # New LAPACK symbols require macOS 13.3.
    return ['cmake', '-S', str(source), '-B', str(build),
            *[f'-D{name}={value}' for name, value in options.items()]]


def stage_cpu_libraries(binary, staging, key):
    """Copy all colocated shared dependencies and CPU modules as regular files."""
    if not key.endswith('-x64-cpu'):
        return
    names = []
    for library in sorted(binary.parent.iterdir()):
        if not re.search(r'\.(?:dll|dylib|so(?:\.[0-9.]+)?)$', library.name, re.I):
            continue
        module = re.match(r'(?:lib)?ggml-(.+?)\.(?:dll|dylib|so(?:\.[0-9.]+)?)$', library.name, re.I)
        if module and module[1].lower() != 'base' and not re.match(r'cpu(?:-|$)', module[1], re.I):
            raise ValueError(f'CPU build contains a non-CPU backend; use a clean build directory: {library.name}')
        # Preserve both SONAME aliases and versioned filenames, but no symlinks
        # may enter the verified portable archive.
        resolved = library.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f'Runtime library is not a regular file: {library}')
        shutil.copy2(resolved, staging / library.name)
        names.append(library.name)
    prefix = '' if key.startswith('windows-') else 'lib'
    suffix = '.dll' if key.startswith('windows-') else '.so'
    baseline = f'{prefix}ggml-cpu-x64{suffix}'
    variants = [name for name in names if name.startswith(f'{prefix}ggml-cpu-') and name.endswith(suffix)]
    if baseline not in variants or len(variants) < 2:
        raise ValueError('CPU runtime must contain the x64 baseline and optimized backend modules')


def stage_windows_crt(staging):
    # The backend's ONNX Runtime needs these even when llama is linked statically.
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    installs = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-property', 'installationPath'], text=True).strip()
    root = Path(installs) / 'VC/Redist/MSVC'
    candidates = sorted(root.glob('*/x64/Microsoft.VC*.CRT'),
                        key=lambda p: tuple(map(int, p.parts[-3].split('.'))), reverse=True)
    names = ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')
    crt = next((p for p in candidates if all((p / n).is_file() for n in names)), None)
    if crt is None:
        raise ValueError('Visual C++ x64 redistributable DLLs are missing from Visual Studio')
    for name in names:
        shutil.copy2(crt / name, staging / name)
    urllib.request.urlretrieve('https://aka.ms/VCRedistLicense', staging / 'LICENSE-Microsoft-Visual-Cpp.html')


def build_runtime(source, build, output, key, release_tag, jobs):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?', release_tag):
        raise ValueError('A versioned project release tag is required')
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
    # Stage outside the build tree so the executable resolves packaged libraries.
    with tempfile.TemporaryDirectory(prefix='llamacpp-stage-', dir=output) as temp:
        staging = Path(temp)
        shutil.copy2(binary, staging / executable)
        stage_cpu_libraries(binary, staging, key)
        shutil.copy2(source / 'LICENSE', staging / 'LICENSE')
        if key.startswith('windows-'):
            stage_windows_crt(staging)
        if key.endswith('-cuda'):
            toolkit = Path(os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME') or '/usr/local/cuda')
            license = next((p for p in (toolkit / 'EULA.txt', toolkit / 'doc/EULA.txt') if p.is_file()), None)
            if license is None:
                raise ValueError('CUDA redistribution requires the toolkit EULA.txt')
            shutil.copy2(license, staging / 'CUDA-EULA.txt')
            if key.startswith('windows-'):
                # NVIDIA provides cuBLAS only as DLLs on Windows. CUDA runtime is
                # static; ship the exact redistributable cuBLAS pair next to it.
                libraries = [p for p in (toolkit / 'bin').rglob('cublas*.dll') if p.is_file()]
                if not any(p.name.startswith('cublasLt') for p in libraries) or not any(p.name.startswith('cublas64') for p in libraries):
                    raise ValueError('CUDA toolkit cuBLAS redistributable DLLs are missing')
                for library in libraries:
                    shutil.copy2(library, staging / library.name)
        for folder in ('licenses', 'vendor', 'ggml'):
            for path in (source / folder).rglob('*'):
                if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
                    dest = staging / 'notices' / path.relative_to(source)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
        (staging / 'build.json').write_text(json.dumps({
            'source': 'https://github.com/ggml-org/llama.cpp',
            'target': key, 'platform': platform.platform(), 'configure': command,
            'runtime_shared_libraries': sorted(path.name for path in staging.iterdir()
                if re.search(r'\.(?:dll|dylib|so(?:\.[0-9.]+)?)$', path.name, re.I)),
        }, indent=2) + '\n', encoding='utf-8')
        filename = f'llamacpp-{key}.zip'
        url = f'https://github.com/Flurry-L/guitarOCR/releases/download/{release_tag}/{filename}'
        return package(source, staging, output, key, url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--release-tag', default='v0.1.0-resources')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 2, 8))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    print(build_runtime(args.source.resolve(), args.build.resolve(), args.output.resolve(),
                        args.target, args.release_tag, args.jobs))


if __name__ == '__main__':
    main()
