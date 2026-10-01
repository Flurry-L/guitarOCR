"""Collect verified local native build inputs; never install or bundle Python.

Supports official ORT tar/ZIP or wheels and PDFium wheels on Linux, Windows and
macOS. Only upstream metadata is fetched; artifact acquisition stays explicit.
The original Linux uv-cache input and pinned local llama build remain supported.
"""
import argparse
import base64
import csv
from hashlib import sha256
import io
import json
import re
import subprocess
import platform
import tarfile
from pathlib import Path, PurePosixPath
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
ORT_VERSION = '1.23.2'
PDFIUM_VERSION = '5.13.0'


def safe_name(name):
    if not name or name.startswith('/') or any(c in name for c in ('\\', ':')) or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError(f'Unsafe native archive path: {name}')
    return name


def upstream_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def native_library(name):
    return bool(re.search(r'(?:\.dll|\.dylib|\.so(?:\.[0-9.]+)?)$', name)) and 'pybind11' not in name


def stage_file(output, component, name, data, role):
    relative = f'{component}/{safe_name(name)}'
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {'path': relative, 'sha256': sha256(data).hexdigest(), 'role': role}


def collect_archive(archive_path, package, version, output, component):
    """Verify a local upstream archive, then copy only native code and notices.

    Acquisition stays explicit: the build operator downloads the archive first.
    PyPI wheels use the full official archive digest plus RECORD; ORT's native
    tar/ZIP uses its official GitHub release digest. No Python modules are copied.
    """
    data = archive_path.read_bytes()
    digest = sha256(data).hexdigest()
    wheel = archive_path.suffix == '.whl'
    if wheel:
        release = upstream_json(f'https://pypi.org/pypi/{package}/{version}/json')
        entry = next((e for e in release['urls'] if e['filename'] == archive_path.name), None)
        if not entry or entry['digests']['sha256'] != digest:
            raise ValueError('Local wheel does not match the official PyPI digest')
        source = entry['url']
        with ZipFile(io.BytesIO(data)) as archive:
            files = {safe_name(i.filename): archive.read(i) for i in archive.infolist() if not i.is_dir()}
            if len(files) != len([i for i in archive.infolist() if not i.is_dir()]):
                raise ValueError('Duplicate wheel archive paths')
        info = f'{package}-{version}.dist-info'
        metadata = files[f'{info}/METADATA'].decode().replace('\r\n', '\n')
        if f'Name: {package}\n' not in metadata or f'Version: {version}\n' not in metadata:
            raise ValueError('Wheel metadata identity mismatch')
        records = {r[0]: r[1:] for r in csv.reader(io.StringIO(files[f'{info}/RECORD'].decode()))}
        selected = {}
        for name, content in files.items():
            library = native_library(name) and name.startswith((package + '/', package + '.libs/', 'pypdfium2_raw/'))
            notice = (name.startswith(f'{info}/licenses/') or PurePosixPath(name).name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE', 'THIRDPARTYNOTICES', 'COPYRIGHT')))
            if not library and not notice:
                continue
            expected, size = records[name]
            actual = 'sha256=' + base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b'=').decode()
            if expected != actual or int(size) != len(content):
                raise ValueError(f'Wheel RECORD integrity failure: {name}')
            selected[name] = (content, 'library' if library else 'license' if 'LICENSE' in name.upper() or name.endswith('/pdfium.txt') else 'notice')
        proof = 'Official PyPI archive SHA-256 and selected wheel RECORD entries'
    else:
        if package != 'onnxruntime':
            raise ValueError('Only ORT accepts native release tar/ZIP archives')
        release = upstream_json(f'https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v{version}')
        entry = next((e for e in release['assets'] if e['name'] == archive_path.name), None)
        if not entry or entry.get('digest') != 'sha256:' + digest:
            raise ValueError('ORT archive lacks a matching official GitHub release digest; use a verified wheel or a reviewed local manifest')
        source = entry['browser_download_url']
        if archive_path.suffix == '.zip':
            with ZipFile(io.BytesIO(data)) as archive:
                files = {safe_name(i.filename): archive.read(i) for i in archive.infolist() if not i.is_dir()}
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
                # Copy regular targets, not platform-specific SONAME symlinks.
                files = {safe_name(i.name): archive.extractfile(i).read() for i in archive.getmembers() if i.isfile()}
        selected = {}
        for name, content in files.items():
            parts = PurePosixPath(name).parts
            name = '/'.join(parts[1:])  # Official archives contain one versioned root.
            library = name.startswith('lib/') and native_library(name)
            notice = PurePosixPath(name).name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE', 'THIRDPARTYNOTICES'))
            if library or notice:
                selected[name] = (content, 'library' if library else 'license' if 'LICENSE' in name.upper() else 'notice')
        proof = 'Official GitHub release asset SHA-256; regular native files and notices only'
    if not any(role == 'library' for _, role in selected.values()) or not any(role == 'license' for _, role in selected.values()):
        raise ValueError(f'Archive lacks native libraries or licenses: {archive_path.name}')
    files = [stage_file(output, component, name, content, role) for name, (content, role) in selected.items()]
    provenance = {'kind': 'native-files-extracted-from-wheel' if wheel else 'official-native-release',
                  'package': package, 'version': version, 'filename': archive_path.name,
                  'url': source, 'sha256': digest, 'verification': proof, 'pythonFilesIncluded': False}
    return files, provenance


def collect_wheel(cache_root, package, version, binary_paths, output, component):
    candidates = list((cache_root / 'wheels-v6/index').glob(f'*/{package}/{version}-*.http'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one cached host wheel for {package}; found {len(candidates)}')
    cache = candidates[0]
    archive = cache.with_suffix('').resolve(strict=True)
    info = archive / f'{package}-{version}.dist-info'
    metadata = (info / 'METADATA').read_text()
    if f'Name: {package}\n' not in metadata or f'Version: {version}\n' not in metadata:
        raise ValueError('Installed wheel identity mismatch')
    record = {row[0]: row[1:] for row in csv.reader(io.StringIO((info / 'RECORD').read_text()))}
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=30) as response:
        release = json.load(response)
    cache_bytes = cache.read_bytes()
    filename = f'{package}-{cache.stem}.whl'
    upstream = next(item for item in release['urls'] if item['filename'] == filename)
    if upstream['digests']['sha256'].encode() not in cache_bytes or archive.name.encode() not in cache_bytes:
        raise ValueError('Cached wheel digest/directory does not match official PyPI metadata')
    selected = list(binary_paths)
    if package == 'onnxruntime':
        selected += ['onnxruntime/LICENSE', 'onnxruntime/ThirdPartyNotices.txt']
    else:
        selected += [str(p.relative_to(archive)) for p in (info / 'licenses/data/linux_x64/BUILD_LICENSES').rglob('*') if p.is_file()]
    files = []
    for name in selected:
        source = archive / name
        data = source.read_bytes()
        digest, size = record[name]
        actual = 'sha256=' + base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b'=').decode()
        if digest != actual or int(size) != len(data):
            raise ValueError(f'Wheel RECORD integrity failure: {name}')
        role = 'library' if name in binary_paths else ('license' if 'LICENSE' in name or name.endswith('/pdfium.txt') else 'notice')
        files.append(stage_file(output, component, name, data, role))
    provenance = {'kind': 'native-files-extracted-from-wheel', 'package': package, 'version': version,
                  'filename': filename, 'url': upstream['url'], 'sha256': upstream['digests']['sha256'],
                  'verification': 'Existing uv cache digest matches official PyPI metadata; every selected file matches wheel RECORD',
                  'pythonFilesIncluded': False}
    return files, provenance


def collect(cache_root, llama_archive, llama_manifest, output, *, target="linux-x64-cpu", ort_archive=None, pdfium_wheel=None):
    output.mkdir(parents=True, exist_ok=True)
    match = re.fullmatch(r'(linux|windows|macos)-(x64|arm64)-(cpu|metal|cuda)', target)
    if not match:
        raise ValueError('Invalid runtime target')
    system, arch, backend = match.groups()
    host = {'Linux': 'linux', 'Darwin': 'macos', 'Windows': 'windows'}.get(platform.system())
    host_arch = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}.get(platform.machine().lower())
    if (system, arch) != (host, host_arch):
        raise ValueError('Collect on the intended native build host')
    if ort_archive and pdfium_wheel:
        ort_files, ort_source = collect_archive(ort_archive, 'onnxruntime', ORT_VERSION, output, 'onnxruntime')
        pdf_files, pdf_source = collect_archive(pdfium_wheel, 'pypdfium2', PDFIUM_VERSION, output, 'pdfium')
    elif cache_root and target == 'linux-x64-cpu' and not ort_archive and not pdfium_wheel:
        ort_files, ort_source = collect_wheel(cache_root, 'onnxruntime', ORT_VERSION,
            ['onnxruntime/capi/libonnxruntime.so.1.23.2', 'onnxruntime/capi/libonnxruntime_providers_shared.so'], output, 'onnxruntime')
        pdf_files, pdf_source = collect_wheel(cache_root, 'pypdfium2', PDFIUM_VERSION,
            ['pypdfium2_raw/libpdfium.so'], output, 'pdfium')
    else:
        raise ValueError('Supply both --ort-archive and --pdfium-wheel, or the existing Linux --cache route')
    entry = json.loads(llama_manifest.read_text())[target]
    data = llama_archive.read_bytes()
    if len(data) != entry['bytes'] or sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('llama local archive checksum mismatch')
    files = []
    with ZipFile(io.BytesIO(data)) as archive:
        build = json.loads(archive.read('build.json'))
        pinned = json.loads((ROOT / 'scripts/llamacpp-runtime.json').read_text())
        if build['commit'] != pinned['commit'] or build['source'] != pinned['source'] or build['target'] != target:
            raise ValueError('llama build does not match pinned upstream revision')
        for item in entry['files']:
            name = item['name']
            safe_name(name)
            data = archive.read(name)
            if sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
                raise ValueError('llama archive entry checksum mismatch')
            relative = 'llama.cpp/' + name
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            role = 'executable' if name == entry['executable'] else ('library' if native_library(name) else 'license' if name == 'LICENSE' else 'notice')
            files.append({'path': relative, 'sha256': item['sha256'], 'role': role})
    manifest = {'schema': 1, 'target': f'{dict(linux="linux", windows="win32", macos="darwin")[system]}-{arch}', 'components': [
        {'name': 'onnxruntime', 'version': '1.23.2', 'source': 'https://github.com/microsoft/onnxruntime', 'distribution': ort_source, 'files': ort_files},
        {'name': 'pdfium', 'version': '153.0.7999.0', 'source': 'https://pdfium.googlesource.com/pdfium', 'distribution': pdf_source, 'files': pdf_files},
        {'name': 'llama.cpp', 'version': build['commit'], 'source': build['source'], 'capabilities': {'cpu': True, 'metal': backend == 'metal', 'cuda': backend == 'cuda'}, 'distribution': {'kind': 'local-pinned-source-build', 'archiveSha256': entry['sha256'], 'build': build}, 'files': files},
    ]}
    if system == 'linux':
        versions = set()
        needed = set()
        for component in manifest['components']:
            for entry in component['files']:
                if entry['role'] not in ('library', 'executable'):
                    continue
                binary = output / entry['path']
                symbols = subprocess.check_output(['readelf', '--version-info', str(binary)], text=True)
                versions.update(re.findall(r'GLIBC_([0-9.]+)', symbols))
                dynamic = subprocess.check_output(['readelf', '-d', str(binary)], text=True)
                needed.update(re.findall(r'NEEDED.*?\[(.*?)\]', dynamic))
        manifest['platformRequirements'] = {'glibc': max(versions, key=lambda version: tuple(map(int, version.split('.')))), 'dynamicLibraries': sorted(needed), 'verification': 'ELF readelf version requirements; not a platform runtime acceptance test'}
    elif system == 'macos':
        minimum = re.search(r'macosx_([0-9]+)_([0-9]+)', pdf_source['filename'])
        if not minimum:
            raise ValueError('Cannot determine PDFium macOS baseline from the verified wheel')
        versions = {'.'.join(minimum.groups())}
        needed = set()
        for component in manifest['components']:
            for item in component['files']:
                if item['role'] not in ('library', 'executable'):
                    continue
                binary = output / item['path']
                commands = subprocess.check_output(['otool', '-l', str(binary)], text=True)
                versions.update(re.findall(r'\bminos ([0-9.]+)', commands))
                versions.update(re.findall(r'cmd LC_VERSION_MIN_MACOSX\s+cmdsize [0-9]+\s+version ([0-9.]+)', commands))
                dependencies = subprocess.check_output(['otool', '-L', str(binary)], text=True)
                needed.update(line.strip().split(' (')[0] for line in dependencies.splitlines()[1:] if ' (' in line)
        manifest['platformRequirements'] = {
            'minimumMacOS': max(versions, key=lambda version: tuple(map(int, version.split('.')))),
            'dynamicLibraries': sorted(needed),
            'verification': 'Verified PDFium wheel tag and every Mach-O minimum OS/dependency load command; not device acceptance'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output / 'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, help='Reuse an existing Linux uv wheel cache')
    parser.add_argument('--ort-archive', type=Path, help='Official ORT native tar/ZIP or wheel, locally downloaded')
    parser.add_argument('--pdfium-wheel', type=Path, help='Official pypdfium2 wheel, locally downloaded')
    parser.add_argument('--target', default='linux-x64-cpu', help='The matching llama build target, e.g. macos-arm64-metal')
    parser.add_argument('--llama-archive', type=Path, required=True)
    parser.add_argument('--llama-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(collect(args.cache, args.llama_archive, args.llama_manifest, args.output, target=args.target, ort_archive=args.ort_archive, pdfium_wheel=args.pdfium_wheel))
