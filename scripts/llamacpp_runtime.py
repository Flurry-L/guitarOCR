"""Pinned, verified llama.cpp distribution; never build code on end-user machines.

Release maintainers populate artifacts with package_llamacpp_runtime.py output
only after native smoke tests. An empty catalog intentionally blocks downloads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from scripts.downloads import download_verified
from shared.model_files import verify_files

ROOT = Path(__file__).resolve().parent.parent
COMMIT = '8019dc563b1ecbae6b161a70c3a1359f1b206c1e'
BUNDLED_DIRECTORY = Path('scripts/llamacpp-bundled')


def read_catalog(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['commit'] != COMMIT or data['source'] != 'https://github.com/ggml-org/llama.cpp' or data['license'] != 'MIT':
        raise ValueError('llama.cpp runtime 来源或版本不匹配')
    return data


def catalog():
    data = read_catalog(ROOT / 'scripts/llamacpp-runtime.json')
    # Only this generated resource can supply embedded archives. Neither manifest
    # may choose a local filename or arbitrary filesystem path.
    data['bundled_artifacts'] = []
    bundled = ROOT / BUNDLED_DIRECTORY / 'manifest.json'
    if bundled.is_file():
        entries = read_catalog(bundled)['artifacts']
        for key, entry in entries.items():
            validate_target(key, entry)
            validate_entry(entry)
            data['artifacts'][key] = entry
            data['bundled_artifacts'].append(key)
    return data


def target(device):
    system = {'Darwin': 'macos', 'Windows': 'windows', 'Linux': 'linux'}.get(platform.system())
    arch = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}.get(platform.machine().lower())
    if not system or not arch:
        raise ValueError('此系统暂未提供本机 runtime，请使用远程识别或仅校对与导出')
    return f'{system}-{arch}-{device}'


def validate_target(key, entry, *, native=False):
    if not re.fullmatch(r'(linux|windows|macos)-(x64|arm64)-(cpu|cuda|metal)', key):
        raise ValueError('runtime 目标平台无效')
    if native and key != target(key.rsplit('-', 1)[1]):
        raise ValueError('runtime 目标平台必须与当前构建机器相同')
    executable = 'llama-server.exe' if key.startswith('windows-') else 'llama-server'
    if entry.get('executable') != executable:
        raise ValueError('runtime 可执行文件与目标平台不匹配')


def select_device(device, *, allow_fallback=False):
    data = catalog()
    key = target(device)
    if key not in data['artifacts']:
        if allow_fallback and device != 'cpu' and target('cpu') in data['artifacts']:
            print(f'{device} runtime 尚未发布，将使用 CPU（较慢）。', flush=True)
            return 'cpu'
        raise ValueError(f'本发行版尚未提供已验证的 {key} runtime（llama.cpp {COMMIT[:12]}）。'
                         '请使用仅校对与导出或连接远程服务；维护者需先构建、测试并发布对应预编译包。'
                         '不会自动编译或下载不明来源程序。')
    return device


def probe(executable):
    try:
        result = subprocess.run([str(executable), '--version'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f'llama-server 无法运行：{error}') from error
    version = result.stdout + result.stderr
    revisions = re.findall(r'(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])', version.lower())
    if result.returncode or not any(COMMIT.startswith(revision) for revision in revisions):
        raise ValueError(f'llama-server 必须来自固定提交 {COMMIT}；检测结果：{version[-800:]}')
    return str(Path(executable).absolute())


def validate_entry(entry):
    url = urlsplit(entry['url'])
    path = unquote(url.path)
    if (url.scheme != 'https' or url.netloc != 'github.com'
            or url.query or url.fragment or '\\' in path
            or not re.fullmatch(r'/Flurry-L/guitarOCR/releases/download/[^/]+/[^/]+', path)
            or any(part in ('.', '..') for part in path.split('/'))):
        raise ValueError('runtime 只接受项目官方 Release 附件')
    for item in [entry, *entry['files']]:
        if not isinstance(item.get('bytes'), int) or item['bytes'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')):
            raise ValueError('runtime 清单缺少可信大小或 SHA-256')
    names = [item['name'] for item in entry['files']]
    if len(names) != len(set(names)) or entry['executable'] not in names or 'LICENSE' not in names:
        raise ValueError('runtime 清单缺少可执行文件或许可证')
    for name in names:
        if (not name or '\\' in name or '\0' in name or name.startswith('/') or ':' in name
                or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('runtime 清单包含不安全路径')


def extract_archive(archive, entry, staging):
    """Verify and probe the same package for both release and embedded installs."""
    validate_entry(entry)
    errors = verify_files(archive.parent, [{**entry, 'name': archive.name}])
    if errors:
        raise ValueError('；'.join(errors))
    with ZipFile(archive) as source:
        names = {item['name'] for item in entry['files']}
        if set(source.namelist()) != names or len(source.namelist()) != len(names):
            raise ValueError('runtime 压缩包内容不匹配清单')
        for item in entry['files']:
            info = source.getinfo(item['name'])
            kind = (info.external_attr >> 16) & 0o170000
            if info.file_size != item['bytes'] or info.is_dir() or kind not in (0, 0o100000):
                raise ValueError('runtime 压缩包文件大小或类型无效')
            destination = staging / item['name']
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as src, destination.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    errors = verify_files(staging, entry['files'])
    if errors:
        raise ValueError('；'.join(errors))
    (staging / entry['executable']).chmod(0o755)
    return probe(staging / entry['executable'])


def acquire(tools, device, explicit=None):
    if explicit:
        executable = shutil.which(str(explicit))
        if not executable:
            raise ValueError('找不到指定的 llama-server')
        return probe(executable)
    data = catalog()
    key = target(device)
    select_device(device)
    entry = data['artifacts'][key]
    validate_entry(entry)
    cache = Path(tools) / 'llamacpp' / COMMIT / key
    if verify_files(cache, entry['files']):
        cache.parent.mkdir(parents=True, exist_ok=True)
        if key in data.get('bundled_artifacts', []):
            archive = ROOT / BUNDLED_DIRECTORY / f'{key}.zip'
        else:
            archive = cache.parent / f'{key}.zip'
            if verify_files(archive.parent, [{**entry, 'name': archive.name}]):
                download_verified(entry['url'], archive, entry)
        with tempfile.TemporaryDirectory(prefix=key + '-', dir=cache.parent) as temp:
            staging = Path(temp)
            extract_archive(archive, entry, staging)
            if cache.exists():
                shutil.rmtree(cache)
            os.replace(staging, cache)
    return probe(cache / entry['executable'])
