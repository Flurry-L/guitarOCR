"""Pinned llama.cpp build and package validation (build machines only)."""
import re
import platform
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile
from research.common.model_files import verify_files

COMMIT = '8019dc563b1ecbae6b161a70c3a1359f1b206c1e'

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


def probe(executable):
    try:
        result = subprocess.run([str(executable), '--version'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f'llama-server 无法运行：{error}') from error
    version = result.stdout + result.stderr
    revisions = re.findall(r'(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])', version.lower())
    if result.returncode or not any(COMMIT.startswith(revision) for revision in revisions):
        raise ValueError(f'llama-server 必须来自固定提交 {COMMIT}；退出码：{result.returncode}；检测结果：{version[-800:]}')
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
