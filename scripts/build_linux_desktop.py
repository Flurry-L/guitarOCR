"""Build a Debian-hosted desktop without installing system packages.

Requires an existing Rust/Cargo, Node/npm, C compiler, apt, dpkg-deb and pkg-config.
Dependencies come from Debian's signed official repository and are extracted as
data into --directory; package maintainer scripts are never executed. The result
targets this Debian release, not older Linux distributions. Use the Ubuntu CI
build for public Linux releases. Nothing is uploaded or published.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ('libwebkit2gtk-4.1-dev', 'libgtk-3-dev', 'libayatana-appindicator3-dev',
            'librsvg2-dev', 'libxdo-dev', 'libssl-dev', 'patchelf')


def run(command, **kwargs):
    print(shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def prepare(directory):
    release = platform.freedesktop_os_release()
    if release.get('ID') != 'debian' or not release.get('VERSION_CODENAME', '').isalnum():
        raise ValueError('This rootless helper supports a matching Debian host only')
    keyring = Path('/usr/share/keyrings/debian-archive-keyring.gpg')
    if not keyring.is_file():
        raise ValueError('The official Debian archive keyring must already be installed')
    for command in ('apt-get', 'dpkg-deb', 'dpkg-architecture', 'gcc', 'pkg-config'):
        if not shutil.which(command):
            raise ValueError(f'Required host tool is missing: {command}')
    directory.mkdir(parents=True, exist_ok=True)
    # A dedicated empty status file downloads a complete dependency closure even
    # when some runtime packages happen to be installed on this build machine.
    for name in ('lists/partial', 'archives/partial', 'root', 'empty'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    (directory / 'status').touch()
    codename = release['VERSION_CODENAME']
    (directory / 'sources.list').write_text(
        f'deb [signed-by={keyring}] https://deb.debian.org/debian {codename} main\n',
        encoding='utf-8')
    config = directory / 'apt.conf'
    settings = {
        'Dir::Etc::main': str(directory / 'empty.conf'),
        'Dir::Etc::parts': str(directory / 'empty'),
        'Dir::Etc::sourcelist': str(directory / 'sources.list'),
        'Dir::Etc::sourceparts': str(directory / 'empty'),
        'Dir::State': str(directory),
        'Dir::State::lists': str(directory / 'lists'),
        'Dir::State::status': str(directory / 'status'),
        'Dir::Cache': str(directory),
        'Dir::Cache::archives': str(directory / 'archives'),
        'Dir::Log': str(directory),
        'APT::Sandbox::User': pwd.getpwuid(os.getuid()).pw_name,
        'APT::Install-Recommends': 'false',
        'APT::Get::List-Cleanup': 'false',
        'Acquire::Languages': 'none',
    }
    (directory / 'empty.conf').touch()
    config.write_text(''.join(f'{key} {json.dumps(value)};\n' for key, value in settings.items()),
                      encoding='utf-8')
    run(['apt-get', '-c', config, 'update'])
    run(['apt-get', '-c', config, '--download-only', '--yes', 'install', *PACKAGES])
    inventory = []
    for package in sorted((directory / 'archives').glob('*.deb')):
        identity = subprocess.check_output(['dpkg-deb', '--show', str(package)], text=True).strip()
        inventory.append({'package': identity, 'file': package.name,
                          'sha256': sha256(package.read_bytes()).hexdigest()})
        run(['dpkg-deb', '--extract', package, directory / 'root'])
    (directory / 'packages.json').write_text(json.dumps({
        'distribution': release, 'repository': 'https://deb.debian.org/debian',
        'verification': 'APT archive signature and package hashes', 'packages': inventory,
    }, indent=2) + '\n', encoding='utf-8')
    return directory / 'root'


def build_environment(sysroot):
    triplet = subprocess.check_output(['gcc', '-dumpmachine'], text=True).strip()
    library = sysroot / 'usr/lib' / triplet
    if not (library / 'pkgconfig/webkit2gtk-4.1.pc').is_file():
        raise ValueError(f'Missing WebKit development files in {sysroot}')
    env = dict(os.environ)
    env['PKG_CONFIG_SYSROOT_DIR'] = str(sysroot)
    env['PKG_CONFIG_LIBDIR'] = os.pathsep.join(map(str, (
        library / 'pkgconfig', sysroot / 'usr/lib/pkgconfig', sysroot / 'usr/share/pkgconfig')))
    env['LIBRARY_PATH'] = os.pathsep.join(filter(None, (str(library), env.get('LIBRARY_PATH'))))
    env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (str(library), env.get('LD_LIBRARY_PATH'))))
    # linuxdeploy's GTK plugin otherwise discovers the host /usr/lib while
    # pkg-config returns sysroot paths, producing invalid AppDir destinations.
    env['LD_GTK_LIBRARY_PATH'] = str(library)
    env['PATH'] = os.pathsep.join(map(str, (
        sysroot / 'usr/bin', library / 'glib-2.0', library / 'libgtk-3-0',
        library / 'gdk-pixbuf-2.0', env['PATH'])))
    env['XDG_CACHE_HOME'] = str(ROOT / 'output/build-cache')
    env['APPIMAGE_EXTRACT_AND_RUN'] = '1'
    env.setdefault('CARGO_BUILD_JOBS', '2')
    return env


def bundle_config(env, destination):
    """Don't let a new-Debian binary claim it installs on older distributions."""
    libc, version = platform.libc_ver()
    if libc != 'glibc' or not re.fullmatch(r'\d+(?:\.\d+)+', version):
        raise ValueError('Cannot determine the build host glibc requirement')
    versions = subprocess.check_output(
        ['pkg-config', '--modversion', 'glib-2.0', 'gtk+-3.0', 'webkit2gtk-4.1'], env=env, text=True).split()
    compiler = subprocess.check_output(['g++', '-dumpfullversion'], text=True).strip()
    if len(versions) != 3 or not all(re.fullmatch(r'\d+(?:\.\d+)+', v) for v in [*versions, compiler]):
        raise ValueError('Cannot determine the native library versions')
    packages = ['libglib2.0-0', 'libgtk-3-0', 'libwebkit2gtk-4.1-0']
    depends = [f'{p} (>= {v})' for p, v in zip(packages, versions)]
    depends += [f'libc6 (>= {version})', f'libstdc++6 (>= {compiler})']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'bundle': {'linux': {'deb': {'depends': depends}}}}, indent=2) + '\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'tools/desktop-sysroot')
    parser.add_argument('--sysroot', type=Path, help='Reuse an already extracted, verified Debian sysroot')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-resources', action='store_true', help='Reuse resources prepared separately, e.g. with a bundled runtime')
    parser.add_argument('--bundles', choices=('deb', 'deb,appimage'), default='deb')
    args = parser.parse_args()
    sysroot = args.sysroot.resolve() if args.sysroot else prepare(args.directory.resolve())
    env = build_environment(sysroot)
    run(['pkg-config', '--modversion', 'glib-2.0', 'gtk+-3.0', 'webkit2gtk-4.1'], env=env)
    if args.prepare_only:
        return
    desktop = ROOT / 'desktop'
    if not (desktop / 'node_modules/.bin/tauri').exists():
        run(['npm', 'ci'], cwd=desktop, env=env)
    run(['cargo', 'fetch', '--locked', '--manifest-path', desktop / 'src-tauri/Cargo.toml'], env=env)
    if not args.skip_resources:
        run(['cargo', 'fetch', '--locked', '--manifest-path', desktop / 'native-service/Cargo.toml'], env=env)
        run(['node', ROOT / 'scripts/prepare_native_desktop.mjs', '--build'], cwd=ROOT, env=env)
    config = bundle_config(env, ROOT / 'output/desktop-debian.conf.json')
    run(['npm', 'run', 'build', '--', '--verbose', '--config', config, '--bundles', args.bundles, '--', '--locked'],
        cwd=desktop, env=env)


if __name__ == '__main__':
    main()
