import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { safeRelative, validateManifest, validateNativeBinary, stageNativeCoreNotices, NATIVE_CORE_NOTICES, isLicenseNotice } from '../../scripts/prepare_native_desktop.mjs';

const manifest = components => ({ schema: 1, target: 'linux-x64', components });
const component = (name = 'onnxruntime') => ({ name, version: '1.23.2', source: 'https://github.com/microsoft/onnxruntime', files: [
  { path: 'lib/libonnxruntime.so', sha256: 'a'.repeat(64), role: 'library' },
  { path: 'LICENSE', sha256: 'b'.repeat(64), role: 'license' },
] });

test('native resource paths reject traversal and cross-platform absolute paths', () => {
  for (const p of ['../x', '/x', 'C:/x', 'a\\b', 'a/../b', 'a//b', './x', '']) assert.throws(() => safeRelative(p));
  assert.equal(safeRelative('lib/libonnxruntime.so'), 'lib/libonnxruntime.so');
});
test('native manifests require exact target, provenance, SHA-256 and licenses', () => {
  assert.doesNotThrow(() => validateManifest(manifest([component()]), 'linux', 'x64'));
  assert.throws(() => validateManifest(manifest([]), 'darwin', 'arm64'));
  for (const changed of [
    { source: 'https://example.com/runtime' },
    { files: [{ path: 'x', sha256: 'bad', role: 'library' }] },
    { files: [{ path: 'x', sha256: 'a'.repeat(64), role: 'library' }] },
    { files: [{ path: '../x', sha256: 'a'.repeat(64), role: 'license' }] },
  ]) assert.throws(() => validateManifest(manifest([{ ...component(), ...changed }]), 'linux', 'x64'));
  assert.throws(() => validateManifest(manifest([component(), component()]), 'linux', 'x64'));
  assert.throws(() => validateManifest(manifest([]), 'linux', 'x64', true));
});
test('Tauri bundle and launcher contain no Python or uv runtime path', async () => {
  const config = JSON.parse(await readFile(new URL('../src-tauri/tauri.conf.json', import.meta.url)));
  assert.deepEqual(Object.values(config.bundle.resources).sort(), ['licenses/', 'native/', 'webapp/static/']);
  const source = await readFile(new URL('../src-tauri/src/main.rs', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /desktop_runtime\.py|managed-python|UV_PYTHON|GUITAROCR_UV|root\.join\("backend"\)/);
  assert.match(source, /guitarocr-native-service/);
  assert.match(source, /GUITAROCR_READY/);
  assert.match(source, /mode != "native"/);
});


test('native binary headers must match platform and architecture', () => {
  const elf = Buffer.alloc(64);
  Buffer.from([127, 69, 76, 70, 2, 1]).copy(elf);
  elf.writeUInt16LE(62, 18);
  assert.doesNotThrow(() => validateNativeBinary(elf, 'linux', 'x64'));
  assert.throws(() => validateNativeBinary(elf, 'linux', 'arm64'));
  assert.throws(() => validateNativeBinary(elf, 'win32', 'x64'));
  assert.throws(() => validateNativeBinary(Buffer.from('script'), 'linux', 'x64'));
  const pe = Buffer.alloc(80);
  pe.write('MZ'); pe.writeUInt32LE(64, 60); pe.writeUInt32LE(0x4550, 64); pe.writeUInt16LE(0x8664, 68);
  assert.doesNotThrow(() => validateNativeBinary(pe, 'win32', 'x64'));
  pe.writeUInt32LE(99999, 60);
  assert.throws(() => validateNativeBinary(pe, 'win32', 'x64'));
  const macho = Buffer.alloc(32);
  macho.writeUInt32LE(0xfeedfacf, 0); macho.writeUInt32LE(0x100000c, 4);
  assert.doesNotThrow(() => validateNativeBinary(macho, 'darwin', 'arm64'));
  assert.throws(() => validateNativeBinary(macho, 'darwin', 'x64'));
});


test('source-informed native notices are copied exactly and missing files fail closed', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'native-notice-contract-'));
  try {
    const destination = path.join(root, 'output');
    const records = await stageNativeCoreNotices(destination);
    assert.deepEqual(records.map(record => record.name), NATIVE_CORE_NOTICES);
    for (const record of records) {
      const copied = await readFile(path.join(destination, record.name));
      const original = await readFile(new URL('../../' + record.source, import.meta.url));
      assert.deepEqual(copied, original);
      assert.equal(createHash('sha256').update(copied).digest('hex'), record.sha256);
    }
    assert.deepEqual(JSON.parse(await readFile(path.join(destination, 'components.json'))), records);
    const fixture = path.join(root, 'fixture');
    await mkdir(path.join(fixture, 'desktop/native-core/licenses'), { recursive: true });
    for (const name of NATIVE_CORE_NOTICES.slice(1)) await writeFile(path.join(fixture, 'desktop/native-core/licenses', name), 'fixture');
    const absentOutput = path.join(root, 'missing-output');
    await assert.rejects(stageNativeCoreNotices(absentOutput, fixture), /ENOENT/);
    await assert.rejects(readdir(absentOutput), /ENOENT/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('license inventory does not count Rust copying APIs as copyright notices', () => {
  for (const name of ['src/copying.rs', 'COPYRIGHT.c', 'LICENSE.json']) assert.equal(isLicenseNotice(name), false);
  for (const name of ['LICENSE', 'LICENSE-UPSTREAM.md', 'LICENSE-MIT', 'COPYING.BSD', 'NOTICE.txt']) assert.equal(isLicenseNotice(name), true);
});

test('Mac supplements retain the exact pinned upstream notice and publication metadata', async () => {
  const base = new URL('../native-licenses/', import.meta.url);
  const names = (await readdir(base)).filter(name => /^(objc2|block2|dispatch2)-/.test(name));
  assert.equal(names.length, 12);
  for (const name of names) {
    const provenance = JSON.parse(await readFile(new URL(name + '/provenance.json', base)));
    assert.match(provenance.upstreamCommit, /^[a-f0-9]{40}$/);
    assert.match(provenance.crateArchiveSha256, /^[a-f0-9]{64}$/);
    for (const entry of provenance.files) {
      const bytes = await readFile(new URL(name + '/' + entry.file, base));
      assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256);
      if (entry.gitBlobSha1) {
        assert.equal(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex'), entry.gitBlobSha1);
        assert.equal(entry.source, `https://raw.githubusercontent.com/madsmtm/objc2/${provenance.upstreamCommit}/LICENSE.md`);
      }
    }
  }
});
