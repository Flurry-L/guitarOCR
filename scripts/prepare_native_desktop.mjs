/** Stage native-only Tauri resources. No downloads, Python runtime, or model execution. */
import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, readdir, lstat, chmod, rm, rename, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SOURCES = {
  onnxruntime: 'https://github.com/microsoft/onnxruntime',
  pdfium: 'https://pdfium.googlesource.com/pdfium',
  'llama.cpp': 'https://github.com/ggml-org/llama.cpp',
};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export function safeRelative(name) {
  if (typeof name !== 'string' || !name || name.includes('\\') || name.includes(':') || name.startsWith('/') || name.split('/').some(p => !p || p === '.' || p === '..')) {
    throw new Error(`Unsafe resource path: ${name}`);
  }
  return name;
}
export function validateManifest(manifest, platform = process.platform, arch = process.arch, required = false) {
  if (manifest.schema !== 1 || manifest.target !== `${platform}-${arch}` || !Array.isArray(manifest.components)) throw new Error('Native manifest schema/target mismatch');
  const names = new Set();
  for (const component of manifest.components) {
    if (names.has(component.name) || SOURCES[component.name] !== component.source || !component.version) throw new Error('Unknown/duplicate native component or untrusted source');
    names.add(component.name);
    if (component.capabilities !== undefined) {
      if (component.name !== 'llama.cpp' || !component.capabilities || typeof component.capabilities !== 'object' || Array.isArray(component.capabilities) || Object.entries(component.capabilities).some(([name, value]) => !['cpu', 'cuda', 'metal'].includes(name) || typeof value !== 'boolean')) throw new Error('Invalid native backend capabilities');
    }
    if (!Array.isArray(component.files) || !component.files.length) throw new Error('Empty component inventory');
    const paths = new Set();
    for (const file of component.files) {
      safeRelative(file.path);
      if (paths.has(file.path) || !/^[a-f0-9]{64}$/.test(file.sha256) || !['library', 'executable', 'license', 'notice'].includes(file.role)) throw new Error('Invalid native file inventory');
      paths.add(file.path);
    }
    if (!component.files.some(f => f.role === 'license')) throw new Error(`Missing license: ${component.name}`);
    if (!component.files.some(f => f.role === (component.name === 'llama.cpp' ? 'executable' : 'library'))) throw new Error(`Missing runtime: ${component.name}`);
  }
  if (required && Object.keys(SOURCES).some(name => !names.has(name))) throw new Error('Inference packaging requires ONNX Runtime, PDFium and llama.cpp with licenses');
  return manifest;
}
export function validateNativeBinary(bytes, platform = process.platform, arch = process.arch) {
  let valid = false;
  if (platform === 'linux' && bytes.length >= 20 && bytes.subarray(0, 4).equals(Buffer.from([127, 69, 76, 70])) && bytes[4] === 2 && bytes[5] === 1) {
    valid = bytes.readUInt16LE(18) === ({ x64: 62, arm64: 183 })[arch];
  } else if (platform === 'win32' && bytes.length >= 64 && bytes.toString('ascii', 0, 2) === 'MZ') {
    const offset = bytes.readUInt32LE(60);
    valid = offset + 6 <= bytes.length && bytes.readUInt32LE(offset) === 0x4550 && bytes.readUInt16LE(offset + 4) === ({ x64: 0x8664, arm64: 0xaa64 })[arch];
  } else if (platform === 'darwin' && bytes.length >= 8 && bytes.readUInt32LE(0) === 0xfeedfacf) {
    valid = bytes.readUInt32LE(4) === ({ x64: 0x1000007, arm64: 0x100000c })[arch];
  }
  if (!valid) throw new Error(`Native binary does not match ${platform}-${arch}; use a target-specific build`);
}
async function walk(dir) {
  const result = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const name = path.join(dir, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Symlink is not allowed in resources: ${name}`);
    if (entry.isDirectory()) result.push(...await walk(name));
    else if (entry.isFile()) result.push(name);
  }
  return result;
}
async function verifiedLocal(root, relative) {
  safeRelative(relative);
  let current = root;
  for (const segment of relative.split('/')) {
    current = path.join(current, segment);
    if ((await lstat(current)).isSymbolicLink()) throw new Error(`Symlink is not allowed: ${relative}`);
  }
  if (!(await lstat(current)).isFile()) throw new Error(`Not a regular file: ${relative}`);
  return current;
}
function cargo(args) {
  const result = spawnSync('cargo', args, { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(`cargo ${args[0]} failed: ${result.error || result.stderr}`);
  return result.stdout;
}
export function isLicenseNotice(filename) {
  // Source APIs named copying.rs/copyright.rs are not license notices.
  return /^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)/i.test(path.basename(filename))
    && !/\.(rs|c|h|cpp|hpp|py|js|mjs|ts|tsx|jsx|toml|json)$/i.test(filename);
}
export async function stageCargoLicenses(destination, overrides, target) {
  await mkdir(destination, { recursive: true });
  if (!target) {
    const host = spawnSync('rustc', ['-vV'], { encoding: 'utf8' });
    target = host.stdout?.match(/^host: (.+)$/m)?.[1];
    if (host.status !== 0 || !target) throw new Error('Cannot determine Rust host target for license inventory');
  }
  const inventory = new Map();
  for (const manifest of ['desktop/src-tauri/Cargo.toml', 'backend/Cargo.toml']) {
    const metadata = JSON.parse(cargo(['metadata', '--locked', '--offline', '--format-version', '1', '--filter-platform', target, '--manifest-path', manifest]));
    const rootId = metadata.packages.find(pkg => path.resolve(pkg.manifest_path) === path.resolve(ROOT, manifest)).id;
    const nodes = new Map(metadata.resolve.nodes.map(node => [node.id, node]));
    const reachable = new Set();
    const visit = id => {
      if (reachable.has(id)) return;
      reachable.add(id);
      for (const dep of nodes.get(id)?.deps || []) visit(dep.pkg);
    };
    visit(rootId);
    for (const pkg of metadata.packages.filter(pkg => reachable.has(pkg.id))) {
      if (!pkg.source) continue;
      if (!pkg.source.startsWith('registry+')) throw new Error(`Unreviewed Cargo source: ${pkg.source}`);
      const key = `${pkg.name}-${pkg.version}`;
      if (inventory.has(key)) continue;
      const dir = path.dirname(pkg.manifest_path);
      const selected = (await walk(dir)).filter(p => isLicenseNotice(p) || (pkg.license_file && p === path.resolve(dir, pkg.license_file)));
      if (!selected.length && overrides) {
        const supplement = path.resolve(overrides, key);
        try {
          const supplementFiles = await walk(supplement);
          if (!supplementFiles.some(file => isLicenseNotice(file))) throw new Error(`No license text in supplement: ${key}`);
          for (const file of supplementFiles) {
            const target = path.join(destination, key, path.relative(supplement, file));
            await mkdir(path.dirname(target), { recursive: true });
            await cp(file, target);
          }
          inventory.set(key, { package: key, license: pkg.license, source: `https://crates.io/crates/${pkg.name}/${pkg.version}`, licenseFiles: supplementFiles.map(p => path.relative(supplement, p)), supplemental: true });
          continue;
        } catch (error) { if (error.code !== 'ENOENT') throw error; }
      }
      for (const file of selected) {
        const target = path.join(destination, key, path.relative(dir, file));
        await mkdir(path.dirname(target), { recursive: true });
        await cp(file, target);
      }
      inventory.set(key, { package: key, license: pkg.license, source: `https://crates.io/crates/${pkg.name}/${pkg.version}`, licenseFiles: selected.map(p => path.relative(dir, p)) });
    }
  }
  await writeFile(path.join(destination, 'components.json'), JSON.stringify([...inventory.values()], null, 2) + '\n');
  return [...inventory.values()].filter(component => !component.licenseFiles.length).map(component => component.package);
}
export const NATIVE_CORE_NOTICES = [
  'OpenCV-4.13.0-resize-NOTICE.txt',
  'OpenCV-4.13.0-LICENSE.txt',
  'Pillow-12.3.0-LICENSE.txt',
  'image-transforms-SOURCES.txt',
];
export async function stageNativeCoreNotices(destination, sourceRoot = ROOT) {
  // Source-informed adaptations have notices outside Cargo's registry graph.
  // Read all required files before writing; a missing file is a hard error.
  const records = [];
  for (const name of NATIVE_CORE_NOTICES) {
    const source = `backend/licenses/${name}`;
    const bytes = await readFile(await verifiedLocal(sourceRoot, source));
    if (!bytes.length) throw new Error(`Empty required native-core notice: ${name}`);
    records.push({ name, source, sha256: sha(bytes), bytes });
  }
  await mkdir(destination, { recursive: true });
  for (const record of records) await writeFile(path.join(destination, record.name), record.bytes);
  const inventory = records.map(({ bytes, ...record }) => record);
  await writeFile(path.join(destination, 'components.json'), JSON.stringify(inventory, null, 2) + '\n');
  return inventory;
}
export async function prepare(options = {}) {
  const destination = path.resolve(options.destination || path.join(ROOT, 'desktop/src-tauri/resources'));
  const staging = destination + '.staging';
  // Build only on explicit request; packaging never downloads an unverified binary.
  if (options.build) cargo(['build', '--locked', '--release', '--manifest-path', 'backend/Cargo.toml']);
  const filename = process.platform === 'win32' ? 'guitarocr-backend.exe' : 'guitarocr-backend';
  const binary = path.resolve(options.binary || path.join(ROOT, 'target/release', filename));
  if (!(await lstat(binary)).isFile() || (await lstat(binary)).isSymbolicLink()) throw new Error('Build the backend first (npm run native:build), or pass --binary PATH');
  validateNativeBinary(await readFile(binary));
  const manifest = options.manifest ? validateManifest(JSON.parse(await readFile(options.manifest, 'utf8')), process.platform, process.arch, options.requireInference) : validateManifest({ schema: 1, target: `${process.platform}-${process.arch}`, components: [] }, process.platform, process.arch, options.requireInference);
  const llama = manifest.components.find(component => component.name === 'llama.cpp');
  if (llama) {
    const pinned = JSON.parse(await readFile(path.join(ROOT, 'scripts/llamacpp-runtime.json'), 'utf8'));
    if (llama.version !== pinned.commit) throw new Error('llama.cpp must match the repository-pinned commit');
  }
  await rm(staging, { recursive: true, force: true });
  await mkdir(path.join(staging, 'native'), { recursive: true });
  await mkdir(path.join(staging, 'licenses'), { recursive: true });
  try {
    // This whitelist intentionally excludes all backend Python, uv, research and model files.
    const staticDir = path.join(ROOT, 'ui');
    for (const file of await walk(staticDir)) {
      const relative = path.relative(staticDir, file);
      if (!/^(workbench|accounts)[\\/]/.test(relative)) continue;
      if (/\.(py|pyc|pyo|whl)$/i.test(relative)) throw new Error(`Unexpected Python resource: ${relative}`);
      const target = path.join(staging, 'ui', relative);
      await mkdir(path.dirname(target), { recursive: true });
      await cp(file, target);
    }
    manifest.modelMetadata = [];
    for (const relative of ['weights/distribution.json', 'weights/score_ocr/merged/score_image_policy.json', 'weights/score_ocr/merged/capabilities.json']) {
      const bytes = await readFile(await verifiedLocal(ROOT, relative));
      JSON.parse(bytes.toString('utf8'));
      const file = 'models/' + path.basename(relative);
      await mkdir(path.join(staging, 'native/models'), { recursive: true });
      await writeFile(path.join(staging, 'native', file), bytes);
      manifest.modelMetadata.push({ file, sha256: sha(bytes), source: relative });
    }
    await cp(binary, path.join(staging, 'native', filename));
    if (process.platform !== 'win32') await chmod(path.join(staging, 'native', filename), 0o755);
    for (const component of manifest.components) {
      for (const file of component.files) {
        const source = await verifiedLocal(path.dirname(path.resolve(options.manifest)), file.path);
        const bytes = await readFile(source);
        if (file.role === 'executable' || file.role === 'library') validateNativeBinary(bytes);
        if (sha(bytes) !== file.sha256) throw new Error(`Native resource checksum mismatch: ${file.path}`);
        const target = path.join(staging, 'native', component.name, file.path);
        await mkdir(path.dirname(target), { recursive: true });
        await writeFile(target, bytes);
        if (file.role === 'executable' && process.platform !== 'win32') await chmod(target, 0o755);
      }
    }
    manifest.nativeCoreNotices = await stageNativeCoreNotices(path.join(staging, 'licenses/native-core'));
    manifest.licenseReviewRequired = await stageCargoLicenses(path.join(staging, 'licenses'), options.licenseOverrides || path.join(ROOT, 'desktop/native-licenses'));
    if (options.requireInference && manifest.licenseReviewRequired.length) throw new Error(`Missing Cargo license texts: ${manifest.licenseReviewRequired.join(', ')}; supply reviewed --license-overrides DIR/<package-version>/ files`);
    await cp(path.join(ROOT, 'THIRD_PARTY_NOTICES.md'), path.join(staging, 'licenses/THIRD_PARTY_NOTICES.md'));
    await cp(path.join(ROOT, 'weights/licenses'), path.join(staging, 'licenses/models'), { recursive: true });
    manifest.service = { file: filename, sha256: sha(await readFile(binary)), source: 'backend', build: 'local-cargo' };
    manifest.version = "0.1.0";
    manifest.inferenceInventoryComplete = Object.keys(SOURCES).every(name => manifest.components.some(component => component.name === name)) && manifest.licenseReviewRequired.length === 0;
    await writeFile(path.join(staging, 'native/manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
    await rm(destination, { recursive: true, force: true });
    await rename(staging, destination);
    return manifest;
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const options = {};
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--build') options.build = true;
    else if (arg === '--require-inference') options.requireInference = true;
    else if (arg === '--license-overrides' && args[i + 1]) options.licenseOverrides = args[++i];
    else if (['--binary', '--manifest', '--destination'].includes(arg) && args[i + 1]) options[arg.slice(2)] = args[++i];
    else throw new Error(`Unknown or incomplete option: ${arg}`);
  }
  const result = await prepare(options);
  if (result.licenseReviewRequired.length) console.warn(`Release license review required: ${result.licenseReviewRequired.length} crates lack bundled license texts (see native/manifest.json)`);
  console.log(`Prepared native-only ${result.target} resources (${result.inferenceInventoryComplete ? 'inference ready' : 'editor only'}).`);
}
