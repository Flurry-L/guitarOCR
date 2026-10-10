/** Stage native-only Tauri resources. No downloads, Python runtime, or model execution. */
import { cp, copyFile, mkdir, readFile, readdir, lstat, chmod, rm, rename, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export async function syncUiAssets({root = ROOT, resources = path.join(root, 'desktop/src-tauri/resources')} = {}) {
  const ui = path.join(root, 'ui'), launcher = path.join(ui, 'launcher');
  for (const name of ['theme.css', 'theme.js'])
    await copyFile(path.join(ui, 'workbench', name), path.join(launcher, name));
  const destination = path.join(resources, 'ui'), staging = path.join(resources, 'ui.next');
  await rm(staging, {recursive: true, force: true});
  try {
    await mkdir(staging, {recursive: true});
    for (const section of ['accounts', 'workbench'])
      await cp(path.join(ui, section), path.join(staging, section), {recursive: true});
    await rm(destination, {recursive: true, force: true});
    await rename(staging, destination);
  } finally {
    await rm(staging, {recursive: true, force: true});
  }
}

const SOURCES = {
  onnxruntime: 'https://github.com/microsoft/onnxruntime',
  pdfium: 'https://pdfium.googlesource.com/pdfium',
  'llama.cpp': 'https://github.com/ggml-org/llama.cpp',
};
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
    if (names.has(component.name) || SOURCES[component.name] !== component.source) throw new Error('Unknown/duplicate native component or untrusted source');
    names.add(component.name);
    if (component.name === 'llama.cpp' &&
        (!component.capabilities || ['cpu', 'cuda', 'metal'].some(name => typeof component.capabilities[name] !== 'boolean'))) {
      throw new Error('Native runtime must declare CPU, CUDA and Metal capabilities');
    }
    if (!Array.isArray(component.files) || !component.files.length) throw new Error('Empty component inventory');
    const paths = new Set();
    for (const file of component.files) {
      safeRelative(file.path);
      if (paths.has(file.path) || !Number.isSafeInteger(file.bytes) || file.bytes < 0 || !['library', 'executable', 'license', 'notice'].includes(file.role)) throw new Error('Invalid native file inventory');
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
function cargo(args, env = process.env) {
  const result = spawnSync('cargo', args, { cwd: ROOT, env, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(`cargo ${args[0]} failed: ${result.error || result.stderr}`);
  return result.stdout;
}
export function isLicenseNotice(filename) {
  // Source APIs named copying.rs/copyright.rs are not license notices.
  return /^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)/i.test(path.basename(filename))
    && !/\.(rs|c|h|cpp|hpp|py|js|mjs|ts|tsx|jsx|toml|json)$/i.test(filename);
}
export async function stageCargoLicenses(destination, target) {
  await mkdir(destination, { recursive: true });
  await cp(path.join(ROOT, 'THIRD_PARTY_NOTICES.md'), path.join(destination, 'THIRD_PARTY_NOTICES.md'));
  const supplements = new Set(JSON.parse(await readFile(path.join(ROOT, 'desktop/license-supplements.json'))));
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
      if (!selected.length && supplements.has(key)) {
        inventory.set(key, { package: key, license: pkg.license, source: `https://crates.io/crates/${pkg.name}/${pkg.version}`, licenseFiles: ['THIRD_PARTY_NOTICES.md'], supplemental: true });
        continue;
      }
      for (const file of selected) {
        const target = path.join(destination, key, path.relative(dir, file));
        await mkdir(path.dirname(target), { recursive: true });
        await cp(file, target);
      }
      inventory.set(key, { package: key, license: pkg.license, source: `https://crates.io/crates/${pkg.name}/${pkg.version}`, licenseFiles: selected.map(p => path.join(key, path.relative(dir, p))) });
    }
  }
  await writeFile(path.join(destination, 'components.json'), JSON.stringify([...inventory.values()], null, 2) + '\n');
  return [...inventory.values()].filter(component => !component.licenseFiles.length).map(component => component.package);
}
export async function readModelMetadata(modelCatalog = null) {
  const records = [];
  for (const relative of ['weights/distribution.json', 'weights/score_ocr/merged/score_image_policy.json', 'weights/score_ocr/merged/capabilities.json']) {
    const customCatalog = relative === 'weights/distribution.json' && modelCatalog;
    const bytes = await readFile(customCatalog || await verifiedLocal(ROOT, relative));
    JSON.parse(bytes.toString('utf8'));
    const file = 'models/' + path.basename(relative);
    records.push({ file, source: customCatalog ? 'build-time-model-catalog' : relative, bytes });
  }
  return records;
}
export async function stageModelMetadata(destination, modelCatalog = null, snapshot = null) {
  const records = snapshot || await readModelMetadata(modelCatalog);
  await mkdir(path.join(destination, 'models'), { recursive: true });
  for (const { file, bytes } of records) await writeFile(path.join(destination, file), bytes);
  return records.map(({ bytes, ...record }) => record);
}
// Declarations are relative to their source model; packaged assets use one root.
function auxiliaryAsset(value) {
  return typeof value === 'string' ? value.match(/(?:^|\/)(auxiliary\/.+)$/)?.[1] : undefined;
}
export function validateDeclaredModelAssets(records) {
  const metadata = Object.fromEntries(records.map(({ file, bytes }) => [path.basename(file), JSON.parse(bytes.toString('utf8'))]));
  const catalog = new Set(Object.values(metadata['distribution.json'].files || {}).flat().map(file => file.path));
  const capabilities = metadata['capabilities.json'];
  if (capabilities.measure_output !== 'observation' || capabilities.measure_images !== 1) {
    throw new Error('Measure recognition requires single-image observation weights');
  }
  const information = capabilities.information_model;
  if (!information || !Array.isArray(information.adapter_tasks) || !information.adapter_tasks.length ||
      new Set(information.adapter_tasks).size !== information.adapter_tasks.length ||
      !information.adapter_tasks.every(task => ['clef', 'staff'].includes(task))) {
    throw new Error('Information recognition requires its model, projector and staff/clef adapter');
  }
  for (const [field, filename] of [['model', 'model-Q8_0.gguf'], ['projector', 'vision-F16.gguf'], ['adapter', 'information-adapter.gguf']]) {
    if (information[field] !== `../../information/gguf/${filename}` || !catalog.has(`weights/information/gguf/${filename}`)) {
      throw new Error(`Information model ${field} is missing or has an unsupported path`);
    }
  }
  if ((metadata['capabilities.json'].staff_label_model == null) !==
      (metadata['capabilities.json'].instrument_text_model == null)) {
    throw new Error('staff_label_model and instrument_text_model must be declared together');
  }
  if (metadata['capabilities.json'].part_continuation_model != null &&
      (metadata['capabilities.json'].staff_label_model == null || metadata['capabilities.json'].staff_profile !== true)) {
    throw new Error('Part continuation requires observed staff label profiles');
  }
  for (const [field, asset, description] of [
    ['layout_quality_model', 'auxiliary/layout_quality.onnx', 'layout quality'],
    ['staff_label_model', 'auxiliary/staff_label.onnx', 'staff label'],
    ['instrument_text_model', 'auxiliary/instrument_text.onnx', 'instrument text'],
    ['part_continuation_model', 'auxiliary/part_continuation.onnx', 'part continuation'],
    ['clef_spatial_model', 'auxiliary/clef_spatial.onnx', 'spatial clef'],
  ]) {
    const declared = metadata['capabilities.json'][field];
    if (declared != null && auxiliaryAsset(declared) !== asset) throw new Error(`Unsupported ${description} model declaration`);
  }
  for (const [capability, declaration] of Object.entries(metadata['capabilities.json'])) {
    const asset = auxiliaryAsset(declaration);
    if (asset != null && !catalog.has(asset)) {
      throw new Error(`Selected model catalog omits declared ${capability}: ${asset}; use --model-catalog with the matching model generation`);
    }
  }
}
export function verifyBinaryModelMetadata(binary, records, run = spawnSync) {
  const result = run(binary, ['--model-metadata'], { cwd: ROOT, timeout: 15000, maxBuffer: 1024 * 1024 });
  let compiled;
  try {
    if (result.error || result.status !== 0 || !Buffer.isBuffer(result.stdout)) throw new Error();
    compiled = JSON.parse(result.stdout.toString('utf8'));
  } catch {
    throw new Error('Backend cannot report its compiled model metadata; rebuild the backend before packaging');
  }
  for (const { file, bytes } of records) {
    const name = path.basename(file);
    if (typeof compiled[name] !== 'string' || !Buffer.from(compiled[name], 'utf8').equals(bytes)) {
      throw new Error(`Backend ${name} does not match packaged metadata; rebuild with the selected inference metadata`);
    }
  }
}
export async function prepare(options = {}) {
  const destination = path.resolve(options.destination || path.join(ROOT, 'desktop/src-tauri/resources'));
  const staging = destination + '.staging';
  const override = options.modelCatalog || process.env.GUITAROCR_BUILD_MODEL_CATALOG;
  if (override && (!options.build || options.binary)) {
    throw new Error('A custom model catalog requires --build and cannot be combined with --binary');
  }
  const modelCatalog = override ? path.resolve(ROOT, override) : null;
  if (modelCatalog) {
    if (!(await lstat(modelCatalog)).isFile() || (await lstat(modelCatalog)).isSymbolicLink()) throw new Error('Model catalog override must be a regular local file');
    JSON.parse(await readFile(modelCatalog, 'utf8'));
  }
  const modelMetadata = await readModelMetadata(modelCatalog);
  validateDeclaredModelAssets(modelMetadata);
  const buildEnv = modelCatalog ? { ...process.env, GUITAROCR_BUILD_MODEL_CATALOG: modelCatalog } : process.env;
  // Build only on explicit request; packaging never downloads an unverified binary.
  if (options.build) cargo(['build', '--locked', '--release', '--manifest-path', 'backend/Cargo.toml'], buildEnv);
  const filename = process.platform === 'win32' ? 'guitarocr-backend.exe' : 'guitarocr-backend';
  const binary = path.resolve(options.binary || path.join(ROOT, 'target/release', filename));
  if (!(await lstat(binary)).isFile() || (await lstat(binary)).isSymbolicLink()) throw new Error('Build the backend first (npm run native:build), or pass --binary PATH');
  const binaryBytes = await readFile(binary);
  validateNativeBinary(binaryBytes);
  verifyBinaryModelMetadata(binary, modelMetadata);
  const manifest = options.manifest ? validateManifest(JSON.parse(await readFile(options.manifest, 'utf8')), process.platform, process.arch, options.requireInference) : validateManifest({ schema: 1, target: `${process.platform}-${process.arch}`, components: [] }, process.platform, process.arch, options.requireInference);
  await rm(staging, { recursive: true, force: true });
  await mkdir(path.join(staging, 'native'), { recursive: true });
  await mkdir(path.join(staging, 'licenses'), { recursive: true });
  try {
    // The same refresh runs before every Tauri build, preventing an older
    // prepared resource tree from silently embedding a stale workbench.
    await syncUiAssets({root: ROOT, resources: staging});
    manifest.modelMetadata = await stageModelMetadata(path.join(staging, 'native'), modelCatalog, modelMetadata);
    await writeFile(path.join(staging, 'native', filename), binaryBytes);
    if (process.platform !== 'win32') await chmod(path.join(staging, 'native', filename), 0o755);
    for (const component of manifest.components) {
      for (const file of component.files) {
        const source = await verifiedLocal(path.dirname(path.resolve(options.manifest)), file.path);
        const bytes = await readFile(source);
        if (file.role === 'executable' || file.role === 'library') validateNativeBinary(bytes);
        if (bytes.length !== file.bytes) throw new Error(`Native resource size mismatch: ${file.path}`);
        const target = path.join(staging, 'native', component.name, file.path);
        await mkdir(path.dirname(target), { recursive: true });
        await writeFile(target, bytes);
        // ONNX Runtime also needs the C++ runtime in the backend's DLL search
        // directory. Keep the inventoried copy beside llama-server as well.
        if (process.platform === 'win32' && ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'].includes(path.basename(file.path).toLowerCase())) {
          await writeFile(path.join(staging, 'native', path.basename(file.path)), bytes);
        }
        if (file.role === 'executable' && process.platform !== 'win32') await chmod(target, 0o755);
      }
    }
    manifest.licenseReviewRequired = await stageCargoLicenses(path.join(staging, 'licenses'));
    if (options.requireInference && manifest.licenseReviewRequired.length) throw new Error(`Missing Cargo license texts: ${manifest.licenseReviewRequired.join(', ')}; add their notices to THIRD_PARTY_NOTICES.md and desktop/license-supplements.json`);
    manifest.service = { file: filename, source: 'backend', build: 'local-cargo' };
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
  if (process.argv.slice(2).join(' ') === '--ui-only') {
    await syncUiAssets();
    console.log('Desktop UI resources refreshed.');
  } else {
    const options = {};
    const args = process.argv.slice(2);
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--build') options.build = true;
      else if (arg === '--require-inference') options.requireInference = true;
      else if (arg === '--model-catalog' && args[i + 1]) options.modelCatalog = args[++i];
      else if (['--binary', '--manifest', '--destination'].includes(arg) && args[i + 1]) options[arg.slice(2)] = args[++i];
      else throw new Error(`Unknown or incomplete option: ${arg}`);
    }
    const result = await prepare(options);
    if (result.licenseReviewRequired.length) console.warn(`Release license review required: ${result.licenseReviewRequired.length} crates lack bundled license texts (see native/manifest.json)`);
    console.log(`Prepared native-only ${result.target} resources (${result.inferenceInventoryComplete ? 'inference ready' : 'editor only'}).`);
  }
}
