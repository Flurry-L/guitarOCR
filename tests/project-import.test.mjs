import test from 'node:test';
import assert from 'node:assert/strict';
import { createProjectImporter, readSample, SAMPLE_ARCHIVE } from '../webapp/static/project-import.js';

const messages = { prompt: 'Replace draft?', loading: 'Loading', opened: 'Opened', background: 'In library' };
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
function setup(options = {}) {
  const state = { sid: 'current', openGeneration: 1, files: [{ name: 'selected.pdf' }], busy: false,
    draft: { title: 'Unsaved title' } };
  const events = [], request = deferred();
  let navigation = 0;
  const run = createProjectImporter({
    ui: state,
    hasDrafts: () => Boolean(state.draft),
    confirm: (prompt) => { events.push(['confirm', prompt]); return true; },
    setBusy: (value) => { state.busy = value; events.push(['busy', value]); },
    navigation: () => navigation,
    importArchive: (file) => { events.push(['import', file]); return request.promise; },
    accept: (saved) => {
      events.push(['accept', saved]);
      state.sid = saved.id;
      state.openGeneration += 1;
      state.files = [];
      state.draft = null;
    },
    notice: (...args) => events.push(['notice', ...args]),
    refresh: async () => { events.push(['refresh']); },
    ...options,
  });
  return { run, state, events, request, leave: () => { navigation += 1; } };
}
const file = new File(['zip'], 'project.zip', { type: 'application/zip' });
const start = (context) => context.run(() => file, messages);
const count = (context, kind) => context.events.filter(([name]) => name === kind).length;

test('a successful import replaces drafts and selected files only after the response', async () => {
  const context = setup();
  const draft = context.state.draft, files = context.state.files;
  const pending = start(context);
  assert.equal(context.state.busy, true);
  assert.equal(context.state.draft, draft);
  assert.equal(context.state.files, files);
  context.request.resolve({ id: 'new-project' });
  assert.equal(await pending, true);
  assert.equal(context.state.sid, 'new-project');
  assert.equal(context.state.draft, null);
  assert.deepEqual(context.state.files, []);
  assert.equal(context.state.busy, false);
  assert.equal(count(context, 'accept'), 1);
  assert.equal(count(context, 'refresh'), 1);
});

test('repeated clicks and other imports cannot duplicate an in-flight import', async () => {
  const context = setup();
  const pending = start(context);
  assert.equal(await start(context), false);
  context.request.resolve({ id: 'new-project' });
  await pending;
  assert.equal(count(context, 'confirm'), 1);
  assert.equal(count(context, 'import'), 1);
});

test('busy jobs reject imports before prompting or reading the archive', async () => {
  const context = setup();
  context.state.busy = true;
  let reads = 0;
  assert.equal(await context.run(() => { reads++; return file; }, messages), false);
  assert.equal(reads, 0);
  assert.deepEqual(context.events, []);
});

test('canceling replacement preserves the editor and allows choosing the same archive again', async () => {
  let approved = false;
  const context = setup({ confirm: () => approved });
  const before = structuredClone(context.state);
  assert.equal(await start(context), false);
  assert.deepEqual(context.state, before);
  approved = true;
  const pending = start(context);
  context.request.resolve({ id: 'new-project' });
  assert.equal(await pending, true);
});

test('selected files alone require replacement confirmation', async () => {
  const context = setup({ confirm: () => false });
  context.state.draft = null;
  assert.equal(await start(context), false);
  assert.equal(count(context, 'import'), 0);
  assert.equal(context.state.files[0].name, 'selected.pdf');
});

for (const failure of ['read', 'import']) {
  test(`${failure} failure preserves current drafts and files, and unlocks the editor`, async () => {
    const context = setup();
    const before = structuredClone(context.state);
    const pending = context.run(() => {
      if (failure === 'read') throw new Error('Cannot read archive');
      return file;
    }, messages);
    if (failure === 'import') context.request.reject(new Error('Import failed'));
    assert.equal(await pending, false);
    assert.deepEqual(context.state, before);
    assert.equal(count(context, 'accept'), 0);
    assert.equal(context.events.at(-2)[0], 'notice');
    assert.equal(context.events.at(-2)[2], true);
  });
}

test('a failed archive load can be retried successfully', async () => {
  const context = setup();
  assert.equal(await context.run(() => { throw new Error('offline'); }, messages), false);
  const pending = start(context);
  context.request.resolve({ id: 'retried-project' });
  assert.equal(await pending, true);
  assert.equal(context.state.sid, 'retried-project');
  assert.equal(context.state.busy, false);
});

test('late import stays in the library after leaving and returning to the same screen', async () => {
  const context = setup();
  const before = structuredClone(context.state);
  const pending = start(context);
  context.leave();
  context.leave();
  context.request.resolve({ id: 'new-project' });
  assert.equal(await pending, true);
  assert.deepEqual(context.state, before);
  assert.equal(count(context, 'accept'), 0);
  assert.equal(count(context, 'refresh'), 1);
  assert.ok(context.events.some(([name, message]) => name === 'notice' && message === 'In library'));
});

for (const success of [true, false]) {
  test(`a late ${success ? 'response' : 'failure'} cannot replace or unlock a newer project`, async () => {
    const context = setup();
    const pending = start(context);
    await Promise.resolve();
    context.state.sid = 'another-project';
    context.state.openGeneration += 1;
    const newer = structuredClone(context.state);
    if (success) context.request.resolve({ id: 'new-project' });
    else context.request.reject(new Error('Import failed'));
    assert.equal(await pending, success);
    assert.deepEqual(context.state, newer);
    assert.equal(count(context, 'accept'), 0);
    assert.equal(count(context, 'notice'), 1);
    assert.equal(context.state.busy, true);
  });
}

test('sample fetch supplies a ZIP to the existing import path and rejects missing assets', async () => {
  const sample = await readSample(async (path) => {
    assert.equal(path, SAMPLE_ARCHIVE);
    return new Response('bundled original preset', { status: 200 });
  });
  assert.equal(sample.name, 'Harbor-Light-synthetic-project.zip');
  assert.equal(sample.type, 'application/zip');
  assert.equal(await sample.text(), 'bundled original preset');
  await assert.rejects(readSample(async () => new Response('', { status: 404 })), /示例文件未能加载/);
  await assert.rejects(readSample(async () => { throw new Error('offline'); }), /暂时无法加载示例/);
});
