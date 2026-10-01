import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const html = await readFile(new URL('../../ui/launcher/index.html', import.meta.url), 'utf8');
const script = await readFile(new URL('../../ui/launcher/app.js', import.meta.url), 'utf8');

function element() {
  return {
    hidden: false, disabled: false, textContent: '', value: '', children: [],
    attributes: new Map(), classes: new Set(),
    setAttribute(name, value) { this.attributes.set(name, value); },
    removeAttribute(name) { this.attributes.delete(name); },
    replaceChildren(...children) { this.children = children; },
    scrollIntoView() {},
    get classList() {
      return { toggle: (name, active) => active ? this.classes.add(name) : this.classes.delete(name) };
    },
  };
}

function documentFixture() {
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, element()]));
  return {
    elements,
    document: {
      getElementById: id => elements.get(id),
      querySelectorAll: () => [...elements.values()],
      createElement: element,
    },
  };
}

async function launcher({ supported = true, startError } = {}) {
  const { document, elements } = documentFixture();
  const calls = [];
  const events = new Map();
  const context = vm.createContext({
    document,
    confirm: () => true,
    window: { __TAURI__: {
      core: { invoke: async (command, args) => {
        calls.push({ command, args });
        if (command === 'settings') return { server: 'https://ocr.example.com', native_available: supported };
        if (command === 'start_local' && startError) throw new Error(startError);
      } },
      event: { listen: async (name, callback) => { events.set(name, callback); } },
    } },
  });
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  return { elements, calls, events };
}

test('opening local music starts the packaged service and restores controls', async () => {
  const state = await launcher();
  await state.elements.get('localEdit').onclick();
  assert.deepEqual(state.calls.map(call => call.command), ['settings', 'start_local']);
  assert.equal(state.calls[1].args.mode, 'native');
  assert.equal(state.elements.get('startupProgress').hidden, true);
  assert.equal(state.elements.get('localEdit').disabled, false);
  assert.equal(state.elements.get('status').textContent, '已打开本机乐谱。');
});

test('unsupported platform leaves remote access but does not offer broken local mode', async () => {
  const { elements } = await launcher({ supported: false });
  assert.equal(elements.get('localEdit').hidden, true);
  assert.equal(elements.get('remote').hidden, false);
});

test('runtime failure stays visible', async () => {
  const { elements } = await launcher({ startError: '缺少原生组件' });
  await elements.get('localEdit').onclick();
  assert.equal(elements.get('startupProgress').hidden, true);
  assert.match(elements.get('status').textContent, /缺少原生组件/);
  assert.equal(elements.get('status').classes.has('error'), true);
});
