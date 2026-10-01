import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const script = (await readFile(new URL('../workbench/workspace.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\n/gm, '').replace('export function initWorkspace', 'function initWorkspace');
function workspace(pathname = '/') {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, { textContent: '', addEventListener() {}, querySelector() { return $(id + '-option'); } });
    return elements.get(id);
  };
  let refreshes = 0;
  const context = vm.createContext({ $, ui: {}, location: { pathname }, document: { querySelectorAll: () => [] },
    window: { dispatchEvent() {} }, Event, action: fn => fn, setInterval() {},
    refresh: async () => { refreshes++; } });
  vm.runInContext(script + '\nglobalThis.workspace = initWorkspace({ refreshConfig: refresh });', context);
  return { api: context.workspace, $, refreshes: () => refreshes };
}
const native = { native: true, inference_enabled: true, device: 'auto', model_ready: false,
  model_cached: true, layout_cached: true, max_pages: 100, max_upload_mb: 200 };

test('cached native models are not confused with missing or verified files', () => {
  const { api, $ } = workspace();
  api.configure(native);
  assert.equal($('settingModel').textContent, '已缓存，待校验');
  assert.match($('settingDevice').textContent, /首次识别/);
  assert.match($('settingHint').textContent, /中断后重试可续传/);
  api.configure({ ...native, device: 'metal', model_ready: true, layout_ready: true });
  assert.equal($('settingDevice').textContent, 'metal');
  assert.equal($('settingModel').textContent, '已就绪');
  assert.match($('settingHint').textContent, /模型已校验/);
});
test('settings refresh does not retain an old disabled state or Python advice', async () => {
  const { api, $, refreshes } = workspace();
  api.configure({ ...native, inference_enabled: false });
  assert.equal($('importAction-option').disabled, true);
  api.configure(native);
  assert.equal($('importAction-option').disabled, false);
  assert.doesNotMatch($('settingHint').textContent, /启动页|安装环境/);
  api.show('settings');
  assert.equal(refreshes(), 1);
});
test('remote model problems direct users to the service administrator', () => {
  const { api, $ } = workspace('/workbench');
  api.configure({ ...native, server: true, model_cached: false });
  assert.match($('settingHint').textContent, /服务管理员/);
});
