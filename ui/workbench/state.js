export const ui = {
  state: null,
  sid: null,
  openGeneration: 0,
  step: 0,
  files: [],
  busy: false,
  boxes: [],
  pageIndex: 0,
  selected: -1,
  pageImage: null,
  scale: 1,
  drag: null,
  measureIndex: 0,
  editorMode: "score",
  boxDirty: false,
  measureDirty: false,
  metadataDirty: false,
};

// A server snapshot is distinct from unsaved form/score/canvas drafts.
export function hasDrafts() {
  return ui.boxDirty || ui.metadataDirty || ui.measureDirty;
}
export function clearDrafts() {
  ui.boxDirty = ui.metadataDirty = ui.measureDirty = false;
}
export function openProject(id, { discardDrafts = false } = {}) {
  if (discardDrafts) {
    try { globalThis.sessionStorage?.removeItem(`guitarocr-draft:${ui.sid}`); } catch {}
  }
  ui.openGeneration += 1;
  ui.sid = id;
  ui.state = null;
  ui.files = [];
  ui.boxes = [];
  ui.pageIndex = ui.measureIndex = 0;
  ui.selected = -1;
  ui.pageImage = ui.drag = null;
  clearDrafts();
}
export function receiveProject(saved, { preserveDrafts = false } = {}) {
  if (saved.id !== ui.sid) openProject(saved.id);
  ui.state = saved;
  if (!preserveDrafts || !ui.boxDirty)
    ui.boxes = structuredClone(saved.boxes || []);
  ui.pageIndex = Math.max(0, Math.min(ui.pageIndex, saved.pages.length - 1));
  if (ui.selected >= ui.boxes.length) ui.selected = -1;
  if (!preserveDrafts) clearDrafts();
}
export const endpoint = (part) => `/api/sessions/${ui.sid}${part}`;

// Keep the draft's original revision, even if a newer snapshot arrives elsewhere.
// A save response may update only the project snapshot from which it was sent.
export async function commitMeasureDraft({ sid, index, revision, body }, send) {
  const snapshot = ui.state;
  const saved = await send(`/api/sessions/${sid}/measures/${index + 1}`, "PUT", body, revision);
  if (ui.sid !== sid || ui.state !== snapshot || saved.id !== sid) return false;
  receiveProject(saved);
  return true;
}

// Saving a measure invalidates the OCR checkpoint; resume must use it directly.
export async function resumeCheckpoint({ dirty, confirm, start }) {
  if (dirty && !confirm("继续会从中断记录恢复，成功会替换当前未保存草稿；失败或取消会保留草稿。继续？")) return false;
  await start("/recognize", { resume: true });
  return true;
}
