export const ui = {
  state: null,
  sid: null,
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
export function openProject(id) {
  ui.sid = id;
  ui.state = null;
  ui.files = [];
  ui.boxes = [];
  ui.pageIndex = ui.measureIndex = 0;
  ui.selected = -1;
  ui.pageImage = ui.drag = null;
  clearDrafts();
}
export function receiveProject(saved) {
  if (saved.id !== ui.sid) openProject(saved.id);
  ui.state = saved;
  ui.boxes = structuredClone(saved.boxes || []);
  ui.pageIndex = Math.min(ui.pageIndex, saved.pages.length - 1);
  ui.selected = -1;
  clearDrafts();
}
export const endpoint = (part) => `/api/sessions/${ui.sid}${part}`;
