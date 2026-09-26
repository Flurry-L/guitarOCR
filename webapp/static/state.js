export const ui = {
  state: null,
  sid:
    new URLSearchParams(location.search).get("project") ||
    (location.pathname === "/workbench" ? null : localStorage.getItem("guitarocr-session")),
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
  editorMode: "table",
  eventData: [],
  boxDirty: false,
  measureDirty: false,
  metadataDirty: false,
};
