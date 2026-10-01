// Small, consistent line icons; no font-dependent toolbar symbols.
const paths = {
  workbench: ["M4 4h16v16H4z", "M4 10h16M10 10v10"],
  library: [
    "M5 3h13a2 2 0 0 1 2 2v16H6a3 3 0 0 1-3-3V5a2 2 0 0 1 2-2Z",
    "M6 17h14M7 7h9M7 11h7",
  ],
  tasks: ["M20 12a8 8 0 1 1-8-8 8 8 0 0 1 8 8Z", "M12 7v5l3 2"],
  settings: ["M4 7h16M4 17h16M8 4v6M16 14v6"],
  admin: ["M4 5h16M4 12h16M4 19h16", "M7 3v4M17 10v4M10 17v4"],
  undo: ["M9 5 4 10l5 5", "M4 10h10a6 6 0 0 1 6 6v2"],
  redo: ["m15 5 5 5-5 5", "M20 10H10a6 6 0 0 0-6 6v2"],
  copy: ["M8 8h12v12H8z", "M16 5V3H3v13h2"],
  paste: ["M9 5H5v16h14V5h-4", "M9 3h6v4H9zM9 12h6M9 16h4"],
  duplicate: ["M4 4h12v12H4z", "M9 20h11V9M7 10h6M10 7v6"],
  rest: ["m13 3-4 4 5 5-5 4 4 2", "M13 18c-6-3-8 1-4 3"],
  addNote: ["M10 16V4l5 3", "M10 16c0-3-6-2-6 1s6 2 6-1Z", "M16 14v6M13 17h6"],
  deleteNote: ["M10 16V4l5 3", "M10 16c0-3-6-2-6 1s6 2 6-1Z", "M13 17h7"],
  addBeat: ["M4 4h16v16H4z", "M8 12h8M12 8v8"],
  deleteBeat: ["M5 7h14M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6"],
  inspector: ["M3 4h18v16H3z", "M15 4v16M18 8v2M18 13v3"],
  focus: ["M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5"],
  upload: ["m8 8 4-4 4 4M12 4v11M4 15v5h16v-5"],
  remote: ["M4 4h16v12H4zM8 20h8M12 16v4", "m9 11 6-4M11 7h4v4"],
};
export function renderIcons() {
  for (const node of document.querySelectorAll("[data-icon]")) {
    const shape = paths[node.dataset.icon];
    if (!shape) continue;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    for (const [name, value] of Object.entries({
      viewBox: "0 0 24 24",
      width: "18",
      height: "18",
      fill: "none",
      stroke: "currentColor",
      "stroke-width": "1.6",
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "aria-hidden": "true",
    }))
      svg.setAttribute(name, value);
    for (const d of shape) {
      const path = document.createElementNS(svg.namespaceURI, "path");
      path.setAttribute("d", d);
      svg.append(path);
    }
    node.replaceChildren(svg);
  }
}
renderIcons();
document.addEventListener("DOMContentLoaded", renderIcons, { once: true });
