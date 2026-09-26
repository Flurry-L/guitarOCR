import { ui } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api, endpoint } from "./api.js";
const colors = { measure: "#28785b", header: "#6189ac", tempo: "#b07628" };
const names = { measure: "小节", header: "谱头", tempo: "速度" };
const modeNames = { tab: "TAB", notation: "五线谱", both: "五线谱 + TAB" };

export function initBoxes({ start, go, render }) {
  function boxMode(box) {
    return $("mode").value !== "auto"
      ? $("mode").value
      : box.mode || ui.state.pages[box.page - 1]?.notation_mode || "";
  }
  function loadPage() {
    if (!ui.state?.pages) return;
    const p = ui.state.pages[ui.pageIndex];
    $("pageLabel").textContent =
      `第 ${ui.pageIndex + 1} / ${ui.state.pages.length} 页${p.notation_mode ? `，${modeNames[p.notation_mode]}` : ""}`;
    ui.pageImage = null;
    $("canvas").setAttribute("aria-busy", "true");
    const img = new Image();
    img.onload = () => {
      if (ui.state.pages[ui.pageIndex] !== p) return;
      ui.pageImage = img;
      $("canvas").setAttribute("aria-busy", "false");
      resizeCanvas();
    };
    img.src = p.url;
    renderBoxList();
  }
  function resizeCanvas() {
    if (!ui.pageImage) return;
    const fit = Math.min(
      1,
      Math.max(200, $("canvasScroll").clientWidth - 32) / ui.pageImage.width,
    );
    ui.scale = fit * (+$("zoom").value / 75);
    $("canvas").width = Math.round(ui.pageImage.width * ui.scale);
    $("canvas").height = Math.round(ui.pageImage.height * ui.scale);
    $("zoomValue").textContent = `${$("zoom").value}%`;
    draw();
  }
  $("zoom").oninput = resizeCanvas;
  window.addEventListener("resize", () => {
    if (ui.step === 1) resizeCanvas();
  });
  function numberOf(index) {
    return ui.boxes.slice(0, index + 1).filter((b) => b.kind === "measure")
      .length;
  }
  function draw() {
    const canvas = $("canvas"),
      ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!ui.pageImage) return;
    ctx.drawImage(ui.pageImage, 0, 0, canvas.width, canvas.height);
    ui.boxes.forEach((box, i) => {
      if (box.page !== ui.pageIndex + 1) return;
      const [x, y, w, h] = box.bbox.map((v) => v * ui.scale);
      ctx.strokeStyle = colors[box.kind];
      ctx.fillStyle = colors[box.kind] + (i === ui.selected ? "24" : "0b");
      ctx.lineWidth = i === ui.selected ? 2.5 : 1.5;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.font = "12px system-ui";
      const text =
        box.kind === "measure" ? `小节 ${numberOf(i)}` : names[box.kind];
      const tw = ctx.measureText(text).width + 10;
      ctx.fillStyle = colors[box.kind];
      ctx.fillRect(x, Math.max(0, y - 20), tw, 20);
      ctx.fillStyle = "white";
      ctx.fillText(text, x + 5, Math.max(14, y - 5));
      if (i === ui.selected) {
        ctx.fillStyle = "white";
        for (const [cx, cy] of [
          [x, y],
          [x + w, y],
          [x, y + h],
          [x + w, y + h],
        ]) {
          ctx.fillRect(cx - 4, cy - 4, 8, 8);
          ctx.strokeRect(cx - 4, cy - 4, 8, 8);
        }
      }
    });
  }
  function renderBoxList() {
    const list = $("boxList");
    list.replaceChildren();
    ui.boxes.forEach((box, i) => {
      if (box.page !== ui.pageIndex + 1) return;
      const b = el(
        "button",
        box.kind === "measure" ? `小节 ${numberOf(i)}，${modeNames[boxMode(box)] || "待识别"}` : names[box.kind],
        i === ui.selected ? "active" : "",
      );
      b.onclick = () => {
        ui.selected = i;
        renderBoxList();
        draw();
      };
      list.append(b);
    });
    $("coords").hidden = ui.selected < 0;
    const selected = ui.boxes[ui.selected];
    $("boxModeField").hidden = !selected || selected.kind !== "measure";
    if (selected?.kind === "measure") {
      $("boxMode").value = boxMode(selected);
      $("boxMode").disabled = $("mode").value !== "auto";
    }
    if (ui.selected >= 0)
      ui.boxes[ui.selected].bbox.forEach(
        (v, i) =>
          ($(["boxX", "boxY", "boxW", "boxH"][i]).value = Math.round(v)),
      );
    $("boxSummary").textContent =
      `${ui.state?.pages.length || 0} 页，${ui.boxes.filter((b) => b.kind === "measure").length} 个小节${ui.boxDirty ? "，有未保存的修改" : ""}`;
    updateBoxControls();
  }
  function updateBoxControls() {
    const selected = ui.boxes[ui.selected];
    $("deleteBox").disabled = ui.busy || !selected;
    $("boxMode").disabled = ui.busy || $("mode").value !== "auto";
    const peers = ui.boxes.map((b, i) => ({ b, i })).filter(({ b }) =>
      selected && b.page === selected.page && b.kind === selected.kind);
    $("earlier").disabled = ui.busy || !selected || peers[0]?.i === ui.selected;
    $("later").disabled = ui.busy || !selected || peers.at(-1)?.i === ui.selected;
  }
  function point(e) {
    const r = $("canvas").getBoundingClientRect();
    return [
      Math.max(
        0,
        Math.min(ui.pageImage.width, (e.clientX - r.left) / ui.scale),
      ),
      Math.max(
        0,
        Math.min(ui.pageImage.height, (e.clientY - r.top) / ui.scale),
      ),
    ];
  }
  $("canvas").onpointerdown = (e) => {
    if (ui.busy || !ui.pageImage) return;
    e.preventDefault();
    const [x, y] = point(e);
    const tool = $("boxTool").value;
    let hit = -1,
      corner = -1;
    if (tool === "select") {
      if (ui.selected >= 0) {
        const [bx, by, w, h] = ui.boxes[ui.selected].bbox;
        [
          [bx, by],
          [bx + w, by],
          [bx, by + h],
          [bx + w, by + h],
        ].forEach(([cx, cy], i) => {
          if (Math.hypot(x - cx, y - cy) < 10 / ui.scale) {
            hit = ui.selected;
            corner = i;
          }
        });
      }
      if (hit < 0) {
        for (let i = ui.boxes.length - 1; i >= 0; i--) {
          const b = ui.boxes[i],
            [bx, by, w, h] = b.bbox;
          if (
            b.page === ui.pageIndex + 1 &&
            x >= bx &&
            x <= bx + w &&
            y >= by &&
            y <= by + h
          ) {
            hit = i;
            break;
          }
        }
      }
      ui.selected = hit;
      if (hit >= 0)
        ui.drag = { x, y, old: [...ui.boxes[hit].bbox], corner, kind: "edit" };
    } else {
      ui.boxes.push({ page: ui.pageIndex + 1, kind: tool, bbox: [x, y, 0, 0] });
      ui.selected = ui.boxes.length - 1;
      ui.drag = { x, y, kind: "new" };
    }
    if (ui.drag) $("canvas").setPointerCapture(e.pointerId);
    renderBoxList();
    draw();
  };
  $("canvas").onpointermove = (e) => {
    if (!ui.drag || !ui.pageImage) return;
    const [x, y] = point(e);
    const b = ui.boxes[ui.selected];
    if (ui.drag.kind === "new")
      b.bbox = [
        Math.min(x, ui.drag.x),
        Math.min(y, ui.drag.y),
        Math.abs(x - ui.drag.x),
        Math.abs(y - ui.drag.y),
      ];
    else {
      const [bx, by, w, h] = ui.drag.old;
      if (ui.drag.corner < 0)
        b.bbox = [
          Math.max(0, Math.min(ui.pageImage.width - w, bx + x - ui.drag.x)),
          Math.max(0, Math.min(ui.pageImage.height - h, by + y - ui.drag.y)),
          w,
          h,
        ];
      else {
        const ax = ui.drag.corner % 2 ? bx : bx + w,
          ay = ui.drag.corner < 2 ? by + h : by;
        b.bbox = [
          Math.min(x, ax),
          Math.min(y, ay),
          Math.abs(x - ax),
          Math.abs(y - ay),
        ];
      }
    }
    ui.boxDirty = true;
    draw();
  };
  function endDrag() {
    if (!ui.drag) return;
    if (
      ui.boxes[ui.selected].bbox[2] < 2 ||
      ui.boxes[ui.selected].bbox[3] < 2
    ) {
      if (ui.drag.kind === "new") ui.boxes.splice(ui.selected, 1);
      else ui.boxes[ui.selected].bbox = ui.drag.old;
      ui.selected = -1;
    }
    ui.boxes.sort((a, b) => a.page - b.page);
    ui.drag = null;
    ui.selected = -1;
    renderBoxList();
    draw();
  }
  $("canvas").onpointerup = endDrag;
  $("canvas").onpointercancel = endDrag;
  $("deleteBox").onclick = () => {
    if (ui.selected < 0) return;
    ui.boxes.splice(ui.selected, 1);
    ui.selected = -1;
    ui.boxDirty = true;
    renderBoxList();
    draw();
  };
  $("applyCoords").onclick = action(() => {
    if (ui.selected < 0) return;
    const b = ["boxX", "boxY", "boxW", "boxH"].map((id) => +$(id).value);
    if (
      b.some((v) => !Number.isFinite(v)) ||
      b[0] < 0 ||
      b[1] < 0 ||
      b[2] < 2 ||
      b[3] < 2 ||
      b[0] + b[2] > ui.pageImage.width ||
      b[1] + b[3] > ui.pageImage.height
    )
      throw new Error("框需要在页面内，宽高至少为 2 像素。");
    ui.boxes[ui.selected].bbox = b;
    ui.boxDirty = true;
    draw();
    renderBoxList();
  });
  function moveBox(delta) {
    if (ui.selected < 0) return;
    const current = ui.boxes[ui.selected];
    let to = ui.selected + delta;
    while (to >= 0 && to < ui.boxes.length) {
      if (
        ui.boxes[to].page === current.page &&
        ui.boxes[to].kind === current.kind
      )
        break;
      to += delta;
    }
    if (to < 0 || to >= ui.boxes.length) return;
    [ui.boxes[ui.selected], ui.boxes[to]] = [
      ui.boxes[to],
      ui.boxes[ui.selected],
    ];
    ui.selected = to;
    ui.boxDirty = true;
    renderBoxList();
    draw();
  }
  $("earlier").onclick = () => moveBox(-1);
  $("later").onclick = () => moveBox(1);
  $("mode").onchange = () => {
    ui.boxDirty = true;
    renderBoxList();
  };
  $("boxMode").onchange = () => {
    if (ui.selected < 0) return;
    ui.boxes[ui.selected].mode = $("boxMode").value;
    ui.boxDirty = true;
    renderBoxList();
  };
  $("detect").onclick = action(async () => {
    if (
      ui.boxes.length &&
      !confirm("自动检测会替换当前区域，并使后续识别结果失效，继续？")
    )
      return;
    await start(
      "/detect",
      { mode: $("mode").value, source: "auto" },
      () => {
        ui.boxDirty = ui.metadataDirty = ui.measureDirty = false;
        renderBoxList();
      },
    );
  });
  $("saveBoxes").onclick = action(async () => {
    if (!ui.boxDirty && ui.state.layout) {
      go(2);
      return;
    }
    ui.state = await api(endpoint("/boxes"), "PUT", {
      boxes: ui.boxes,
      mode: $("mode").value,
    });
    ui.boxDirty = ui.metadataDirty = ui.measureDirty = false;
    render();
    notice("区域已保存。");
    go(2);
  });

  return { loadPage, renderBoxList, updateBoxControls };
}
