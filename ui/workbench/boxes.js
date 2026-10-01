import { ui, endpoint, receiveProject } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api } from "./api.js";
const colors = { measure: "#4267c5", header: "#6189ac", title: "#6189ac", subtitle: "#6189ac", credit: "#6189ac", tuning: "#6189ac", header_text: "#6189ac", tempo: "#b07628", clef: "#7756a4", transposition: "#2468aa", annotation: "#2468aa" };
const names = { measure: "小节", header: "谱头", title: "曲名", subtitle: "副标题", credit: "署名", tuning: "调弦文字", header_text: "其他谱头文字", tempo: "速度", clef: "谱号", transposition: "标记候选", annotation: "标记候选" };
const modeNames = { tab: "TAB", notation: "五线谱", both: "五线谱 + TAB" };
function boxName(box) {
  if (["annotation", "transposition"].includes(box.kind)) {
    const [x, y, w, h] = box.bbox;
    if (ui.boxes.some(b => b.kind === "header" && b.page === box.page &&
      x >= b.bbox[0] && y >= b.bbox[1] && x+w <= b.bbox[0]+b.bbox[2] && y+h <= b.bbox[1]+b.bbox[3]))
      return "谱头文字";
  }
  return names[box.kind];
}

export function initBoxes({ start, go, render, setBusy }) {
  let undo = [], redo = [], dragStart, savedRevision, pageRequest = 0, imageTimer;
  $("boxKind").replaceChildren(...Object.entries(names).filter(([kind]) => kind !== "transposition").map(([kind, name]) => {
    const option = new Option(name, kind);
    option.hidden = kind === "header";
    return option;
  }));
  const snapshot = () => ({boxes:structuredClone(ui.boxes),selected:ui.selected});
  function remember(before = snapshot()) {
    undo.push(before); if(undo.length>50)undo.shift(); redo=[];
  }
  function restore(from,to) {
    if(ui.busy||!from.length)return;
    to.push(snapshot());const previous=from.pop();
    ui.boxes=previous.boxes;ui.selected=previous.selected;
    if(ui.boxes[ui.selected]?.page!==ui.pageIndex+1)ui.selected=-1;
    ui.boxDirty=true;renderBoxList();draw();
  }
  $("undoBox").onclick=()=>restore(undo,redo);
  $("redoBox").onclick=()=>restore(redo,undo);
  function boxMode(box) {
    return $("mode").value !== "auto"
      ? $("mode").value
      : box.mode || ui.state.pages[box.page - 1]?.notation_mode || "";
  }
  function loadPage() {
    if (!ui.state?.pages) return;
    const revision=`${ui.sid}:${ui.state.revision}`;
    if(revision!==savedRevision){undo=[];redo=[];savedRevision=revision;}
    const p = ui.state.pages[ui.pageIndex];
    $("pageLabel").textContent =
      `第 ${ui.pageIndex + 1} / ${ui.state.pages.length} 页${p.notation_mode ? `，${modeNames[p.notation_mode]}` : ""}`;
    const request = ++pageRequest;
    clearTimeout(imageTimer);
    ui.pageImage = null;
    ui.drag = null;
    $("canvas").hidden = true;
    $("boxPreview").hidden = true;
    $("canvas").setAttribute("aria-busy", "true");
    $("pageImageStatus").hidden = false;
    $("pageImageMessage").textContent = "正在加载页面图片…";
    $("retryPageImage").hidden = true;
    const img = new Image();
    const current = () => request === pageRequest && ui.state?.pages?.[ui.pageIndex] === p;
    const failed = () => {
      if (!current()) return;
      clearTimeout(imageTimer);
      img.onload = img.onerror = null;
      $("canvas").setAttribute("aria-busy", "false");
      $("pageImageMessage").textContent = "页面图片未能加载，区域修改仍保留。请检查连接后重试。";
      $("retryPageImage").hidden = false;
      updateBoxControls();
    };
    img.onload = () => {
      if (!current()) return;
      clearTimeout(imageTimer);
      ui.pageImage = img;
      $("canvas").hidden = false;
      $("canvas").setAttribute("aria-busy", "false");
      $("pageImageStatus").hidden = true;
      resizeCanvas();
      updateBoxControls();
    };
    img.onerror = failed;
    imageTimer = setTimeout(failed, 20000);
    img.src = p.url;
    renderBoxList();
  }
  $("retryPageImage").onclick = loadPage;
  function resizeCanvas() {
    if (!ui.pageImage) return;
    const fit = Math.min(
      1,
      Math.max(200, $("canvasScroll").clientWidth - 32) / ui.pageImage.width,
    );
    ui.scale = fit * (+$("zoom").value / 100);
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
        box.kind === "measure" ? `小节 ${numberOf(i)}` : boxName(box);
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
    const selected=ui.boxes[ui.selected];
    $("boxPreview").hidden=!selected;
    if(selected && selected.page===ui.pageIndex+1){
      const [x,y,w,h]=selected.bbox;
      if(w>1&&h>1){
        const preview=$("cropPreview"),scale=Math.min(1,360/w);
        preview.width=Math.max(1,Math.round(w*scale));preview.height=Math.max(1,Math.round(h*scale));
        preview.getContext("2d").drawImage(ui.pageImage,x,y,w,h,0,0,preview.width,preview.height);
      }
    }
  }
  function renderBoxList() {
    const list = $("boxList");
    list.replaceChildren();
    ui.boxes.forEach((box, i) => {
      if (box.page !== ui.pageIndex + 1) return;
      const b = el(
        "button",
        box.kind === "measure" ? `小节 ${numberOf(i)}，${modeNames[boxMode(box)] || "待识别"}` : boxName(box),
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
    if (selected) $("boxKind").value = selected.kind === "transposition" ? "annotation" : selected.kind;
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
    $("applyCoords").disabled = ui.busy || !selected || !ui.pageImage;
    $("discardBoxes").hidden = !ui.boxDirty;
    $("discardBoxes").disabled = ui.busy;
    $("saveBoxes").disabled = ui.busy || !ui.boxes.some(box => box.kind === "measure");
    $("deleteBox").disabled = ui.busy || !selected;
    $("undoBox").disabled = ui.busy || !undo.length;
    $("redoBox").disabled = ui.busy || !redo.length;
    $("boxKind").disabled = ui.busy;
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
    $("canvas").focus({preventScroll:true});
    dragStart={...snapshot(),dirty:ui.boxDirty};
    const [x, y] = point(e);
    const tool = $("boxTool").value;
    let hit = -1,
      corner = -1;
    if (tool === "select") {
      if (ui.selected >= 0 && ui.boxes[ui.selected].page===ui.pageIndex+1) {
        const [bx, by, w, h] = ui.boxes[ui.selected].bbox;
        [
          [bx, by],
          [bx + w, by],
          [bx, by + h],
          [bx + w, by + h],
        ].forEach(([cx, cy], i) => {
          if (Math.hypot(x - cx, y - cy) < 14 / ui.scale) {
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
    const selected=ui.boxes[ui.selected];
    ui.boxes.sort((a, b) => a.page - b.page);
    ui.drag = null;
    ui.selected = selected?ui.boxes.indexOf(selected):-1;
    if(JSON.stringify(dragStart.boxes)!==JSON.stringify(ui.boxes)){
      remember(dragStart);ui.boxDirty=true;
    }else ui.boxDirty=dragStart.dirty;
    $("boxTool").value="select";
    renderBoxList();
    draw();
  }
  $("canvas").onpointerup = endDrag;
  $("canvas").onpointercancel = () => {
    if(!ui.drag)return;
    ui.boxes=dragStart.boxes;ui.selected=dragStart.selected;ui.boxDirty=dragStart.dirty;
    ui.drag=null;renderBoxList();draw();
  };
  $("deleteBox").onclick = () => {
    if (ui.selected < 0) return;
    remember();
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
    remember();ui.boxes[ui.selected].bbox = b;
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
    remember();
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
  $("boxKind").onchange = () => {
    if (ui.busy || ui.selected < 0) return;
    remember();
    ui.boxes[ui.selected].kind = $("boxKind").value;
    ui.boxDirty = true;
    renderBoxList();
    draw();
  };
  $("boxMode").onchange = () => {
    if (ui.selected < 0) return;
    remember();
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
  $("discardBoxes").onclick = () => {
    if (ui.busy || !confirm("放弃未保存的区域修改，恢复已保存的区域？")) return;
    ui.boxes = structuredClone(ui.state.boxes || []);
    ui.selected = -1;
    ui.boxDirty = false;
    undo = []; redo = [];
    $("mode").value = ui.state.mode_setting || ui.state.mode;
    renderBoxList(); draw();
    notice("已恢复保存的区域。");
  };
  $("saveBoxes").onclick = action(async () => {
    if (!ui.boxDirty && ui.state.layout) {
      go(2);
      return;
    }
    if(ui.state.recognition && !confirm("保存区域修改后需要重新读取谱面信息和小节，现有识别及校对结果会失效。继续？"))return;
    const navigation = ui.navigation;
    setBusy(true);
    try {
      const saved = await api(endpoint("/boxes"), "PUT", {
        boxes: ui.boxes,
        mode: $("mode").value,
      }, ui.state.revision);
      receiveProject(saved);
      render();
    } finally {
      setBusy(false);
    }
    go(2, ui.navigation === navigation);
    if (ui.navigation === navigation) notice("区域已保存，请核对谱面信息。");
  });
  $("canvas").onkeydown=e=>{
    if(ui.busy)return;
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){
      e.preventDefault();restore(e.shiftKey?redo:undo,e.shiftKey?undo:redo);
    }else if(e.key==="Delete"||e.key==="Backspace"){
      e.preventDefault();$("deleteBox").click();
    }
  };

  return { loadPage, renderBoxList, updateBoxControls };
}
