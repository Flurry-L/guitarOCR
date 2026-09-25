import { ui } from "./state.js";
import { $, el, notice, action } from "./dom.js";
import { api, endpoint } from "./api.js";
import { initPages } from "./pages.js";
import { initBoxes } from "./boxes.js";
import { initMeasures } from "./measure-editor.js";
const { renderPages, updatePageNavigation } = initPages(() => loadPage());
const { loadPage, renderBoxList } = initBoxes({ start, go, render });
const { renderMeasures } = initMeasures({ start, go, renderExport });
function setBusy(value) {
  ui.busy = value;
  document
    .querySelectorAll("button,input,select,textarea")
    .forEach((n) => (n.disabled = value));
  $("notice").classList.toggle("busy", value);
  $("cancelJob").hidden = !value;
  $("cancelJob").disabled = !value;
  updatePageNavigation();
}
function go(next) {
  if (next > 0 && ui.files.length)
    return notice("已选择新文件，请先点击「导入乐谱」，完成后再继续。", true);
  if (next > 0 && !ui.state?.pages) return notice("请先导入乐谱。", true);
  if (next > 1 && !ui.state.layout) return notice("请先保存页面区域。", true);
  if (next > 2 && !ui.state.info) return notice("请先保存谱面信息。", true);
  if (next > 3 && !ui.state.recognition) return notice("请先识别小节。", true);
  if (
    ((ui.step === 1 && ui.boxDirty) ||
      (ui.step === 2 && ui.metadataDirty) ||
      (ui.step === 3 && ui.measureDirty)) &&
    next !== ui.step
  ) {
    notice("当前修改尚未保存，请先保存再切换步骤。", true);
    return;
  }
  ui.step = next;
  document
    .querySelectorAll("[data-panel]")
    .forEach((p) => p.classList.toggle("active", +p.dataset.panel === ui.step));
  document
    .querySelectorAll("[data-step]")
    .forEach((p) => p.classList.toggle("active", +p.dataset.step === ui.step));
  if (ui.step === 1) loadPage();
  if (ui.step === 4) renderExport();
}
$("steps").onclick = (e) => {
  const button = e.target.closest("[data-step]");
  if (button) go(+button.dataset.step);
};
function renderFiles() {
  const list = $("fileList");
  list.replaceChildren();
  ui.files.forEach((f, i) => {
    const row = el("div", undefined, "file-row");
    row.append(
      el("span", String(i + 1).padStart(2, "0"), "muted"),
      el("span", f.name, "name"),
      el("span", `${(f.size / 1024 / 1024).toFixed(1)} MB`, "muted"),
    );
    for (const [label, delta] of [
      ["↑", -1],
      ["↓", 1],
      ["×", 0],
    ]) {
      const b = el("button", label, "quiet");
      b.type = "button";
      b.setAttribute(
        "aria-label",
        `${label === "×" ? "移除" : label === "↑" ? "提前" : "延后"} ${f.name}`,
      );
      b.onclick = () => {
        if (delta && ui.files[i + delta])
          [ui.files[i], ui.files[i + delta]] = [
            ui.files[i + delta],
            ui.files[i],
          ];
        else if (!delta) ui.files.splice(i, 1);
        renderFiles();
      };
      row.append(b);
    }
    list.append(row);
  });
}
function chooseFiles(incoming) {
  if (!incoming.length || ui.busy) return;
  if (
    (ui.boxDirty || ui.metadataDirty || ui.measureDirty) &&
    !confirm("当前编辑尚未保存，仍要导入新乐谱？")
  )
    return;
  ui.boxDirty = ui.metadataDirty = ui.measureDirty = false;
  ui.files.push(...incoming);
  renderFiles();
  go(0);
  $("currentDocument").hidden = true;
  notice(
    `已选择 ${ui.files.length} 个文件。点击「导入乐谱」上传并展开全部 PDF 页面。`,
  );
}
$("files").onchange = (e) => {
  chooseFiles([...e.target.files]);
  e.target.value = "";
};
for (const event of ["dragover", "dragleave", "drop"])
  document.addEventListener(event, (e) => {
    if (!Array.from(e.dataTransfer?.types || []).includes("Files")) return;
    e.preventDefault();
    $("dropzone").classList.toggle("drag", event === "dragover");
    if (event === "drop") chooseFiles([...e.dataTransfer.files]);
  });
async function watch(after) {
  setBusy(true);
  try {
    for (;;) {
      const current = await api(endpoint(""));
      const job = current.job;
      if (job && ["queued", "running"].includes(job.status)) {
        if (current.pages) ui.state = current;
        $("cancelJob").hidden = !job.cancellable;
        notice((job.status === "queued" ? "排队中 · " : "") + job.message);
        await new Promise((resolve) => setTimeout(resolve, 1200));
        continue;
      }
      if (!current.pages && !["queued", "running"].includes(job?.status)) {
        throw new Error(job?.error || "页面导入未完成，请重新上传乐谱。");
      }
      if (current.pages) {
        ui.state = current;
        render();
      }
      if (after && current.pages) after();
      if (job?.status === "failed") throw new Error(job.error);
      notice(
        ["cancelled", "interrupted"].includes(job?.status)
          ? job.message
          : "已完成并保存。",
      );
      break;
    }
  } finally {
    setBusy(false);
  }
}
$("cancelJob").onclick = action(async () => {
  await api(endpoint("/cancel"), "POST");
  notice("正在停止，当前小节处理结束后生效。已完成部分会保留。");
});
async function start(path, body, after) {
  if (ui.busy) return;
  setBusy(true);
  try {
    await api(endpoint(path), "POST", body);
    await watch(after);
  } finally {
    setBusy(false);
  }
}
$("upload").onclick = action(async () => {
  if (!ui.files.length) throw new Error("请先选择 PDF 或图片。");
  const form = new FormData();
  ui.files.forEach((f) => form.append("files", f));
  setBusy(true);
  notice("正在上传乐谱…");
  try {
    const result = await api("/api/sessions", "POST", form);
    ui.sid = result.id;
    ui.state = null;
    localStorage.setItem("guitarocr-session", ui.sid);
    ui.boxDirty = ui.measureDirty = ui.metadataDirty = false;
    ui.pageIndex = ui.measureIndex = 0;
    ui.files = [];
    renderFiles();
    await watch(() => go(1));
    notice(
      `导入成功：${documentName()}，共 ${ui.state.pages.length} 页。当前显示第 1 页，可用翻页按钮或左侧缩略图查看其他页。`,
    );
  } finally {
    setBusy(false);
  }
});
$("importProject").onchange = action(async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (
    (ui.boxDirty || ui.measureDirty || ui.metadataDirty) &&
    !confirm("放弃未保存的编辑，恢复项目？")
  )
    return;
  const form = new FormData();
  form.append("file", file);
  setBusy(true);
  try {
    ui.state = await api("/api/projects/import", "POST", form);
    ui.sid = ui.state.id;
    ui.files = [];
    ui.boxDirty = ui.measureDirty = ui.metadataDirty = false;
    ui.pageIndex = ui.measureIndex = 0;
    renderFiles();
    render();
    go(ui.state.recognition ? 3 : 1);
    notice("项目已恢复。原始页面、已保存的编辑和导出结果均已载入。");
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});
$("newProject").onclick = () => {
  if (
    (ui.boxDirty || ui.measureDirty || ui.metadataDirty) &&
    !confirm("当前有未保存的编辑，仍然新建项目？")
  )
    return;
  localStorage.removeItem("guitarocr-session");
  location.href = "/";
};
$("deleteProject").onclick = action(async () => {
  if (!confirm("删除当前项目的上传文件、识别结果和 GP5？此操作无法撤销。"))
    return;
  await api(endpoint(""), "DELETE");
  localStorage.removeItem("guitarocr-session");
  location.href = "/";
});
function documentName() {
  return (
    ui.state.input_names || ui.state.inputs.map((p) => p.split(/[\\/]/).pop())
  ).join("、");
}
function render() {
  localStorage.setItem("guitarocr-session", ui.sid);
  history.replaceState(null, "", `?project=${ui.sid}`);
  $("currentDocument").hidden = false;
  $("documentName").textContent = documentName();
  $("documentPages").textContent = ` · 已导入 ${ui.state.pages.length} 页`;
  $("documentId").textContent = ui.sid.slice(0, 8);
  ui.boxes = structuredClone(ui.state.boxes || []);
  ui.pageIndex = Math.min(ui.pageIndex, ui.state.pages.length - 1);
  ui.selected = -1;
  $("mode").value = ui.state.mode;
  $("projectId").textContent =
    `项目 ${ui.sid.slice(0, 8)} · ${ui.state.pages.length} 页`;
  $("deleteProject").hidden = false;
  renderPages();
  renderMetadata();
  renderMeasures();
  renderExport();
  if (ui.step === 1) loadPage();
}
function renderMetadata() {
  const m = ui.state.metadata;
  $("title").value = m?.title || "";
  $("artist").value = m?.artist || "";
  $("tempo").value = m?.document_metadata?.tempo_quarter || 120;
  $("capo").value = m?.capo || 0;
  $("tuning").value = (m?.tuning_used || [64, 59, 55, 50, 45, 40]).join(",");
  const preset = Array.from($("tuningPreset").options).some(
    (o) => o.value === $("tuning").value,
  );
  $("tuningPreset").value = preset ? $("tuning").value : "custom";
  const warnings = m?.document_metadata?.warnings || [];
  $("infoWarnings").hidden = !warnings.length;
  $("infoWarnings").textContent = warnings.join("\n");
}
$("tuningPreset").onchange = () => {
  if ($("tuningPreset").value !== "custom")
    $("tuning").value = $("tuningPreset").value;
  ui.metadataDirty = true;
};
for (const id of ["title", "artist", "tempo", "capo", "tuning"])
  $(id).oninput = () => {
    ui.metadataDirty = true;
  };
$("readInfo").onclick = action(async () => {
  if (
    (ui.metadataDirty || ui.state.info) &&
    !confirm("自动读取会替换当前信息；调弦改变时需重新识别小节，继续？")
  )
    return;
  await start("/information", undefined, () => {
    ui.metadataDirty = ui.measureDirty = false;
  });
});
$("saveInfo").onclick = action(async () => {
  if (!ui.metadataDirty && ui.state.info) {
    go(3);
    return;
  }
  const tuning = $("tuning")
    .value.split(",")
    .map((v) => Number(v.trim()));
  ui.state = await api(endpoint("/metadata"), "PUT", {
    title: $("title").value || "未命名乐谱",
    artist: $("artist").value,
    tempo_quarter: +$("tempo").value,
    capo: +$("capo").value,
    tuning_used: tuning,
  });
  ui.metadataDirty = ui.measureDirty = false;
  render();
  notice("谱面信息已保存。");
  go(3);
});
function renderExport() {
  if (!ui.state) return;
  const count = ui.state.measures?.length || 0,
    review = ui.state.review_measures?.length || 0;
  $("exportSummary").replaceChildren();
  for (const [value, label] of [
    [ui.state.pages.length, "页乐谱"],
    [count, "个小节"],
    [review, "个待检查"],
  ]) {
    const stat = el("div", undefined, "stat");
    stat.append(el("strong", value), el("span", label));
    $("exportSummary").append(stat);
  }
  $("downloadProject").hidden = false;
  $("downloadProject").href = endpoint("/archive");
  $("downloadM2").hidden = !ui.state.m2_url;
  if (ui.state.m2_url) {
    $("downloadM2").href = ui.state.m2_url;
    $("downloadM2").download = "prediction.m2";
  }
  $("downloadGP5").hidden = !ui.state.gp5_url;
  $("export").textContent = ui.state.gp5_url ? "重新生成 GP5" : "生成 GP5";
  if (ui.state.gp5_url) {
    $("downloadGP5").href = ui.state.gp5_url;
    $("downloadGP5").download = "score.gp5";
  }
  $("exportWarning").hidden = !review;
  $("exportWarning").textContent = review
    ? `还有 ${review} 个小节需要检查：${ui.state.review_measures.join("、")}。请回到「校对小节」保存确认。`
    : "";
}
$("export").onclick = action(async () => {
  ui.state = await api(endpoint("/export"), "POST");
  renderExport();
  notice("GP5 已生成，可以下载。");
  if (ui.state.encoding_url) {
    const report = await api(ui.state.encoding_url);
    if (report.replacements?.length) {
      $("exportWarning").hidden = false;
      $("exportWarning").textContent =
        "部分文字无法写入 GP5 的字符编码，小节文本保留了原文。";
    }
  }
});
window.addEventListener("beforeunload", (e) => {
  if (ui.boxDirty || ui.metadataDirty || ui.measureDirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
async function renderRecent() {
  const projects = await api("/api/sessions");
  $("recentProjects").hidden = !projects.length;
  $("projectList").replaceChildren();
  projects.forEach((project) => {
    const button = el(
      "button",
      `${project.title} · ${project.pages} 页 · ${project.stage} · ${project.id.slice(0, 8)}`,
    );
    button.onclick = action(async () => {
      if (
        (ui.boxDirty || ui.metadataDirty || ui.measureDirty) &&
        !confirm("放弃当前未保存的修改，打开这个项目？")
      )
        return;
      ui.sid = project.id;
      localStorage.setItem("guitarocr-session", ui.sid);
      ui.boxDirty = ui.metadataDirty = ui.measureDirty = false;
      ui.files = [];
      renderFiles();
      ui.pageIndex = ui.measureIndex = 0;
      await watch(() => go(ui.state.recognition ? 3 : 1));
      notice(
        `已打开已有项目：${documentName()} · ${ui.state.pages.length} 页。上传新 PDF 请点击右上角「上传新乐谱」。`,
      );
    });
    $("projectList").append(button);
  });
}
(async () => {
  try {
    await renderRecent();
    if (ui.sid) {
      await watch(() => {
        go(ui.state.recognition ? 3 : ui.state.info ? 3 : 1);
      });
      notice(
        `已恢复上次项目：${documentName()} · ${ui.state.pages.length} 页。上传新 PDF 请点击右上角「上传新乐谱」。`,
      );
    }
    const config = await api("/api/config");
    if (!config.model_ready)
      notice(
        "识别环境尚未就绪，请运行 install.bat（Windows）或 bash install.sh（Linux）完成自动安装。",
      );
  } catch (e) {
    setBusy(false);
    notice(e.message, true);
  }
})();
