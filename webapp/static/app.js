import {
  ui,
  endpoint,
  hasDrafts,
  clearDrafts,
  openProject,
  receiveProject,
} from "./state.js";
import { $, el, notice, action } from "./dom.js";
import { api } from "./api.js";
import { initWorkspace } from "./workspace.js";
import { initMetadata } from "./metadata-editor.js";
import { initPages } from "./pages.js";
import { initBoxes } from "./boxes.js";
import { initMeasures } from "./measure-editor.js";
const { renderPages, updatePageNavigation } = initPages(() => loadPage());
const { loadPage, renderBoxList, updateBoxControls } = initBoxes({
  start,
  go,
  render,
  setBusy,
});
const { renderMeasures, updateMeasureControls, openIssue } = initMeasures({
  start,
  go,
  renderExport,
  setBusy,
});
const { renderMetadata } = initMetadata({ start, go, render, setBusy });
openProject(
  new URLSearchParams(location.search).get("project") ||
    (location.pathname === "/workbench"
      ? null
      : localStorage.getItem("guitarocr-session")),
);

const workspace = initWorkspace({
  open: openExisting,
  resume: () =>
    start(
      location.pathname === "/workbench" ? "/retry" : "/process",
      undefined,
      () => go(3),
    ),
});
let uploadConfig = { max_upload_mb: 200, max_pages: 100 };

async function openExisting(id) {
  if (ui.busy) return false;
  if (hasDrafts() && !confirm("放弃当前未保存的修改，打开这个项目？"))
    return false;
  openProject(id);
  renderFiles();
  await watch(() =>
    go(ui.state.recognition || ui.state.info ? 3 : ui.state.layout ? 2 : 1),
  );
  return true;
}

function setBusy(value) {
  ui.busy = value;
  document
    .querySelectorAll(
      "[data-panel] button,[data-panel] input,[data-panel] select,[data-panel] textarea,[data-step],#newProject,#deleteProject",
    )
    .forEach((n) => (n.disabled = value));
  $("notice").classList.toggle("busy", value);
  workspace.renderJob(ui.state?.job);
  renderFiles();
  updatePageNavigation();
  updateMeasureControls();
  updateBoxControls();
  renderExport();
  document.querySelectorAll(".project-card").forEach((node) => {
    node.disabled = value;
  });
}
function go(next) {
  if (next > 0 && ui.files.length)
    return notice("请先处理已选择的文件，再切换到编辑步骤。", true);
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
  if (!ui.busy) notice("");
  ui.step = next;
  workspace.show("workbench");
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
  $("upload").disabled = ui.busy || !ui.files.length;
  $("upload").textContent =
    $("importAction").value === "full" ? "开始识别" : "导入并检查区域";
  $("importMode").disabled = ui.busy || $("importAction").value !== "full";
  $("actionDescription").textContent =
    $("importAction").value === "full"
      ? "自动完成页面展开、小节检测、谱面信息和音符识别。"
      : "导入后先检查小节位置与阅读顺序，再继续识别。";
  $("importAction").querySelector('[value="full"]').disabled =
    uploadConfig.inference_enabled === false;
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
      b.disabled = ui.busy || (!!delta && !ui.files[i + delta]);
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
  if (hasDrafts() && !confirm("当前编辑尚未保存，仍要导入新乐谱？")) return;
  const invalid = incoming.find(
    (file) => !/\.(pdf|png|jpe?g|bmp|tiff?)$/i.test(file.name),
  );
  if (invalid)
    return notice(`不支持 ${invalid.name}，请选择 PDF 或乐谱图片。`, true);
  const combined = [...ui.files, ...incoming];
  if (combined.length > uploadConfig.max_pages)
    return notice(`最多选择 ${uploadConfig.max_pages} 个文件。`, true);
  if (
    combined.reduce((sum, file) => sum + file.size, 0) >
    uploadConfig.max_upload_mb * 1024 ** 2
  )
    return notice(
      `总上传大小不能超过 ${uploadConfig.max_upload_mb} MB。`,
      true,
    );
  clearDrafts();
  ui.files.push(...incoming);
  renderFiles();
  go(0);
  $("currentDocument").hidden = true;
  notice(`已选择 ${ui.files.length} 个文件，可调整顺序后开始处理。`);
}
$("importAction").onchange = renderFiles;
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
      workspace.renderJob(job);
      if (job && ["queued", "running"].includes(job.status)) {
        if (current.pages) ui.state = current;
        $("cancelJob").hidden = !job.cancellable;
        notice("");
        await new Promise((resolve) => setTimeout(resolve, 1200));
        continue;
      }
      if (!current.pages && !["queued", "running"].includes(job?.status)) {
        throw new Error(job?.error || "页面导入未完成，请重新上传乐谱。");
      }
      if (current.pages) {
        receiveProject(current);
        render();
      }
      if (after && current.pages) {
        if (["failed", "cancelled", "interrupted"].includes(job?.status))
          go(current.recognition || current.info ? 3 : current.layout ? 2 : 1);
        else after();
      }
      notice(
        job?.status === "failed"
          ? job.error
          : ["cancelled", "interrupted"].includes(job?.status)
            ? job.message
            : "已完成并保存。",
        job?.status === "failed",
      );
      break;
    }
  } finally {
    setBusy(false);
    workspace.refresh();
  }
}
$("cancelJob").onclick = action(async () => {
  await api(endpoint("/cancel"), "POST", undefined, ui.state?.revision);
  window.dispatchEvent(new Event("guitarocr:cancel"));
  notice("正在停止，当前小节处理结束后生效。已完成部分会保留。");
});
async function start(path, body, after) {
  if (ui.busy) return;
  setBusy(true);
  try {
    await api(endpoint(path), "POST", body, ui.state.revision);
    await watch(after);
  } finally {
    setBusy(false);
  }
}
$("upload").onclick = action(async () => {
  if (!ui.files.length) throw new Error("请先选择 PDF 或图片。");
  const form = new FormData();
  ui.files.forEach((f) => form.append("files", f));
  const automatic = $("importAction").value === "full";
  form.append("action", automatic ? "full" : "import");
  form.append("mode", $("importMode").value);
  setBusy(true);
  notice("正在上传乐谱…");
  try {
    const result = await api("/api/sessions", "POST", form);
    openProject(result.id);
    localStorage.setItem("guitarocr-session", ui.sid);
    renderFiles();
    await watch(() => go(automatic ? 3 : 1));
  } finally {
    setBusy(false);
  }
});
$("importProject").onchange = action(async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (hasDrafts() && !confirm("放弃未保存的编辑，恢复项目？")) return;
  const form = new FormData();
  form.append("file", file);
  setBusy(true);
  try {
    receiveProject(await api("/api/projects/import", "POST", form));
    renderFiles();
    render();
    go(ui.state.recognition ? 3 : 1);
    notice("项目已恢复。");
    workspace.refresh();
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});
$("newProject").onclick = () => {
  if (hasDrafts() && !confirm("当前有未保存的编辑，仍然新建项目？")) return;
  localStorage.removeItem("guitarocr-session");
  location.href = "/";
};
$("deleteProject").onclick = action(async () => {
  if (!confirm("删除当前项目的上传文件、识别结果和 GP5？此操作无法撤销。"))
    return;
  await api(endpoint(""), "DELETE", undefined, ui.state.revision);
  localStorage.removeItem("guitarocr-session");
  location.href = "/";
});
function documentName() {
  return (
    ui.state.input_names ||
    (ui.state.inputs || []).map((p) => p.split(/[\\/]/).pop())
  ).join("、");
}
function render() {
  localStorage.setItem("guitarocr-session", ui.sid);
  history.replaceState(null, "", `?project=${ui.sid}`);
  $("currentDocument").hidden = false;
  $("documentName").textContent = documentName();
  $("documentPages").textContent = `，共 ${ui.state.pages.length} 页`;
  $("intro").hidden = true;
  $("steps").hidden = false;
  document.querySelectorAll("[data-step]").forEach((node) => {
    const n = +node.dataset.step;
    node.classList.toggle(
      "done",
      [
        true,
        ui.state.layout,
        ui.state.info,
        ui.state.recognition,
        ui.state.export,
      ][n],
    );
    node.disabled = ui.busy;
  });
  $("newProject").hidden = false;
  $("mode").value = ui.state.mode_setting || ui.state.mode;
  $("deleteProject").hidden = false;
  renderPages();
  renderMetadata();
  renderMeasures();
  renderExport();
  if (ui.step === 1) loadPage();
}
function renderExport() {
  if (!ui.state) return;
  const measures = ui.state.measures || [];
  const count = new Set(measures.map((m, i) => m.bar_index ?? i)).size,
    parts = new Set(measures.map((m) => m.part_id || "part-1")).size,
    review = ui.state.review_measures?.length || 0;
  $("exportSummary").replaceChildren();
  for (const [value, label] of [
    [ui.state.pages.length, "页乐谱"],
    [count, "个小节"],
    ...(parts > 1 ? [[parts, "条音轨"]] : []),
    [review, "个待检查"],
  ]) {
    const stat = el("div", undefined, "stat");
    stat.append(el("strong", value), el("span", label));
    $("exportSummary").append(stat);
  }
  $("downloadProject").hidden = false;
  $("downloadProject").href = endpoint("/archive");
  $("downloadScoreText").hidden = !ui.state.score_text_url;
  $("downloadScoreDocument").hidden = !ui.state.score_document_url;
  $("downloadMusicXML").hidden = !ui.state.musicxml_url;
  if (ui.state.musicxml_url) {
    $("downloadMusicXML").href = ui.state.musicxml_url;
    $("downloadMusicXML").download = "score.musicxml";
  }
  if (ui.state.score_document_url) {
    $("downloadScoreDocument").href = ui.state.score_document_url;
    $("downloadScoreDocument").download = "score.json";
  }
  if (ui.state.score_text_url) {
    $("downloadScoreText").href = ui.state.score_text_url;
    $("downloadScoreText").download = "score.txt";
  }
  $("downloadGP5").hidden = !ui.state.gp5_url;
  $("export").hidden = !!ui.state.gp5_url;
  $("export").disabled = ui.busy || !count || !!review;
  if (ui.state.gp5_url) {
    $("downloadGP5").href = ui.state.gp5_url;
    $("downloadGP5").download = ui.state.gp5_name;
  }
  $("exportWarning").hidden = !review;
  $("reviewBeforeExport").hidden = !review;
  $("exportWarning").textContent = review
    ? `还有 ${review} 个小节待检查，确认后即可导出。`
    : "";
}
$("reviewBeforeExport").onclick = () => {
  go(3);
  openIssue();
};
$("export").onclick = action(async () => {
  setBusy(true);
  try {
    receiveProject(
      await api(endpoint("/export"), "POST", undefined, ui.state.revision),
    );
  } finally {
    setBusy(false);
  }
  renderExport();
  if (!ui.state.gp5_url) {
    notice("部分小节需要检查，请核对延音、音高和时值。");
    go(3);
    openIssue();
    return;
  }
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
  if (hasDrafts()) {
    e.preventDefault();
    e.returnValue = "";
  }
});
(async () => {
  try {
    renderFiles();
    $("steps").hidden = !ui.sid;
    const config = await api("/api/config");
    uploadConfig = config;
    workspace.configure(config);
    renderFiles();
    await workspace.refresh();
    if (ui.sid) {
      await watch(() => {
        go(ui.state.recognition ? 3 : ui.state.info ? 3 : 1);
      });
      if (
        !["failed", "cancelled", "interrupted"].includes(ui.state?.job?.status)
      )
        notice("");
    }
    if (config.inference_enabled === false) {
      document.body.dataset.inference = "disabled";
      if (!ui.state?.pages)
        notice("当前为校对模式。可恢复项目备份、编辑并导出乐谱。");
    } else if (!config.model_ready)
      notice(
        "识别环境尚未就绪，请运行 install.bat（Windows）或 bash install.sh（Linux）完成自动安装。",
      );
  } catch (e) {
    setBusy(false);
    notice(e.message, true);
  }
})();
