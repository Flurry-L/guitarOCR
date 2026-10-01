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
import { modelOptions } from "./model-setup.js";
import { initWorkspace } from "./workspace.js";
import { initMetadata } from "./metadata-editor.js";
import { initPages } from "./pages.js";
import { initBoxes } from "./boxes.js";
import { initMeasures } from "./measure-editor.js";
import { createProjectImporter, readSample } from "./project-import.js";
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
  refreshConfig,
  resume: () =>
    start(
      "/process",
      undefined,
      () => go(3),
    ),
});
let uploadConfig = { max_upload_mb: 200, max_pages: 100 };
async function refreshConfig() {
  const config = await api("/api/config");
  uploadConfig = config;
  workspace.configure(config);
  document.body.dataset.inference = config.inference_enabled === false ? "disabled" : "enabled";
  renderFiles();
  return config;
}
function nativeModelOptions(body) {
  return modelOptions(uploadConfig, body);
}


async function openExisting(id) {
  if (ui.busy && !["queued", "running"].includes(ui.state?.job?.status)) return false;
  if (hasDrafts() && !confirm("放弃当前未保存的修改，打开这个项目？"))
    return false;
  window.dispatchEvent(new Event("guitarocr:leave-project"));
  openProject(id, { discardDrafts: hasDrafts() });
  ui.step = 0;
  document.querySelectorAll("[data-panel]").forEach((panel) =>
    panel.classList.toggle("active", panel.dataset.panel === "0"));
  $("currentDocument").hidden = true;
  $("steps").hidden = true;
  $("reviewEditor").hidden = true;
  workspace.show("workbench");
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
    node.disabled = value && !["queued", "running"].includes(ui.state?.job?.status);
  });
}
function go(next, reveal = true) {
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
  if (reveal) workspace.show("workbench");
  document
    .querySelectorAll("[data-panel]")
    .forEach((p) => p.classList.toggle("active", +p.dataset.panel === ui.step));
  document
    .querySelectorAll("[data-step]")
    .forEach((p) => p.classList.toggle("active", +p.dataset.step === ui.step));
  if (ui.step === 1) loadPage();
  if (ui.step === 4) renderExport();
}
document.addEventListener("click", (e) => {
  const button = e.target.closest("[data-step]");
  if (button) {
    go(+button.dataset.step);
    button.closest("details")?.removeAttribute("open");
  }
});
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
  if (uploadConfig.native && uploadConfig.pdf_enabled === false && incoming.some(file => /\.pdf$/i.test(file.name)))
    return notice("当前安装包缺少 PDF 组件。请使用完整原生安装包，或先导入乐谱图片、恢复项目 ZIP。", true);
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
  sessionStorage.removeItem(`guitarocr-draft:${ui.sid}`);
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
  const sid = ui.sid;
  const generation = ui.openGeneration;
  const screen = workspace.screen();
  setBusy(true);
  try {
    for (;;) {
      const current = await api(`/api/sessions/${sid}`);
      if (ui.sid !== sid || ui.openGeneration !== generation) return;
      const job = current.job;
      workspace.renderJob(job);
      if (job && ["queued", "running"].includes(job.status)) {
        if (current.pages) {
          if (!ui.state) ui.state = current;
          else ui.state = { ...ui.state, job, revision: current.revision };
        }
        $("cancelJob").hidden = !job.cancellable;
        notice("");
        await new Promise((resolve) => setTimeout(resolve, 1200));
        continue;
      }
      if (!current.pages && !["queued", "running"].includes(job?.status)) {
        throw new Error(job?.error || "页面导入未完成，请重新上传乐谱。");
      }
      if (current.pages) {
        const failed = ["failed", "cancelled", "interrupted"].includes(job?.status);
        receiveProject(current, { preserveDrafts: failed });
        render();
      }
      if (after && current.pages && workspace.screen() === screen) {
        if (["failed", "cancelled", "interrupted"].includes(job?.status))
          go(current.recognition || current.info ? 3 : current.layout ? 2 : 1);
        else after();
      } else if (after && current.pages) {
        go(current.recognition || current.info ? 3 : current.layout ? 2 : 1, false);
      }
      notice(
        job?.status === "failed"
          ? job.error
          : ["cancelled", "interrupted"].includes(job?.status)
            ? job.message
            : "已完成并保存。",
        job?.status === "failed",
      );
      // Model preparation/device selection can finish even when a later OCR step fails.
      if (uploadConfig.native) await refreshConfig().catch(() => {});
      break;
    }
  } finally {
    if (ui.sid === sid && ui.openGeneration === generation) setBusy(false);
    workspace.refresh();
  }
}
$("cancelJob").onclick = action(async () => {
  const sid = ui.sid;
  const result = await api(endpoint("/cancel"), "POST", undefined, ui.state?.revision);
  const status = result.job || (await api(`/api/sessions/${sid}`)).job;
  if (ui.sid !== sid) return;
  window.dispatchEvent(new Event("guitarocr:cancel"));
  notice(status?.status === "cancelled"
    ? "已停止排队。原有结果与人工修改已保留。"
    : "正在停止，当前小节处理结束后生效。原有结果与人工修改会保留。");
});
async function start(path, body, after) {
  if (ui.busy) return;
  body = nativeModelOptions(body);
  const sid = ui.sid;
  const generation = ui.openGeneration;
  setBusy(true);
  try {
    await api(`/api/sessions/${sid}${path}`, "POST", body, ui.state.revision);
    if (ui.sid === sid && ui.openGeneration === generation) await watch(after);
  } finally {
    if (ui.sid === sid && ui.openGeneration === generation) setBusy(false);
  }
}
$("upload").onclick = action(async () => {
  if (!ui.files.length) throw new Error("请先选择 PDF 或图片。");
  const form = new FormData();
  ui.files.forEach((f) => form.append("files", f));
  const automatic = $("importAction").value === "full";
  if (automatic && uploadConfig.native) {
    const options = nativeModelOptions({});
    form.append("allow_download", String(options.allow_download));
  }
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
const importProject = createProjectImporter({
  ui, hasDrafts, confirm, setBusy, notice,
  navigation: () => workspace.navigation(),
  importArchive: (file) => {
    const form = new FormData();
    form.append("file", file);
    return api("/api/projects/import", "POST", form);
  },
  accept: (saved) => {
    window.dispatchEvent(new Event("guitarocr:leave-project"));
    openProject(saved.id, { discardDrafts: true });
    receiveProject(saved);
    renderFiles();
    render();
    go(ui.state.recognition ? 3 : 1);
  },
  refresh: () => workspace.refresh(),
});
$("importProject").onchange = async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  await importProject(() => file, {
    prompt: "恢复项目后将替换当前未保存的编辑和已选文件，继续？",
    loading: "正在恢复项目…",
    opened: "项目已恢复。",
    background: "项目已恢复到项目库，可稍后打开。当前编辑与已选文件已保留。",
  });
};
$("openSample").onclick = () => importProject(readSample, {
  prompt: "打开示例后将替换当前未保存的编辑和已选文件，继续？",
  loading: "正在打开原创示例…",
  opened: "已打开原创预设示例（非 OCR 结果）。可编辑音符、试听，检查后导出。",
  background: "原创示例已加入项目库，可稍后打开。当前编辑与已选文件已保留。",
});
$("newProject").onclick = () => {
  if (hasDrafts() && !confirm("当前有未保存的编辑，仍然新建项目？")) return;
  sessionStorage.removeItem(`guitarocr-draft:${ui.sid}`);
  localStorage.removeItem("guitarocr-session");
  location.href = "/";
};
$("deleteProject").onclick = action(async () => {
  if (!confirm("删除当前项目的上传文件、识别结果和 GP5？此操作无法撤销。"))
    return;
  await api(endpoint(""), "DELETE", undefined, ui.state.revision);
  sessionStorage.removeItem(`guitarocr-draft:${ui.sid}`);
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
  $("currentDocument").title = documentName();
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
  if (!ui.metadataDirty) renderMetadata();
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
  const initialGeneration = ui.openGeneration;
  try {
    renderFiles();
    $("steps").hidden = !ui.sid;
    const config = await refreshConfig();
    await workspace.refresh();
    if (ui.sid && ui.openGeneration === initialGeneration && !ui.busy) {
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
      if (!ui.state?.pages && !ui.busy)
        notice(config.native_runtime_error
          ? `本机识别组件未通过校验：${config.native_runtime_error}。可继续校对与导出项目。`
          : "当前为校对模式。可打开原创示例或恢复项目备份，编辑并导出乐谱。");
    } else if (!config.model_ready && !ui.state?.pages && !ui.busy)
      notice(
        config.server
          ? "服务端识别模型尚未就绪。可先打开原创示例体验校对与导出。"
          : "识别模型尚未准备。可先打开示例；首次识别时会确认模型下载。",
      );
  } catch (e) {
    if (ui.openGeneration === initialGeneration && !ui.busy)
      notice(e.message, true);
  }
})();
