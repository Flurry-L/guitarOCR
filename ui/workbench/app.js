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
const { renderMetadata, updateMetadataControls } = initMetadata({ start, go, render, setBusy });
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
let configRequest = 0;
async function refreshConfig() {
  const request = ++configRequest;
  const config = await api("/api/config");
  if (request !== configRequest) return uploadConfig;
  uploadConfig = config;
  workspace.configure(config);
  document.body.dataset.inference = config.inference_enabled === false ? "disabled" : "enabled";
  renderFiles();
  return config;
}
let downloadAllowed = false;
function nativeModelOptions(body) {
  if (uploadConfig.inference_enabled === false)
    throw new Error("识别暂不可用，请在设置中查看原因。");
  if (uploadConfig.requires_model_confirmation && !downloadAllowed) {
    const size = ((uploadConfig.model_download_bytes || 0) / 1e9).toFixed(1);
    const device = uploadConfig.server ? "服务器" : "这台电脑";
    if (!confirm(`${device}可能需要下载约 ${size} GB 识别资源，已有的文件会继续使用。请预留 6 GB 空间。开始识别？`))
      throw new Error("已取消识别，所选文件和编辑仍保留。");
    downloadAllowed = true;
  }
  return { ...body, allow_download: downloadAllowed };
}


let opening = 0;
let importReturnStep = 0;
const activeJob = (job) => ["queued", "running"].includes(job?.status);
const readyStep = (state) => state.recognition || state.info ? 3 : state.layout ? 2 : 1;
async function openExisting(id) {
  if (ui.busy && !activeJob(ui.state?.job)) return false;
  if (id === ui.sid && ui.state?.pages) {
    workspace.show("workbench");
    return true;
  }
  if ((hasDrafts() || ui.files.length) && !confirm("打开成功后将放弃当前未保存的修改和已选文件，继续？"))
    return false;
  const ticket = ++opening, generation = ui.openGeneration;
  const navigation = workspace.navigation();
  const wasBusy = ui.busy;
  setBusy(true);
  notice("正在打开项目…");
  try {
    const current = await api(`/api/sessions/${id}`);
    if (ticket !== opening || generation !== ui.openGeneration || navigation !== workspace.navigation()) return false;
    if (!current.pages && !activeJob(current.job))
      throw new Error(current.job?.error || "这个项目尚未完成导入，请在项目列表查看。");
    window.dispatchEvent(new Event("guitarocr:leave-project"));
    openProject(id, { discardDrafts: true });
    ui.step = 0;
    ui.state = current;
    $("reviewEditor").hidden = true;
    go(0);
    renderFiles();
    await watch(() => go(readyStep(ui.state)), current);
    return true;
  } catch (error) {
    if (ticket === opening && navigation === workspace.navigation()) notice(error.message, true);
    return false;
  } finally {
    if (ticket === opening && generation === ui.openGeneration) {
      setBusy(wasBusy && activeJob(ui.state?.job));
      if (navigation !== workspace.navigation()) notice("");
    }
  }
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
  updateMetadataControls();
  renderExport();
  document.querySelectorAll("[data-open-project]").forEach((node) => {
    node.disabled = node.dataset.unavailable === "true" || (value && !["queued", "running"].includes(ui.state?.job?.status));
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
    next !== 0 && ((ui.step === 1 && ui.boxDirty) ||
      (ui.step === 2 && ui.metadataDirty) ||
      (ui.step === 3 && ui.measureDirty)) &&
    next !== ui.step
  ) {
    notice("当前修改尚未保存，请先保存或放弃修改，再切换步骤。", true);
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
  $("currentDocument").hidden = !ui.state?.pages || ui.step === 0;
  $("steps").hidden = !ui.state?.pages || ui.step === 0;
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
      ? "完成后对照原谱校对，再导出。"
      : "先调整小节框与阅读顺序，再识别音符。";
  $("importAction").querySelector('[value="full"]').disabled =
    uploadConfig.inference_enabled === false;
  $("cancelImport").hidden = !ui.state?.pages || ui.step !== 0;
  $("cancelImport").disabled = ui.busy;
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
  if (uploadConfig.native && uploadConfig.pdf_enabled === false && incoming.some(file => /\.pdf$/i.test(file.name)))
    return notice("当前无法打开 PDF，请更新应用或先将 PDF 转为图片。", true);
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
  if (ui.step !== 0) importReturnStep = ui.step;
  ui.files.push(...incoming);
  go(0);
  renderFiles();
  $("currentDocument").hidden = true;
  notice(`已选择 ${ui.files.length} 个文件，可调整顺序后开始处理。`);
}
$("cancelImport").onclick = () => {
  if (ui.busy) return;
  ui.files = [];
  go(importReturnStep || readyStep(ui.state));
  renderFiles();
  notice(hasDrafts() ? "已返回原项目，未保存修改仍保留。" : "已取消导入。");
};
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
async function watch(after, initial, navigation = workspace.navigation()) {
  const sid = ui.sid;
  const generation = ui.openGeneration;
  setBusy(true);
  try {
    for (;;) {
      let current;
      try {
        current = initial || await api(`/api/sessions/${sid}`);
        initial = null;
      } catch (error) {
        if (ui.sid !== sid || ui.openGeneration !== generation) return;
        if (!activeJob(ui.state?.job)) throw error;
        if (workspace.screen() === "workbench")
          notice(`${error.message} 正在重新连接；任务可能仍在后台运行，可在项目列表查看。`, true);
        await new Promise(resolve => setTimeout(resolve, 3000));
        continue;
      }
      if (ui.sid !== sid || ui.openGeneration !== generation) return;
      const job = current.job;
      if (!ui.state) ui.state = current;
      workspace.renderJob(job);
      if (job && ["queued", "running"].includes(job.status)) {
        if (ui.step === 0) {
          document.querySelector('[data-panel="0"]').classList.remove("active");
          $("currentDocument").hidden = false;
          $("scoreTitle").textContent = current.metadata?.title || "正在导入乐谱";
          $("scoreCredits").textContent = current.metadata?.artist || "";
          $("documentName").textContent = (current.input_names || []).join("、");
          $("documentPages").textContent = current.pages?.length ? `，共 ${current.pages.length} 页` : "";
        }
        ui.state = { ...ui.state, job, revision: current.revision };
        $("cancelJob").hidden = !job.cancellable;
        if (workspace.screen() === "workbench") notice("");
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
      if (after && current.pages && workspace.navigation() === navigation) {
        if (["failed", "cancelled", "interrupted"].includes(job?.status))
          go(readyStep(current));
        else after();
      } else if (after && current.pages) {
        go(readyStep(current), false);
      }
      if (workspace.screen() === "workbench") notice("");
      // Model preparation/device selection can finish even when a later OCR step fails.
      if (uploadConfig.native) await refreshConfig().catch(() => {});
      break;
    }
  } catch (error) {
    if (ui.sid === sid && ui.openGeneration === generation && workspace.screen() === "workbench")
      notice(error.message, true);
    throw error;
  } finally {
    if (ui.sid === sid && ui.openGeneration === generation) setBusy(false);
    workspace.refresh();
  }
}
$("cancelJob").onclick = action(async () => {
  const sid = ui.sid;
  $("cancelJob").disabled = true;
  let result;
  try {
    result = await api(endpoint("/cancel"), "POST", undefined, ui.state?.revision);
  } finally {
    if (ui.sid === sid) workspace.renderJob(ui.state?.job);
  }
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
  const navigation = workspace.navigation();
  setBusy(true);
  notice("正在提交任务…");
  try {
    await api(`/api/sessions/${sid}${path}`, "POST", body, ui.state.revision);
    if (ui.sid === sid && ui.openGeneration === generation) await watch(after, undefined, navigation);
  } finally {
    if (ui.sid === sid && ui.openGeneration === generation) setBusy(false);
  }
}
$("upload").onclick = action(async () => {
  if (ui.busy) return;
  if (!ui.files.length) throw new Error("请先选择 PDF 或图片。");
  if (hasDrafts() && !confirm("开始新项目后将放弃原项目未保存的修改，继续？")) return;
  let generation = ui.openGeneration;
  const navigation = workspace.navigation();
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
    if (generation !== ui.openGeneration) return;
    if (navigation !== workspace.navigation()) {
      notice("新项目已提交，可在项目列表查看。原项目编辑和所选文件仍保留。");
      await workspace.refresh();
      return;
    }
    window.dispatchEvent(new Event("guitarocr:leave-project"));
    openProject(result.id, { discardDrafts: true });
    generation = ui.openGeneration;
    localStorage.setItem("guitarocr-session", ui.sid);
    renderFiles();
    await watch(() => go(automatic ? 3 : 1));
  } finally {
    if (generation === ui.openGeneration) setBusy(false);
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
    background: "项目已恢复到项目列表，可稍后打开。当前编辑仍保留。",
  });
};
$("openSample").onclick = () => importProject(readSample, {
  prompt: "打开示例后将替换当前未保存的编辑和已选文件，继续？",
  loading: "正在打开示例…",
  opened: "已打开示例。点选音符可编辑，对照右侧原谱核对后导出。",
  background: "示例已加入项目列表，可稍后打开。当前编辑仍保留。",
});
window.addEventListener("guitarocr:auth-required", () => { $("authRecovery").hidden = false; });
window.addEventListener("guitarocr:authenticated", ({ detail: user }) => {
  $("authRecovery").hidden = true;
  $("accountLinks").hidden = false;
  $("accountLink").textContent = `账号 · ${user.username}`;
  $("adminLink").hidden = !user.admin;
});
$("newProject").onclick = () => {
  if ((hasDrafts() || ui.files.length) && !confirm("放弃当前未保存的编辑和已选文件，新建项目？")) return;
  sessionStorage.removeItem(`guitarocr-draft:${ui.sid}`);
  localStorage.removeItem("guitarocr-session");
  clearDrafts();
  location.href = "/";
};
$("deleteProject").onclick = action(async () => {
  if (ui.busy) return;
  if (!confirm("删除当前项目的上传文件、识别结果和 GP5？此操作无法撤销。"))
    return;
  setBusy(true);
  try {
    await api(endpoint(""), "DELETE", undefined, ui.state.revision);
    sessionStorage.removeItem(`guitarocr-draft:${ui.sid}`);
    localStorage.removeItem("guitarocr-session");
    clearDrafts();
    location.href = "/";
  } catch (error) {
    setBusy(false);
    throw error;
  }
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
  $("keyboardHelp").hidden = workspace.screen() !== "workbench" || !ui.state.measures?.length;
  $("mode").value = ui.state.mode_setting || ui.state.mode;
  $("deleteProject").hidden = workspace.screen() !== "workbench";
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
  const navigation = workspace.navigation(), generation = ui.openGeneration;
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
    go(3, workspace.navigation() === navigation);
    openIssue();
    if (workspace.navigation() === navigation) notice("部分小节需要检查，请核对延音、音高和时值。");
    return;
  }
  if (workspace.navigation() === navigation) notice("GP5 已生成，可以下载。");
  if (ui.state.encoding_url) {
    const report = await api(ui.state.encoding_url);
    if (ui.openGeneration !== generation) return;
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
  if (new URLSearchParams(location.search).get("view") === "projects") workspace.show("library");
  const initialGeneration = ui.openGeneration;
  const initialNavigation = workspace.navigation();
  try {
    renderFiles();
    $("steps").hidden = !ui.sid;
    const config = await refreshConfig();
    await workspace.refresh();
    if (ui.sid && ui.openGeneration === initialGeneration && !ui.busy) {
      await watch(() => {
        go(ui.state.recognition ? 3 : ui.state.info ? 3 : 1);
      }, undefined, initialNavigation);
      if (
        !["failed", "cancelled", "interrupted"].includes(ui.state?.job?.status)
      )
        notice("");
    }
    if (config.inference_enabled === false && !ui.state?.pages && !ui.busy)
      notice("识别暂不可用。可打开示例或恢复项目进行编辑，详情见设置。");
  } catch (e) {
    if (ui.openGeneration === initialGeneration && !ui.busy)
      notice(e.message, true);
  }
})();
