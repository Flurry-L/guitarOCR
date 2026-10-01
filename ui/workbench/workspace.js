import "./theme.js";
import { ui } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api } from "./api.js";

const active = (job) => ["queued", "running"].includes(job?.status);
const interrupted = (job) => ["failed", "cancelled", "interrupted"].includes(job?.status);
const statusName = (project) => ({
  queued: "排队中", running: "识别中", failed: "识别失败", cancelled: "已停止", interrupted: "已中断",
})[project.job?.status] || project.stage;
const dateLabel = (timestamp) => timestamp ? new Date(timestamp * 1000).toLocaleString("zh-CN", {
  month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
}) : "";

export function initWorkspace({ open, resume, refreshConfig }) {
  let projects = [], screen = "workbench", refreshing = false;
  const pending = new Set();
  function show(name) {
    if (name !== screen) {
      ui.navigation += 1;
      notice("");
    }
    screen = name;
    $("deleteProject").hidden = name !== "workbench" || !ui.state?.pages;
    $("keyboardHelp").hidden = name !== "workbench" || !ui.state?.measures?.length;
    document.querySelectorAll("[data-screen]").forEach(node => { node.hidden = node.dataset.screen !== name; });
    document.querySelectorAll("[data-screen-link]").forEach(node => {
      node.classList.toggle("active", node.dataset.screenLink === name);
      if (node.dataset.screenLink === name) node.setAttribute("aria-current", "page");
      else node.removeAttribute("aria-current");
    });
    $("screenTitle").textContent = { workbench: "导入乐谱", library: "项目", settings: "设置" }[name];
    if (name === "library") refresh();
    if (name === "settings") {
      const navigation = ui.navigation;
      refreshConfig().catch(error => { if (ui.navigation === navigation) notice(error.message, true); });
    }
    renderJob(ui.state?.job);
    window.dispatchEvent(new Event("resize"));
  }
  document.querySelectorAll("[data-screen-link]").forEach(node => {
    node.onclick = () => show(node.dataset.screenLink);
  });
  $("allProjects").onclick = () => show("library");
  $("keyboardHelp").onclick = () => $("shortcutDialog").showModal();
  $("closeShortcuts").onclick = () => $("shortcutDialog").close();
  $("refreshProjects").onclick = refresh;
  for (const id of ["projectSearch", "projectFilter", "projectSort"])
    $(id).addEventListener("input", renderLibrary);

  function projectAction(project, label, run) {
    const button = el("button", label);
    button.dataset.projectAction = label;
    button.disabled = pending.has(project.id);
    button.onclick = action(async () => {
      if (pending.has(project.id)) return;
      pending.add(project.id);
      renderLibrary();
      try { await run(); }
      finally { pending.delete(project.id); renderLibrary(); await refresh(); }
    });
    return button;
  }
  function card(project) {
    const row = el("article", undefined, "project-card");
    row.dataset.project = project.id;
    const job = project.job;
    const button = el("button", undefined, "project-open");
    button.dataset.openProject = project.id;
    button.dataset.unavailable = String(pending.has(project.id) || (!project.pages && !active(job)));
    button.disabled = (ui.busy && !active(ui.state?.job)) || pending.has(project.id) || (!project.pages && !active(job));
    button.setAttribute("aria-label", `打开 ${project.title || "未命名乐谱"}`);
    button.onclick = action(() => open(project.id));
    const top = el("div", undefined, "card-top");
    top.append(el("h3", project.title || "未命名乐谱"), el("span", statusName(project), "pill"));
    button.append(top, el("span", `${project.pages || 0} 页 · ${dateLabel(project.updated || project.created)}`, "card-meta"));
    row.append(button);
    if (active(job) || interrupted(job)) {
      const message = job.error || job.message;
      if (message) row.append(el("p", message, job.status === "failed" ? "error" : "card-meta"));
      if (active(job)) {
        const progress = el("progress");
        progress.setAttribute("aria-label", "识别进度");
        if (job.total) { progress.max = job.total; progress.value = job.done; }
        row.append(progress);
      }
      const actions = el("div", undefined, "row");
      if (active(job) && job.cancellable) actions.append(projectAction(project, "停止识别", async () => {
        const state = await api(`/api/sessions/${project.id}`);
        await api(`/api/sessions/${project.id}/cancel`, "POST", undefined, state.revision);
      }));
      if (interrupted(job) && project.pages && job.action === "full") {
        const retry = el("button", "继续识别");
        retry.dataset.projectAction = "继续识别";
        retry.onclick = action(async () => {
          if (await open(project.id) && ui.sid === project.id && screen === "workbench") await resume();
        });
        retry.disabled ||= ui.busy;
        actions.append(retry);
      }
      if (interrupted(job) && !project.pages) actions.append(projectAction(project, "删除未完成导入", async () => {
        if (confirm(`删除「${project.title}」的未完成导入？`)) await api(`/api/sessions/${project.id}`, "DELETE");
      }));
      if (actions.childElementCount) row.append(actions);
    }
    return row;
  }
  function renderLibrary() {
    const query = $("projectSearch").value.trim().toLocaleLowerCase(), filter = $("projectFilter").value;
    let filtered = projects.filter(p => (p.title || "").toLocaleLowerCase().includes(query));
    if (filter !== "all") filtered = filtered.filter(p => ({
      active: active(p.job), review: p.stage === "待校对" && !active(p.job),
      exported: p.stage === "已导出", failed: interrupted(p.job),
    })[filter]);
    if ($("projectSort").value === "title") filtered.sort((a, b) => (a.title || "").localeCompare(b.title || "", "zh-CN"));
    const focused = document.activeElement;
    const project = focused?.closest("[data-project]")?.dataset.project;
    const control = focused?.dataset.projectAction;
    $("libraryList").replaceChildren(...filtered.map(card));
    $("libraryEmpty").hidden = !!filtered.length;
    $("libraryEmpty").querySelector("strong").textContent = projects.length ? "没有匹配的项目" : "还没有项目";
    $("libraryEmpty").querySelector("p").textContent = projects.length ? "试试其他关键词或状态。" : "选择「新建项目」导入乐谱，或恢复项目备份。";
    $("recentProjects").hidden = !projects.length;
    $("projectList").replaceChildren(...projects.slice(0, 3).map(card));
    if (project) {
      const row = document.querySelector(`[data-screen]:not([hidden]) [data-project="${CSS.escape(project)}"]`);
      const target = control && row?.querySelector(`[data-project-action="${CSS.escape(control)}"]:not(:disabled)`);
      (target || row?.querySelector(".project-open:not(:disabled)"))?.focus({ preventScroll: true });
    }
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    const navigation = ui.navigation;
    $("refreshProjects").disabled = true;
    try {
      projects = await api("/api/sessions");
      renderLibrary();
    } catch (error) {
      if (screen === "library" && ui.navigation === navigation) notice(error.message, true);
    } finally {
      refreshing = false;
      $("refreshProjects").disabled = false;
    }
  }
  setInterval(() => { if (!document.hidden && screen === "library") refresh(); }, 4000);

  function renderJob(job) {
    $("newProject").hidden = screen === "workbench" && !ui.sid;
    $("newProject").disabled = ui.busy && !active(job);
    $("newProject").title = active(job) ? "当前识别会继续，可另建项目" : "导入另一份乐谱";
    $("jobBanner").hidden = screen !== "workbench" || !job || (!active(job) && !interrupted(job));
    $("jobBanner").dataset.status = job?.status || "";
    $("jobTitle").textContent = statusName({ job }) || "";
    $("jobMessage").textContent = job?.error || job?.message || "";
    const progress = $("jobProgress");
    progress.hidden = !active(job);
    if (job?.total) { progress.max = job.total; progress.value = job.done; }
    else progress.removeAttribute("value");
    $("cancelJob").hidden = !active(job) || !job?.cancellable;
    $("cancelJob").disabled = !job?.cancellable;
    $("continueJob").hidden = !interrupted(job) || !ui.state?.pages || job.action !== "full";
    $("continueJob").disabled = ui.busy;
  }
  $("continueJob").onclick = action(resume);
  function configure(config) {
    const remote = config.server === true || location.pathname === "/workbench";
    const editOnly = config.inference_enabled === false;
    const ready = config.model_ready && config.layout_ready;
    const cached = config.model_cached && config.layout_cached;
    $("environmentLabel").textContent = remote ? "服务器识别" : "本机识别";
    $("settingConnection").textContent = remote ? "服务器" : "这台电脑";
    $("settingDevice").textContent = editOnly ? "暂不可用" : ({ auto: "自动选择", cpu: "CPU", cuda: "NVIDIA GPU", metal: "Apple GPU" })[config.device] || config.device || "自动选择";
    $("settingModel").textContent = ready ? "已就绪" : cached ? "已下载" : "尚未下载";
    $("uploadLimits").textContent = `最多 ${config.max_pages} 页 · ${config.max_upload_mb} MB`;
    $("importAction").querySelector('[value="full"]').disabled = editOnly;
    if (editOnly) $("importAction").value = "import";
    $("taskHint").textContent = editOnly ? "当前可打开示例或恢复项目进行编辑。" : remote
      ? "文件上传到服务器处理；关闭网页后识别继续。"
      : "文件在本机处理；退出应用会停止识别。";
    $("settingHint").textContent = editOnly
      ? remote ? "识别暂不可用，请联系管理员。已保存的项目仍可编辑。" : "识别暂不可用，请重新安装应用。已保存的项目仍可编辑。"
      : ready || cached ? "" : `本机识别需下载约 ${((config.model_download_bytes || 0) / 1e9).toFixed(1)} GB。`;
    $("settingHint").hidden = !$("settingHint").textContent;
    $("environmentDetails").hidden = !config.native_runtime_error;
    $("environmentError").textContent = config.native_runtime_error || "";
  }
  return { show, refresh, renderJob, configure, screen: () => screen, navigation: () => ui.navigation };
}
