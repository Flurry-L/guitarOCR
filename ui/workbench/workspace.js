import "./theme.js";
import { ui } from "./state.js";
import { $, el, action, notice } from "./dom.js";
import { api } from "./api.js";

const active = (job) => ["queued", "running"].includes(job?.status);
const interrupted = (job) =>
  ["failed", "cancelled", "interrupted"].includes(job?.status);
const statusName = (project) =>
  active(project.job)
    ? project.job.status === "queued"
      ? "排队中"
      : "处理中"
    : interrupted(project.job)
      ? { failed: "处理失败", cancelled: "已停止", interrupted: "已中断" }[
          project.job.status
        ]
      : project.stage;
const dateLabel = (timestamp) =>
  timestamp
    ? new Date(timestamp * 1000).toLocaleString("zh-CN", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

export function initWorkspace({ open, resume, refreshConfig }) {
  let projects = [],
    screen = "workbench",
    navigation = 0,
    refreshing = false;
  function show(name) {
    if (name !== screen) {
      navigation += 1;
      ui.navigation = navigation;
      notice("");
    }
    screen = name;
    $("deleteProject").hidden = name !== "workbench" || !ui.state?.pages;
    document.querySelectorAll("[data-screen]").forEach((node) => {
      node.hidden = node.dataset.screen !== name;
    });
    document.querySelectorAll("[data-screen-link]").forEach((node) => {
      node.classList.toggle("active", node.dataset.screenLink === name);
      if (node.dataset.screenLink === name)
        node.setAttribute("aria-current", "page");
      else node.removeAttribute("aria-current");
    });
    $("screenTitle").textContent = {
      workbench: "工作台",
      library: "项目库",
      tasks: "任务中心",
      settings: "设置",
    }[name];
    if (["library", "tasks"].includes(name)) refresh();
    if (name === "settings" && refreshConfig) {
      const startedAt = navigation;
      refreshConfig().catch(error => { if (navigation === startedAt) notice(error.message, true); });
    }
    renderJob(ui.state?.job);
    window.dispatchEvent(new Event("resize"));
  }
  document.querySelectorAll("[data-screen-link]").forEach((node) => {
    node.onclick = () => show(node.dataset.screenLink);
  });
  $("allProjects").onclick = () => show("library");
  $("keyboardHelp").onclick = () => $("shortcutDialog").showModal();
  $("closeShortcuts").onclick = () => $("shortcutDialog").close();
  $("refreshProjects").onclick = () => refresh();
  $("refreshTasks").onclick = () => refresh();
  for (const id of ["projectSearch", "projectFilter", "projectSort"])
    $(id).addEventListener("input", renderLibrary);

  function openButton(project, label = "打开项目") {
    const button = el("button", label);
    button.disabled = ui.busy && !active(ui.state?.job);
    button.onclick = action(async () => {
      await open(project.id);
    });
    return button;
  }
  function card(project) {
    const button = openButton(project, "");
    button.className = "project-card";
    if (!project.pages && !active(project.job))
      button.onclick = () => show("tasks");
    const top = el("div", undefined, "card-top");
    top.append(
      el("span", "♫", "file-icon"),
      el("span", statusName(project), "pill"),
    );
    const foot = el("div", undefined, "card-foot");
    foot.append(
      el("span", dateLabel(project.updated || project.created)),
      el("span", "打开项目 →"),
    );
    button.append(
      top,
      el("h3", project.title || "未命名乐谱"),
      el(
        "span",
        `${project.pages || 0} 页 · ${active(project.job) ? project.job.message : "识别与校对项目"}`,
        "card-meta",
      ),
      foot,
    );
    return button;
  }
  function renderLibrary() {
    const query = $("projectSearch").value.trim().toLocaleLowerCase(),
      filter = $("projectFilter").value;
    let filtered = projects.filter((p) =>
      (p.title || "").toLocaleLowerCase().includes(query),
    );
    if (filter !== "all")
      filtered = filtered.filter(
        (p) =>
          ({
            active: active(p.job),
            review: p.stage === "待校对" && !active(p.job),
            exported: p.stage === "已导出",
            failed: interrupted(p.job),
          })[filter],
      );
    if ($("projectSort").value === "title")
      filtered.sort((a, b) =>
        (a.title || "").localeCompare(b.title || "", "zh-CN"),
      );
    $("libraryList").replaceChildren(...filtered.map(card));
    $("libraryEmpty").hidden = !!filtered.length;
    $("libraryEmpty").querySelector("strong").textContent = projects.length
      ? "没有匹配的项目"
      : "还没有项目";
    $("libraryEmpty").querySelector("p").textContent = projects.length
      ? "试试其他关键词或状态。"
      : "在工作台导入乐谱，或恢复项目 ZIP。";
    $("recentProjects").hidden = !projects.length;
    $("projectList").replaceChildren(...projects.slice(0, 3).map(card));
  }
  function renderTasks() {
    const tasks = projects
      .filter((p) => p.job)
      .sort(
        (a, b) =>
          Number(active(b.job)) - Number(active(a.job)) ||
          (b.job.created || b.updated || 0) - (a.job.created || a.updated || 0),
      );
    $("tasksEmpty").hidden = !!tasks.length;
    $("taskList").replaceChildren(
      ...tasks.map((project) => {
        const job = project.job,
          row = el("article", undefined, "task-card"),
          info = el("div"),
          buttons = el("div", undefined, "row");
        info.append(
          el("h3", project.title),
          el(
            "p",
            `${statusName(project)} · ${job.error || job.message || ""}`,
            job.status === "failed" ? "error" : "",
          ),
        );
        if (active(job)) {
          const progress = el("progress");
          progress.setAttribute("aria-label", "任务进度");
          if (job.total) {
            progress.max = job.total;
            progress.value = job.done;
          }
          info.append(progress);
        }
        if (project.pages || active(job))
          buttons.append(
            openButton(project, active(job) ? "查看进度" : "打开项目"),
          );
        if (active(job) && job.cancellable) {
          const stop = el("button", "停止");
          stop.onclick = action(async () => {
            const state = await api(`/api/sessions/${project.id}`);
            await api(
              `/api/sessions/${project.id}/cancel`,
              "POST",
              undefined,
              state.revision,
            );
            await refresh();
          });
          buttons.append(stop);
        }
        if (interrupted(job) && !project.pages) {
          const remove = el("button", "删除未完成导入");
          remove.onclick = action(async () => {
            if (!confirm(`删除「${project.title}」的未完成导入文件？`)) return;
            await api(`/api/sessions/${project.id}`, "DELETE");
            await refresh();
          });
          buttons.append(remove);
        }
        if (interrupted(job) && project.pages && job.action === "full") {
          const retry = el("button", "继续识别");
          retry.disabled = ui.busy;
          retry.onclick = action(async () => {
            if (await open(project.id) && ui.sid === project.id && screen === "workbench") {
              await resume();
            }
          });
          buttons.append(retry);
        }
        row.append(info, buttons);
        return row;
      }),
    );
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      projects = await api("/api/sessions");
      renderLibrary();
      renderTasks();
    } catch (error) {
      if (screen !== "workbench") notice(error.message, true);
    } finally {
      refreshing = false;
    }
  }
  setInterval(() => {
    if (!document.hidden && ["library", "tasks"].includes(screen)) refresh();
  }, 4000);
  function renderJob(job) {
    $("newProject").disabled = ui.busy && !active(job);
    $("newProject").title = active(job)
      ? "当前任务会在后台继续，可新建另一份乐谱"
      : "导入另一份乐谱";
    $("jobBanner").hidden = screen !== "workbench" || !job || (!active(job) && !interrupted(job));
    $("jobBanner").dataset.status = job?.status || "";
    $("jobTitle").textContent = active(job) ? "正在处理乐谱" : "任务已暂停";
    if (job?.status === "failed") $("jobTitle").textContent = "处理失败";
    $("jobMessage").textContent = job?.error || job?.message || "";
    const progress = $("jobProgress");
    progress.hidden = !active(job);
    if (job?.total) {
      progress.max = job.total;
      progress.value = job.done;
    } else progress.removeAttribute("value");
    $("cancelJob").hidden = !active(job) || !job?.cancellable;
    $("cancelJob").disabled = !job?.cancellable;
    $("continueJob").hidden =
      !interrupted(job) || !ui.state?.pages || job.action !== "full";
    $("continueJob").disabled = ui.busy;
  }
  $("continueJob").onclick = action(resume);
  function configure(config) {
    const remote = config.server === true || location.pathname === "/workbench";
    const editOnly = config.inference_enabled === false;
    $("environmentLabel").textContent = remote
      ? "远程工作台"
      : editOnly
        ? "本机校对模式"
        : "本机工作台";
    $("settingConnection").textContent = remote ? "远程服务" : "本机服务";
    $("settingDevice").textContent = editOnly
      ? "校对与导出"
      : config.device === "auto" ? "自动选择（首次识别时检测）"
        : config.device || (config.gpu_available ? "服务器 GPU" : "未启用 GPU");
    $("settingModel").textContent = config.model_ready ? "已就绪" : config.model_cached ? "已缓存，待校验" : "未准备";
    $("settingLayout").textContent = config.layout_ready ? "已就绪" : config.layout_cached ? "已缓存，待校验" : "未准备";
    $("settingLimits").textContent =
      `${config.max_pages} 页 / ${config.max_upload_mb} MB`;
    $("uploadLimits").textContent =
      `每个项目最多 ${config.max_pages} 页 · ${config.max_upload_mb} MB`;
    $("importAction").querySelector('[value="full"]').disabled = editOnly;
    if (editOnly) $("importAction").value = "import";
    $("taskHint").textContent = editOnly
      ? "校对模式：导入并手动框选，或恢复项目备份后继续编辑与导出。"
      : remote
        ? "识别在服务器运行，关闭网页后可回来继续校对。"
        : "识别在本机后台运行。退出桌面应用会停止任务，已保存结果可继续处理。";
    $("settingHint").textContent = remote
      ? config.model_ready
        ? "服务器模型已就绪，所有账号共用模型，项目分别保存。"
        : "服务器模型未就绪，首次识别可确认下载；运行组件问题请联系服务管理员。"
      : editOnly
        ? "当前安装包缺少可用的识别组件。可继续校对与导出；本机识别需要完整安装包。"
        : config.model_ready
          ? "模型已校验。项目与模型保存在本机，更新应用后可继续使用。"
          : "开始识别时会校验模型缓存，缺失或损坏时经确认后下载；中断后重试可续传。";
  }
  return { show, refresh, renderJob, configure, screen: () => screen, navigation: () => navigation };
}
