import { api, auth, setAuth, element as el } from "./http.js";
import { BrowserRunner } from "./browser-runner.js";
const $ = (id) => document.getElementById(id);
let config,
  registering = false,
  view = "projects",
  refreshing = false;
const runners = new Map(),
  browserMessages = new Map();
const signedIn = () => Boolean(auth?.user && !auth.user.guest);
const engine = () => document.querySelector('[name="engine"]:checked')?.value;
function notice(text, error = false) {
  if ($("auth").open) {
    $("authNotice").textContent = text;
    $("authNotice").hidden = !text;
    return;
  }
  $("notice").textContent = text;
  $("notice").hidden = !text;
  $("notice").classList.toggle("error", error);
}
function action(fn) {
  return async (event) => {
    event?.preventDefault();
    const button = event?.currentTarget;
    if (button?.tagName === "BUTTON") button.disabled = true;
    try {
      await fn(event);
    } catch (error) {
      notice(error.message || "无法连接服务，请稍后重试。", true);
    } finally {
      if (button?.tagName === "BUTTON") button.disabled = false;
    }
  };
}
function show(name) {
  view = name;
  document
    .querySelectorAll("section[data-view]")
    .forEach(
      (n) =>
        (n.hidden =
          n.dataset.view !== name || (name !== "projects" && !signedIn())),
    );
  document
    .querySelectorAll("nav [data-view]")
    .forEach((n) => n.classList.toggle("selected", n.dataset.view === name));
}
function button(label, fn) {
  const n = el("button", label);
  n.type = "button";
  n.onclick = action(fn);
  return n;
}
function link(label, url) {
  const n = el("a", label, "button");
  n.href = url;
  return n;
}
function minutes(seconds) {
  return seconds < 60
    ? `${Math.round(seconds)} 秒`
    : `${Math.round(seconds / 60)} 分钟`;
}
function engineHint() {
  if (!config) return;
  $("engineHint").textContent =
    engine() === "gpu"
      ? "提交后可以关闭网页，服务器会继续处理。重新登录即可查看进度和结果，也可取消排队或识别中的任务。"
      : "识别时请保持此页面打开。关闭网页会暂停，再次打开任务可继续；登录后可从账号找回任务，继续时仍需保持页面打开。";
  $("uploadButton").textContent =
    engine() === "gpu" && !signedIn() ? "登录后使用 GPU" : "上传并识别";
  $("uploadButton").disabled = !(engine() === "browser"
    ? config.browser_ready
    : config.gpu_available);
}
function openAuth(reason) {
  $("authReason").textContent =
    reason || "登录后可使用服务器 GPU，并在账号中保存任务和结果。";
  $("authNotice").hidden = true;
  $("auth").showModal();
}
function renderIdentity() {
  $("loginButton").hidden = signedIn();
  $("navigation").hidden = !signedIn();
  $("adminTab").hidden = !auth?.user?.admin;
  $("identity").textContent = signedIn() ? auth.user.username : "";
  $("historyTitle").textContent = signedIn() ? "我的乐谱" : "识别记录";
  $("saveAccount").hidden = signedIn() || !auth;
  $("sessionHint").textContent = signedIn()
    ? "任务和结果已保存到账号。GPU 任务关闭网页后继续，浏览器任务会暂停。"
    : auth
      ? `当前记录凭此浏览器的 Cookie 访问，保留至 ${new Date(auth.expires * 1000).toLocaleDateString("zh-CN")}。清除 Cookie 后无法找回，请及时下载，或登录保存到账号。`
      : `无需登录即可使用浏览器推理。访客记录保留 ${config.session_days} 天，请在同一浏览器查看；登录后可保存到账号。`;
  engineHint();
}
function startBrowser(job) {
  if (runners.has(job.id)) return;
  const runner = new BrowserRunner(
    job.id,
    (message) => {
      browserMessages.set(job.id, message);
    },
    (error) => notice(error, true),
  );
  runners.set(job.id, runner);
  runner.start().finally(() => {
    runners.delete(job.id);
    browserMessages.delete(job.id);
    refresh();
  });
}
function renderProjects(projects) {
  $("empty").hidden = projects.length > 0;
  const nodes = [];
  for (const project of projects) {
    const row = el("article", undefined, "project"),
      info = el("div"),
      actions = el("div", undefined, "actions");
    info.append(el("h2", project.title));
    const job = project.job,
      busy = job && ["queued", "running"].includes(job.status);
    let status = project.stage;
    if (busy)
      status = !job.cancellable
        ? "正在取消，已完成部分会保留"
        : job.status === "queued"
          ? project.engine === "browser"
            ? "等待浏览器运行"
            : `排队中，前面约 ${Math.max(0, job.position - 1)} 个任务`
          : job.message;
    if (job?.status === "failed") status = job.error;
    if (job?.status === "cancelled") status = "已取消，可继续处理";
    info.append(
      el(
        "p",
        `${project.engine === "gpu" ? "服务器 GPU" : "浏览器 CPU"} · ${project.pages} 页 · ${status}`,
      ),
    );
    if (job?.status === "failed") info.append(el("p", `任务编号 ${job.id}`));
    if (browserMessages.has(job?.id))
      info.append(el("p", browserMessages.get(job.id)));
    if (busy && job.total) {
      const p = el("progress");
      p.max = job.total;
      p.value = job.done;
      p.setAttribute("aria-label", "识别进度");
      info.append(p);
    }
    if (
      project.engine === "browser" &&
      busy &&
      job.cancellable &&
      !runners.has(job.id)
    )
      actions.append(button("在此页继续", () => startBrowser(job)));
    if (!busy && project.pages)
      actions.append(link("校对 / 下载", `/workbench?project=${project.id}`));
    if (busy) {
      const cancelButton = button(
        job.cancellable ? "取消任务" : "正在取消…",
        async () => {
          await api(`/api/sessions/${project.id}/cancel`, "POST");
          runners.get(job.id)?.stop();
          notice(
            project.engine === "browser"
              ? "已停止浏览器推理，正在取消任务。已完成部分会保留。"
              : "已请求取消，当前步骤结束后停止。已完成部分会保留。",
          );
          await refresh();
        },
      );
      cancelButton.disabled = !job.cancellable;
      actions.append(cancelButton);
    } else {
      if (["failed", "cancelled"].includes(job?.status))
        actions.append(
          button("重试", async () => {
            const data = await api(`/api/sessions/${project.id}/retry`, "POST");
            if (project.engine === "browser") startBrowser(data.job);
            await refresh();
          }),
        );
      actions.append(
        button("删除", async () => {
          if (!confirm(`删除「${project.title}」及其结果？`)) return;
          await api(`/api/sessions/${project.id}`, "DELETE");
          await refresh();
        }),
      );
    }
    row.append(info, actions);
    nodes.push(row);
  }
  $("projects").replaceChildren(...nodes);
}
async function renderUsage() {
  const usage = await api("/api/usage");
  $("usage").replaceChildren();
  for (const engine of usage.engines) {
    const n = el("div", undefined, "stat");
    n.append(
      el("span", engine.engine === "gpu" ? "服务器 GPU" : "浏览器 CPU"),
      el("strong", `${engine.jobs} 次任务`),
      el("span", minutes(engine.seconds)),
    );
    $("usage").append(n);
  }
  const n = el("div", undefined, "stat");
  n.append(
    el("span", "已保存"),
    el("strong", `${usage.projects} 份乐谱`),
    el("span", `${usage.pages} 页，${(usage.bytes / 1024 ** 2).toFixed(0)} MB`),
  );
  $("usage").append(n);
}
async function renderAdmin() {
  const [status, users] = await Promise.all([
    api("/api/admin/status"),
    api("/api/admin/users"),
  ]);
  $("registration").checked = status.registration;
  const update = status.update;
  $("updateMessage").textContent = update.message || "等待首次检查";
  $("updateVersion").textContent = [
    update.current ? `当前提交 ${update.current.slice(0, 8)}` : "",
    update.latest && update.latest !== update.current
      ? `可用提交 ${update.latest.slice(0, 8)} ${update.subject || ""}`
      : "",
  ]
    .filter(Boolean)
    .join("，");
  $("updateError").hidden = !update.error;
  $("updateError").textContent = update.error || "";
  $("applyUpdate").hidden = update.state !== "available";
  $("checkUpdate").disabled = [
    "requested",
    "preparing",
    "draining",
    "applying",
  ].includes(update.state);
  const userRows = users.map((user) => {
    const tr = el("tr"),
      controls = el("td");
    tr.append(
      el(
        "td",
        user.username +
          (user.admin ? "（管理员）" : user.disabled ? "（已停用）" : ""),
      ),
      el("td", user.projects),
      el("td", minutes(user.gpu_seconds)),
    );
    if (!user.admin) {
      controls.append(
        button(user.disabled ? "启用" : "停用", async () => {
          await api(`/api/admin/users/${user.id}`, "PATCH", {
            disabled: !user.disabled,
          });
          await renderAdmin();
        }),
      );
      controls.append(
        button("重设密码", async () => {
          const password = prompt(
            `为 ${user.username} 设置新密码，至少 10 个字符。`,
          );
          if (!password) return;
          await api(`/api/admin/users/${user.id}`, "PATCH", { password });
          notice("密码已修改，用户需要重新登录。");
        }),
      );
    }
    tr.append(controls);
    return tr;
  });
  $("users").replaceChildren(...userRows);
  $("adminJobs").replaceChildren(
    ...status.jobs.map((job) => {
      const row = el("div", undefined, "project");
      row.append(
        el(
          "span",
          `${job.username}，${job.engine === "gpu" ? "GPU" : "浏览器"}，${job.message}`,
        ),
        button("取消任务", async () => {
          await api(`/api/admin/jobs/${job.id}/cancel`, "POST");
          await renderAdmin();
        }),
      );
      return row;
    }),
  );
  if (!status.jobs.length)
    $("adminJobs").append(el("p", "没有待处理任务。", "muted"));
}
async function refresh() {
  if (!auth || refreshing) return;
  refreshing = true;
  try {
    if (view === "projects") renderProjects(await api("/api/sessions"));
    else if (view === "account") await renderUsage();
    else if (auth.user.admin) await renderAdmin();
  } finally {
    refreshing = false;
  }
}
async function enterSession(data) {
  setAuth(data);
  $("auth").close();
  renderIdentity();
  show("projects");
  notice("");
  await refresh();
}
$("authForm").onsubmit = action(async () => {
  $("authSubmit").disabled = true;
  try {
    const data = await api(
      `/api/auth/${registering ? "register" : "login"}`,
      "POST",
      { username: $("username").value, password: $("password").value },
    );
    $("password").value = "";
    await enterSession(data);
  } finally {
    $("authSubmit").disabled = false;
  }
});
$("authToggle").onclick = () => {
  registering = !registering;
  $("authTitle").textContent = registering ? "创建账号" : "登录";
  $("authSubmit").textContent = registering ? "创建账号" : "登录";
  $("authToggle").textContent = registering ? "已有账号，去登录" : "创建账号";
  $("password").autocomplete = registering
    ? "new-password"
    : "current-password";
};
$("loginButton").onclick = () => openAuth();
$("saveAccount").onclick = () =>
  openAuth("登录或创建账号后，当前访客任务和结果会保存到账号。");
$("closeAuth").onclick = () => $("auth").close();
$("auth").addEventListener("close", () => {
  $("password").value = "";
});
$("logout").onclick = action(async () => {
  for (const r of runners.values()) r.stop();
  await api("/api/auth/logout", "POST");
  localStorage.removeItem("guitarocr-session");
  location.reload();
});
$("engines").onchange = engineHint;
$("uploadForm").onsubmit = action(async () => {
  const selectedEngine = engine();
  if (selectedEngine === "gpu" && !signedIn()) {
    openAuth("登录后可使用服务器 GPU；已选择的文件会保留。");
    return;
  }
  $("uploadButton").disabled = true;
  $("engines").disabled = true;
  try {
    if (!auth) {
      setAuth(await api("/api/auth/guest", "POST"));
      renderIdentity();
      $("uploadButton").disabled = true;
    }
    const form = new FormData();
    for (const file of $("files").files) form.append("files", file);
    form.append("engine", selectedEngine);
    form.append("action", "full");
    notice("正在上传…");
    const result = await api("/api/sessions", "POST", form);
    $("files").value = "";
    notice("");
    if (selectedEngine === "browser") startBrowser(result.job);
    await refresh();
  } finally {
    $("engines").disabled = false;
    engineHint();
  }
});
$("passwordForm").onsubmit = action(async (event) => {
  const data = Object.fromEntries(new FormData(event.currentTarget));
  const result = await api("/api/auth/password", "PUT", data);
  setAuth(result);
  event.currentTarget.reset();
  notice("密码已修改，其他页面需要重新登录。");
});
for (const n of document.querySelectorAll("nav [data-view]"))
  n.onclick = action(async () => {
    show(n.dataset.view);
    notice("");
    await refresh();
  });
$("registration").onchange = action(async () => {
  await api(
    `/api/admin/registration?enabled=${$("registration").checked}`,
    "PUT",
  );
});
$("checkUpdate").onclick = action(async () => {
  await api("/api/admin/updates/check", "POST");
  notice("已开始检查更新。");
});
$("applyUpdate").onclick = action(async () => {
  await api("/api/admin/updates/apply", "POST");
  await renderAdmin();
});
(async () => {
  try {
    config = await api("/api/config");
    $("authToggle").hidden = !config.registration;
    $("gpuEngine").disabled = !config.gpu_available;
    $("browserEngine").disabled = !config.browser_ready;
    $("gpuUnavailable").hidden = config.gpu_available;
    $("browserUnavailable").hidden = config.browser_ready;
    if (!config.browser_ready && config.gpu_available)
      $("gpuEngine").checked = true;
    if (config.browser_ready)
      $("browserDownload").textContent =
        `首次需下载约 ${(config.browser_download_bytes / 1024 ** 3).toFixed(1)} GB 模型，下载时间另计。`;
    $("limits").textContent =
      `每份乐谱最多 ${config.max_pages} 页、${config.max_upload_mb} MB。`;
    const session = await api("/api/auth/me");
    if (session.user) await enterSession(session);
    else {
      renderIdentity();
      renderProjects([]);
    }
  } catch (error) {
    notice(error.message, true);
  }
})();
setInterval(() => {
  if (!document.hidden)
    refresh().catch(() => notice("暂时无法连接服务，正在等待恢复。", true));
}, 3000);
window.addEventListener("beforeunload", (event) => {
  if (runners.size) {
    event.preventDefault();
    event.returnValue = "";
  }
});
