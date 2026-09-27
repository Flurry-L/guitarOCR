import { api, auth, setAuth, element as el } from "./http.js";
const $ = (id) => document.getElementById(id);
let config,
  registering = false,
  view = "projects",
  refreshing = false;
const signedIn = () => Boolean(auth?.user && !auth.user.guest);
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
function uploadState() {
  if (!config) return;
  $("uploadButton").textContent = signedIn() ? "上传并识别" : "登录后识别";
  $("uploadButton").disabled = !config.gpu_available;
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
  $("history").hidden = !auth;
  $("sessionHint").textContent = signedIn()
    ? "任务和结果已保存到账号。关闭网页后继续处理。"
    : "登录后可将已有记录保存到账号，并使用服务器 GPU 继续识别。";
  uploadState();
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
          ? `排队中，前面约 ${Math.max(0, job.position - 1)} 个任务`
          : job.message;
    if (job?.status === "failed") status = job.error;
    if (job?.status === "cancelled") status = "已取消，可继续处理";
    info.append(
      el(
        "p",
        `${project.pages} 页，${status}`,
      ),
    );
    if (job?.status === "failed") info.append(el("p", `任务编号 ${job.id}`));
    if (busy && job.total) {
      const p = el("progress");
      p.max = job.total;
      p.value = job.done;
      p.setAttribute("aria-label", "识别进度");
      info.append(p);
    }
    if (!busy && project.pages)
      actions.append(link("校对 / 下载", `/workbench?project=${project.id}`));
    if (busy) {
      const cancelButton = button(
        job.cancellable ? "取消任务" : "正在取消…",
        async () => {
          await api(`/api/sessions/${project.id}/cancel`, "POST");
          notice("已请求取消，当前步骤结束后停止。已完成部分会保留。");
          await refresh();
        },
      );
      cancelButton.disabled = !job.cancellable;
      actions.append(cancelButton);
    } else {
      if (signedIn() && ["failed", "cancelled"].includes(job?.status))
        actions.append(
          button("重试", async () => {
            await api(`/api/sessions/${project.id}/retry`, "POST");
            await refresh();
          }),
        );
      if (signedIn()) actions.append(
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
      el("span", "服务器 GPU"),
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
          `${job.username}，${job.message}`,
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
  await api("/api/auth/logout", "POST");
  localStorage.removeItem("guitarocr-session");
  location.reload();
});
$("uploadForm").onsubmit = action(async () => {
  if (!signedIn()) {
    openAuth("登录后可使用服务器 GPU；已选择的文件会保留。");
    return;
  }
  $("uploadButton").disabled = true;
  try {
    const form = new FormData();
    for (const file of $("files").files) form.append("files", file);
    form.append("action", "full");
    notice("正在上传…");
    await api("/api/sessions", "POST", form);
    $("files").value = "";
    notice("");
    await refresh();
  } finally {
    uploadState();
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
    $("gpuUnavailable").hidden = config.gpu_available;
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
