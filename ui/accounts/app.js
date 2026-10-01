import { api, auth, setAuth, element as el } from "./http.js";
const $ = id => document.getElementById(id);
let registering = false, view = "account", navigation = 0, refreshing = false, refreshAgain = false;
function notice(text, error = false) {
  const node = $(!$("auth").hidden ? "authNotice" : "notice");
  node.textContent = text;
  node.hidden = !text;
  node.classList.toggle("error", error);
}
function action(fn) {
  let pending = false;
  return async event => {
    event?.preventDefault();
    if (pending) return;
    pending = true;
    const button = event?.currentTarget;
    if (button?.tagName === "BUTTON") button.disabled = true;
    try { await fn(event); }
    catch (error) { notice(error.message, true); }
    finally {
      pending = false;
      if (button?.tagName === "BUTTON") button.disabled = false;
    }
  };
}
function show(name) {
  if (name !== "admin" || !auth?.user.admin) name = "account";
  if (name !== view) navigation += 1;
  view = name;
  history.replaceState(null, "", `/#${name}`);
  document.querySelectorAll("section[data-view]").forEach(node => { node.hidden = !auth || node.dataset.view !== name; });
  document.querySelectorAll("nav [data-view]").forEach(node => {
    node.classList.toggle("active", node.dataset.view === name);
    if (node.dataset.view === name) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  $("screenTitle").textContent = name === "admin" ? "用户与任务" : "账号设置";
}
function identity() {
  document.body.classList.toggle("signed-out", !auth);
  $("auth").hidden = !!auth;
  $("adminTab").hidden = !auth?.user.admin;
  $("accountIdentity").textContent = auth?.user.username || "";
}
function button(label, fn) {
  const node = el("button", label);
  node.type = "button";
  node.onclick = action(fn);
  return node;
}
function minutes(seconds) {
  return seconds < 60 ? `${Math.round(seconds)} 秒` : `${Math.round(seconds / 60)} 分钟`;
}
async function renderUsage() {
  const session = auth, startedAt = navigation;
  const usage = await api("/api/usage");
  if (auth !== session || navigation !== startedAt) return;
  $("usage").textContent = `已保存 ${usage.projects} 个项目 · ${(usage.bytes / 1024 ** 2).toFixed(1)} MB`;
}
async function renderAdmin() {
  const session = auth, startedAt = navigation;
  const [status, users] = await Promise.all([
    api("/api/admin/status"),
    api("/api/admin/users"),
  ]);
  if (auth !== session || navigation !== startedAt) return;
  if (!$("registration").disabled) $("registration").checked = status.registration;
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
      const row = el("div", undefined, "admin-job");
      row.append(
        el("span", `${job.username}，${job.message}`),
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
  if (!auth) return;
  if (refreshing) { refreshAgain = true; return; }
  refreshing = true;
  try {
    do {
      refreshAgain = false;
      if (view === "account") await renderUsage();
      else if (auth?.user.admin) await renderAdmin();
    } while (refreshAgain && auth);
  } finally {
    refreshing = false;
  }
}
async function enterSession(session) {
  setAuth(session);
  const requested = location.hash.slice(1);
  if (!["account", "admin"].includes(requested)) {
    location.replace(["library", "tasks"].includes(requested) ? "/workbench?view=projects" : "/workbench");
    return;
  }
  identity();
  show(requested);
  notice("");
  await refresh();
}
$("authForm").onsubmit = action(async () => {
  const controls = Array.from($("authForm").elements);
  controls.forEach(node => { node.disabled = true; });
  try {
    const session = await api(`/api/auth/${registering ? "register" : "login"}`, "POST", {
      username: $("username").value, password: $("password").value,
    });
    $("password").value = "";
    await enterSession(session);
  } finally {
    controls.forEach(node => { node.disabled = false; });
  }
});
$("authToggle").onclick = () => {
  registering = !registering;
  $("authTitle").textContent = $("authSubmit").textContent = registering ? "创建账号" : "登录";
  $("authToggle").textContent = registering ? "返回登录" : "创建账号";
  $("password").autocomplete = registering ? "new-password" : "current-password";
  $("usernameHint").hidden = $("passwordHint").hidden = !registering;
  notice("");
};
window.addEventListener("guitarocr:auth-required", () => {
  identity();
  document.querySelectorAll("section[data-view]").forEach(node => { node.hidden = true; });
  $("authReason").hidden = false;
  $("authReason").textContent = "登录已失效，请重新登录。";
});
$("logout").onclick = action(async () => {
  await api("/api/auth/logout", "POST");
  localStorage.removeItem("guitarocr-session");
  location.replace("/");
});
$("passwordForm").onsubmit = action(async event => {
  const form = event.currentTarget, data = Object.fromEntries(new FormData(form));
  const controls = Array.from(form.elements);
  controls.forEach(node => { node.disabled = true; });
  try {
    const result = await api("/api/auth/password", "PUT", data);
    setAuth(result);
    form.reset();
    notice("密码已修改。");
  } finally { controls.forEach(node => { node.disabled = false; }); }
});
for (const node of document.querySelectorAll("nav [data-view]")) node.onclick = action(async () => {
  show(node.dataset.view);
  notice("");
  await refresh();
});
$("registration").onchange = action(async () => {
  const desired = $("registration").checked;
  $("registration").disabled = true;
  try { await api(`/api/admin/registration?enabled=${desired}`, "PUT"); }
  catch (error) { $("registration").checked = !desired; throw error; }
  finally { $("registration").disabled = false; }
});
(async () => {
  try {
    const config = await api("/api/config");
    $("authToggle").hidden = !config.registration;
    const session = await api("/api/auth/me");
    if (session.user) await enterSession(session);
  } catch (error) { notice(error.message, true); }
})();
setInterval(() => {
  if (!document.hidden) refresh().catch(error => notice(error.message, true));
}, 4000);
window.addEventListener("hashchange", action(async () => {
  if (auth) { show(location.hash.slice(1)); await refresh(); }
}));
