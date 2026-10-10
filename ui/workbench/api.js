// Keep the deadline active while reading the response body, too.
export async function request(path, options = {}, format = "json") {
  const controller = new AbortController();
  const writing = options.method && options.method !== "GET";
  const timeout = options.body instanceof FormData ? 120000 : writing ? 60000 : 20000;
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const data = response.ok
      ? await response[format]()
      : await response.json().catch(() => null);
    return { ok: response.ok, status: response.status, data };
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(writing
        ? "请求超时，尚未收到保存或提交的确认。当前编辑和所选文件已保留，请检查项目或任务状态后再试。"
        : "连接超时，请检查服务是否仍在运行后重试。");
    }
    throw new Error(error instanceof SyntaxError
      ? "服务返回的内容无法读取，请重试。"
      : "暂时无法连接服务。当前编辑和所选文件已保留，请检查连接后重试。");
  } finally {
    clearTimeout(timer);
  }
}

const serverMode = location.pathname === "/workbench";
let csrf;
let userId;
let authentication;
async function authenticate() {
  if (!serverMode) return;
  if (!authentication) authentication = (async () => {
    const auth = await request("/api/auth/me");
    const session = auth.ok ? auth.data : null;
    if (!session?.user || (userId !== undefined && userId !== session.user.id)) {
      csrf = undefined;
      window.dispatchEvent(new Event("guitarocr:auth-required"));
      throw new Error("登录已失效或账号已切换。当前编辑已保留，请在新窗口登录原账号后重试。");
    }
    userId = session.user.id;
    csrf = session.csrf;
    window.dispatchEvent(new CustomEvent("guitarocr:authenticated", { detail: session.user }));
    return session;
  })().finally(() => { authentication = null; });
  return authentication;
}
function requestError(detail) {
  return typeof detail === "string" ? detail : "请求失败，请重试。";
}
export async function api(path, method = "GET", body, revision) {
  const opts = { method };
  if (serverMode && (!csrf || method !== "GET")) await authenticate();
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers = { "Content-Type": "application/json" };
  }
  if (revision !== undefined && method !== "GET") {
    opts.headers = { ...opts.headers, "If-Match": String(revision) };
  }
  if (serverMode && method !== "GET") {
    opts.headers = { ...opts.headers, "X-CSRF-Token": csrf };
  }
  const r = await request(path, opts);
  if (!r.ok) {
    if (serverMode && [401, 403].includes(r.status)) {
      csrf = undefined;
      if (r.status === 401) {
        window.dispatchEvent(new Event("guitarocr:auth-required"));
        throw new Error("登录已失效。当前编辑已保留，请在新窗口登录后重试。");
      }
    }
    throw new Error(requestError(r.data?.detail));
  }
  return r.data;
}
