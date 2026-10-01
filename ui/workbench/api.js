import { request } from "./http.js";
const serverMode = location.pathname === "/workbench";
let csrf;
let userId;
let authentication;
export async function authenticate() {
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
    window.dispatchEvent(new Event("guitarocr:authenticated"));
    return session;
  })().finally(() => { authentication = null; });
  return authentication;
}
function requestError(detail) {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return "请求失败，请重试。";
  const fields = {
    title: "曲名最多 500 个字符。",
    artist: "作者最多 500 个字符。",
    tempo_quarter: "速度须为 20 至 400 的整数。",
    capo: "变调夹须为 0 至 24 的整数。",
    transpose: "记谱移调须为 -36 至 36 的整数，留空则自动读取。",
    tuning_used: "请填写 1 至 12 个弦的 MIDI 音高（0 至 127），用逗号分隔。",
    boxes: "区域格式有误，请检查页码和位置。",
    mode: "请选择谱面类型。",
    measures: "请选择要识别的小节。",
  };
  return [...new Set(detail.map((error) =>
    fields[error.loc?.[1]] || "输入格式有误，请检查填写的内容。",
  ))].join("\n");
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
