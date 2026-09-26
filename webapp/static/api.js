import { ui } from "./state.js";
function requestError(detail) {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return "请求失败，请重试。";
  const fields = {
    title: "曲名最多 500 个字符。",
    artist: "作者最多 500 个字符。",
    tempo_quarter: "速度须为 20 至 400 的整数。",
    capo: "变调夹须为 0 至 24 的整数。",
    tuning_used: "请填写 1 至 7 个弦的 MIDI 音高（0 至 127），用逗号分隔。",
    boxes: "区域格式有误，请检查页码和位置。",
    mode: "请选择谱面类型。",
    measures: "请选择要识别的小节。",
  };
  return [...new Set(detail.map((error) =>
    fields[error.loc?.[1]] || "输入格式有误，请检查填写的内容。",
  ))].join("\n");
}
export async function api(path, method = "GET", body) {
  const opts = { method };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers = { "Content-Type": "application/json" };
  }
  if (
    method !== "GET" &&
    ui.state &&
    path.startsWith(`/api/sessions/${ui.sid}`)
  ) {
    opts.headers = { ...opts.headers, "If-Match": String(ui.state.revision) };
  }
  let r;
  try {
    r = await fetch(path, opts);
  } catch {
    throw new Error("无法连接工作台，请确认启动窗口仍在运行后重试。");
  }
  if (!r.ok) {
    let data;
    try {
      data = await r.json();
    } catch {
      data = { detail: "请求失败，请查看启动窗口中的错误后重试。" };
    }
    throw new Error(requestError(data.detail));
  }
  return r.json();
}
export const endpoint = (part) => `/api/sessions/${ui.sid}${part}`;
