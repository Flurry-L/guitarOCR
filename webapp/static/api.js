import { ui } from "./state.js";
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
  const r = await fetch(path, opts);
  if (!r.ok) {
    let data;
    try {
      data = await r.json();
    } catch {
      data = { detail: r.statusText };
    }
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  }
  return r.json();
}
export const endpoint = (part) => `/api/sessions/${ui.sid}${part}`;
