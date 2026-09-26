export let auth = null;
export function setAuth(value) {
  auth = value;
}
export async function api(path, method = "GET", body) {
  const options = { method, headers: {} };
  if (auth && method !== "GET") options.headers["X-CSRF-Token"] = auth.csrf;
  if (body instanceof FormData) options.body = body;
  else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    let data;
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    if (response.status === 401 && auth) {
      setAuth(null);
      location.assign("/");
    }
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "填写的内容有误，请检查后重试。",
    );
  }
  return response.json();
}
export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
