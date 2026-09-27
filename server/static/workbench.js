import { api, setAuth, element } from "./http.js";
const response = await fetch("/api/auth/me");
const session = response.ok ? await response.json() : null;
if (!session?.user) location.replace("/");
else {
  setAuth(session);
  const back = element("a", "返回识别记录");
  back.href = "/";
  document.querySelector("header").append(back);
  document.querySelector(".project-import").hidden = true;
  const config = await api("/api/config");
  document.querySelector('[data-panel="0"] .footer .muted').textContent =
    `每个项目最多 ${config.max_pages} 页、${config.max_upload_mb} MB`;
}
