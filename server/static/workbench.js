import { api, setAuth, element } from "./http.js";
import { BrowserRunner } from "./browser-runner.js";
import { ui } from "/static/state.js";
const response = await fetch("/api/auth/me");
if (!response.ok) location.replace("/");
else {
  setAuth(await response.json());
  const back = element("a", "返回我的乐谱");
  back.href = "/";
  document.querySelector("header").append(back);
  document.querySelector(".project-import").hidden = true;
  const config = await api("/api/config");
  document.querySelector('[data-panel="0"] .footer .muted').textContent =
    `每个项目最多 ${config.max_pages} 页、${config.max_upload_mb} MB`;
  let runner;
  setInterval(async () => {
    try {
      if (!ui.sid) return;
      const state = await api(`/api/sessions/${ui.sid}`);
      if (
        state.engine === "browser" &&
        ["queued", "running"].includes(state.job?.status) &&
        !runner
      ) {
        const status = document.getElementById("notice");
        runner = new BrowserRunner(
          state.job.id,
          (message) => {
            status.hidden = false;
            status.textContent = message;
          },
          (message) => {
            status.hidden = false;
            status.textContent = message;
          },
        );
        runner.start().finally(() => {
          runner = null;
        });
      }
    } catch {
      /* The editor displays request errors. */
    }
  }, 2000);
}
