import { api, setAuth, element } from "./http.js";
import { BrowserRunner } from "./browser-runner.js";
import { ui } from "/static/state.js";
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
  let runner;
  window.addEventListener("guitarocr:cancel", () => runner?.stop());
  window.addEventListener("beforeunload", (event) => {
    if (runner && !runner.stopped) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  setInterval(async () => {
    try {
      if (!ui.sid) return;
      const state = await api(`/api/sessions/${ui.sid}`);
      if (
        state.engine === "browser" &&
        ["queued", "running"].includes(state.job?.status) &&
        state.job.cancellable &&
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
