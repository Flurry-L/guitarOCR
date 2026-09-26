import { api } from "./http.js";
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export class BrowserRunner {
  constructor(job, onProgress, onError) {
    this.job = job;
    this.onProgress = onProgress;
    this.onError = onError;
    this.token = crypto.randomUUID();
    this.stopped = false;
    this.worker = null;
    this.call = null;
    this.pending = null;
    this.kind = null;
  }
  stop() {
    this.stopped = true;
    this.worker?.terminate();
    this.worker = null;
  }
  async start() {
    try {
      if (
        !globalThis.WebAssembly ||
        !globalThis.Worker ||
        !globalThis.OffscreenCanvas
      )
        throw new Error(
          "当前浏览器不支持此推理环境，请使用新版桌面 Chrome 或 Edge。",
        );
      const config = await api("/api/config");
      if (!config.browser_ready) throw new Error("浏览器模型尚未安装");
      while (!this.stopped) {
        try {
          const data = await api(`/api/browser/jobs/${this.job}/poll`, "POST", {
            token: this.token,
          });
          if (data.cancel || !["queued", "running"].includes(data.job.status))
            break;
          if (this.pending) {
            await api(
              `/api/browser/jobs/${this.job}/calls/${this.call}`,
              "POST",
              { token: this.token, ...this.pending },
            );
            this.pending = null;
          }
          if (data.call && data.call.id !== this.call) {
            this.call = data.call.id;
            if (this.kind !== data.call.kind || !this.worker) {
              this.worker?.terminate();
              this.kind = data.call.kind;
              this.worker = new Worker("/server-static/inference.js", {
                type: "module",
              });
              this.worker.onmessage = (event) => {
                if (event.data.progress) this.onProgress(event.data.progress);
                else this.pending = event.data.result;
              };
              this.worker.onerror = () => {
                this.pending = { error: "浏览器推理进程退出，可能内存不足" };
                this.onError(
                  "浏览器推理进程退出。请关闭其他占用内存的页面后重试。",
                );
              };
            }
            this.worker.postMessage({
              ...data.call,
              model_root: data.call.model_root || config.browser_model_root,
            });
          }
        } catch (error) {
          if (/另一个网页|请先登录|任务不存在/.test(error.message)) throw error;
          this.onProgress("连接中断，正在重连…");
        }
        await delay(1500);
      }
    } catch (error) {
      this.onError(error.message);
    } finally {
      this.stop();
    }
  }
}
