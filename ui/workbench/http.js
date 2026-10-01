// Keep the deadline active while reading the response body, too.
export async function request(path, options = {}, format = "json", fetcher = fetch) {
  const controller = new AbortController();
  const writing = options.method && options.method !== "GET";
  const timeout = options.body instanceof FormData ? 120000 : writing ? 60000 : 20000;
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetcher(path, { ...options, signal: controller.signal });
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
