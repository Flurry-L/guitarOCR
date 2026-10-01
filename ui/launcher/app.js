const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = id => document.getElementById(id);
let busy = false, stopping = false, generation = 0;
function status(text, error = false) {
  $("status").hidden = !text;
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
function controls() {
  for (const node of document.querySelectorAll("button,input")) node.disabled = busy || stopping;
  $("stop").disabled = stopping;
}
async function action(fn) {
  if (busy || stopping) return;
  busy = true;
  controls();
  try { await fn(); }
  catch (error) { status(String(error), true); }
  finally { busy = false; controls(); }
}
$("remote").onsubmit = event => {
  event.preventDefault();
  action(async () => {
    status("正在连接…");
    await invoke("connect_server", { address: $("serverUrl").value });
    status("已打开服务器。");
  });
};
$("localEdit").onclick = () => action(async () => {
  const startedAt = ++generation;
  status("正在打开乐谱…");
  $("startupProgress").hidden = false;
  $("logs").textContent = "";
  $("logPanel").hidden = false;
  $("logPanel").open = false;
  $("stop").hidden = false;
  try {
    await invoke("start_local", { mode: "native" });
    if (startedAt !== generation) { await invoke("stop_local"); return; }
    status("已打开本机乐谱。");
  } catch (error) {
    if (startedAt !== generation) return;
    $("logPanel").open = true;
    $("stop").hidden = true;
    throw error;
  } finally { $("startupProgress").hidden = true; }
});
$("stop").onclick = async () => {
  if (stopping || !confirm("停止本机识别并关闭乐谱窗口？已保存的项目会保留。")) return;
  generation += 1;
  stopping = true;
  controls();
  try {
    await invoke("stop_local");
    $("startupProgress").hidden = true;
    $("stop").hidden = true;
    status("已停止，项目仍保留。");
  } catch (error) { status(String(error), true); }
  finally { stopping = false; controls(); }
};
$("showData").onclick = () => action(() => invoke("open_data"));
(async () => {
  await listen("runtime-log", event => {
    const out = $("logs");
    out.textContent = (out.textContent + event.payload + "\n").slice(-50000);
    out.scrollTop = out.scrollHeight;
  });
  await listen("runtime-exit", event => {
    $("startupProgress").hidden = true;
    $("stop").hidden = true;
    if (!stopping) status(event.payload, true);
  });
  const state = await invoke("settings");
  $("serverUrl").value = state.server || "";
  $("remoteOptions").open = !!state.server;
  if (!state.native_available) {
    $("localEdit").hidden = $("modelHint").hidden = true;
    $("localHint").textContent = "这台设备请通过服务器识别。";
    $("remoteOptions").open = true;
  }
})().catch(error => status(String(error), true));
