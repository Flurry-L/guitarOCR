const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
let busy=false;
function status(text,error=false){$("status").hidden=!text;$("status").textContent=text;$("status").classList.toggle("error",error);}
function controls(){for(const n of document.querySelectorAll("button,input"))n.disabled=busy;$("stop").disabled=false;}
async function action(fn){if(busy)return;busy=true;controls();try{await fn();}catch(e){status(String(e),true);}finally{busy=false;controls();}}
$("remote").onsubmit=(e)=>{e.preventDefault();action(async()=>{status("正在连接…");await invoke("connect_server",{address:$("serverUrl").value});status("");});};
async function local(mode){await action(async()=>{
 status(mode==="gpu"?"正在准备本机 GPU 环境，进度见下方日志。":"正在准备校对环境…");$("logs").textContent="";$("logPanel").hidden=false;$("logPanel").open=true;$("stop").hidden=false;
 await invoke("start_local",{mode});status("本机工作台已打开。");
});}
$("localGpu").onclick=()=>local("gpu");$("localEdit").onclick=()=>local("edit");
$("stop").onclick=async()=>{if(!confirm("停止本机服务会中断本机任务，已保存的项目会保留。继续？"))return;try{await invoke("stop_local");status("本机服务已停止，项目已保留。");$("stop").hidden=true;}catch(e){status(String(e),true);}};
$("showData").onclick=()=>action(()=>invoke("open_data"));
(async()=>{
 await listen("runtime-log",e=>{const out=$("logs");out.textContent=(out.textContent+e.payload+"\n").slice(-50000);out.scrollTop=out.scrollHeight;});
 await listen("runtime-exit",e=>{status(e.payload,true);$("stop").hidden=true;});
 const state=await invoke("settings");$("serverUrl").value=state.server||"";
 if(!state.local_gpu){$("localGpu").hidden=true;$("localHint").textContent="这台设备可使用远程 GPU 识别，也可在本机校对项目备份和导出 GP5。";}
})().catch(e=>status(String(e),true));
