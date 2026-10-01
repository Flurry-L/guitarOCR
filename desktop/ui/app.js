import { StartupProgress } from './progress.mjs';

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
let busy=false;
const progress = new StartupProgress(document);
function status(text,error=false){$("status").hidden=!text;$("status").textContent=text;$("status").classList.toggle("error",error);}
function controls(){for(const n of document.querySelectorAll("button,input"))n.disabled=busy;$("stop").disabled=false;}
async function action(fn){if(busy)return;busy=true;controls();try{await fn();}catch(e){status(String(e),true);}finally{busy=false;controls();}}
$("remote").onsubmit=(e)=>{e.preventDefault();action(async()=>{status("正在连接…");await invoke("connect_server",{address:$("serverUrl").value});status("");});};
async function local(mode){await action(async()=>{
 status("");
 if(!await invoke("confirm_local_setup",{mode})){status("已取消本机启动，未开始准备。");return;}
 status("");progress.begin();$("logs").textContent="";$("logPanel").hidden=false;$("logPanel").open=false;$("stop").hidden=false;
 try { await invoke("start_local",{mode});progress.end(true);status("本机工作台已打开。"); }
 catch (error) { progress.end(false);$("logPanel").open=true;throw error; }
});}
$("localEdit").onclick=()=>local("native");
$("stop").onclick=async()=>{if(!confirm("停止本机服务会中断本机任务，已保存的项目会保留。继续？"))return;try{await invoke("stop_local");status("本机服务已停止，项目已保留。");$("stop").hidden=true;}catch(e){status(String(e),true);}};
$("showData").onclick=()=>action(()=>invoke("open_data"));
(async()=>{
 await listen("runtime-log",e=>{const out=$("logs");out.textContent=(out.textContent+e.payload+"\n").slice(-50000);out.scrollTop=out.scrollHeight;});
 await listen("runtime-exit",e=>{if(progress.active)progress.end(false);status(e.payload,true);$("stop").hidden=true;});
 const state=await invoke("settings");$("serverUrl").value=state.server||"";
 if(!state.native_available){$("localEdit").hidden=true;$("modelHint").hidden=true;$("runtimeHint").hidden=true;$("localHint").textContent="当前系统或架构尚不支持原生客户端。可连接远程服务。";}

})().catch(e=>status(String(e),true));
