//! Conservative native device selection, independent of operating-system guesses.
//! A declared backend, a live
//! device listing and sufficient reported memory are all required for GPU use.
use crate::llama::Device;
use serde_json::Value;
use std::{
    fs,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
const MIB: u64 = 1024 * 1024;
const MAX_PROBE_BYTES: u64 = 128 * 1024;

#[derive(Clone, Debug, Default)]
pub struct RuntimeCapabilities {
    pub cpu: Option<bool>,
    pub cuda: Option<bool>,
    pub metal: Option<bool>,
}
impl RuntimeCapabilities {
    /// Caller must have checked the packaged files first.
    /// Missing flags remain unknown; notably Metal is never inferred from macOS.
    pub fn from_component(component: &Value) -> Result<Self, String> {
        if component["name"] != "llama.cpp" {
            return Err("Not a llama runtime component".into());
        }
        let explicit = &component["capabilities"];
        if !explicit.is_null() && !explicit.is_object() {
            return Err("Invalid native runtime capabilities".into());
        }
        let build = &component["distribution"]["build"];
        let get = |key: &str, flag: &str| -> Result<Option<bool>, String> {
            let declared = match explicit.get(key) {
                None | Some(Value::Null) => None,
                Some(Value::Bool(value)) => Some(*value),
                _ => return Err(format!("Invalid native {key} capability")),
            };
            let mut configured = None;
            if let Some(args) = build["configure"].as_array() {
                for arg in args.iter().filter_map(Value::as_str) {
                    if let Some((name, value)) =
                        arg.strip_prefix("-D").and_then(|a| a.split_once('='))
                    {
                        if name.split(':').next() == Some(flag) {
                            configured = Some(match value.to_ascii_uppercase().as_str() {
                                "ON" | "1" | "TRUE" => true,
                                "OFF" | "0" | "FALSE" => false,
                                _ => return Err(format!("Invalid build capability {flag}")),
                            });
                        }
                    }
                }
            }
            if declared.is_some() && configured.is_some() && declared != configured {
                return Err(format!(
                    "Native {key} capability contradicts its build flags"
                ));
            }
            Ok(declared.or(configured))
        };
        Ok(Self {
            cpu: get("cpu", "GGML_CPU")?,
            cuda: get("cuda", "GGML_CUDA")?,
            metal: get("metal", "GGML_METAL")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MemoryBudget {
    pub required_bytes: u64,
}
impl MemoryBudget {
    pub fn from_files(
        model: &Path,
        projector: &Path,
        context_per_slot: usize,
        parallel: usize,
    ) -> Result<Self, String> {
        Self::estimate(
            fs::metadata(model).map_err(|e| e.to_string())?.len(),
            fs::metadata(projector).map_err(|e| e.to_string())?.len(),
            context_per_slot,
            parallel,
        )
    }
    /// Conservative admission estimate, not an allocation guarantee: weights +
    /// 25%, 64 KiB per context token/slot, and 1 GiB for scratch/vision buffers.
    pub fn estimate(
        model_bytes: u64,
        projector_bytes: u64,
        context_per_slot: usize,
        parallel: usize,
    ) -> Result<Self, String> {
        if model_bytes == 0
            || projector_bytes == 0
            || !(2048..=131072).contains(&context_per_slot)
            || !(1..=4).contains(&parallel)
        {
            return Err("Invalid native model memory estimate inputs".into());
        }
        let weights = model_bytes
            .checked_add(projector_bytes)
            .ok_or("Model size overflow")?;
        let required_bytes = weights
            .checked_add(weights / 4)
            .and_then(|v| v.checked_add(context_per_slot as u64 * parallel as u64 * 64 * 1024))
            .and_then(|v| v.checked_add(1024 * MIB))
            .ok_or("Model memory estimate overflow")?;
        Ok(Self { required_bytes })
    }
}

#[derive(Clone, Debug)]
pub struct DetectedDevice {
    pub device: Device,
    pub selector: String,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
}
#[derive(Clone, Debug)]
pub struct Selection {
    pub device: Device,
    pub device_selector: Option<String>,
    pub reason: String,
    pub required_bytes: u64,
}

pub fn valid_device_selector(device: Device, selector: &str) -> bool {
    let prefix = match device {
        Device::Cuda => "CUDA",
        Device::Metal => "MTL",
        Device::Cpu => return selector == "none",
    };
    selector.strip_prefix(prefix).is_some_and(|index| {
        !index.is_empty() && index.len() <= 4 && index.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Parse the runtime's common_print_available_devices output. Only local
/// CUDA/Metal names are admitted; RPC/Vulkan/other backends are never selected.
pub fn parse_device_listing(output: &str) -> Result<Vec<DetectedDevice>, String> {
    if !output
        .lines()
        .any(|line| line.trim() == "Available devices:")
    {
        return Err("原生设备探测未返回有效设备清单".into());
    }
    let mut devices = Vec::new();
    for line in output.lines() {
        let Some((selector, details)) = line.trim().split_once(": ") else {
            continue;
        };
        let device = if valid_device_selector(Device::Cuda, selector) {
            Device::Cuda
        } else if valid_device_selector(Device::Metal, selector) {
            Device::Metal
        } else {
            continue;
        };
        if devices
            .iter()
            .any(|d: &DetectedDevice| d.selector == selector)
        {
            return Err("原生设备清单包含重复设备".into());
        }
        let memory = details
            .rsplit_once(" (")
            .and_then(|(_, v)| v.strip_suffix(" MiB free)"))
            .and_then(|v| v.split_once(" MiB, "))
            .and_then(|(total, free)| {
                Some((
                    total.parse::<u64>().ok()?.checked_mul(MIB)?,
                    free.parse::<u64>().ok()?.checked_mul(MIB)?,
                ))
            })
            .filter(|(total, free)| *total > 0 && free <= total);
        devices.push(DetectedDevice {
            device,
            selector: selector.into(),
            total_bytes: memory.map(|p| p.0),
            free_bytes: memory.map(|p| p.1),
        });
    }
    Ok(devices)
}

/// Probe the installed Windows driver before downloading CUDA libraries.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub fn cuda_available(required_bytes: u64, minimum_driver: i32) -> bool {
    unsafe fn detect(required_bytes: u64, minimum_driver: i32) -> Option<bool> {
        // LOAD_LIBRARY_SEARCH_SYSTEM32: never load a DLL from the project or PATH.
        let driver: libloading::Library =
            libloading::os::windows::Library::load_with_flags("nvcuda.dll", 0x00000800)
                .ok()?
                .into();
        let init = driver
            .get::<unsafe extern "system" fn(u32) -> i32>(b"cuInit\0")
            .ok()?;
        let version = driver
            .get::<unsafe extern "system" fn(*mut i32) -> i32>(b"cuDriverGetVersion\0")
            .ok()?;
        let count = driver
            .get::<unsafe extern "system" fn(*mut i32) -> i32>(b"cuDeviceGetCount\0")
            .ok()?;
        let capability = driver
            .get::<unsafe extern "system" fn(*mut i32, *mut i32, i32) -> i32>(
                b"cuDeviceComputeCapability\0",
            )
            .ok()?;
        let memory = driver
            .get::<unsafe extern "system" fn(*mut usize, i32) -> i32>(b"cuDeviceTotalMem_v2\0")
            .ok()?;
        let (mut driver_version, mut devices) = (0, 0);
        if init(0) != 0
            || version(&mut driver_version) != 0
            || driver_version < minimum_driver
            || count(&mut devices) != 0
        {
            return Some(false);
        }
        for device in 0..devices.min(64) {
            let (mut major, mut minor, mut bytes) = (0, 0, 0usize);
            if capability(&mut major, &mut minor, device) == 0
                && (major, minor) >= (7, 5)
                && memory(&mut bytes, device) == 0
                && bytes as u64 / 100 * 80 >= required_bytes
            {
                return Some(true);
            }
        }
        Some(false)
    }
    unsafe { detect(required_bytes, minimum_driver).unwrap_or(false) }
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
pub fn cuda_available(_required_bytes: u64, _minimum_driver: i32) -> bool {
    false
}

/// Fixed short, local, no-model probe. No shell, nvidia-smi, uploads or downloads.
pub fn probe_devices(
    executable: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<DetectedDevice>, String> {
    if cancelled() {
        return Err("设备探测已取消".into());
    }
    let mut command = Command::new(executable);
    crate::llama::restrict_runtime_environment(&mut command);
    command
        .arg("--list-devices")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(parent) = executable.parent() {
        command.current_dir(parent);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("原生设备探测无法启动: {e}"))?;
    let read = |pipe: Box<dyn Read + Send>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            pipe.take(MAX_PROBE_BYTES + 1)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if cancelled() || Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout.join();
            let _ = stderr.join();
            return Err(if cancelled() {
                "设备探测已取消"
            } else {
                "设备探测超时，未确认GPU可用"
            }
            .into());
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(25)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout.join();
                let _ = stderr.join();
                return Err(format!("设备探测失败: {error}"));
            }
        }
    };
    let stdout = stdout
        .join()
        .map_err(|_| "设备探测读取失败")?
        .map_err(|e| e.to_string())?;
    let stderr = stderr
        .join()
        .map_err(|_| "设备探测读取失败")?
        .map_err(|e| e.to_string())?;
    if !status.success()
        || stdout.len() as u64 > MAX_PROBE_BYTES
        || stderr.len() as u64 > MAX_PROBE_BYTES
    {
        return Err("原生设备探测失败或输出异常，未确认GPU可用".into());
    }
    parse_device_listing(&String::from_utf8_lossy(&stdout))
}

fn cpu_selection(
    caps: &RuntimeCapabilities,
    budget: &MemoryBudget,
    reason: String,
) -> Result<Selection, String> {
    if caps.cpu == Some(false) {
        return Err(format!("{reason}；安装包未提供CPU后端。不会转发到远程服务"));
    }
    let uncertainty = if caps.cpu.is_none() {
        "；CPU能力未声明，将验证本机CPU加载结果"
    } else {
        ""
    };
    Ok(Selection {
        device: Device::Cpu,
        device_selector: None,
        reason: format!("{reason}{uncertainty}；仅在本机运行"),
        required_bytes: budget.required_bytes,
    })
}

pub fn select_from_probe(
    caps: &RuntimeCapabilities,
    budget: &MemoryBudget,
    detected: &Result<Vec<DetectedDevice>, String>,
) -> Result<Selection, String> {
    let gpu_allowed = |device| match device {
        Device::Cuda => caps.cuda == Some(true),
        Device::Metal => caps.metal == Some(true),
        Device::Cpu => false,
    };
    let mut reasons = Vec::new();
    if caps.cuda != Some(true) {
        reasons.push("CUDA后端未在已校验安装包中声明可用".to_string());
    }
    if caps.metal != Some(true) {
        reasons.push("Metal后端未在已校验安装包中声明可用".to_string());
    }
    let mut candidates = Vec::new();
    match detected {
        Err(error) => reasons.push(error.clone()),
        Ok(devices) => {
            for device in devices {
                if !gpu_allowed(device.device) {
                    continue;
                }
                if !valid_device_selector(device.device, &device.selector) {
                    continue;
                }
                let Some((free, total)) = device
                    .free_bytes
                    .zip(device.total_bytes)
                    .filter(|(free, total)| *free > 0 && free <= total)
                else {
                    reasons.push(format!("{}可用内存未知，保守使用CPU", device.selector));
                    continue;
                };
                // Metal's value is recommended working-set headroom, not dedicated
                // free VRAM. Reserve 40% there, 20% on CUDA, beyond the model budget.
                let percent = if device.device == Device::Metal {
                    60
                } else {
                    80
                };
                let usable = free / 100 * percent;
                if usable < budget.required_bytes {
                    reasons.push(format!(
                        "{}报告可用{} MiB/总量{} MiB，保留余量后不足估计的{} MiB",
                        device.selector,
                        free / MIB,
                        total / MIB,
                        budget.required_bytes / MIB
                    ));
                    continue;
                }
                candidates.push((usable, device));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.selector.cmp(&b.1.selector)));
    if let Some((_, device)) = candidates.first() {
        return Ok(Selection { device: device.device, device_selector: Some(device.selector.clone()), reason: format!("安装包后端与本机设备探测一致，选择{}；报告可用{} MiB，预计需{} MiB并额外保留内存余量；实际分配仍以加载结果为准", device.selector,device.free_bytes.unwrap()/MIB,budget.required_bytes/MIB), required_bytes: budget.required_bytes });
    }
    if reasons.is_empty() {
        reasons.push("原生运行组件未发现符合条件的本机GPU".into());
    }
    let reason = reasons.join("；");
    cpu_selection(caps, budget, format!("{reason}；自动回退CPU"))
}

pub fn choose_runtime(
    executable: &Path,
    caps: &RuntimeCapabilities,
    budget: &MemoryBudget,
    cancelled: &dyn Fn() -> bool,
) -> Result<Selection, String> {
    if cancelled() {
        return Err("设备选择已取消".into());
    }
    let detected = if caps.cuda != Some(true) && caps.metal != Some(true) {
        Ok(Vec::new())
    } else {
        probe_devices(executable, cancelled)
    };
    if cancelled() {
        return Err("设备选择已取消".into());
    }
    select_from_probe(caps, budget, &detected)
}

/// Retry local CPU once after GPU startup fails. Never
/// retry completed/in-flight inference on a different backend without its caller.
pub fn cpu_fallback_after_start_failure(
    caps: &RuntimeCapabilities,
    selection: &Selection,
) -> Result<Selection, String> {
    if selection.device == Device::Cpu {
        return Err("本机模型启动失败；不会重复CPU重试或转发远程服务".into());
    }
    cpu_selection(
        caps,
        &MemoryBudget {
            required_bytes: selection.required_bytes,
        },
        format!(
            "{}启动失败，自动模式仅重试一次本机CPU",
            selection.device_selector.as_deref().unwrap_or("GPU")
        ),
    )
}
