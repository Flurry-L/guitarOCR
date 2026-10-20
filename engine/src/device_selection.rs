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
    pub cpu: bool,
    pub cuda: bool,
    pub metal: bool,
}
impl RuntimeCapabilities {
    /// Caller must have checked the packaged files first.
    pub fn from_component(component: &Value) -> Result<Self, String> {
        if component["name"] != "llama.cpp" {
            return Err("Not a llama runtime component".into());
        }
        let get = |key: &str| {
            component["capabilities"][key]
                .as_bool()
                .ok_or_else(|| format!("Missing or invalid native {key} capability"))
        };
        Ok(Self {
            cpu: get("cpu")?,
            cuda: get("cuda")?,
            metal: get("metal")?,
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
    pub total_bytes: u64,
    pub free_bytes: u64,
}
#[derive(Clone, Debug)]
pub struct Selection {
    pub device: Device,
    pub device_selector: Option<String>,
    pub reason: String,
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
        let (total_bytes, free_bytes) = details
            .rsplit_once(" (")
            .and_then(|(_, v)| v.strip_suffix(" MiB free)"))
            .and_then(|v| v.split_once(" MiB, "))
            .and_then(|(total, free)| {
                Some((
                    total.parse::<u64>().ok()?.checked_mul(MIB)?,
                    free.parse::<u64>().ok()?.checked_mul(MIB)?,
                ))
            })
            .filter(|(total, free)| *total > 0 && free <= total)
            .ok_or_else(|| format!("原生设备清单缺少有效内存信息：{selector}"))?;
        devices.push(DetectedDevice {
            device,
            selector: selector.into(),
            total_bytes,
            free_bytes,
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

/// List local devices without loading a model.
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
    // Driver initialization on the first launch can take several seconds.
    let deadline = Instant::now() + Duration::from_secs(30);
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

fn cpu_selection(caps: &RuntimeCapabilities, reason: String) -> Result<Selection, String> {
    if !caps.cpu {
        return Err(format!("{reason}；安装包未提供CPU后端。不会转发到远程服务"));
    }
    Ok(Selection {
        device: Device::Cpu,
        device_selector: None,
        reason: format!("{reason}；仅在本机运行"),
    })
}

pub fn select_from_probe(
    caps: &RuntimeCapabilities,
    budget: &MemoryBudget,
    devices: &[DetectedDevice],
) -> Result<Selection, String> {
    let gpu_allowed = |device| match device {
        Device::Cuda => caps.cuda,
        Device::Metal => caps.metal,
        Device::Cpu => false,
    };
    let mut reasons = Vec::new();
    if !caps.cuda {
        reasons.push("CUDA后端未在已校验安装包中声明可用".to_string());
    }
    if !caps.metal {
        reasons.push("Metal后端未在已校验安装包中声明可用".to_string());
    }
    let mut candidates = Vec::new();
    for device in devices {
        if !gpu_allowed(device.device) {
            continue;
        }
        let (free, total) = (device.free_bytes, device.total_bytes);
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
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.selector.cmp(&b.1.selector)));
    if let Some((_, device)) = candidates.first() {
        return Ok(Selection { device: device.device, device_selector: Some(device.selector.clone()), reason: format!("安装包后端与本机设备探测一致，选择{}；报告可用{} MiB，预计需{} MiB并额外保留内存余量；实际分配仍以加载结果为准", device.selector,device.free_bytes/MIB,budget.required_bytes/MIB) });
    }
    if reasons.is_empty() {
        reasons.push("原生运行组件未发现符合条件的本机GPU".into());
    }
    let reason = reasons.join("；");
    cpu_selection(caps, format!("{reason}；选择CPU"))
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
    let detected = if !caps.cuda && !caps.metal {
        Vec::new()
    } else {
        match probe_devices(executable, cancelled) {
            Ok(devices) => devices,
            Err(error) => {
                if cancelled() {
                    return Err("设备选择已取消".into());
                }
                return cpu_selection(caps, format!("{error}；选择CPU"));
            }
        }
    };
    if cancelled() {
        return Err("设备选择已取消".into());
    }
    select_from_probe(caps, budget, &detected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn failed_gpu_probe_uses_the_available_cpu_backend() {
        let mut caps = RuntimeCapabilities {
            cpu: true,
            metal: true,
            ..Default::default()
        };
        let budget = MemoryBudget { required_bytes: 1 };
        let selection = choose_runtime(Path::new(""), &caps, &budget, &|| false).unwrap();
        assert_eq!(selection.device, Device::Cpu);
        assert!(selection.device_selector.is_none());
        assert!(selection.reason.contains("探测无法启动"));

        caps.cpu = false;
        assert!(choose_runtime(Path::new(""), &caps, &budget, &|| false).is_err());
    }

    #[test]
    fn cancelling_a_probe_does_not_start_cpu_inference() {
        let caps = RuntimeCapabilities {
            cpu: true,
            metal: true,
            ..Default::default()
        };
        let budget = MemoryBudget { required_bytes: 1 };
        let calls = AtomicUsize::new(0);
        let cancelled = || calls.fetch_add(1, Ordering::SeqCst) > 0;
        let error = choose_runtime(Path::new(""), &caps, &budget, &cancelled).unwrap_err();
        assert_eq!(error, "设备选择已取消");
    }
}
