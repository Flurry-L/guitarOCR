//! Probe the installed Windows driver before downloading CUDA libraries.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub fn available(required_bytes: u64) -> bool {
    unsafe fn detect(required_bytes: u64) -> Option<bool> {
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
            || driver_version < 12080
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
    unsafe { detect(required_bytes).unwrap_or(false) }
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
pub fn available(_required_bytes: u64) -> bool {
    false
}
