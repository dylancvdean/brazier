use std::path::Path;

use serde::Serialize;

use crate::runtime_settings::RuntimeTarget;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeTargetInfo {
    pub id: RuntimeTarget,
    pub name: &'static str,
    pub available: bool,
    pub recommended: bool,
    pub managed_install: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub os: &'static str,
    pub architecture: &'static str,
    pub logical_cpus: usize,
    pub memory_bytes: Option<u64>,
    pub gpu: Option<String>,
    pub targets: Vec<RuntimeTargetInfo>,
    pub recommended_target: RuntimeTarget,
}

fn command_exists(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| {
            let candidate = directory.join(name);
            candidate.is_file()
                || (cfg!(windows) && directory.join(format!("{name}.exe")).is_file())
        })
    })
}

#[cfg(target_os = "linux")]
fn linux_gpu() -> (bool, bool, Option<String>) {
    let mut nvidia = Path::new("/proc/driver/nvidia/version").exists();
    let mut amd = false;
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let device = entry.path().join("device");
            let vendor = std::fs::read_to_string(device.join("vendor")).unwrap_or_default();
            nvidia |= vendor.trim() == "0x10de";
            amd |= vendor.trim() == "0x1002";
            if let Ok(name) = std::fs::read_to_string(device.join("product_name")) {
                let name = name.trim();
                if !name.is_empty() && !names.iter().any(|known| known == name) {
                    names.push(name.to_owned());
                }
            }
        }
    }
    nvidia |= command_exists("nvidia-smi");
    amd |= command_exists("rocminfo") || command_exists("rocm-smi");
    (nvidia, amd, (!names.is_empty()).then(|| names.join(", ")))
}

#[cfg(not(target_os = "linux"))]
fn linux_gpu() -> (bool, bool, Option<String>) {
    (false, false, None)
}

fn memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        Some(kib * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

use std::sync::OnceLock;

static HARDWARE_CACHE: OnceLock<HardwareInfo> = OnceLock::new();

/// Detect hardware once per daemon process (PATH/sysfs scans are expensive).
pub fn detect() -> HardwareInfo {
    HARDWARE_CACHE.get_or_init(detect_uncached).clone()
}

fn detect_uncached() -> HardwareInfo {
    let (nvidia, amd, gpu_name) = linux_gpu();
    let metal = cfg!(target_os = "macos");
    let vulkan = command_exists("vulkaninfo")
        || Path::new("/usr/lib/libvulkan.so").exists()
        || Path::new("/usr/lib64/libvulkan.so").exists();
    let recommended_target = if metal {
        RuntimeTarget::Metal
    } else if nvidia {
        RuntimeTarget::Cuda
    } else if amd {
        RuntimeTarget::Rocm
    } else if vulkan {
        RuntimeTarget::Vulkan
    } else {
        RuntimeTarget::Cpu
    };
    let target = |id, name, available, managed_install, detail: &str| RuntimeTargetInfo {
        id,
        name,
        available,
        recommended: id == recommended_target,
        managed_install,
        detail: detail.to_owned(),
    };
    let mut targets = vec![target(
        RuntimeTarget::Cpu,
        "CPU",
        true,
        true,
        "Compatible fallback using system memory",
    )];
    if cfg!(target_os = "linux") || nvidia {
        targets.push(target(
            RuntimeTarget::Cuda,
            "NVIDIA CUDA",
            nvidia,
            cfg!(all(target_os = "linux", target_arch = "x86_64")),
            if nvidia {
                "NVIDIA GPU or driver detected"
            } else {
                "No NVIDIA GPU or driver detected"
            },
        ));
    }
    if cfg!(target_os = "linux") || amd {
        targets.push(target(
            RuntimeTarget::Rocm,
            "AMD ROCm",
            amd,
            cfg!(all(target_os = "linux", target_arch = "x86_64")),
            if amd {
                "AMD GPU or ROCm tooling detected"
            } else {
                "No AMD GPU or ROCm runtime detected"
            },
        ));
    }
    if metal {
        targets.push(target(
            RuntimeTarget::Metal,
            "Apple Metal",
            true,
            true,
            "Apple GPU acceleration",
        ));
    }
    if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
        targets.push(target(
            RuntimeTarget::Vulkan,
            "Vulkan",
            vulkan,
            cfg!(all(target_os = "linux", target_arch = "x86_64")),
            if vulkan {
                "Vulkan loader detected"
            } else {
                "No Vulkan loader detected"
            },
        ));
    }
    HardwareInfo {
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        logical_cpus: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        memory_bytes: memory_bytes(),
        gpu: gpu_name.or_else(|| {
            nvidia
                .then(|| "NVIDIA GPU".to_owned())
                .or_else(|| amd.then(|| "AMD GPU".to_owned()))
                .or_else(|| metal.then(|| "Apple GPU".to_owned()))
        }),
        targets,
        recommended_target,
    }
}
