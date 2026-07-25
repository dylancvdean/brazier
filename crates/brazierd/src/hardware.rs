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
    pub gpu_arch: Option<String>,
    pub targets: Vec<RuntimeTargetInfo>,
    pub recommended_target: RuntimeTarget,
}

/// gfx architectures compiled into the ROCm llama.cpp releases Brazier installs.
///
/// A GPU outside this list still enumerates as a ROCm device, so llama.cpp
/// commits to the HIP backend and then dispatches a kernel that has no code
/// object for the hardware. That wedges the HSA queue and the runtime aborts
/// with "HW Exception ... GPU Hang" instead of failing cleanly, so ROCm must
/// not be offered for those GPUs. AMD APUs (gfx90c, gfx902, gfx1010) are the
/// common case: an AMD vendor ID alone says nothing about ROCm support.
const ROCM_SUPPORTED_ARCHES: &[&str] = &[
    "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151",
    "gfx1200", "gfx1201",
];

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

#[cfg(target_os = "windows")]
fn windows_gpu() -> (bool, Option<String>) {
    let nvidia = command_exists("nvidia-smi");
    (nvidia, nvidia.then(|| "NVIDIA GPU".to_owned()))
}

#[cfg(not(target_os = "windows"))]
fn windows_gpu() -> (bool, Option<String>) {
    (false, None)
}

fn gpu_capabilities() -> (bool, bool, Option<String>) {
    let (mut nvidia, mut amd, mut gpu_name) = linux_gpu();
    let (windows_nvidia, windows_gpu_name) = windows_gpu();
    nvidia |= windows_nvidia;
    if gpu_name.is_none() {
        gpu_name = windows_gpu_name;
    }
    (nvidia, amd, gpu_name)
}

/// Render a KFD `gfx_target_version` as a gfx architecture name.
///
/// The version packs `major * 10000 + minor * 100 + step`, where minor and step
/// are single hex digits in the name: 90012 is gfx90c, 90010 is gfx90a.
fn gfx_arch_name(version: u32) -> Option<String> {
    if version == 0 {
        return None;
    }
    let major = version / 10000;
    let minor = (version / 100) % 100;
    let step = version % 100;
    (minor <= 0xf && step <= 0xf).then(|| format!("gfx{major}{minor:x}{step:x}"))
}

/// gfx architectures of every AMD GPU the kernel exposes through KFD topology.
///
/// The amdgpu driver publishes this without any ROCm userspace installed. The
/// CPU node reports a zero version and is skipped.
#[cfg(target_os = "linux")]
fn amd_gfx_arches() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
        return Vec::new();
    };
    let mut arches = Vec::new();
    for entry in entries.flatten() {
        let Ok(properties) = std::fs::read_to_string(entry.path().join("properties")) else {
            continue;
        };
        let arch = properties
            .lines()
            .find_map(|line| line.strip_prefix("gfx_target_version "))
            .and_then(|value| value.trim().parse::<u32>().ok())
            .and_then(gfx_arch_name);
        if let Some(arch) = arch
            && !arches.contains(&arch)
        {
            arches.push(arch);
        }
    }
    arches
}

#[cfg(not(target_os = "linux"))]
fn amd_gfx_arches() -> Vec<String> {
    Vec::new()
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
    let (nvidia, amd, gpu_name) = gpu_capabilities();
    let metal = cfg!(target_os = "macos");
    let vulkan = command_exists("vulkaninfo")
        || Path::new("/usr/lib/libvulkan.so").exists()
        || Path::new("/usr/lib64/libvulkan.so").exists()
        || (cfg!(target_os = "windows")
            && Path::new("C:\\Windows\\System32\\vulkan-1.dll").exists());
    let gfx_arches = amd_gfx_arches();
    let rocm_supported = gfx_arches
        .iter()
        .any(|arch| ROCM_SUPPORTED_ARCHES.contains(&arch.as_str()));
    // An architecture we read and did not recognise fails as a GPU hang, so hide
    // ROCm entirely. An architecture we could not read stays selectable, because
    // the read is the uncertain part, but it is never the recommendation.
    let rocm_available = amd && (rocm_supported || gfx_arches.is_empty());
    let recommended_target = if metal {
        RuntimeTarget::Metal
    } else if nvidia {
        RuntimeTarget::Cuda
    } else if amd && rocm_supported {
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
        let detail = if !amd {
            "No AMD GPU or ROCm runtime detected".to_owned()
        } else if rocm_supported {
            "AMD GPU or ROCm tooling detected".to_owned()
        } else if gfx_arches.is_empty() {
            "AMD GPU detected but its architecture could not be read — prefer Vulkan".to_owned()
        } else {
            format!(
                "{} is not built into the ROCm releases Brazier installs — use Vulkan",
                gfx_arches.join(", ")
            )
        };
        targets.push(target(
            RuntimeTarget::Rocm,
            "AMD ROCm",
            rocm_available,
            cfg!(all(target_os = "linux", target_arch = "x86_64")),
            &detail,
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
        gpu_arch: (!gfx_arches.is_empty()).then(|| gfx_arches.join(", ")),
        targets,
        recommended_target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_gfx_target_versions() {
        assert_eq!(gfx_arch_name(90012).as_deref(), Some("gfx90c"));
        assert_eq!(gfx_arch_name(90010).as_deref(), Some("gfx90a"));
        assert_eq!(gfx_arch_name(90008).as_deref(), Some("gfx908"));
        assert_eq!(gfx_arch_name(90402).as_deref(), Some("gfx942"));
        assert_eq!(gfx_arch_name(100300).as_deref(), Some("gfx1030"));
        assert_eq!(gfx_arch_name(110001).as_deref(), Some("gfx1101"));
        assert_eq!(gfx_arch_name(120001).as_deref(), Some("gfx1201"));
    }

    #[test]
    fn ignores_the_cpu_topology_node() {
        assert_eq!(gfx_arch_name(0), None);
    }

    #[test]
    fn apu_architectures_are_not_rocm_capable() {
        // gfx90a (CDNA2) is supported and gfx90c (Renoir APU) is not, despite
        // decoding from adjacent version numbers.
        assert!(ROCM_SUPPORTED_ARCHES.contains(&"gfx90a"));
        for apu in ["gfx90c", "gfx902", "gfx1010"] {
            assert!(
                !ROCM_SUPPORTED_ARCHES.contains(&apu),
                "{apu} must not qualify"
            );
        }
    }
}
