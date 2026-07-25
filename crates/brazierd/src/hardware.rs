use std::path::Path;

use serde::Serialize;

use crate::{rocm, runtime_settings::RuntimeTarget};

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

/// One AMD GPU as the kernel describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdGpu {
    pub arch: String,
    /// No local VRAM: an APU sharing system memory. Used for wording only —
    /// whether ROCm works is decided by the build, not by this. Some APUs are
    /// covered by the ROCm builds and some discrete cards are not.
    pub integrated: bool,
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
// KFD is a Linux interface, but the decoding is worth testing on any host.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn gfx_arch_name(version: u32) -> Option<String> {
    if version == 0 {
        return None;
    }
    let major = version / 10000;
    let minor = (version / 100) % 100;
    let step = version % 100;
    (minor <= 0xf && step <= 0xf).then(|| format!("gfx{major}{minor:x}{step:x}"))
}

/// Read one numeric field out of a KFD node's `properties` file.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn kfd_property(properties: &str, key: &str) -> Option<u64> {
    properties
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|value| value.trim().parse().ok())
}

/// Every AMD GPU the kernel exposes through KFD topology.
///
/// The amdgpu driver publishes this without any ROCm userspace installed. The
/// CPU node reports a zero architecture version and is skipped.
#[cfg(target_os = "linux")]
fn amd_gpus() -> Vec<AmdGpu> {
    let Ok(entries) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
        return Vec::new();
    };
    let mut gpus: Vec<AmdGpu> = Vec::new();
    for entry in entries.flatten() {
        let Ok(properties) = std::fs::read_to_string(entry.path().join("properties")) else {
            continue;
        };
        let Some(arch) = kfd_property(&properties, "gfx_target_version ")
            .and_then(|version| u32::try_from(version).ok())
            .and_then(gfx_arch_name)
        else {
            continue;
        };
        if gpus.iter().any(|gpu| gpu.arch == arch) {
            continue;
        }
        let integrated = kfd_property(&properties, "local_mem_size ").unwrap_or(0) == 0;
        gpus.push(AmdGpu { arch, integrated });
    }
    gpus
}

#[cfg(not(target_os = "linux"))]
fn amd_gpus() -> Vec<AmdGpu> {
    Vec::new()
}

/// gfx architectures of the AMD GPUs on this machine, for the ROCm check.
pub fn amd_gfx_arches() -> Vec<String> {
    amd_gpus().into_iter().map(|gpu| gpu.arch).collect()
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

/// Where an installed managed ROCm llama.cpp build puts its binaries.
fn rocm_install_bin(data_dir: &Path) -> std::path::PathBuf {
    crate::llama::managed_engine_dir(data_dir)
        .join(RuntimeTarget::Rocm.as_str())
        .join("bin")
}

/// Check this machine's GPUs against an installed ROCm build, if there is one.
///
/// Detection runs before any data directory is known, so the path is resolved
/// from the same default the daemon uses. A missing build simply reports
/// `Unknown`, which is neither a pass nor a failure.
pub fn rocm_support(gfx_arches: &[String]) -> rocm::Support {
    let Some(data_dir) = default_data_dir() else {
        return rocm::Support::Unknown;
    };
    rocm::support(
        gfx_arches,
        &rocm::install_arches(&rocm_install_bin(&data_dir)),
    )
}

fn default_data_dir() -> Option<std::path::PathBuf> {
    if let Some(value) = std::env::var_os("BRAZIER_DATA_DIR") {
        return Some(std::path::PathBuf::from(value));
    }
    dirs_data_dir()
}

#[cfg(target_os = "linux")]
fn dirs_data_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/share")))
        .map(|base| base.join("brazier"))
}

#[cfg(not(target_os = "linux"))]
fn dirs_data_dir() -> Option<std::path::PathBuf> {
    None
}

/// What to tell the user about the ROCm target.
fn rocm_detail(amd: bool, gpus: &[AmdGpu], support: rocm::Support) -> String {
    if !amd {
        return "No AMD GPU or ROCm runtime detected".to_owned();
    }
    match support {
        rocm::Support::Covered { arch } => {
            format!("Verified: the installed ROCm build has device code for {arch}")
        }
        rocm::Support::Uncovered { build_arches, .. } => format!(
            "The installed ROCm build covers {} and not this GPU — use Vulkan",
            build_arches.join(", ")
        ),
        rocm::Support::Unknown => {
            let integrated = gpus.iter().any(|gpu| gpu.integrated);
            let named = if gpus.is_empty() {
                "AMD GPU detected".to_owned()
            } else {
                format!(
                    "AMD {} detected",
                    gpus.iter()
                        .map(|gpu| gpu.arch.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            // The APU caveat the recommendation cannot make for itself: the
            // ROCm releases carry device code for discrete parts, and running
            // an uncovered GPU hangs it rather than failing.
            let caveat = if integrated {
                "This looks like integrated graphics (an APU), which the ROCm builds usually do \
                 not cover — Vulkan is the safe choice. Install ROCm to check for certain."
            } else {
                "Not for integrated graphics (APUs): the ROCm builds carry device code for a \
                 limited set of architectures, and an uncovered GPU hangs rather than failing. \
                 Brazier checks against the build when you install it."
            };
            format!("{named}. {caveat}")
        }
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
    let gpus = amd_gpus();
    let gfx_arches: Vec<String> = gpus.iter().map(|gpu| gpu.arch.clone()).collect();
    // Verified only against an installed ROCm build, which is the only thing
    // that knows which architectures it carries device code for. Until one is
    // installed there is nothing to check, so ROCm is offered but not advised:
    // Vulkan runs on every one of these GPUs, ROCm does not.
    let rocm_verified = matches!(rocm_support(&gfx_arches), rocm::Support::Covered { .. });
    let rocm_available = amd;
    let recommended_target = if metal {
        RuntimeTarget::Metal
    } else if nvidia {
        RuntimeTarget::Cuda
    } else if amd && rocm_verified {
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
        let detail = rocm_detail(amd, &gpus, rocm_support(&gfx_arches));
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
    fn reads_a_kfd_node_as_the_kernel_writes_it() {
        // A Renoir APU node: shares system memory, so no local VRAM.
        let apu = "cpu_cores_count 0\nsimd_count 28\ngfx_target_version 90012\nlocal_mem_size 0\n";
        assert_eq!(kfd_property(apu, "gfx_target_version "), Some(90012));
        assert_eq!(kfd_property(apu, "local_mem_size "), Some(0));
        assert_eq!(kfd_property(apu, "absent_key "), None);

        let discrete = "gfx_target_version 110000\nlocal_mem_size 25753026560\n";
        assert_eq!(kfd_property(discrete, "local_mem_size "), Some(25753026560));
    }

    /// The wording carries the APU caveat, but the verdict never comes from it:
    /// some APUs are covered by the ROCm builds and some discrete cards are not,
    /// so only the build itself decides.
    #[test]
    fn an_unverified_amd_gpu_is_offered_with_the_apu_caveat() {
        let apu = [AmdGpu {
            arch: "gfx90c".to_owned(),
            integrated: true,
        }];
        let detail = rocm_detail(true, &apu, rocm::Support::Unknown);
        assert!(detail.contains("gfx90c"), "{detail}");
        assert!(detail.contains("APU"), "{detail}");
        assert!(detail.contains("Vulkan"), "{detail}");

        let discrete = [AmdGpu {
            arch: "gfx1100".to_owned(),
            integrated: false,
        }];
        let detail = rocm_detail(true, &discrete, rocm::Support::Unknown);
        assert!(detail.contains("gfx1100"), "{detail}");
        assert!(detail.contains("APUs"), "{detail}");
    }

    #[test]
    fn a_verified_build_says_so_and_an_uncovered_one_points_at_vulkan() {
        let gpus = [AmdGpu {
            arch: "gfx1100".to_owned(),
            integrated: false,
        }];
        let covered = rocm_detail(
            true,
            &gpus,
            rocm::Support::Covered {
                arch: "gfx1100".to_owned(),
            },
        );
        assert!(covered.starts_with("Verified"), "{covered}");
        assert!(covered.contains("gfx1100"), "{covered}");

        let uncovered = rocm_detail(
            true,
            &gpus,
            rocm::Support::Uncovered {
                gpu_arches: vec!["gfx90c".to_owned()],
                build_arches: vec!["gfx1030".to_owned(), "gfx1100".to_owned()],
            },
        );
        assert!(uncovered.contains("gfx1030, gfx1100"), "{uncovered}");
        assert!(uncovered.contains("Vulkan"), "{uncovered}");

        assert_eq!(
            rocm_detail(false, &[], rocm::Support::Unknown),
            "No AMD GPU or ROCm runtime detected"
        );
    }
}
