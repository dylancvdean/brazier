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
    /// Dedicated video memory on a discrete GPU, when there is one.
    ///
    /// Absent on unified-memory machines (Apple Silicon, APUs), where the GPU
    /// draws on system memory and reporting a separate figure would be a
    /// fiction.
    pub vram_bytes: Option<u64>,
    /// Memory that a Vulkan backend can use for model weights. On a discrete
    /// GPU this is dedicated VRAM; on an AMD APU it is the kernel-reported GTT
    /// aperture, which is a safer placement ceiling than all system RAM.
    pub gpu_offload_memory_bytes: Option<u64>,
    /// The memory a model actually has to fit in: video memory on a discrete
    /// GPU, system memory otherwise.
    ///
    /// This is what model recommendations are sized against, so the answer
    /// lives here rather than being re-derived by each caller.
    pub usable_model_memory_bytes: Option<u64>,
    pub gpu: Option<String>,
    pub gpu_arch: Option<String>,
    /// An AMD GPU without local memory, so Vulkan allocations share system RAM.
    ///
    /// sd.cpp needs more conservative placement defaults on these devices than
    /// it does on a discrete Vulkan GPU.
    pub amd_apu: bool,
    /// An Intel GPU without local memory, sharing system RAM the way an AMD APU
    /// does. Intel integrated parts expose no VRAM or GTT aperture in sysfs, so
    /// the shared system pool is their placement budget.
    pub intel_igpu: bool,
    pub targets: Vec<RuntimeTargetInfo>,
    pub recommended_target: RuntimeTarget,
}

impl HardwareInfo {
    /// Whether the machine's accelerator draws on shared system memory.
    ///
    /// A unified-memory device is not a hard placement limit: model weights
    /// come out of the same RAM the OS and everything else use, so falling back
    /// to system memory is legitimate rather than an over-recommendation.
    pub fn integrated_gpu(&self) -> bool {
        self.amd_apu || self.intel_igpu
    }
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

/// A DRM VRAM figure below this is an integrated device's firmware carve-out,
/// not a model-placement budget. An AMD APU or Intel iGPU that reports a small
/// value is treated as unified memory; a multi-gigabyte value is decisive
/// evidence of a discrete card even if KFD topology said otherwise.
const MIN_DEDICATED_VRAM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

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

/// Whether an Intel GPU is present, and whether it is integrated.
///
/// Integrated Intel parts share system memory, so they get the same
/// unified-memory treatment as AMD APUs. A discrete Arc card is told apart by
/// its VRAM region: `mem_info_vram_total` reports a multi-gigabyte value on a
/// discrete part, and is absent or a small stolen carve-out on an iGPU.
#[cfg(target_os = "linux")]
fn intel_gpu() -> Option<bool> {
    let mut present = false;
    let mut integrated = true;
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let device = entry.path().join("device");
            let vendor = std::fs::read_to_string(device.join("vendor")).unwrap_or_default();
            if vendor.trim() != "0x8086" {
                continue;
            }
            present = true;
            let vram = std::fs::read_to_string(device.join("mem_info_vram_total"))
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
                .unwrap_or(0);
            if vram >= MIN_DEDICATED_VRAM_BYTES {
                integrated = false;
            }
        }
    }
    present.then_some(integrated)
}

#[cfg(not(target_os = "linux"))]
fn intel_gpu() -> Option<bool> {
    None
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
    let (mut nvidia, amd, mut gpu_name) = linux_gpu();
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

/// Total system memory.
///
/// Every platform is covered rather than Linux alone: model recommendations are
/// chosen by how much memory a machine has, and a machine that reports none
/// gets no recommendation at all — which on macOS and Windows would have been
/// every machine.
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
    #[cfg(target_os = "macos")]
    {
        run_first_line("sysctl", &["-n", "hw.memsize"])?
            .trim()
            .parse()
            .ok()
    }
    #[cfg(target_os = "windows")]
    {
        // No Windows API crate is linked, so this asks the OS the way the rest
        // of the module shells out for GPU facts.
        let output = run_first_line(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ],
        )?;
        output.trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Run a command and return its first non-empty line of output.
#[allow(dead_code)]
fn run_first_line(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

/// Run a command and return every non-empty line of output.
fn run_all_lines(program: &str, args: &[&str]) -> Option<Vec<String>> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
    )
}

/// Dedicated video memory on a discrete GPU.
///
/// Unified-memory machines deliberately report `None`: on Apple Silicon and on
/// APUs the GPU allocates out of system memory, so a separate VRAM figure would
/// either double-count it or understate what a model can use.
///
/// The DRM sysfs node `mem_info_vram_total` exists on AMD APUs too, where it
/// reports only a small firmware carveout. The caller decides whether a small
/// value is meaningful after it has combined this result with KFD topology.
fn vram_bytes() -> Option<u64> {
    if let Some(lines) = run_all_lines(
        "nvidia-smi",
        &["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    ) {
        // nvidia-smi reports mebibytes. The largest card wins, since that is
        // the one a model would be loaded onto.
        let mut best = 0_u64;
        for line in lines {
            if let Ok(mib) = line.trim().parse::<u64>() {
                best = best.max(mib);
            }
        }
        if best > 0 {
            return Some(best * 1024 * 1024);
        }
    }
    #[cfg(target_os = "linux")]
    {
        // AMD and Intel expose it through the DRM sysfs nodes. The largest card
        // wins, since that is the one a model would be loaded onto.
        let mut best = 0_u64;
        for entry in std::fs::read_dir("/sys/class/drm")
            .into_iter()
            .flatten()
            .flatten()
        {
            let total = entry.path().join("device").join("mem_info_vram_total");
            if let Ok(text) = std::fs::read_to_string(total)
                && let Ok(bytes) = text.trim().parse::<u64>()
            {
                best = best.max(bytes);
            }
        }
        if best > 0 {
            return Some(best);
        }
    }
    None
}

/// The amount of shared system memory that amdgpu exposes to GPU clients.
///
/// This is intentionally not used as `vram_bytes`: it is an allocation budget
/// for offloading model weights, not dedicated video memory. The largest value
/// wins on hosts with more than one AMD GPU.
#[cfg(target_os = "linux")]
fn amd_gtt_bytes() -> Option<u64> {
    let mut best = 0_u64;
    for entry in std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten()
    {
        let total = entry.path().join("device").join("mem_info_gtt_total");
        if let Ok(text) = std::fs::read_to_string(total)
            && let Ok(bytes) = text.trim().parse::<u64>()
        {
            best = best.max(bytes);
        }
    }
    (best > 0).then_some(best)
}

#[cfg(not(target_os = "linux"))]
fn amd_gtt_bytes() -> Option<u64> {
    None
}

/// Memory a model has to fit inside on this machine.
///
/// A discrete GPU's own memory is the binding constraint when there is one;
/// otherwise the model shares system memory, and that is the figure that
/// matters. Recommendations are sized against this.
pub fn usable_model_memory_bytes(vram: Option<u64>, system: Option<u64>) -> Option<u64> {
    vram.or(system)
}

/// Memory budget for choosing a downloadable model.
///
/// A discrete Linux or Windows GPU is a hard placement limit: falling back to
/// host RAM when its VRAM cannot be read recommends files that cannot run on
/// the accelerator. Unified-memory Macs and AMD APUs legitimately use system
/// memory instead.
pub fn recommendation_memory_bytes(hardware: &HardwareInfo) -> Option<u64> {
    let accelerator = hardware.gpu_offload_memory_bytes.or(hardware.vram_bytes);
    if hardware.os != "macos" && hardware.gpu.is_some() && !hardware.integrated_gpu() {
        return accelerator;
    }
    accelerator.or(hardware.memory_bytes)
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
    // The Metal backend is Apple Silicon-only; an Intel Mac is a CPU machine.
    let metal = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let vulkan = command_exists("vulkaninfo")
        || Path::new("/usr/lib/libvulkan.so").exists()
        || Path::new("/usr/lib64/libvulkan.so").exists()
        || (cfg!(target_os = "windows")
            && Path::new("C:\\Windows\\System32\\vulkan-1.dll").exists());
    let gpus = amd_gpus();
    let amd_apu_only = !gpus.is_empty() && gpus.iter().all(|gpu| gpu.integrated);
    let intel = intel_gpu();
    let raw_vram = vram_bytes();
    // An integrated device's small firmware carveout is not a model-placement
    // budget. A multi-gigabyte DRM value, however, is decisive evidence of a
    // discrete card even if KFD topology happened to report it as integrated.
    let integrated_only = amd_apu_only || matches!(intel, Some(true));
    let vram = if integrated_only && raw_vram.is_some_and(|bytes| bytes < MIN_DEDICATED_VRAM_BYTES)
    {
        None
    } else {
        raw_vram
    };
    let amd_apu = amd_apu_only && vram.is_none();
    // An Intel iGPU with no discrete VRAM anywhere draws on system memory, the
    // same situation an AMD APU is in. A discrete Intel Arc (or any discrete
    // card) leaves this false.
    let intel_igpu = matches!(intel, Some(true)) && vram.is_none();
    let gfx_arches: Vec<String> = gpus.iter().map(|gpu| gpu.arch.clone()).collect();
    // Verified only against an installed ROCm build, which is the only thing
    // that knows which architectures it carries device code for. Until one is
    // installed there is nothing to check, so ROCm is offered but not advised:
    // Vulkan runs on every one of these GPUs, ROCm does not.
    let rocm_verified = matches!(rocm_support(&gfx_arches), rocm::Support::Covered { .. });
    let rocm_available = amd;
    // An Intel iGPU with no discrete VRAM gets no benefit from the generic
    // Vulkan path that every other integrated GPU falls back to: llama.cpp's
    // oneAPI SYCL build is the tuned backend for Intel GPUs, so prefer it when
    // one exists.
    let recommended_target = if metal {
        RuntimeTarget::Metal
    } else if nvidia {
        RuntimeTarget::Cuda
    } else if amd && rocm_verified {
        RuntimeTarget::Rocm
    } else if intel_igpu {
        RuntimeTarget::Sycl
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
    // llama.cpp publishes managed prebuilts for these platform/target pairs.
    // CUDA and ROCm are x86_64 Linux only; Vulkan also ships for ARM64 Linux.
    let linux_managed_accelerated = cfg!(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ));
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
            linux_managed_accelerated,
            if vulkan {
                "Vulkan loader detected"
            } else {
                "No Vulkan loader detected"
            },
        ));
    }
    // llama.cpp publishes a oneAPI SYCL prebuilt for Intel GPUs on x86_64
    // Linux and Windows. It is only offered on hosts that report an Intel GPU;
    // anything else would install a build that cannot reach a device.
    targets.push(target(
        RuntimeTarget::Sycl,
        "Intel SYCL",
        intel.is_some(),
        cfg!(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "windows", target_arch = "x86_64")
        )),
        if intel.is_some() {
            "Intel GPU detected — oneAPI SYCL backend for Intel integrated and Arc GPUs"
        } else {
            "No Intel GPU detected"
        },
    ));
    let system_memory = memory_bytes();
    let gpu_offload_memory = vram.or_else(|| {
        if amd_apu {
            amd_gtt_bytes()
        } else if intel_igpu {
            // Intel exposes no GTT aperture in sysfs, so the shared system
            // pool is the honest ceiling — the same figure unified-memory Macs
            // draw on.
            system_memory
        } else {
            None
        }
    });
    HardwareInfo {
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        logical_cpus: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        memory_bytes: system_memory,
        vram_bytes: vram,
        gpu_offload_memory_bytes: gpu_offload_memory,
        usable_model_memory_bytes: usable_model_memory_bytes(vram, system_memory),
        gpu: gpu_name.or_else(|| {
            nvidia
                .then(|| "NVIDIA GPU".to_owned())
                .or_else(|| amd.then(|| "AMD GPU".to_owned()))
                .or_else(|| intel.is_some().then(|| "Intel GPU".to_owned()))
                .or_else(|| metal.then(|| "Apple GPU".to_owned()))
        }),
        gpu_arch: (!gfx_arches.is_empty()).then(|| gfx_arches.join(", ")),
        amd_apu,
        intel_igpu,
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

    /// A discrete GPU's own memory is what a model has to fit inside; without
    /// one the model shares system memory, and that is the figure to size
    /// against.
    #[test]
    fn usable_memory_prefers_video_memory_when_there_is_some() {
        let system = Some(64 * 1024 * 1024 * 1024);
        let vram = Some(24 * 1024 * 1024 * 1024);
        assert_eq!(usable_model_memory_bytes(vram, system), vram);
        assert_eq!(usable_model_memory_bytes(None, system), system);
        assert_eq!(usable_model_memory_bytes(None, None), None);
    }

    #[test]
    fn recommendations_never_substitute_host_ram_for_a_discrete_gpu() {
        let hardware = HardwareInfo {
            os: "linux",
            architecture: "x86_64",
            logical_cpus: 8,
            memory_bytes: Some(64 * 1024 * 1024 * 1024),
            vram_bytes: None,
            gpu_offload_memory_bytes: None,
            usable_model_memory_bytes: Some(64 * 1024 * 1024 * 1024),
            gpu: Some("AMD GPU".into()),
            gpu_arch: None,
            amd_apu: false,
            intel_igpu: false,
            targets: Vec::new(),
            recommended_target: RuntimeTarget::Vulkan,
        };
        assert_eq!(recommendation_memory_bytes(&hardware), None);
    }

    /// An Intel iGPU shares system memory the way an AMD APU does, so it is
    /// treated as unified memory rather than as a discrete hard limit.
    #[test]
    fn an_intel_igpu_sizes_recommendations_against_system_memory() {
        let hardware = HardwareInfo {
            os: "linux",
            architecture: "x86_64",
            logical_cpus: 16,
            memory_bytes: Some(32 * 1024 * 1024 * 1024),
            vram_bytes: None,
            gpu_offload_memory_bytes: Some(32 * 1024 * 1024 * 1024),
            usable_model_memory_bytes: Some(32 * 1024 * 1024 * 1024),
            gpu: Some("Intel GPU".into()),
            gpu_arch: None,
            amd_apu: false,
            intel_igpu: true,
            targets: Vec::new(),
            recommended_target: RuntimeTarget::Vulkan,
        };
        assert!(hardware.integrated_gpu());
        assert_eq!(
            recommendation_memory_bytes(&hardware),
            Some(32 * 1024 * 1024 * 1024)
        );
    }

    /// A discrete Intel Arc is not unified memory: it has its own VRAM and is a
    /// hard placement limit like any other discrete card.
    #[test]
    fn a_discrete_intel_card_is_not_unified_memory() {
        let hardware = HardwareInfo {
            os: "linux",
            architecture: "x86_64",
            logical_cpus: 16,
            memory_bytes: Some(64 * 1024 * 1024 * 1024),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
            gpu_offload_memory_bytes: Some(16 * 1024 * 1024 * 1024),
            usable_model_memory_bytes: Some(16 * 1024 * 1024 * 1024),
            gpu: Some("Intel Arc".into()),
            gpu_arch: None,
            amd_apu: false,
            intel_igpu: false,
            targets: Vec::new(),
            recommended_target: RuntimeTarget::Vulkan,
        };
        assert!(!hardware.integrated_gpu());
        assert_eq!(
            recommendation_memory_bytes(&hardware),
            Some(16 * 1024 * 1024 * 1024)
        );
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
