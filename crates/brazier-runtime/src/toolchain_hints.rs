//! Platform-specific install commands for build prerequisites.

use std::path::{Path, PathBuf};

use crate::runtime_settings::RuntimeTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsFamily {
    Linux,
    Windows,
    Macos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageManager {
    Pacman,
    Apt,
    Dnf,
    Zypper,
    Apk,
    Xbps,
    Brew,
    Winget,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct OsInfo {
    pub family: OsFamily,
    pub id: String,
    pub id_like: Vec<String>,
    pub pretty_name: String,
    pub package_manager: PackageManager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolchainPackage {
    Git,
    Cmake,
    CppBuild,
    RocmHip,
    Cuda,
    Vulkan,
    Uv,
    Ffmpeg,
}

/// The small set of host capabilities that can make an optional tool useful.
///
/// Managed runtimes do not need the source-build toolchain. Keeping this
/// decision explicit lets the welcome screen ask what the person wants before
/// reporting scary-looking prerequisites that do not apply to them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolchainNeeds {
    pub custom_runtimes: bool,
    pub voice: bool,
    pub computer_use: bool,
    pub video: bool,
}

pub fn required_tool_ids(needs: ToolchainNeeds) -> Vec<&'static str> {
    let mut ids = Vec::new();
    let os = detect_os();

    // uv is shipped/used by the Python engines. On macOS it is part of the
    // supported baseline; elsewhere it is only needed when the chosen flow
    // actually asks for a Python runtime.
    if matches!(os.family, OsFamily::Macos)
        || needs.custom_runtimes
        || needs.voice
        || needs.computer_use
    {
        ids.push("uv");
    }
    if needs.custom_runtimes {
        ids.extend(["git", "cmake", "cpp"]);
    }
    if needs.video {
        ids.push("ffmpeg");
    }
    ids
}

pub fn detect_os() -> OsInfo {
    if cfg!(target_os = "windows") {
        return OsInfo {
            family: OsFamily::Windows,
            id: "windows".into(),
            id_like: Vec::new(),
            pretty_name: "Windows".into(),
            package_manager: PackageManager::Winget,
        };
    }
    if cfg!(target_os = "macos") {
        return OsInfo {
            family: OsFamily::Macos,
            id: "macos".into(),
            id_like: Vec::new(),
            pretty_name: "macOS".into(),
            package_manager: PackageManager::Brew,
        };
    }

    let content = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    let mut id = String::new();
    let mut pretty_name = String::new();
    let mut id_like = Vec::new();
    for line in content.lines() {
        if let Some(value) = line.strip_prefix("ID=") {
            id = trim_os_value(value);
        } else if let Some(value) = line.strip_prefix("PRETTY_NAME=") {
            pretty_name = trim_os_value(value);
        } else if let Some(value) = line.strip_prefix("ID_LIKE=") {
            id_like = trim_os_value(value)
                .split_whitespace()
                .map(str::to_owned)
                .collect();
        }
    }
    if pretty_name.is_empty() {
        pretty_name = if id.is_empty() {
            "Linux".into()
        } else {
            id.clone()
        };
    }
    let package_manager = infer_package_manager(&id, &id_like);
    OsInfo {
        family: OsFamily::Linux,
        id,
        id_like,
        pretty_name,
        package_manager,
    }
}

fn trim_os_value(value: &str) -> String {
    value.trim().trim_matches('"').to_owned()
}

fn infer_package_manager(id: &str, id_like: &[String]) -> PackageManager {
    let id = id.to_ascii_lowercase();
    if id == "arch" || id == "manjaro" || id == "endeavouros" || id == "garuda" || id == "cachyos" {
        return PackageManager::Pacman;
    }
    if id == "ubuntu"
        || id == "debian"
        || id == "linuxmint"
        || id == "pop"
        || id == "elementary"
        || id == "zorin"
        || id == "kali"
    {
        return PackageManager::Apt;
    }
    if id == "fedora"
        || id == "centos"
        || id == "rhel"
        || id == "rocky"
        || id == "almalinux"
        || id == "nobara"
        || id == "ultramarine"
        || id == "azurelinux"
    {
        return PackageManager::Dnf;
    }
    if id == "opensuse-leap" || id == "opensuse-tumbleweed" || id == "suse" {
        return PackageManager::Zypper;
    }
    if id == "alpine" {
        return PackageManager::Apk;
    }
    if id == "void" {
        return PackageManager::Xbps;
    }
    if id == "nixos" {
        return PackageManager::Unknown;
    }
    for like in id_like {
        match like.as_str() {
            "arch" => return PackageManager::Pacman,
            "debian" | "ubuntu" => return PackageManager::Apt,
            "fedora" | "rhel" | "centos" => return PackageManager::Dnf,
            "suse" => return PackageManager::Zypper,
            _ => {}
        }
    }
    PackageManager::Unknown
}

pub fn install_command(package: ToolchainPackage, os: &OsInfo) -> Option<String> {
    match (os.package_manager, package) {
        (PackageManager::Pacman, ToolchainPackage::Git) => Some("sudo pacman -S git".into()),
        (PackageManager::Pacman, ToolchainPackage::Cmake) => Some("sudo pacman -S cmake".into()),
        (PackageManager::Pacman, ToolchainPackage::CppBuild) => {
            Some("sudo pacman -S base-devel".into())
        }
        (PackageManager::Pacman, ToolchainPackage::RocmHip) => Some(
            "sudo pacman -S rocm-hip-sdk rocm-opencl-sdk hipsparselt base-devel cmake git".into(),
        ),
        (PackageManager::Pacman, ToolchainPackage::Cuda) => Some(
            "Enable the `extra` repo if needed, then: sudo pacman -S cuda".into(),
        ),
        (PackageManager::Pacman, ToolchainPackage::Vulkan) => {
            Some("sudo pacman -S spirv-headers vulkan-devel cmake extra/glslang".into())
        }
        (PackageManager::Pacman, ToolchainPackage::Ffmpeg) => Some("sudo pacman -S ffmpeg".into()),

        (PackageManager::Apt, ToolchainPackage::Git) => Some("sudo apt install git".into()),
        (PackageManager::Apt, ToolchainPackage::Cmake) => Some("sudo apt install cmake".into()),
        (PackageManager::Apt, ToolchainPackage::CppBuild) => {
            Some("sudo apt install build-essential".into())
        }
        (PackageManager::Apt, ToolchainPackage::RocmHip) => Some(
            "Follow AMD's ROCm install guide for your Ubuntu/Debian release (https://rocm.docs.amd.com), then install git, cmake, and build-essential.".into(),
        ),
        (PackageManager::Apt, ToolchainPackage::Cuda) => Some(
            "Install the NVIDIA CUDA toolkit for your distro from https://developer.nvidia.com/cuda-downloads".into(),
        ),
        (PackageManager::Apt, ToolchainPackage::Vulkan) => {
            Some("sudo apt install vulkan-tools libvulkan-dev glslang-tools mesa-vulkan-drivers".into())
        }
        (PackageManager::Apt, ToolchainPackage::Ffmpeg) => Some("sudo apt install ffmpeg".into()),

        (PackageManager::Dnf, ToolchainPackage::Git) => Some("sudo dnf install git".into()),
        (PackageManager::Dnf, ToolchainPackage::Cmake) => Some("sudo dnf install cmake".into()),
        (PackageManager::Dnf, ToolchainPackage::CppBuild) => {
            Some("sudo dnf groupinstall \"Development Tools\"".into())
        }
        (PackageManager::Dnf, ToolchainPackage::RocmHip) => Some(
            "sudo dnf install rocm-hip-sdk rocm-opencl-sdk git cmake gcc-c++".into(),
        ),
        (PackageManager::Dnf, ToolchainPackage::Cuda) => Some(
            "Install the NVIDIA CUDA repo for Fedora/RHEL, then: sudo dnf install cuda-toolkit".into(),
        ),
        (PackageManager::Dnf, ToolchainPackage::Vulkan) => {
            Some("sudo dnf install vulkan-tools vulkan-loader-devel glslang mesa-vulkan-drivers".into())
        }
        (PackageManager::Dnf, ToolchainPackage::Ffmpeg) => Some("sudo dnf install ffmpeg".into()),

        (PackageManager::Zypper, ToolchainPackage::Git) => Some("sudo zypper install git".into()),
        (PackageManager::Zypper, ToolchainPackage::Cmake) => {
            Some("sudo zypper install cmake".into())
        }
        (PackageManager::Zypper, ToolchainPackage::CppBuild) => {
            Some("sudo zypper install -t pattern devel_C_C++".into())
        }
        (PackageManager::Zypper, ToolchainPackage::RocmHip) => Some(
            "sudo zypper install rocm-hip-sdk rocm-opencl-sdk git cmake".into(),
        ),
        (PackageManager::Zypper, ToolchainPackage::Cuda) => {
            Some("Install NVIDIA CUDA for openSUSE from NVIDIA's repository".into())
        }
        (PackageManager::Zypper, ToolchainPackage::Vulkan) => {
            Some("sudo zypper install vulkan-tools vulkan-devel".into())
        }
        (PackageManager::Zypper, ToolchainPackage::Ffmpeg) => {
            Some("sudo zypper install ffmpeg".into())
        }

        (PackageManager::Apk, ToolchainPackage::Git) => Some("sudo apk add git".into()),
        (PackageManager::Apk, ToolchainPackage::Cmake) => Some("sudo apk add cmake".into()),
        (PackageManager::Apk, ToolchainPackage::CppBuild) => {
            Some("sudo apk add build-base linux-headers".into())
        }
        (PackageManager::Apk, ToolchainPackage::RocmHip) => Some(
            "ROCm on Alpine is unsupported for most GPUs. Use CPU or Vulkan instead.".into(),
        ),
        (PackageManager::Apk, ToolchainPackage::Cuda) => Some(
            "Install NVIDIA drivers and CUDA from Alpine/community docs, or use CPU/Vulkan.".into(),
        ),
        (PackageManager::Apk, ToolchainPackage::Vulkan) => {
            Some("sudo apk add vulkan-dev vulkan-loader mesa-dev".into())
        }
        (PackageManager::Apk, ToolchainPackage::Ffmpeg) => Some("sudo apk add ffmpeg".into()),

        (PackageManager::Xbps, ToolchainPackage::Git) => Some("sudo xbps-install -S git".into()),
        (PackageManager::Xbps, ToolchainPackage::Cmake) => {
            Some("sudo xbps-install -S cmake".into())
        }
        (PackageManager::Xbps, ToolchainPackage::CppBuild) => {
            Some("sudo xbps-install -S base-devel".into())
        }
        (PackageManager::Xbps, ToolchainPackage::RocmHip) => Some(
            "Install ROCm packages from Void repositories if available, or use CPU/Vulkan.".into(),
        ),
        (PackageManager::Xbps, ToolchainPackage::Cuda) => Some(
            "Install NVIDIA drivers/CUDA from Void docs, or use CPU/Vulkan.".into(),
        ),
        (PackageManager::Xbps, ToolchainPackage::Vulkan) => {
            Some("sudo xbps-install -S vulkan-loader vulkan-headers mesa-dri".into())
        }
        (PackageManager::Xbps, ToolchainPackage::Ffmpeg) => {
            Some("sudo xbps-install -S ffmpeg".into())
        }

        (PackageManager::Brew, ToolchainPackage::Git) => Some("brew install git".into()),
        (PackageManager::Brew, ToolchainPackage::Cmake) => Some("brew install cmake".into()),
        (PackageManager::Brew, ToolchainPackage::CppBuild) => Some(
            "xcode-select --install   # Apple Command Line Tools (clang, make, headers)".into(),
        ),
        (PackageManager::Brew, ToolchainPackage::RocmHip) => None,
        (PackageManager::Brew, ToolchainPackage::Cuda) => None,
        (PackageManager::Brew, ToolchainPackage::Vulkan) => {
            Some("brew install molten-vk vulkan-loader   # optional; Metal is preferred on macOS".into())
        }
        (PackageManager::Brew, ToolchainPackage::Uv) => {
            Some("brew install uv   # or: curl -LsSf https://astral.sh/uv/install.sh | sh".into())
        }
        (PackageManager::Brew, ToolchainPackage::Ffmpeg) => Some("brew install ffmpeg".into()),

        (PackageManager::Winget, ToolchainPackage::Git) => Some(
            "winget install --id Git.Git -e --source winget".into(),
        ),
        (PackageManager::Winget, ToolchainPackage::Cmake) => Some(
            "winget install --id Kitware.CMake -e --source winget".into(),
        ),
        (PackageManager::Winget, ToolchainPackage::CppBuild) => Some(
            "winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget   # then add the \"Desktop development with C++\" workload in Visual Studio Installer".into(),
        ),
        (PackageManager::Winget, ToolchainPackage::RocmHip) => None,
        (PackageManager::Winget, ToolchainPackage::Cuda) => Some(
            "Install the NVIDIA CUDA toolkit from https://developer.nvidia.com/cuda-downloads (Windows x86_64)".into(),
        ),
        (PackageManager::Winget, ToolchainPackage::Vulkan) => Some(
            "Install the LunarG Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows and ensure your GPU driver is up to date".into(),
        ),
        (PackageManager::Winget, ToolchainPackage::Uv) => Some(
            "winget install --id astral-sh.uv -e --source winget".into(),
        ),
        (PackageManager::Winget, ToolchainPackage::Ffmpeg) => Some(
            "winget install --id Gyan.FFmpeg -e --source winget".into(),
        ),

        (PackageManager::Pacman, ToolchainPackage::Uv)
        | (PackageManager::Apt, ToolchainPackage::Uv)
        | (PackageManager::Dnf, ToolchainPackage::Uv)
        | (PackageManager::Zypper, ToolchainPackage::Uv)
        | (PackageManager::Apk, ToolchainPackage::Uv)
        | (PackageManager::Xbps, ToolchainPackage::Uv)
        | (PackageManager::Unknown, ToolchainPackage::Uv) => Some(
            "Install uv from https://docs.astral.sh/uv/ and ensure it is on your PATH.".into(),
        ),

        (PackageManager::Unknown, ToolchainPackage::Git)
        | (PackageManager::Unknown, ToolchainPackage::Cmake)
        | (PackageManager::Unknown, ToolchainPackage::CppBuild) if os.id == "nixos" => {
            Some(
                "nix-shell -p git cmake gcc   # or add git, cmake, and a C++ toolchain to your NixOS configuration".into(),
            )
        }
        (PackageManager::Unknown, _) => None,
    }
}

pub fn install_hint(package: ToolchainPackage) -> String {
    let os = detect_os();
    install_hint_for_os(package, &os)
}

pub fn install_hint_for_os(package: ToolchainPackage, os: &OsInfo) -> String {
    let fallback = generic_fallback(package, os.family);
    if let Some(command) = install_command(package, os) {
        format!("On {}: `{}`.", os.pretty_name, command)
    } else {
        fallback
    }
}

fn generic_fallback(package: ToolchainPackage, family: OsFamily) -> String {
    match (package, family) {
        (ToolchainPackage::Git, OsFamily::Windows) => {
            "Install Git for Windows and ensure `git.exe` is on your PATH.".into()
        }
        (ToolchainPackage::Git, _) => "Install Git and ensure it is on your PATH.".into(),
        (ToolchainPackage::Cmake, OsFamily::Windows) => {
            "Install CMake 3.20+ for Windows and ensure `cmake.exe` is on your PATH.".into()
        }
        (ToolchainPackage::Cmake, _) => {
            "Install CMake 3.20+ and ensure it is on your PATH.".into()
        }
        (ToolchainPackage::CppBuild, OsFamily::Windows) => {
            "Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload, or full Visual Studio with C++.".into()
        }
        (ToolchainPackage::CppBuild, OsFamily::Macos) => {
            "Install Xcode Command Line Tools (`xcode-select --install`) or Xcode from the App Store.".into()
        }
        (ToolchainPackage::CppBuild, _) => {
            "Install a C/C++ toolchain (build-essential, Xcode CLT, or Visual Studio Build Tools).".into()
        }
        (ToolchainPackage::RocmHip, OsFamily::Windows | OsFamily::Macos) => {
            "ROCm builds are Linux-only. Switch the build target to CPU, CUDA, Vulkan, or Metal.".into()
        }
        (ToolchainPackage::RocmHip, _) => {
            "Install AMD ROCm with the HIP compiler (hipcc), or switch the build target to CPU.".into()
        }
        (ToolchainPackage::Cuda, OsFamily::Macos) => {
            "CUDA builds are not supported on macOS. Use CPU or Metal instead.".into()
        }
        (ToolchainPackage::Cuda, _) => {
            "Install the NVIDIA CUDA toolkit, or switch the build target to CPU.".into()
        }
        (ToolchainPackage::Vulkan, OsFamily::Macos) => {
            "Vulkan is optional on macOS; Metal is the recommended GPU target.".into()
        }
        (ToolchainPackage::Vulkan, _) => {
            "Install Vulkan headers/loader and drivers, or switch the build target to CPU.".into()
        }
        (ToolchainPackage::Uv, OsFamily::Macos) => {
            "Install uv with Homebrew (`brew install uv`) or Astral's installer script, then ensure `uv` is on your PATH.".into()
        }
        (ToolchainPackage::Uv, _) => {
            "Install uv from https://docs.astral.sh/uv/ and ensure it is on your PATH.".into()
        }
        (ToolchainPackage::Ffmpeg, OsFamily::Macos) => {
            "Install ffmpeg with Homebrew (`brew install ffmpeg`) and ensure `ffmpeg` / `ffprobe` are on your PATH.".into()
        }
        (ToolchainPackage::Ffmpeg, _) => {
            "Install ffmpeg (and ffprobe) from your package manager and ensure both are on your PATH.".into()
        }
    }
}

/// Snapshot of host toolchain tools used by the welcome / setup screen.
pub fn toolchain_status() -> serde_json::Value {
    toolchain_status_for(None)
}

/// Return the host-tool snapshot, optionally narrowed to tools relevant to a
/// first-run intent. `None` preserves the complete diagnostic view used by
/// Manage and support bundles.
pub fn toolchain_status_for(needs: Option<ToolchainNeeds>) -> serde_json::Value {
    let os = detect_os();
    let tool =
        |id: &str, label: &str, available: bool, package: ToolchainPackage, summary: &str| {
            serde_json::json!({
                "id": id,
                "label": label,
                "available": available,
                "required_for": summary,
                // Which copy was found, so "installed" can be checked against
                // the one the user thinks they installed.
                "path": resolve_command(id).map(|path| path.display().to_string()),
                "install_hint": if available {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(install_hint_for_os(package, &os))
                },
            })
        };
    let ffmpeg = command_on_path("ffmpeg") && command_on_path("ffprobe");
    let mut tools = vec![
        tool(
            "git",
            "Git",
            command_on_path("git"),
            ToolchainPackage::Git,
            "Cloning engine sources for llama.cpp, whisper.cpp, and MLX builds",
        ),
        tool(
            "cmake",
            "CMake",
            command_on_path("cmake"),
            ToolchainPackage::Cmake,
            "Configuring llama.cpp and whisper.cpp source builds",
        ),
        tool(
            "cpp",
            "C/C++ toolchain",
            cpp_compiler_available(),
            ToolchainPackage::CppBuild,
            "Compiling llama.cpp and whisper.cpp from source",
        ),
        tool(
            "uv",
            "uv",
            command_on_path("uv"),
            ToolchainPackage::Uv,
            "Creating Python environments for MLX, streaming ASR, PersonaPlex, and computer use",
        ),
        tool(
            "ffmpeg",
            "ffmpeg",
            ffmpeg,
            ToolchainPackage::Ffmpeg,
            "Video frame sampling and converting audio for transcription",
        ),
    ];
    if let Some(needs) = needs {
        let required = required_tool_ids(needs);
        tools.retain(|entry| {
            entry["id"]
                .as_str()
                .is_some_and(|id| required.contains(&id))
        });
    }

    serde_json::json!({
        "os": {
            "family": match os.family {
                OsFamily::Linux => "linux",
                OsFamily::Windows => "windows",
                OsFamily::Macos => "macos",
            },
            "id": os.id,
            "pretty_name": os.pretty_name,
        },
        "tools": tools,
        "platforms": {
            "mlx": matches!(os.family, OsFamily::Macos) && cfg!(target_arch = "aarch64"),
            "streaming_asr": !matches!(os.family, OsFamily::Windows),
            "whisper_cpp": true,
            "llama_cpp": true,
        }
    })
}

pub fn validate_build_target(target: RuntimeTarget) -> Option<String> {
    validate_build_target_for_os(target, &detect_os())
}

pub fn validate_build_target_for_os(target: RuntimeTarget, os: &OsInfo) -> Option<String> {
    match (os.family, target) {
        (OsFamily::Macos, RuntimeTarget::Rocm | RuntimeTarget::Cuda) => Some(format!(
            "{} builds are not supported on macOS. Use CPU or Metal.",
            target.as_str().to_ascii_uppercase()
        )),
        (OsFamily::Macos, RuntimeTarget::Sycl) => {
            Some("SYCL builds are not supported on macOS. Use CPU or Metal.".into())
        }
        (OsFamily::Windows, RuntimeTarget::Rocm) => {
            Some("ROCm builds are Linux-only. Use CPU, CUDA, or Vulkan on Windows.".into())
        }
        (OsFamily::Linux | OsFamily::Windows, RuntimeTarget::Metal) => Some(
            "Metal builds require macOS with Apple Silicon or an Intel Mac with Metal support."
                .into(),
        ),
        _ => None,
    }
}

pub fn cpp_compiler_available() -> bool {
    ["cl", "clang", "clang++", "g++", "c++"]
        .into_iter()
        .any(command_on_path)
}

pub fn cpp_compiler_preflight_message() -> Option<String> {
    if cpp_compiler_available() {
        return None;
    }
    Some(format!(
        "No C/C++ compiler was found on PATH. {}",
        install_hint(ToolchainPackage::CppBuild)
    ))
}

pub fn windows_vs_environment_hint() -> Option<String> {
    if !matches!(detect_os().family, OsFamily::Windows) || command_on_path("cl") {
        return None;
    }
    Some(
        "Visual Studio may be installed but `cl.exe` is not on PATH. Open **x64 Native Tools Command Prompt for VS 2022**, or run Brazier from a Developer PowerShell session, then retry the build.".into(),
    )
}

pub fn macos_clt_hint() -> Option<String> {
    if !matches!(detect_os().family, OsFamily::Macos) || cpp_compiler_available() {
        return None;
    }
    Some(install_hint(ToolchainPackage::CppBuild))
}

pub fn rocm_path_hint() -> Option<String> {
    if !matches!(detect_os().family, OsFamily::Linux) || hip_compiler_available() {
        return None;
    }
    Some(
        "After installing ROCm, open a new shell (or log out/in) so `/opt/rocm/bin` is on your PATH, then retry the build.".into(),
    )
}

pub fn hip_compiler_available() -> bool {
    command_on_path("hipcc") || Path::new("/opt/rocm/bin/hipcc").is_file()
}

pub fn rocm_preflight_message() -> Option<String> {
    if !matches!(detect_os().family, OsFamily::Linux) {
        return Some(
            "ROCm builds are Linux-only. Switch the build target to CPU, CUDA, or Vulkan.".into(),
        );
    }
    if hip_compiler_available() {
        return None;
    }
    Some(format!(
        "ROCm HIP compiler (hipcc) was not found. {}",
        install_hint(ToolchainPackage::RocmHip)
    ))
}

pub fn missing_hipsparselt(log_lower: &str, target: RuntimeTarget) -> bool {
    if !matches!(target, RuntimeTarget::Rocm) {
        return false;
    }
    if !log_lower.contains("hipsparselt") {
        return false;
    }
    [
        "not found",
        "could not find",
        "target was not found",
        "optional package hipsparselt not found",
    ]
    .iter()
    .any(|marker| log_lower.contains(marker))
}

pub fn hipsparselt_install_hint() -> Option<String> {
    if !matches!(detect_os().family, OsFamily::Linux) {
        return None;
    }
    match detect_os().package_manager {
        PackageManager::Pacman => Some("sudo pacman -S hipsparselt".into()),
        PackageManager::Apt | PackageManager::Dnf | PackageManager::Zypper => Some(
            "Install the ROCm hipSPARSELt development package for your distro (often named hipsparselt or rocm-sparselt).".into(),
        ),
        _ => Some(
            "Install the ROCm hipSPARSELt library that matches your installed ROCm version.".into(),
        ),
    }
}

pub fn missing_rocm_hip(log_lower: &str, message_lower: &str, target: RuntimeTarget) -> bool {
    if !matches!(target, RuntimeTarget::Rocm) {
        return false;
    }
    if message_lower.contains("hip compiler") || message_lower.contains("hipcc") {
        return true;
    }
    let mentions_hip = ["hip", "rocm", "ggml_hip"]
        .iter()
        .any(|marker| log_lower.contains(marker) || message_lower.contains(marker));
    if !mentions_hip {
        return false;
    }
    [
        "not found",
        "notfound",
        "could not find",
        "couldn't find",
        "missing:",
        "command not found",
        "no such file",
        "unable to find",
        "compiler not found",
        "failed to find",
        "cmake_hip_compiler",
        "hip_compiler-notfound",
    ]
    .iter()
    .any(|marker| log_lower.contains(marker) || message_lower.contains(marker))
}

pub fn missing_cpp_compiler(log_lower: &str, message_lower: &str) -> bool {
    [
        "no cmake_cxx_compiler",
        "no cmake_c_compiler",
        "cmake_cxx_compiler",
        "cmake_c_compiler",
        "c++: command not found",
        "g++: command not found",
        "clang++: command not found",
        "unable to find a c++ compiler",
        "could not find any instance of visual studio",
        "no CMAKE_CXX_COMPILER",
        "no CMAKE_C_COMPILER",
        "nmake : fatal error",
        "cl is not recognized",
        "cl.exe is not recognized",
        "cannot open include file: 'windows.h'",
        "xcrun: error",
        "xcode-select: error",
        "command 'clang++' failed",
        "ld: library not found",
    ]
    .iter()
    .any(|marker| log_lower.contains(marker) || message_lower.contains(marker))
}

pub fn missing_cuda(log_lower: &str) -> bool {
    [
        "could not find cuda",
        "cudatoolkit not found",
        "cuda_toolkit",
        "cuda_toolkit_root_dir",
        "findcuda",
    ]
    .iter()
    .any(|marker| log_lower.contains(marker))
}

pub fn missing_vulkan(log_lower: &str, target: RuntimeTarget) -> bool {
    let mentions_vulkan_stack = ["vulkan", "spirv", "glslang", "shaderc"]
        .iter()
        .any(|component| log_lower.contains(component));
    let reports_missing_component = [
        "not found",
        "could not find",
        "missing:",
        "no package configuration file",
    ]
    .iter()
    .any(|marker| log_lower.contains(marker));
    matches!(target, RuntimeTarget::Vulkan) && mentions_vulkan_stack && reports_missing_component
}

pub fn missing_cmake_or_vs_generator(log_lower: &str) -> bool {
    log_lower.contains("could not find any instance of visual studio")
        || log_lower.contains("generator")
            && log_lower.contains("visual studio")
            && log_lower.contains("could not find")
        || log_lower.contains("no CMAKE_CXX_COMPILER could be found")
}

fn command_on_path(name: &str) -> bool {
    resolve_command(name).is_some()
}

/// Where tools live when they are not on the inherited `PATH`.
///
/// A desktop application does not get the shell's environment. Launched from
/// Finder, macOS hands it `/usr/bin:/bin:/usr/sbin:/sbin`, so Homebrew's cmake
/// in `/opt/homebrew/bin` is invisible and Brazier reports a toolchain missing
/// on a machine that has had it for years — then prints an install command for
/// something already installed. The same applies to `~/.local/bin` and
/// `~/.cargo/bin` on Linux, which is where user-scoped installs go precisely so
/// that nothing needs elevation.
///
/// These are searched after `PATH`, so a deliberate environment still wins.
fn user_scoped_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    if let Some(home) = &home {
        directories.push(home.join(".local/bin"));
        directories.push(home.join(".cargo/bin"));
        directories.push(home.join("bin"));
    }
    if cfg!(target_os = "macos") {
        directories.push(PathBuf::from("/opt/homebrew/bin"));
        directories.push(PathBuf::from("/usr/local/bin"));
        directories.push(PathBuf::from("/opt/local/bin"));
        if let Some(home) = &home {
            directories.push(home.join("homebrew/bin"));
        }
    }
    if cfg!(target_os = "linux") {
        directories.push(PathBuf::from("/usr/local/bin"));
        directories.push(PathBuf::from("/var/lib/flatpak/exports/bin"));
        if let Some(home) = &home {
            directories.push(home.join(".local/share/flatpak/exports/bin"));
            directories.push(home.join(".linuxbrew/bin"));
            directories.push(PathBuf::from("/home/linuxbrew/.linuxbrew/bin"));
        }
    }
    if cfg!(windows)
        && let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    {
        directories.push(local.join("Microsoft/WindowsApps"));
        directories.push(local.join("Programs"));
    }
    directories
}

/// The full path of a tool, searching `PATH` and then the user-scoped places.
///
/// Returned rather than a boolean so callers can *run* what was found: knowing
/// ffmpeg exists in `/opt/homebrew/bin` is no use to a `Command::new("ffmpeg")`
/// that will search the same empty `PATH` and fail.
pub fn resolve_command(name: &str) -> Option<PathBuf> {
    let mut directories: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    directories.extend(user_scoped_directories());
    find_command_in(&directories, name)
}

/// The first directory holding an executable called `name`.
fn find_command_in(directories: &[PathBuf], name: &str) -> Option<PathBuf> {
    for directory in directories {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        if cfg!(windows) {
            for extension in ["exe", "cmd", "bat"] {
                let candidate = directory.join(format!("{name}.{extension}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The failure this prevents: a desktop application launched from Finder
    /// gets `/usr/bin:/bin:/usr/sbin:/sbin`, reports Homebrew's tools missing,
    /// and offers to install what is already there.
    #[test]
    fn finds_a_tool_outside_the_inherited_path() {
        let dir = tempfile::tempdir().unwrap();
        let user_bin = dir.path().join("homebrew/bin");
        std::fs::create_dir_all(&user_bin).unwrap();
        let tool = user_bin.join("cmake");
        std::fs::write(&tool, b"#!/bin/sh\n").unwrap();

        let empty = dir.path().join("usr/bin");
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(find_command_in(std::slice::from_ref(&empty), "cmake"), None);
        assert_eq!(find_command_in(&[empty, user_bin], "cmake"), Some(tool));
    }

    /// `PATH` is searched first, so an environment set on purpose still decides.
    #[test]
    fn prefers_what_the_environment_points_at() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(first.join("git"), b"#!/bin/sh\n").unwrap();
        std::fs::write(second.join("git"), b"#!/bin/sh\n").unwrap();
        assert_eq!(
            find_command_in(&[first.clone(), second], "git"),
            Some(first.join("git"))
        );
    }

    #[test]
    fn toolchain_status_lists_core_tools() {
        let status = toolchain_status();
        let tools = status["tools"].as_array().unwrap();
        let ids: Vec<&str> = tools
            .iter()
            .filter_map(|tool| tool["id"].as_str())
            .collect();
        assert!(ids.contains(&"git"));
        assert!(ids.contains(&"cmake"));
        assert!(ids.contains(&"cpp"));
        assert!(ids.contains(&"uv"));
        assert!(ids.contains(&"ffmpeg"));
        assert!(status["platforms"]["llama_cpp"].as_bool().unwrap());
    }

    #[test]
    fn intent_selects_only_the_tools_that_can_be_needed() {
        let managed = required_tool_ids(ToolchainNeeds::default());
        assert!(!managed.contains(&"cmake"));
        assert!(!managed.contains(&"git"));
        assert!(!managed.contains(&"cpp"));

        let custom = required_tool_ids(ToolchainNeeds {
            custom_runtimes: true,
            ..ToolchainNeeds::default()
        });
        assert!(custom.contains(&"cmake"));
        assert!(custom.contains(&"git"));
        assert!(custom.contains(&"cpp"));
        assert!(custom.contains(&"uv"));

        let python = required_tool_ids(ToolchainNeeds {
            voice: true,
            computer_use: true,
            ..ToolchainNeeds::default()
        });
        assert!(python.contains(&"uv"));
        assert!(!python.contains(&"cmake"));
        assert!(!python.contains(&"git"));
    }

    #[test]
    fn arch_rocm_hint_uses_pacman() {
        let os = OsInfo {
            family: OsFamily::Linux,
            id: "arch".into(),
            id_like: vec![],
            pretty_name: "Arch Linux".into(),
            package_manager: PackageManager::Pacman,
        };
        let cmd = install_command(ToolchainPackage::RocmHip, &os).unwrap();
        assert!(cmd.contains("pacman"));
        assert!(cmd.contains("rocm-hip-sdk"));
    }

    #[test]
    fn arch_vulkan_hint_includes_spirv_headers() {
        let os = OsInfo {
            family: OsFamily::Linux,
            id: "arch".into(),
            id_like: vec![],
            pretty_name: "Arch Linux".into(),
            package_manager: PackageManager::Pacman,
        };
        let cmd = install_command(ToolchainPackage::Vulkan, &os).unwrap();
        assert!(cmd.contains("spirv-headers"));
        assert!(cmd.contains("vulkan-devel"));
    }

    #[test]
    fn windows_git_uses_winget() {
        let os = OsInfo {
            family: OsFamily::Windows,
            id: "windows".into(),
            id_like: vec![],
            pretty_name: "Windows".into(),
            package_manager: PackageManager::Winget,
        };
        let cmd = install_command(ToolchainPackage::Git, &os).unwrap();
        assert!(cmd.contains("winget"));
        assert!(cmd.contains("Git.Git"));
    }

    #[test]
    fn debian_uses_apt() {
        let os = OsInfo {
            family: OsFamily::Linux,
            id: "debian".into(),
            id_like: vec![],
            pretty_name: "Debian GNU/Linux".into(),
            package_manager: PackageManager::Apt,
        };
        let cmd = install_command(ToolchainPackage::CppBuild, &os).unwrap();
        assert!(cmd.contains("apt"));
        assert!(cmd.contains("build-essential"));
    }

    #[test]
    fn fedora_id_like_maps_to_dnf() {
        let os = OsInfo {
            family: OsFamily::Linux,
            id: "nobara".into(),
            id_like: vec!["rhel".into(), "fedora".into()],
            pretty_name: "Nobara Linux".into(),
            package_manager: PackageManager::Dnf,
        };
        let cmd = install_command(ToolchainPackage::Cmake, &os).unwrap();
        assert!(cmd.contains("dnf"));
    }

    #[test]
    fn rejects_rocm_on_windows() {
        let os = OsInfo {
            family: OsFamily::Windows,
            id: "windows".into(),
            id_like: vec![],
            pretty_name: "Windows".into(),
            package_manager: PackageManager::Winget,
        };
        let message = validate_build_target_for_os(RuntimeTarget::Rocm, &os).unwrap();
        assert!(message.contains("Linux-only"));
    }

    #[test]
    fn rejects_cuda_on_macos() {
        let os = OsInfo {
            family: OsFamily::Macos,
            id: "macos".into(),
            id_like: vec![],
            pretty_name: "macOS".into(),
            package_manager: PackageManager::Brew,
        };
        let message = validate_build_target_for_os(RuntimeTarget::Cuda, &os).unwrap();
        assert!(message.contains("macOS"));
        let sycl = validate_build_target_for_os(RuntimeTarget::Sycl, &os).unwrap();
        assert!(sycl.contains("SYCL"));
    }

    #[test]
    fn detects_missing_rocm_from_cmake_log() {
        assert!(missing_rocm_hip(
            "cmake error: could not find hip (missing: hip_library hip_include_dir)",
            "configure failed with 1",
            RuntimeTarget::Rocm,
        ));
    }

    #[test]
    fn detects_missing_hipsparselt_from_cmake_log() {
        let log = r#"
            optional package hipsparselt not found
            cmake error: the link interface of target "torch_hip_library" contains:
              roc::hipsparselt
            but the target was not found.
        "#;
        assert!(missing_hipsparselt(
            &log.to_ascii_lowercase(),
            RuntimeTarget::Rocm,
        ));
        assert!(!missing_hipsparselt(
            &log.to_ascii_lowercase(),
            RuntimeTarget::Cuda,
        ));
    }

    #[test]
    fn detects_missing_spirv_headers_from_cmake_log() {
        let log = r#"
            -- Found Vulkan: /usr/lib/libvulkan.so
            Could not find a package configuration file provided by "SPIRV-Headers"
            with any of the following names:
              SPIRV-HeadersConfig.cmake
        "#;
        assert!(missing_vulkan(
            &log.to_ascii_lowercase(),
            RuntimeTarget::Vulkan,
        ));
        assert!(!missing_vulkan(
            &log.to_ascii_lowercase(),
            RuntimeTarget::Cpu,
        ));
    }

    #[test]
    fn detects_missing_msvc_from_log() {
        assert!(missing_cpp_compiler(
            "cmake error: could not find any instance of visual studio",
            "configure failed",
        ));
    }

    #[test]
    fn preflight_message_mentions_install_when_hipcc_missing() {
        if !matches!(detect_os().family, OsFamily::Linux) || hip_compiler_available() {
            return;
        }
        let message = rocm_preflight_message().unwrap();
        assert!(message.contains("hipcc"));
    }
}
