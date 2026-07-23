//! OS/distro-specific install commands for build prerequisites.

use std::path::Path;

use crate::runtime_settings::RuntimeTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageManager {
    Pacman,
    Apt,
    Dnf,
    Zypper,
    Brew,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct OsInfo {
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
}

pub fn detect_os() -> OsInfo {
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
            std::env::consts::OS.to_owned()
        } else {
            id.clone()
        };
    }
    let package_manager = infer_package_manager(&id, &id_like);
    OsInfo {
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
    if id == "arch" || id == "manjaro" || id == "endeavouros" {
        return PackageManager::Pacman;
    }
    if id == "ubuntu" || id == "debian" || id == "linuxmint" || id == "pop" {
        return PackageManager::Apt;
    }
    if id == "fedora" || id == "centos" || id == "rhel" || id == "rocky" || id == "almalinux" {
        return PackageManager::Dnf;
    }
    if id == "opensuse-leap" || id == "opensuse-tumbleweed" || id == "suse" {
        return PackageManager::Zypper;
    }
    if id == "macos" || cfg!(target_os = "macos") {
        return PackageManager::Brew;
    }
    for like in id_like {
        match like.as_str() {
            "arch" => return PackageManager::Pacman,
            "debian" | "ubuntu" => return PackageManager::Apt,
            "fedora" | "rhel" => return PackageManager::Dnf,
            "suse" => return PackageManager::Zypper,
            _ => {}
        }
    }
    PackageManager::Unknown
}

pub fn install_command(package: ToolchainPackage, os: &OsInfo) -> Option<String> {
    match (os.package_manager, package) {
        (PackageManager::Pacman, ToolchainPackage::Git) => {
            Some("sudo pacman -S git".into())
        }
        (PackageManager::Pacman, ToolchainPackage::Cmake) => {
            Some("sudo pacman -S cmake".into())
        }
        (PackageManager::Pacman, ToolchainPackage::CppBuild) => {
            Some("sudo pacman -S base-devel".into())
        }
        (PackageManager::Pacman, ToolchainPackage::RocmHip) => Some(
            "sudo pacman -S rocm-hip-sdk rocm-opencl-sdk base-devel cmake git".into(),
        ),
        (PackageManager::Pacman, ToolchainPackage::Cuda) => Some(
            "Enable the `extra` repo if needed, then: sudo pacman -S cuda".into(),
        ),
        (PackageManager::Pacman, ToolchainPackage::Vulkan) => {
            Some("sudo pacman -S vulkan-devel cmake extra/glslang".into())
        }

        (PackageManager::Apt, ToolchainPackage::Git) => Some("sudo apt install git".into()),
        (PackageManager::Apt, ToolchainPackage::Cmake) => Some("sudo apt install cmake".into()),
        (PackageManager::Apt, ToolchainPackage::CppBuild) => {
            Some("sudo apt install build-essential".into())
        }
        (PackageManager::Apt, ToolchainPackage::RocmHip) => Some(
            "Install ROCm for your Ubuntu/Debian release from AMD's docs, then: sudo apt install rocm-hip-sdk git cmake build-essential".into(),
        ),
        (PackageManager::Apt, ToolchainPackage::Cuda) => Some(
            "Install the NVIDIA CUDA toolkit for your distro from https://developer.nvidia.com/cuda-downloads".into(),
        ),
        (PackageManager::Apt, ToolchainPackage::Vulkan) => {
            Some("sudo apt install vulkan-tools libvulkan-dev glslang-tools".into())
        }

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
            Some("sudo dnf install vulkan-tools vulkan-loader-devel glslang".into())
        }

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
        (PackageManager::Zypper, ToolchainPackage::Cuda) => Some(
            "Install NVIDIA CUDA for openSUSE from NVIDIA's repository".into(),
        ),
        (PackageManager::Zypper, ToolchainPackage::Vulkan) => {
            Some("sudo zypper install vulkan-tools vulkan-devel".into())
        }

        (PackageManager::Brew, ToolchainPackage::Git) => Some("brew install git".into()),
        (PackageManager::Brew, ToolchainPackage::Cmake) => Some("brew install cmake".into()),
        (PackageManager::Brew, ToolchainPackage::CppBuild) => {
            Some("xcode-select --install".into())
        }
        (PackageManager::Brew, ToolchainPackage::RocmHip) => None,
        (PackageManager::Brew, ToolchainPackage::Cuda) => None,
        (PackageManager::Brew, ToolchainPackage::Vulkan) => None,

        (PackageManager::Unknown, _) => None,
    }
}

pub fn install_hint(package: ToolchainPackage) -> String {
    let os = detect_os();
    install_hint_for_os(package, &os)
}

pub fn install_hint_for_os(package: ToolchainPackage, os: &OsInfo) -> String {
    let fallback = match package {
        ToolchainPackage::Git => "Install Git and ensure it is on your PATH.",
        ToolchainPackage::Cmake => "Install CMake 3.20+ and ensure it is on your PATH.",
        ToolchainPackage::CppBuild => "Install a C/C++ toolchain (build-essential, Xcode CLT, or Visual Studio Build Tools).",
        ToolchainPackage::RocmHip => {
            "Install AMD ROCm with the HIP compiler (hipcc), or switch the build target to CPU."
        }
        ToolchainPackage::Cuda => {
            "Install the NVIDIA CUDA toolkit, or switch the build target to CPU."
        }
        ToolchainPackage::Vulkan => {
            "Install Vulkan headers/loader and drivers, or switch the build target to CPU."
        }
    };
    if let Some(command) = install_command(package, os) {
        format!("On {}: `{}`.", os.pretty_name, command)
    } else {
        fallback.to_owned()
    }
}

pub fn hip_compiler_available() -> bool {
    command_on_path("hipcc") || Path::new("/opt/rocm/bin/hipcc").is_file()
}

pub fn rocm_preflight_message() -> Option<String> {
    if hip_compiler_available() {
        return None;
    }
    Some(format!(
        "ROCm HIP compiler (hipcc) was not found. {}",
        install_hint(ToolchainPackage::RocmHip)
    ))
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

fn command_on_path(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| {
            let candidate = directory.join(name);
            candidate.is_file()
                || (cfg!(windows) && directory.join(format!("{name}.exe")).is_file())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_rocm_hint_uses_pacman() {
        let os = OsInfo {
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
    fn detects_missing_rocm_from_cmake_log() {
        assert!(missing_rocm_hip(
            "cmake error: could not find hip (missing: hip_library hip_include_dir)",
            "configure failed with 1",
            RuntimeTarget::Rocm,
        ));
    }

    #[test]
    fn preflight_message_mentions_install_when_hipcc_missing() {
        if hip_compiler_available() {
            assert!(rocm_preflight_message().is_none());
        } else {
            let message = rocm_preflight_message().unwrap();
            assert!(message.contains("hipcc"));
        }
    }
}
