//! Whether the ROCm llama.cpp build Brazier installs can actually run on this
//! GPU.
//!
//! The failure this prevents is not a clean error. A GPU outside the build's
//! compiled architectures still enumerates as a ROCm device, so llama.cpp
//! commits to the HIP backend and then dispatches a kernel with no code object
//! for the hardware. That wedges the HSA queue and the process aborts with
//! "HW Exception ... GPU Hang".
//!
//! The only thing that knows which architectures are covered is the build
//! itself. HIP embeds its device code as a fat binary inside the ELF, and every
//! bundled code object carries its target id as a plain string
//! (`hipv4-amdgcn-amd-amdhsa--gfx1030`). Reading those out of the installed
//! files answers the question exactly, for whatever llama.cpp shipped, without a
//! list in this repository that would go stale every time AMD releases a part.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

/// Target-id marker that precedes an architecture in a HIP fat binary.
const TARGET_MARKER: &[u8] = b"amdgcn-amd-amdhsa--";

/// Longest architecture token worth reading after the marker (`gfx1201:xnack-`).
const MAX_ARCH_LEN: usize = 24;

/// Read in chunks so a multi-hundred-megabyte HIP binary is never held whole.
const CHUNK: usize = 1 << 20;

/// Architecture names a single file carries device code for.
///
/// Returns an empty list for a file with no HIP payload, which is the normal
/// answer for the CPU and Vulkan builds.
pub fn code_object_arches(path: &Path) -> Vec<String> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    let mut reader = BufReader::new(file);
    let mut arches: Vec<String> = Vec::new();
    // Carry the tail of each chunk forward so a marker split across the
    // boundary is still matched.
    let overlap = TARGET_MARKER.len() + MAX_ARCH_LEN;
    let mut window: Vec<u8> = Vec::with_capacity(CHUNK + overlap);
    let mut chunk = vec![0_u8; CHUNK];
    loop {
        let read = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => read,
            Err(_) => break,
        };
        window.extend_from_slice(&chunk[..read]);
        for arch in arches_in(&window) {
            if !arches.contains(&arch) {
                arches.push(arch);
            }
        }
        let keep = window.len().saturating_sub(overlap);
        window.drain(..keep);
    }
    arches.sort();
    arches
}

/// Every `gfx…` target id following a HIP target marker in one buffer.
fn arches_in(buffer: &[u8]) -> Vec<String> {
    let mut found = Vec::new();
    let mut offset = 0;
    while let Some(index) = find(&buffer[offset..], TARGET_MARKER) {
        let start = offset + index + TARGET_MARKER.len();
        offset = start;
        let end = (start + MAX_ARCH_LEN).min(buffer.len());
        let token: String = buffer[start..end]
            .iter()
            .take_while(|byte| byte.is_ascii_alphanumeric())
            .map(|byte| *byte as char)
            .collect();
        // Feature suffixes (`:xnack-`, `:sramecc+`) stop at the colon above, so
        // the bare architecture is what remains.
        if token.starts_with("gfx") && token.len() > 3 {
            found.push(token);
        }
    }
    found
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Architectures an installed llama.cpp build covers, across its binary and the
/// shared libraries beside it — HIP device code usually lives in `libggml-hip`
/// rather than in `llama-server` itself.
pub fn install_arches(bin_dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(bin_dir) else {
        return Vec::new();
    };
    let mut arches: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        // Skip obvious non-code payloads; everything else is cheap to scan.
        if name.ends_with(".txt") || name.ends_with(".md") || name.ends_with(".json") {
            continue;
        }
        for arch in code_object_arches(&path) {
            if !arches.contains(&arch) {
                arches.push(arch);
            }
        }
    }
    arches.sort();
    arches
}

/// Outcome of checking a GPU against an installed build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Support {
    /// The build carries device code for one of this machine's GPUs.
    Covered { arch: String },
    /// The build was read and covers none of them. Running it would hang.
    Uncovered {
        gpu_arches: Vec<String>,
        build_arches: Vec<String>,
    },
    /// Nothing to check against yet: no ROCm build installed, or no GPU
    /// architecture readable. Never treated as either a pass or a failure.
    Unknown,
}

/// Compare the GPUs the kernel reports against the architectures a build ships.
pub fn support(gpu_arches: &[String], build_arches: &[String]) -> Support {
    if gpu_arches.is_empty() || build_arches.is_empty() {
        return Support::Unknown;
    }
    match gpu_arches.iter().find(|arch| build_arches.contains(arch)) {
        Some(arch) => Support::Covered { arch: arch.clone() },
        None => Support::Uncovered {
            gpu_arches: gpu_arches.to_vec(),
            build_arches: build_arches.to_vec(),
        },
    }
}

impl Support {
    /// Message for a refusal, or `None` when there is nothing to refuse.
    pub fn rejection(&self) -> Option<String> {
        match self {
            Self::Uncovered {
                gpu_arches,
                build_arches,
            } => Some(format!(
                "This ROCm build has device code for {} and none for this machine's {}. \
                 Running it would not fail cleanly — the GPU hangs mid-dispatch. Use the Vulkan \
                 target instead, which supports this GPU.",
                build_arches.join(", "),
                gpu_arches.join(", ")
            )),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A stand-in for the target-id strings a HIP fat binary carries.
    fn write_fatbin(path: &Path, targets: &[&str]) {
        let mut file = File::create(path).unwrap();
        file.write_all(&[0_u8; 512]).unwrap();
        for target in targets {
            file.write_all(b"__CLANG_OFFLOAD_BUNDLE__").unwrap();
            file.write_all(format!("hipv4-amdgcn-amd-amdhsa--{target}").as_bytes())
                .unwrap();
            file.write_all(&[0_u8; 64]).unwrap();
        }
        file.write_all(&[0_u8; 512]).unwrap();
    }

    #[test]
    fn reads_targets_out_of_a_hip_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("libggml-hip.so");
        write_fatbin(&path, &["gfx1030", "gfx1100", "gfx1101"]);
        assert_eq!(
            code_object_arches(&path),
            vec!["gfx1030", "gfx1100", "gfx1101"]
        );
    }

    #[test]
    fn strips_feature_suffixes_and_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("libggml-hip.so");
        write_fatbin(
            &path,
            &["gfx90a:sramecc+:xnack-", "gfx90a", "gfx942:xnack-"],
        );
        assert_eq!(code_object_arches(&path), vec!["gfx90a", "gfx942"]);
    }

    #[test]
    fn a_build_with_no_device_code_reports_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("llama-server");
        std::fs::write(&path, b"a plain CPU build with no offload bundles").unwrap();
        assert!(code_object_arches(&path).is_empty());
    }

    /// The scan must not miss a target that straddles a read boundary.
    #[test]
    fn finds_a_target_split_across_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("libggml-hip.so");
        let mut file = File::create(&path).unwrap();
        let marker = b"hipv4-amdgcn-amd-amdhsa--gfx1100";
        // Land the marker so it begins a few bytes before the chunk boundary.
        let padding = CHUNK - 8;
        file.write_all(&vec![0_u8; padding]).unwrap();
        file.write_all(marker).unwrap();
        file.write_all(&[0_u8; 32]).unwrap();
        drop(file);
        assert_eq!(code_object_arches(&path), vec!["gfx1100"]);
    }

    #[test]
    fn scans_every_file_in_the_install() {
        let dir = tempfile::tempdir().unwrap();
        write_fatbin(&dir.path().join("libggml-hip.so"), &["gfx1030"]);
        write_fatbin(&dir.path().join("llama-server"), &["gfx1100"]);
        std::fs::write(dir.path().join("notes.md"), b"amdgcn-amd-amdhsa--gfx999").unwrap();
        // The markdown is skipped, so its bogus target does not count.
        assert_eq!(install_arches(dir.path()), vec!["gfx1030", "gfx1100"]);
        assert!(install_arches(&dir.path().join("missing")).is_empty());
    }

    #[test]
    fn matches_a_gpu_against_what_the_build_ships() {
        let build = vec!["gfx1030".to_owned(), "gfx1100".to_owned()];
        assert_eq!(
            support(&["gfx1100".to_owned()], &build),
            Support::Covered {
                arch: "gfx1100".to_owned()
            }
        );
        // The APU case this exists for.
        let uncovered = support(&["gfx90c".to_owned()], &build);
        assert!(matches!(uncovered, Support::Uncovered { .. }));
        let rejection = uncovered.rejection().unwrap();
        assert!(rejection.contains("gfx90c"));
        assert!(rejection.contains("Vulkan"));
    }

    #[test]
    fn an_unreadable_side_is_never_a_verdict() {
        let build = vec!["gfx1100".to_owned()];
        assert_eq!(support(&[], &build), Support::Unknown);
        assert_eq!(support(&["gfx1100".to_owned()], &[]), Support::Unknown);
        assert_eq!(Support::Unknown.rejection(), None);
        assert_eq!(
            Support::Covered {
                arch: "gfx1100".to_owned()
            }
            .rejection(),
            None
        );
    }

    /// One GPU covered is enough, even in a mixed machine.
    #[test]
    fn a_covered_gpu_wins_in_a_mixed_machine() {
        let build = vec!["gfx1100".to_owned()];
        assert_eq!(
            support(&["gfx90c".to_owned(), "gfx1100".to_owned()], &build),
            Support::Covered {
                arch: "gfx1100".to_owned()
            }
        );
    }
}
