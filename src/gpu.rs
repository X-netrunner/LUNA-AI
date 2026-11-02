//! gpu.rs — notice when a GPU is present but Luna's models still run on the CPU.
//!
//! Luna never drives the GPU directly. Ollama (the LLM tiers) and whisper.cpp
//! (speech to text) pick it up on their own, but only when their GPU backend is
//! installed. On Arch the plain `ollama` and `ggml` packages are CPU-only, so a
//! machine with an NVIDIA card silently runs every model on the CPU. This module
//! spots that and says which package fixes it. It only looks; it installs nothing.

use std::path::{Path, PathBuf};

/// Directories where Ollama keeps its compute backends (distro package, official installer).
const OLLAMA_LIB_DIRS: &[&str] = &["/usr/lib/ollama", "/usr/local/lib/ollama"];
/// Directory where the split `ggml` package (used by whisper-cli) keeps its backends.
const GGML_LIB_DIR: &str = "/usr/lib/ggml";

/// Is an NVIDIA GPU with a working driver present?
pub fn nvidia_present() -> bool {
    Path::new("/proc/driver/nvidia/version").exists()
}

/// Does this backend directory hold any GPU backend (CUDA, ROCm or Vulkan)?
fn has_gpu_backend(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|e| {
        let name = e.file_name().to_string_lossy().to_lowercase();
        name.starts_with("cuda_")
            || name.starts_with("rocm")
            || name.starts_with("vulkan")
            || name.starts_with("libggml-cuda")
            || name.starts_with("libggml-hip")
            || name.starts_with("libggml-vulkan")
    })
}

/// Whether Ollama has a GPU backend installed. None when no Ollama lib dir is found
/// (unknown install layout, e.g. a container), so we stay quiet instead of guessing.
fn ollama_has_gpu_backend(dirs: &[PathBuf]) -> Option<bool> {
    let existing: Vec<&PathBuf> = dirs.iter().filter(|d| d.is_dir()).collect();
    if existing.is_empty() {
        return None;
    }
    Some(existing.iter().any(|d| has_gpu_backend(d)))
}

/// Whether the shared ggml library has a GPU backend. None when whisper-cli does not
/// use the split ggml package (it may bundle its own backends), so we stay quiet.
fn ggml_has_gpu_backend(dir: &Path) -> Option<bool> {
    dir.is_dir().then(|| has_gpu_backend(dir))
}

/// From Ollama's `/api/ps` reply: true when models are loaded and none of them is on the GPU.
fn ps_says_cpu_only(ps: &serde_json::Value) -> bool {
    let Some(models) = ps["models"].as_array() else {
        return false;
    };
    !models.is_empty()
        && models
            .iter()
            .all(|m| m["size_vram"].as_u64().unwrap_or(0) == 0)
}

/// Problems to fix, each with the command that fixes it. Empty = nothing to report.
fn hints(
    nvidia: bool,
    ollama_gpu: Option<bool>,
    ollama_ps_cpu: bool,
    ggml_gpu: Option<bool>,
) -> Vec<String> {
    let mut out = Vec::new();
    if !nvidia {
        return out;
    }
    if ollama_gpu == Some(false) || ollama_ps_cpu {
        out.push(
            "Ollama is running on the CPU even though an NVIDIA GPU is present. \
             Install its CUDA backend and restart it: \
             `sudo pacman -S ollama-cuda && sudo systemctl restart ollama`"
                .to_string(),
        );
    }
    if ggml_gpu == Some(false) {
        out.push(
            "whisper-cli (voice input) is running on the CPU even though an NVIDIA GPU is present. \
             Install the CUDA backend for ggml: `sudo pacman -S ggml-cuda`"
                .to_string(),
        );
    }
    out
}

/// Same check from the installed files alone (no network), for the setup guide.
pub fn check_installed() -> Vec<String> {
    let dirs: Vec<PathBuf> = OLLAMA_LIB_DIRS.iter().map(PathBuf::from).collect();
    hints(
        nvidia_present(),
        ollama_has_gpu_backend(&dirs),
        false,
        ggml_has_gpu_backend(Path::new(GGML_LIB_DIR)),
    )
}

/// Check this machine and return the GPU setup hints (empty when the GPU is used or absent).
pub async fn check(ollama_base_url: &str) -> Vec<String> {
    let nvidia = nvidia_present();
    if !nvidia {
        return Vec::new();
    }
    let dirs: Vec<PathBuf> = OLLAMA_LIB_DIRS.iter().map(PathBuf::from).collect();
    let ollama_gpu = ollama_has_gpu_backend(&dirs);
    let ggml_gpu = ggml_has_gpu_backend(Path::new(GGML_LIB_DIR));

    let url = format!("{}/api/ps", ollama_base_url.trim_end_matches('/'));
    let ps_cpu = match reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) => r
            .json::<serde_json::Value>()
            .await
            .map(|v| ps_says_cpu_only(&v))
            .unwrap_or(false),
        Err(_) => false,
    };
    hints(nvidia, ollama_gpu, ps_cpu, ggml_gpu)
}

/// Log the hints once at startup. Goes through tracing so it never corrupts the TUI.
pub fn warn_in_background(ollama_base_url: String) {
    tokio::spawn(async move {
        for h in check(&ollama_base_url).await {
            tracing::warn!("GPU: {}", h);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmpdir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("luna_gpu_test_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn cpu_only_ollama_dir_is_flagged() {
        let d = tmpdir("cpu");
        std::fs::write(d.join("libggml-cpu-haswell.so"), b"").unwrap();
        assert_eq!(ollama_has_gpu_backend(&[d.clone()]), Some(false));
        std::fs::create_dir(d.join("cuda_v13")).unwrap();
        assert_eq!(ollama_has_gpu_backend(&[d.clone()]), Some(true));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn unknown_layout_stays_quiet() {
        assert_eq!(
            ollama_has_gpu_backend(&[PathBuf::from("/nonexistent/luna/ollama")]),
            None
        );
        assert_eq!(
            ggml_has_gpu_backend(Path::new("/nonexistent/luna/ggml")),
            None
        );
    }

    #[test]
    fn ggml_cuda_backend_is_recognised() {
        let d = tmpdir("ggml");
        std::fs::write(d.join("libggml-cpu-zen4.so"), b"").unwrap();
        assert_eq!(ggml_has_gpu_backend(&d), Some(false));
        std::fs::write(d.join("libggml-cuda.so"), b"").unwrap();
        assert_eq!(ggml_has_gpu_backend(&d), Some(true));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn ps_reply_cpu_vs_gpu() {
        assert!(ps_says_cpu_only(
            &json!({"models":[{"name":"a","size_vram":0}]})
        ));
        assert!(!ps_says_cpu_only(
            &json!({"models":[{"name":"a","size_vram":3100198829u64}]})
        ));
        assert!(
            !ps_says_cpu_only(&json!({"models":[]})),
            "nothing loaded is not evidence"
        );
        assert!(!ps_says_cpu_only(&json!({})));
    }

    #[test]
    fn no_nvidia_means_no_hints() {
        assert!(hints(false, Some(false), true, Some(false)).is_empty());
    }

    #[test]
    fn hints_name_the_fixing_package() {
        let h = hints(true, Some(false), false, Some(false));
        assert_eq!(h.len(), 2);
        assert!(h[0].contains("ollama-cuda"));
        assert!(h[1].contains("ggml-cuda"));
        assert!(hints(true, Some(true), false, Some(true)).is_empty());
        assert!(hints(true, None, false, None).is_empty());
        assert_eq!(
            hints(true, Some(true), true, None).len(),
            1,
            "loaded on CPU despite backend"
        );
    }
}
