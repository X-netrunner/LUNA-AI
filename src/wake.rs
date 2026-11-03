//! src/wake.rs — the always-on "hey luna" listener
//!
//! Runs as a user systemd service (`luna-wake.service`) and does two jobs:
//!
//! 1. **Wake her from anywhere.** Sleeps on the tolerant
//!    [`listen_for_wake_word`](crate::audio::capture::listen_for_wake_word)
//!    engine until "hey luna" is heard, then starts Luna in the configured
//!    launch mode:
//!      * `headless` (the final product) — runs the voice session right here
//!        in this process. No windows: she just listens, and answers through
//!        the speakers (TTS). One-breath commands ("hey luna, what's the
//!        weather") are processed directly.
//!      * `tui` (great for debugging) — spawns a terminal (ghostty, kitty,
//!        …) running the full TUI, handing over any inline command via a
//!        pending-voice marker the TUI picks up at startup.
//! 2. **Stand down while a session is alive.** If a Luna agent (TUI,
//!    headless, manual `luna`) is already running, this process doesn't
//!    double-launch and doesn't fight for the microphone.
//!
//! While listening / thinking, the desktop overlay
//! (see [`crate::overlay`]) animates — driven over the unix socket by both
//! this process and the spawned TUI.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::config::LunaConfig;

/// Entry point for `luna --wake-daemon`.
pub async fn run(config: LunaConfig) -> Result<()> {
    tracing::info!(
        "Wake daemon starting (launch_mode={}, overlay={})",
        config.wake.launch_mode,
        config.wake.overlay_enabled
    );
    if !config.wake.enabled {
        tracing::warn!("[wake] enabled = false — not listening for \"hey luna\"");
        return Ok(());
    }

    if config.wake.overlay_enabled && crate::overlay::start() {
        tracing::info!("wake: desktop listening indicator running");
    }

    match config.wake.launch_mode.as_str() {
        "tui" => run_tui_launcher(config).await,
        "headless" => crate::agent::run_headless(&config).await,
        other => anyhow::bail!(
            "[wake] launch_mode {:?} is invalid — use \"headless\" or \"tui\"",
            other
        ),
    }
}

// ── Debug mode: launch the TUI in a fresh terminal ───────────────────────────

async fn run_tui_launcher(config: LunaConfig) -> Result<()> {
    let stt = crate::stt::whisper::WhisperStt::with_prompt(
        &config.voice.whisper_model.to_string_lossy(),
        // Keep this SHORT — Whisper can echo prompt text back on near-silence.
        Some("Luna, open, close, run, search, volume, terminal, browser.".into()),
    );
    let aliases = config.audio.wake_aliases.clone();
    let sample_rate = config.audio.sample_rate;
    let silence_ms = config.audio.vad_silence_ms;

    loop {
        // Stand down while any Luna agent session is already running.
        if luna_active() {
            tracing::debug!("wake: a Luna session is already running — standing by");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        let text = match crate::audio::capture::listen_for_wake_word(
            sample_rate,
            silence_ms,
            &aliases,
            &stt,
        )
        .await
        {
            Ok(t) => t,
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(300)).await;
                continue;
            }
        };
        tracing::info!("wake: \"{}\"", text);
        crate::overlay::signal("listening");

        let inline = crate::audio::capture::strip_wake_word(&text, &aliases)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        write_pending_voice(&inline);

        match spawn_terminal(config.wake.terminal_cmd.as_str()) {
            Ok(()) => {
                tracing::info!("wake: launched TUI terminal");
                // Debounce: leftover audio / slow boot must not re-trigger,
                // and the TUI now owns the overlay (it signals itself).
                tokio::time::sleep(Duration::from_secs(config.wake.min_interval_secs)).await;
                crate::overlay::signal("idle");
            }
            Err(e) => {
                tracing::error!("wake: failed to launch terminal: {}", e);
                crate::overlay::signal("idle");
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
    }
}

/// Is any Luna agent process already alive? Scans /proc cmdlines for a
/// `luna` binary running an interactive session. The wake daemon itself
/// (`--wake-daemon`) and the watchdog daemon (`--daemon`) are excluded —
/// they're different jobs, not interactive sessions.
fn luna_active() -> bool {
    let self_pid = std::process::id();
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return false;
    };
    for entry in rd.flatten() {
        let pid: i64 = match entry.file_name().to_string_lossy().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        if pid == self_pid as i64 {
            continue;
        }
        let cmdline = std::fs::read(entry.path().join("cmdline")).unwrap_or_default();
        if cmdline.is_empty() {
            continue;
        }
        let cmd = String::from_utf8_lossy(&cmdline).replace('\0', " ");
        if cmd.contains("luna") && !cmd.contains("--wake-daemon") && !cmd.contains("--daemon") {
            return true;
        }
    }
    false
}

// ── Pending-voice handoff (TUI mode) ─────────────────────────────────────────

fn pending_voice_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".local/share/luna/pending_voice.txt"))
}

/// The wake daemon writes the inline command it stripped from the wake
/// utterance ("what's the weather" out of "hey luna, what's the weather")
/// here; the TUI reads and deletes it at startup and processes it as its
/// first voice utterance.
fn write_pending_voice(text: &str) {
    let Some(path) = pending_voice_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, text);
    tracing::debug!("wake: pending voice marker written: {:?}", path);
}

/// Read-and-consume the pending voice marker (TUI startup). Returns None when
/// there's nothing waiting.
pub fn read_pending_voice() -> Option<String> {
    let path = pending_voice_path()?;
    let text = std::fs::read_to_string(&path).ok()?;
    let _ = std::fs::remove_file(&path);
    Some(text)
}

// ── Terminal spawning ─────────────────────────────────────────────────────────

/// Launch a fresh terminal running `luna --tui`. The terminal binary is the
/// one named in `[wake] terminal_cmd`, or the first available of
/// ghostty/kitty/alacritty/foot/konsole.
fn spawn_terminal(preferred: &str) -> Result<()> {
    let exe = std::env::current_exe()
        .context("failed to resolve luna binary path")?
        .to_string_lossy()
        .to_string();

    let mut choice: Option<(PathBuf, Vec<String>)> = None;
    if !preferred.is_empty() {
        if let Some(p) = which(preferred) {
            choice = Some((p, term_args(preferred, &exe)));
        } else {
            tracing::warn!(
                "wake: configured terminal_cmd {:?} not on PATH — auto-detecting",
                preferred
            );
        }
    }
    if choice.is_none() {
        for t in ["ghostty", "kitty", "alacritty", "foot", "konsole"] {
            if let Some(p) = which(t) {
                choice = Some((p, term_args(t, &exe)));
                break;
            }
        }
    }

    let (bin, args) = choice.context(
        "no terminal emulator found — set one with \"terminal_cmd\" in the [wake] section",
    )?;
    Command::new(&bin).args(&args).spawn().with_context(|| {
        format!(
            "failed to spawn terminal {} with {:?}",
            bin.display(),
            args
        )
    })?;
    tracing::info!("wake: spawned {} {}", bin.display(), args.join(" "));
    Ok(())
}

/// Per-terminal argument layout. ghostty takes the command as ONE string
/// (it shells out); others take argv items.
fn term_args(term: &str, exe: &str) -> Vec<String> {
    if term == "foot" {
        vec![exe.to_string(), "--tui".into()]
    } else if term == "ghostty" {
        vec!["-e".into(), format!("{} --tui", exe)]
    } else {
        vec!["-e".into(), exe.to_string(), "--tui".into()]
    }
}

fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var("PATH").unwrap_or_default();
    for dir in path.split(':') {
        let cand = Path::new(dir).join(bin);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_args_shape() {
        assert_eq!(
            term_args("ghostty", "/home/u/.local/bin/luna"),
            vec!["-e".to_string(), "/home/u/.local/bin/luna --tui".to_string()]
        );
        assert_eq!(
            term_args("kitty", "/home/u/.local/bin/luna"),
            vec!["-e".to_string(), "/home/u/.local/bin/luna".to_string(), "--tui".to_string()]
        );
        assert_eq!(
            term_args("foot", "/home/u/.local/bin/luna"),
            vec!["/home/u/.local/bin/luna".to_string(), "--tui".to_string()]
        );
    }

    #[test]
    fn which_finds_shell() {
        // The test binary itself runs under a known shell environment — /bin/sh
        // or /usr/bin/sh exists on every POSIX system.
        let found = which("sh").or_else(|| which("bash"));
        assert!(found.is_some(), "expected sh or bash on PATH");
    }

    #[test]
    fn pending_voice_roundtrip() {
        let home = std::env::temp_dir().join(format!(
            "luna-wake-test-{}",
            std::process::id()
        ));
        // Override $HOME so dirs::home_dir() points at a scratch dir.
        std::env::set_var("HOME", &home);
        write_pending_voice("what's the weather");
        assert_eq!(read_pending_voice().as_deref(), Some("what's the weather"));
        // Second read must be None — the marker is consumed.
        assert_eq!(read_pending_voice(), None);
        let _ = std::fs::remove_dir_all(&home);
    }
}