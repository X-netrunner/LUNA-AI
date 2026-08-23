//! daemon/cleanup.rs — Disk hygiene
//!
//! Measures reclaimable space and either cleans (auto mode) or reports
//! (notify mode). Only ever touches four safe locations: the pacman
//! package cache, ~/.cache files older than N days, the trash, and
//! systemd journals.

use crate::config::LunaConfig;
use anyhow::Result;
use std::time::{Duration, Instant};

use super::sh;

struct DiskUsage {
    used_pct: u32,
    pacman_cache_mb: u64,
    user_cache_mb: u64,
    trash_mb: u64,
}

/// One disk-hygiene cycle: measure, then act according to cleanup_mode.
pub(crate) async fn disk_cycle(
    config: &LunaConfig,
    last_cleanup: &mut Option<Instant>,
    last_notify: &mut Option<Instant>,
) -> Result<()> {
    let usage = measure_disk(&config.daemon.pacman_cache_keep.to_string()).await;

    // Auto mode: actually clean when / crosses the proactive threshold.
    // One-hour cooldown between runs.
    if config.daemon.cleanup_mode == "auto" && usage.used_pct >= config.proactive.disk_full_threshold
    {
        let cooled = last_cleanup.map(|t| t.elapsed() > Duration::from_secs(3600)).unwrap_or(true);
        if cooled {
            let sudo = config.agent.sudo_password.as_deref();
            let freed = run_cleanup(config, sudo).await;
            *last_cleanup = Some(Instant::now());
            if freed.reclaimed_mb > 0 {
                super::notify(
                    "Luna daemon",
                    &format!(
                        "Disk was {}% full — cleaned {} MiB:\n{}",
                        usage.used_pct,
                        freed.reclaimed_mb,
                        freed.actions.join("\n")
                    ),
                )
                .await;
            }
        }
        return Ok(());
    }

    // Notify mode: report reclaimable space, change nothing.
    if config.daemon.cleanup_mode != "auto" {
        let potential = usage.pacman_cache_mb + usage.user_cache_mb + usage.trash_mb;
        let worth_it = potential >= config.daemon.min_notify_mb;
        let cooled = last_notify
            .map(|t| t.elapsed() > Duration::from_secs(24 * 3600))
            .unwrap_or(true);
        if worth_it && cooled {
            super::notify(
                "Luna daemon",
                &format!(
                    "Reclaimable space: {} MiB\n- pacman cache: {} MiB (say 'clean pacman cache')\n\
                     - ~/.cache: {} MiB older than {}d (say 'clean caches')\n\
                     - trash: {} MiB older than {}d (say 'empty trash')",
                    potential,
                    usage.pacman_cache_mb,
                    usage.user_cache_mb,
                    config.daemon.cache_max_age_days,
                    usage.trash_mb,
                    config.daemon.trash_max_age_days,
                ),
            )
            .await;
            *last_notify = Some(Instant::now());
        }
    }

    Ok(())
}

async fn measure_disk(_keep: &str) -> DiskUsage {
    let used_pct = sh("df -h / | awk 'NR==2 {print $5}' | tr -d '%'")
        .await
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    DiskUsage {
        used_pct,
        pacman_cache_mb: size_of("/var/cache/pacman/pkg").await,
        user_cache_mb: size_of(&format!("{}/.cache", home)).await,
        trash_mb: size_of(&format!("{}/.local/share/Trash", home)).await,
    }
}

struct CleanupResult {
    reclaimed_mb: u64,
    actions: Vec<String>,
}

/// Actually delete things — only ever touches the four safe locations.
async fn run_cleanup(config: &LunaConfig, sudo: Option<&str>) -> CleanupResult {
    let mut result = CleanupResult { reclaimed_mb: 0, actions: Vec::new() };
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());

    let before_pacman = size_of("/var/cache/pacman/pkg").await;
    let before_cache = size_of(&format!("{}/.cache", home)).await;
    let before_trash = size_of(&format!("{}/.local/share/Trash", home)).await;

    // 1. pacman cache — official paccache tool keeps N versions
    if let Some(pass) = sudo {
        let keep = config.daemon.pacman_cache_keep;
        let cmd = format!(
            "echo '{}' | sudo -S paccache -rk{} >/dev/null 2>&1; \
             echo '{}' | sudo -S paccache -ruk0 >/dev/null 2>&1",
            pass, keep, pass
        );
        if sh(&cmd).await.is_ok() {
            let after = size_of("/var/cache/pacman/pkg").await;
            if before_pacman > after {
                result.actions.push(format!(
                    "pacman cache: {} MiB freed",
                    before_pacman - after
                ));
                result.reclaimed_mb += before_pacman - after;
            }
        }
    } else {
        tracing::debug!("No sudo password in config — skipping pacman cache");
    }

    // 2. user cache — files untouched for cache_max_age_days
    let days = config.daemon.cache_max_age_days;
    let _ = sh(&format!(
        "find '{}/.cache' -type f -mtime +{} -delete 2>/dev/null",
        home, days
    ))
    .await;
    let after_cache = size_of(&format!("{}/.cache", home)).await;
    if before_cache > after_cache {
        result.actions.push(format!(
            "~/.cache: {} MiB freed",
            before_cache - after_cache
        ));
        result.reclaimed_mb += before_cache - after_cache;
    }

    // 3. trash — items older than trash_max_age_days (files + info sidecars)
    let tdays = config.daemon.trash_max_age_days;
    let _ = sh(&format!(
        "find '{}/.local/share/Trash/files' -mindepth 1 -mtime +{} \
         -exec rm -rf {{}} + 2>/dev/null; \
         find '{}/.local/share/Trash/info' -name '*.trashinfo' -mtime +{} -delete 2>/dev/null",
        home, tdays, home, tdays
    ))
    .await;
    let after_trash = size_of(&format!("{}/.local/share/Trash", home)).await;
    if before_trash > after_trash {
        result.actions.push(format!(
            "trash: {} MiB freed",
            before_trash - after_trash
        ));
        result.reclaimed_mb += before_trash - after_trash;
    }

    // 4. journal vacuum — system logs (needs sudo)
    if let Some(pass) = sudo {
        let days = config.daemon.journal_vacuum_days;
        let cmd = format!(
            "echo '{}' | sudo -S journalctl --vacuum-time={}d 2>&1 | tail -1",
            pass, days
        );
        if let Ok(out) = sh(&cmd).await {
            if !out.trim().is_empty() && !out.contains("No files") {
                result.actions.push(format!("journals: {}", out.trim()));
            }
        }
    }

    tracing::info!(
        "Cleanup reclaimed ~{} MiB: {}",
        result.reclaimed_mb,
        result.actions.join("; ")
    );
    result
}

async fn size_of(path: &str) -> u64 {
    sh(&format!("du -sm '{}' 2>/dev/null | cut -f1", path))
        .await
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}
