//! daemon/digest.rs — Luna's daily self-audit
//!
//! Once a day (job-gated in the main loop) Luna checks the system on her own
//! initiative and reports one concise digest: disk, reclaimable space, pending
//! updates, orphaned packages, backup status and top memory users. Fully
//! deterministic — no LLM, no hallucinations, pure /proc + system tools.

use crate::config::LunaConfig;
use std::collections::HashMap;

use super::{sh, NameStats};

/// Build today's system-check digest. Empty string = nothing noteworthy.
pub(crate) async fn build(config: &LunaConfig, by_name: &HashMap<String, NameStats>) -> String {
    let mut lines: Vec<String> = Vec::new();

    // ── Root disk + reclaimable space ─────────────────────────────────────
    let disk = sh("df -h / | awk 'NR==2 {print $5}' | tr -d '%'")
        .await
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let pacman = size_of("/var/cache/pacman/pkg").await;
    let cache = size_of(&format!("{home}/.cache")).await;
    let trash = size_of(&format!("{home}/.local/share/Trash")).await;
    let reclaimable = pacman + cache + trash;

    lines.push(format!("Disk: {disk}% of / used"));
    if reclaimable >= config.daemon.min_notify_mb {
        lines.push(format!(
            "Reclaimable: {reclaimable} MiB (pacman cache {pacman} · ~/.cache {cache} · trash {trash})"
        ));
    }

    // ── Pending pacman updates ────────────────────────────────────────────
    if config.proactive.check_updates {
        if let Ok(out) = sh("checkupdates 2>/dev/null | wc -l").await {
            if let Ok(n) = out.trim().parse::<u32>() {
                if n > 0 {
                    lines.push(format!("Updates: {n} package(s) available"));
                }
            }
        }
    }

    // ── Orphaned packages ─────────────────────────────────────────────────
    let orphans = sh("pacman -Qtdq 2>/dev/null").await.unwrap_or_default();
    let orphan_count = orphans
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter(|p| !config.daemon.orphan_keep.iter().any(|k| k == p))
        .count();
    if orphan_count > 0 {
        lines.push(format!("Orphans: {orphan_count} package(s)"));
    }

    // ── Backup status ─────────────────────────────────────────────────────
    if config.daemon.backup_days > 0 {
        let drive = std::path::Path::new("/dev/sda1").exists();
        match (drive, super::job_marker_age("backup")) {
            (true, None) => lines.push("Backup: never run yet — drive is connected, overdue".into()),
            (true, Some(age)) if age >= config.daemon.backup_days as u64 => {
                lines.push(format!("Backup: {age} day(s) ago — due, drive connected"))
            }
            (true, Some(age)) => lines.push(format!(
                "Backup: {age} day(s) ago (due every {}d)",
                config.daemon.backup_days
            )),
            (false, _) => lines.push("Backup: drive not connected".into()),
        }
    }

    // ── Top memory users ──────────────────────────────────────────────────
    let mut hogs: Vec<(&str, u64)> = by_name
        .iter()
        .filter(|(name, s)| {
            s.max_rss_mb > 0 && !config.daemon.ignore_processes.iter().any(|p| p == *name)
        })
        .map(|(name, s)| (name.as_str(), s.max_rss_mb))
        .collect();
    hogs.sort_by(|a, b| b.1.cmp(&a.1));
    hogs.truncate(3);
    if !hogs.is_empty() {
        lines.push(format!(
            "Top memory: {}",
            hogs.iter()
                .map(|(n, mb)| format!("{n} ({mb} MiB)"))
                .collect::<Vec<_>>()
                .join(" · ")
        ));
    }

    lines.join("\n")
}

async fn size_of(path: &str) -> u64 {
    sh(&format!("du -sm '{}' 2>/dev/null | cut -f1", path))
        .await
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}