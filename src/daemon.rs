//! daemon.rs — Luna background daemon (`luna --daemon`)
//!
//! A lightweight watchdog that runs as a systemd *user* service next to
//! the interactive agent. Two jobs:
//!
//!   1. Process watchdog — scans /proc for RAM/CPU hogs and fires a
//!      desktop notification with the exact kill command. Processes are
//!      NEVER killed automatically; the user always decides.
//!   2. Disk hygiene — measures reclaimable space (pacman cache,
//!      ~/.cache, trash, journals). In "notify" mode it only reports;
//!      in "auto" mode it cleans those safe locations when / crosses
//!      the disk_full_threshold from [proactive].

use crate::config::LunaConfig;
use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(config: LunaConfig) -> Result<()> {
    if !config.daemon.enabled {
        tracing::info!("Daemon disabled in config — exiting");
        return Ok(());
    }

    let interval = Duration::from_secs(config.daemon.check_interval_mins.max(1) * 60);
    tracing::info!("Luna daemon started (scan every {} min)", interval.as_secs() / 60);

    let mut prev_jiffies: HashMap<u32, (u64, Instant)> = HashMap::new();
    // pid -> reason we flagged it; cleared when the process behaves again
    let mut flagged: HashMap<u32, String> = HashMap::new();
    let mut last_cleanup: Option<Instant> = None;
    let mut last_cleanup_notify: Option<Instant> = None;

    loop {
        tokio::time::sleep(interval).await;

        // ── 1. Process scan ──────────────────────────────────────────────
        match scan_processes(&config, &mut prev_jiffies).await {
            Ok(procs) => {
                let offenders: Vec<_> = procs
                    .iter()
                    .filter(|p| {
                        p.rss_mb >= config.daemon.ram_threshold_mb
                            || p.cpu_percent
                                .map(|c| c >= config.daemon.cpu_threshold_percent)
                                .unwrap_or(false)
                    })
                    .collect();

                // Notify once per (pid, condition); re-notify if the reason changed.
                for p in &offenders {
                    let reason = format!(
                        "{} MiB RAM{}",
                        p.rss_mb,
                        p.cpu_percent
                            .map(|c| format!(", {:.0}% CPU", c))
                            .unwrap_or_default()
                    );
                    let msg = format!(
                        "Process '{}' (pid {}) is using {}.\nTo end it, tell me: kill {}",
                        p.name, p.pid, reason, p.pid
                    );
                    let changed = flagged.get(&p.pid).map(|r| r != &reason).unwrap_or(true);
                    if changed {
                        notify("Luna daemon", &msg).await;
                        flagged.insert(p.pid, reason);
                    }
                }
                // Clear flags for processes that finished or calmed down
                flagged.retain(|pid, _| offenders.iter().any(|o| o.pid == *pid));
                prev_jiffies.retain(|pid, _| flagged.contains_key(pid) || procs.iter().any(|p| p.pid == *pid));
            }
            Err(e) => tracing::warn!("Process scan failed: {}", e),
        }

        // ── 2. Disk hygiene ──────────────────────────────────────────────
        if config.daemon.disk_cleanup {
            if let Err(e) = disk_cycle(
                &config,
                &mut last_cleanup,
                &mut last_cleanup_notify,
            )
            .await
            {
                tracing::warn!("Disk cycle failed: {}", e);
            }
        }

        // ── 3. Orphaned packages ─────────────────────────────────────────
        if let Ok(n) = sh("pacman -Qtdq 2>/dev/null | wc -l").await {
            if let Ok(count) = n.trim().parse::<u32>() {
                if count > 0 {
                    static ONCE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        notify(
                            "Luna daemon",
                            &format!(
                                "{} orphaned package(s) found. Tell me 'remove orphans' to clean them.",
                                count
                            ),
                        )
                        .await;
                    }
                }
            }
        }
    }
}

// ── Process scanning ──────────────────────────────────────────────────────────

#[derive(Debug)]
struct ProcInfo {
    pid: u32,
    name: String,
    rss_mb: u64,
    /// CPU% since the previous scan cycle (None on the first cycle)
    cpu_percent: Option<f32>,
}

fn page_size() -> u64 {
    std::process::Command::new("getconf")
        .arg("PAGESIZE")
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(4096)
}

async fn scan_processes(
    config: &LunaConfig,
    prev_jiffies: &mut HashMap<u32, (u64, Instant)>,
) -> Result<Vec<ProcInfo>> {
    const CLK_TCK: f64 = 100.0; // kernel CONFIG_HZ — universal on x86_64
    let pgsize = page_size();

    let entries = tokio::fs::read_dir("/proc").await?;
    let mut out: Vec<ProcInfo> = Vec::new();
    let now = Instant::now();

    // Read every numeric /proc entry; unreadable ones are skipped
    // (kernel threads and other users' processes).
    let mut read_set = Vec::new();
    let mut dir = entries;
    while let Some(entry) = dir.next_entry().await? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.chars().all(|c| c.is_ascii_digit()) {
            read_set.push(name.to_string());
        }
    }

    for pid_str in read_set {
        let pid: u32 = match pid_str.parse() { Ok(p) => p, Err(_) => continue };
        let base = PathBuf::from("/proc").join(&pid_str);

        let comm = match tokio::fs::read_to_string(base.join("comm")).await {
            Ok(c) => c.trim().to_string(),
            Err(_) => continue,
        };

        // Skip our own ignore list
        if config.daemon.ignore_processes.iter().any(|i| i == &comm) {
            continue;
        }

        let stat = match tokio::fs::read_to_string(base.join("stat")).await {
            Ok(s) => s,
            Err(_) => continue,
        };

        // comm can contain spaces/parens — parse fields after the LAST ')'
        let after = match stat.rfind(')') {
            Some(i) => &stat[i + 1..],
            None => continue,
        };
        let fields: Vec<&str> = after.split_whitespace().collect();
        if fields.len() < 13 {
            continue;
        }

        // Zombie processes hold no memory worth reporting
        if fields[0] == "Z" {
            continue;
        }

        let rss_pages: u64 = match tokio::fs::read_to_string(base.join("statm")).await {
            Ok(s) => s
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            Err(_) => continue,
        };
        let rss_mb = rss_pages * pgsize / 1024 / 1024;

        // CPU jiffies delta since last cycle
        let total_jiffies: u64 = fields[11].parse::<u64>().unwrap_or(0)
            + fields[12].parse::<u64>().unwrap_or(0);
        let cpu_percent = match prev_jiffies.remove(&pid) {
            Some((prev, at)) => {
                let elapsed = now.duration_since(at).as_secs_f64();
                if elapsed > 0.0 && total_jiffies >= prev {
                    Some(((total_jiffies - prev) as f64 / (elapsed * CLK_TCK) * 100.0) as f32)
                } else {
                    None
                }
            }
            None => None,
        };
        prev_jiffies.insert(pid, (total_jiffies, now));

        out.push(ProcInfo {
            pid,
            name: comm,
            rss_mb,
            cpu_percent,
        });
    }

    // Biggest consumers first so the notification reads naturally
    out.sort_by_key(|p| std::cmp::Reverse(p.rss_mb));
    Ok(out)
}

// ── Disk hygiene ──────────────────────────────────────────────────────────────

struct DiskUsage {
    used_pct: u32,
    pacman_cache_mb: u64,
    user_cache_mb: u64,
    trash_mb: u64,
}

/// One disk-hygiene cycle: measure, then act according to cleanup_mode.
async fn disk_cycle(
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
                notify(
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
            notify(
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

    let du = |path: &str| {
        let p = path.to_string();
        async move {
            sh(&format!("du -sm '{}' 2>/dev/null | cut -f1", p))
                .await
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        }
    };

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    DiskUsage {
        used_pct,
        pacman_cache_mb: du("/var/cache/pacman/pkg").await,
        user_cache_mb: du(&format!("{}/.cache", home)).await,
        trash_mb: du(&format!("{}/.local/share/Trash", home)).await,
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

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn sh(cmd: &str) -> Result<String> {
    let output = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

async fn notify(title: &str, body: &str) {
    // Direct exec — no shell, so no quoting/escaping concerns
    let _ = tokio::process::Command::new("notify-send")
        .args(["-u", "normal", title, body])
        .output()
        .await;
}
