//! daemon/mod.rs — Luna background daemon (`luna --daemon`)
//!
//! A lightweight watchdog that runs as a systemd *user* service next to
//! the interactive agent. Three jobs:
//!
//!   1. Process watchdog — scans /proc for RAM/CPU hogs and fires a
//!      desktop notification with the exact kill command. Learned
//!      daily-use processes are silently ignored.
//!   2. Usage learning — idle non-daily processes become auto-kill
//!      candidates: user-approved ones get terminated, others earn an
//!      opt-in suggestion. Stateful apps are always protected.
//!   3. Disk hygiene — measures reclaimable space; cleans only safe
//!      locations and only in "auto" mode.

pub mod cleanup;
pub mod tracker;
pub mod watchdog;

use crate::config::LunaConfig;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracker::Tracker;

/// One scan cycle's view of a process NAME (all pids merged)
struct NameStats {
    pids: Vec<u32>,
    total_jiffies: u64,
    max_rss_mb: u64,
}

pub async fn run(config: LunaConfig) -> Result<()> {
    if !config.daemon.enabled {
        tracing::info!("Daemon disabled in config — exiting");
        return Ok(());
    }

    let interval = Duration::from_secs(config.daemon.check_interval_mins.max(1) * 60);
    let interval_mins = config.daemon.check_interval_mins.max(1);
    tracing::info!("Luna daemon started (scan every {} min)", interval_mins);

    let mut prev_jiffies: HashMap<u32, (u64, Instant)> = HashMap::new();
    let mut flagged: HashMap<u32, String> = HashMap::new();
    // Pids we already SIGTERMed once — if still alive next cycle, SIGKILL
    let mut term_killed: HashSet<u32> = HashSet::new();
    // name -> last time we suggested opt-in auto-kill (24h cooldown)
    let mut last_suggest: HashMap<String, Instant> = HashMap::new();
    let mut last_cleanup: Option<Instant> = None;
    let mut last_cleanup_notify: Option<Instant> = None;
    let mut tracker = Tracker::load();

    loop {
        // ── 1. Scan processes ────────────────────────────────────────────
        let procs = match watchdog::scan_processes(&config, &mut prev_jiffies).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Process scan failed: {}", e);
                Vec::new()
            }
        };

        // Aggregate by name so learning works across multi-process apps
        let mut by_name: HashMap<String, NameStats> = HashMap::new();
        for p in &procs {
            let entry = by_name.entry(p.name.clone()).or_insert(NameStats {
                pids: Vec::new(),
                total_jiffies: 0,
                max_rss_mb: 0,
            });
            entry.pids.push(p.pid);
            entry.total_jiffies += p.total_jiffies;
            entry.max_rss_mb = entry.max_rss_mb.max(p.rss_mb);
        }

        // ── 2. Feed the learner ──────────────────────────────────────────
        let learning = config.daemon.learning_enabled;
        for (name, stats) in &by_name {
            if learning {
                tracker.observe(name, stats.total_jiffies);
            }
        }

        // ── 3. Resource offenders (with dynamic ignore) ──────────────────
        for p in &procs {
            let over =
                p.rss_mb >= config.daemon.ram_threshold_mb
                    || p.cpu_percent
                        .map(|c| c >= config.daemon.cpu_threshold_percent)
                        .unwrap_or(false);
            if !over {
                continue;
            }
            // Daily-use apps never nag — this is the learned ignore list
            if learning
                && tracker
                    .classify_daily_use(
                        &p.name,
                        config.daemon.daily_use_days_per_week,
                        3,
                    )
                    .unwrap_or(false)
            {
                continue;
            }
            let reason = format!(
                "{} MiB RAM{}",
                p.rss_mb,
                p.cpu_percent
                    .map(|c| format!(", {:.0}% CPU", c))
                    .unwrap_or_default()
            );
            let changed = flagged.get(&p.pid).map(|r| r != &reason).unwrap_or(true);
            if changed {
                notify(
                    "Luna daemon",
                    &format!(
                        "Process '{}' (pid {}) is using {}.\nTo end it, tell me: kill {}",
                        p.name, p.pid, reason, p.pid
                    ),
                )
                .await;
                flagged.insert(p.pid, reason);
            }
        }
        flagged.retain(|pid, _| procs.iter().any(|o| o.pid == *pid));

        // ── 4. Idle process policy ───────────────────────────────────────
        if learning {
            handle_idle(&config, &procs, &by_name, &mut tracker, &mut term_killed, &mut last_suggest).await;
        }

        // ── 5. Disk hygiene ──────────────────────────────────────────────
        if config.daemon.disk_cleanup {
            if let Err(e) =
                cleanup::disk_cycle(&config, &mut last_cleanup, &mut last_cleanup_notify).await
            {
                tracing::warn!("Disk cycle failed: {}", e);
            }
        }

        // ── 6. Orphaned packages ─────────────────────────────────────────
        if let Ok(n) = sh("pacman -Qtdq 2>/dev/null | wc -l").await {            if let Ok(count) = n.trim().parse::<u32>() {
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

        // ── 7. Monthly shell-history learning ────────────────────────────
        if config.daemon.history_learn_days > 0 {
            match crate::memory::workflow::learn_if_due(config.daemon.history_learn_days) {
                Ok(Some(summary)) => {
                    tracing::info!("{}", summary);
                    notify("Luna daemon", &summary).await;
                }
                Ok(None) => {}
                Err(e) => tracing::warn!("Workflow learning failed: {}", e),
            }
        }

        tracker.save_if_dirty();
        prev_jiffies.retain(|pid, _| procs.iter().any(|p| p.pid == *pid));

        tokio::time::sleep(interval).await;
    }
}

// ── Idle handling ─────────────────────────────────────────────────────────────

async fn handle_idle(
    config: &LunaConfig,
    procs: &[watchdog::ProcInfo],
    by_name: &HashMap<String, NameStats>,
    tracker: &mut Tracker,
    term_killed: &mut HashSet<u32>,
    last_suggest: &mut HashMap<String, Instant>,
) {
    let interval_mins = config.daemon.check_interval_mins.max(1);

    for (name, stats) in by_name {
        // Protected stateful apps are untouchable, full stop
        if config
            .daemon
            .protected_processes
            .iter()
            .any(|p| p == name)
        {
            continue;
        }
        // Daily-use apps are doing their job just by existing
        if tracker
            .classify_daily_use(name, config.daemon.daily_use_days_per_week, 3)
            .unwrap_or(false)
        {
            continue;
        }

        let idle_mins = tracker.idle_minutes(name, interval_mins);
        if idle_mins < config.daemon.idle_kill_minutes.min(config.daemon.suggest_autokill_after_mins) {
            continue;
        }

        let allowlisted = tracker::load_allowlist().iter().any(|a| a == name);

        if allowlisted && idle_mins >= config.daemon.idle_kill_minutes {
            // Escalate TERM -> KILL for pids that survived the previous cycle
            let survivors: Vec<u32> = stats
                .pids
                .iter()
                .filter(|pid| term_killed.contains(pid))
                .copied()
                .collect();
            for pid in &survivors {
                let _ = sh(&format!("kill -9 {} 2>/dev/null", pid)).await;
                tracing::info!("SIGKILL idle process '{}' (pid {})", name, pid);
            }

            // Fresh targets get SIGTERM
            let targets: Vec<u32> = stats
                .pids
                .iter()
                .filter(|pid| !term_killed.contains(pid))
                .copied()
                .collect();
            for pid in &targets {
                let _ = sh(&format!("kill -TERM {} 2>/dev/null", pid)).await;
                tracing::info!("SIGTERM idle process '{}' (pid {})", name, pid);
            }
            for pid in &stats.pids {
                term_killed.insert(*pid);
            }

            if !targets.is_empty() || !survivors.is_empty() {
                notify(
                    "Luna daemon",
                    &format!(
                        "Ended idle '{}' ({} min without activity, {} MiB).",
                        name, idle_mins, stats.max_rss_mb
                    ),
                )
                .await;
            }
        } else if !allowlisted && idle_mins >= config.daemon.suggest_autokill_after_mins {
            // Opt-in suggestion — at most once per day per process
            let cooled = last_suggest
                .get(name)
                .map(|t| t.elapsed() > Duration::from_secs(24 * 3600))
                .unwrap_or(true);
            if cooled {
                notify(
                    "Luna daemon",
                    &format!(
                        "'{}' has been idle {} min ({} MiB).\nSay 'allow auto-kill {}' to let me \
                         end it automatically when idle.",
                        name, idle_mins, stats.max_rss_mb, name
                    ),
                )
                .await;
                last_suggest.insert(name.clone(), Instant::now());
            }
        }
        let _ = procs; // pids come from by_name aggregation
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

pub(crate) async fn sh(cmd: &str) -> Result<String> {
    let output = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub(crate) async fn notify(title: &str, body: &str) {
    // Direct exec — no shell, so no quoting/escaping concerns
    let _ = tokio::process::Command::new("notify-send")
        .args(["-u", "normal", title, body])
        .output()
        .await;
}
