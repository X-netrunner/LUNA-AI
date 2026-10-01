//! daemon/mod.rs — Luna background daemon (`luna --daemon`)
//!
//! A lightweight watchdog that runs as a systemd *user* service next to
//! the interactive agent. Three jobs:
//!
//!   1. Process watchdog — scans /proc for RAM/CPU hogs and fires a
//!      desktop notification with the exact kill command. Learned
//!      daily-use processes are silently ignored.
//!   2. Usage learning — idle non-daily processes are stopped autonomously
//!      past `auto_stop_after_mins`, but only after one notify-send announcing
//!      exactly what is about to stop (longest idle first) and one "done"
//!      summary. Stateful apps are always protected; allowlisted apps still
//!      stop earlier at `idle_kill_minutes`, others earn an opt-in suggestion.
//!   3. Disk hygiene — measures reclaimable space; cleans only safe
//!      locations and only in "auto" mode.

pub mod cleanup;
pub mod digest;
pub mod tracker;
pub mod watchdog;

use crate::config::LunaConfig;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracker::Tracker;

/// One scan cycle's view of a process NAME (all pids merged)
pub(crate) struct NameStats {
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

    // Heartbeat stats — surfaced in the periodic "I'm alive" notification
    let started = Instant::now();
    let mut last_heartbeat = Instant::now();
    let mut cycles: u64 = 0;
    let mut autokills: u64 = 0;
    let mut reminders_fired: u64 = 0;

    loop {
        // ── 0. Due reminders — checked FIRST every cycle so heavyweight
        // jobs below (disk du, system re-index) never delay a firing
        // reminder past its time ─────────────────────────────────────────
        match crate::tools::reminders::fire_due() {
            Ok(due) if !due.is_empty() => {
                for r in due {
                    tracing::info!("Reminder fired: {}", r.text);
                    reminders_fired += 1;
                    notify("Luna — Reminder", &r.text).await;
                }
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("Reminder check failed: {}", e),
        }

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
            autokills +=
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
// ── 6. Orphaned packages (grace-period auto-removal) ─────────────
        if let Err(e) = cleanup::orphans_cycle(&config).await {
            tracing::warn!("Orphaned-package check failed: {}", e);
        }

        // ── 6b. Daily system-check digest — Luna audits on her own ────────
        // She looks around once a day and reports one concise state-of-the-
        // system, on her own initiative (no prompt needed).
        if config.daemon.system_digest_days > 0
            && job_due("sys_digest", config.daemon.system_digest_days)
        {
            let body = digest::build(&config, &by_name).await;
            if !body.is_empty() {
                notify("Luna — system check", &body).await;
                tracing::info!("System digest:\n{}", body);
            }
            job_done("sys_digest");
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

        // ── 8. Monthly system re-index (detached — find over home can
        // take minutes on first run; don't stall the watchdog) ────────────
        if config.daemon.index_learn_days > 0 {
            let days = config.daemon.index_learn_days;
            let sudo = config.agent.sudo_password.clone();
            tokio::spawn(async move {
                match crate::memory::workflow::index_if_due(days, sudo.as_deref()).await {
                    Ok(Some(summary)) => {
                        tracing::info!("{}", summary);
                        notify("Luna daemon", &summary).await;
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!("System indexing failed: {}", e),
                }
            });
        }

        // ── 9. Weekly safety check (detached — ClamAV + pacman -Syu can
        // take up to an hour; never stall the watchdog) ────────────────────
        if config.daemon.safety_check_days > 0 {
            let due = job_due("safety_check", config.daemon.safety_check_days);
            let running = crate::tools::safety::is_running();
            if due && !running {
                let sudo = config.agent.sudo_password.clone();
                tokio::spawn(async move {
                    match crate::tools::safety::run("light", sudo.as_deref(), false).await {
                        Ok(summary) => {
                            tracing::info!("Safety check: {}", summary);
                            job_done("safety_check");
                            notify("Luna daemon — Safety Check", &summary).await;
                        }
                        Err(e) => tracing::warn!("Safety check failed: {}", e),
                    }
                });
            }
        }

        // ── 10. Weekly backup (/dev/sda1 only — skipped while unplugged) ──
        if config.daemon.backup_days > 0
            && job_due("backup", config.daemon.backup_days)
            && std::path::Path::new("/dev/sda1").exists()
        {
            let sudo = config.agent.sudo_password.clone();
            tokio::spawn(async move {
                match crate::tools::backup::run(sudo.as_deref()).await {
                    Ok(summary) => {
                        tracing::info!("Backup: {}", summary);
                        job_done("backup");
                        notify("Luna daemon — Backup", &summary).await;
                    }
                    Err(e) => tracing::warn!("Backup failed: {}", e),
                }
            });
        }

        tracker.save_if_dirty();
        prev_jiffies.retain(|pid, _| procs.iter().any(|p| p.pid == *pid));
        cycles += 1;

        // ── Periodic "I'm alive" notification (notify_hours, 0 = off) ────
        if config.daemon.notify_hours > 0
            && last_heartbeat.elapsed() >= Duration::from_secs(config.daemon.notify_hours as u64 * 3600)
        {
            last_heartbeat = Instant::now();
            let facts = crate::memory::permanent::PermanentMemory::load()
                .map(|p| p.all_facts().len())
                .unwrap_or(0);

            // Proactive Todoist summary (if token configured)
            let mut todoist_line = String::new();
            if let Some(token) = &config.todoist.api_token {
                if !token.is_empty() {
                    if let Ok(Some(summary)) = crate::tools::todoist::proactive_task_summary(token).await {
                        todoist_line = format!(" · {}", summary);
                    }
                }
            }

            notify(
                "Luna daemon",
                &format!(
                    "Alive {} · {} cycles · {} auto-kills · {} reminders fired · {} known facts{}",
                    fmt_uptime(started.elapsed()),
                    cycles,
                    autokills,
                    reminders_fired,
                    facts,
                    todoist_line
                ),
            )
            .await;
        }

        // ── 11. Keep the desktop notification corner bounded ─────────────
        if config.daemon.notif_cap > 0 {
            enforce_notif_cap(config.daemon.notif_cap).await;
        }

        // Sleep until the next scan OR the next reminder, whichever first —
        // so a "remind me in 2 minutes" doesn't wait out the whole interval.
        let next_reminder = crate::tools::reminders::next_in()
            .map(|d| d.min(interval))
            .unwrap_or(interval);
        tokio::time::sleep(next_reminder.max(Duration::from_secs(5))).await;
    }
}

/// Keep the desktop notification corner bounded. Reads the running shell's
/// stored notification list (caelestia/Quickshell) and, when it exceeds the
/// configured cap, tells the shell to clear it. no-op when cap == 0.
async fn enforce_notif_cap(cap: u32) {
    if cap == 0 {
        return;
    }
    let state = dirs::state_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("caelestia")
        .join("notifs.json");
    let Ok(text) = tokio::fs::read_to_string(&state).await else {
        return; // shell not present or not running — nothing to do
    };
    let count = serde_json::from_str::<Vec<serde_json::Value>>(&text)
        .map(|v| v.len())
        .unwrap_or(0);
    if count <= cap as usize {
        return;
    }
    tracing::info!("Notification corner has {count} items (cap {cap}) — clearing via shell");
    let _ = tokio::process::Command::new("qs")
        .args(["-c", "caelestia", "ipc", "call", "notifs", "clear"])
        .output()
        .await;
}

// ── Idle handling ─────────────────────────────────────────────────────────────

/// Idle-process policy (autonomous mode).
///
/// Any non-protected, non-important, non-daily-use process idle past
/// `auto_stop_after_mins` gets stopped on Luna's own — but only after ONE
/// notify-send announcing exactly what she is about to stop (full list,
/// longest-idle first, "in order of time"), and a single "done" summary once
/// the stops have run. Allowlisted processes still stop earlier at
/// `idle_kill_minutes`. Every stop is fed back to the tracker so a quick
/// Byte-exact protected-process matcher (the idle-kill short-circuit).
///
/// The comparison is deliberately a strict `==` against `/proc/<pid>/comm`:
/// kernel comm is case-sensitive and NOT lowercased, so a config entry only
/// protects a process when it byte-matches the comm string exactly. This is
/// load-bearing — OnlyOffice's comm is `DesktopEditors` (camelCase) and its
/// helpers are `editors_helper`; a lowercased `desktopeditors` entry does NOT
/// protect them and they will be idle-killed. Any config for a protected app
/// must byte-match the kernel comm (see the README Operating Notes).
fn is_protected(config: &LunaConfig, name: &str) -> bool {
    config
        .daemon
        .protected_processes
        .iter()
        .any(|p| p == name)
}

/// Any non-protected, non-important, non-daily-use process idle past
/// `auto_stop_after_mins` gets stopped on Luna's own — but only after ONE
/// notify-send announcing exactly what she is about to stop (full list,
/// longest-idle first, "in order of time"), and a single "done" summary once
/// the stops have run. Allowlisted processes still stop earlier at
/// `idle_kill_minutes`. Every stop is fed back to the tracker so a quick
/// reopen teaches her that the app matters.
async fn handle_idle(
    config: &LunaConfig,
    procs: &[watchdog::ProcInfo],
    by_name: &HashMap<String, NameStats>,
    tracker: &mut Tracker,
    term_killed: &mut HashSet<u32>,
    last_suggest: &mut HashMap<String, Instant>,
) -> u64 {
    let interval_mins = config.daemon.check_interval_mins.max(1);
    let idle_kill = config.daemon.idle_kill_minutes;
    let suggest_after = config.daemon.suggest_autokill_after_mins;
    let auto_stop = config.daemon.auto_stop_after_mins;
    let allowlist = tracker::load_allowlist();

    fn daily_use(tracker: &Tracker, config: &LunaConfig, name: &str) -> bool {
        tracker
            .classify_daily_use(name, config.daemon.daily_use_days_per_week, 3)
            .unwrap_or(false)
    }
    // Learned importance (usage intensity, keep markers, reopen-after-stop)
    // plus the daily-use frequency rule → never targeted by idle policy.
    fn important(tracker: &Tracker, config: &LunaConfig, name: &str) -> bool {
        tracker.is_important(name) || daily_use(tracker, config, name)
    }
    // Will this name be handled by the autonomous/allowlist kill pass below?
    let will_kill = |name: &str, idle_mins: u32| {
        let allowlisted = allowlist.iter().any(|a| a == name);
        (allowlisted && idle_kill > 0 && idle_mins >= idle_kill)
            || (auto_stop > 0 && idle_mins >= auto_stop)
    };

    // ── Suggestion pass — processes not being stopped this cycle still earn
    // the opt-in suggestion while they idle (at most once per day per name).
    for (name, stats) in by_name {
        if is_protected(config, name) || important(tracker, config, name) {
            continue;
        }
        let idle_mins = tracker.idle_minutes(name, interval_mins);
        if idle_mins < suggest_after || will_kill(name, idle_mins) {
            continue;
        }
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

    // ── Kill pass — collect this cycle's targets, announce intent ONCE
    // (longest idle first), then act, then one done summary.
    struct Target<'a> {
        name: &'a str,
        stats: &'a NameStats,
        idle_mins: u32,
    }
    let mut targets: Vec<Target> = Vec::new();
    for (name, stats) in by_name {
        if is_protected(config, name) || important(tracker, config, name) {
            continue;
        }
        let idle_mins = tracker.idle_minutes(name, interval_mins);
        if !will_kill(name, idle_mins) {
            continue;
        }
        targets.push(Target {
            name,
            stats,
            idle_mins,
        });
    }
    if targets.is_empty() {
        return 0;
    }

    // "in order of time" — longest idle first
    targets.sort_by(|a, b| b.idle_mins.cmp(&a.idle_mins));

    let period = if auto_stop > 0 && auto_stop % 60 == 0 {
        format!("{}h", auto_stop / 60)
    } else {
        format!("{} min", auto_stop.max(1))
    };
    let lines: Vec<String> = targets
        .iter()
        .enumerate()
        .map(|(i, t)| {
            format!(
                "  {}. {} — idle {} ({} MiB)",
                i + 1,
                t.name,
                fmt_idle_mins(t.idle_mins),
                t.stats.max_rss_mb
            )
        })
        .collect();
    let names: Vec<&str> = targets.iter().map(|t| t.name).collect();

    // ANNOUNCE BEFORE ACTING — one message listing everything about to stop
    notify(
        "Luna — about to act",
        &format!(
            "I am about to stop {} unused program(s) idle ≥ {}:\n{}",
            targets.len(),
            period,
            lines.join("\n")
        ),
    )
    .await;
    tracing::info!(
        "About to stop {} idle program(s): {}",
        targets.len(),
        names.join(", ")
    );

    // Act: SIGTERM fresh pids; SIGKILL survivors that ignored last cycle
    let mut kills: u64 = 0;
    for t in &targets {
        let survivors: Vec<u32> = t
            .stats
            .pids
            .iter()
            .filter(|pid| term_killed.contains(pid))
            .copied()
            .collect();
        for pid in &survivors {
            let _ = sh(&format!("kill -9 {} 2>/dev/null", pid)).await;
            tracing::info!("SIGKILL idle process '{}' (pid {})", t.name, pid);
            kills += 1;
        }
        let fresh: Vec<u32> = t
            .stats
            .pids
            .iter()
            .filter(|pid| !term_killed.contains(pid))
            .copied()
            .collect();
        for pid in &fresh {
            let _ = sh(&format!("kill -TERM {} 2>/dev/null", pid)).await;
            tracing::info!("SIGTERM idle process '{}' (pid {})", t.name, pid);
            kills += 1;
        }
        for pid in &t.stats.pids {
            term_killed.insert(*pid);
        }
        // Record the stop so a quick reopen teaches "this app matters"
        tracker.mark_stopped(t.name);
    }

    // One done summary — replaces the old per-process confirmations
    notify(
        "Luna — done",
        &format!(
            "Stopped {} unused program(s): {}.",
            targets.len(),
            names.join(", ")
        ),
    )
    .await;

    let _ = procs; // pids come from by_name aggregation
    kills
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

/// "2h05m" / "47m" / "<1m" for the heartbeat digest
fn fmt_uptime(d: Duration) -> String {
    let mins = d.as_secs() / 60;
    if mins >= 60 {
        format!("{}h{:02}m", mins / 60, mins % 60)
    } else {
        format!("{}m", mins.max(1))
    }
}

/// "5h24m" / "47m" — idle-time formatter for the announce-before-act list
fn fmt_idle_mins(mins: u32) -> String {
    if mins >= 60 {
        format!("{}h{:02}m", mins / 60, mins % 60)
    } else {
        format!("{}m", mins.max(1))
    }
}

// ── Marker-gated jobs (safety check, backup) ─────────────────────────────────

/// Path of a job marker in ~/.local/share/luna/last_<name>.
fn job_marker_path(name: &str) -> std::path::PathBuf {
    crate::memory::workflow::marker_dir().join(format!("last_{}", name))
}

/// True when `days` have passed since the marker was last written (or it has never run).
fn job_due(name: &str, days: u32) -> bool {
    if days == 0 {
        return false;
    }
    let path = job_marker_path(name);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(ts) = std::fs::read_to_string(&path) {
        let last: u64 = ts.trim().parse().unwrap_or(0);
        if now.saturating_sub(last) < days as u64 * 86400 {
            return false;
        }
    }
    true
}

/// Stamp a job marker so its window restarts.
fn job_done(name: &str) {
    let path = job_marker_path(name);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default();
    let _ = std::fs::write(&path, now);
}

/// Days since the marker was last stamped (None if it has never been stamped).
fn job_marker_age(name: &str) -> Option<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let last: u64 = std::fs::read_to_string(job_marker_path(name))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(now.saturating_sub(last) / 86400)
}

/// Forget a job marker so its window starts fresh.
fn job_forget(name: &str) {
    let _ = std::fs::remove_file(job_marker_path(name));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_time_formatting() {
        assert_eq!(fmt_idle_mins(0), "1m");
        assert_eq!(fmt_idle_mins(45), "45m");
        assert_eq!(fmt_idle_mins(60), "1h00m");
        assert_eq!(fmt_idle_mins(125), "2h05m");
        assert_eq!(fmt_idle_mins(324), "5h24m");
    }

    // Regression guard: a byte-exact protected comm must NEVER be selected as
    // an idle-kill target. This is the invariant whose absence let OnlyOffice
    // (`DesktopEditors` + `editors_helper`) and `micro` get SIGTERM'd: the
    // matcher is a strict, case-sensitive `==` against /proc/<pid>/comm, and a
    // lowercased config entry silently fails to protect the real camelCase
    // process. If a future edit makes the matcher case-insensitive (or the
    // defaults drop these names), this test fails.
    #[test]
    fn protected_comms_are_never_kill_targets() {
        // Use the SHIPPED defaults so the exact names that bit us are locked in.
        let config = LunaConfig::default();

        // Every protected default is recognized as protected (byte-exact).
        for p in &config.daemon.protected_processes {
            assert!(
                is_protected(&config, p),
                "default protected comm {:?} must be recognized as protected",
                p
            );
        }

        // The names that were actually SIGTERM'd must be in the defaults and
        // must be recognized byte-exactly.
        for comm in ["DesktopEditors", "editors_helper", "micro"] {
            assert!(
                config.daemon.protected_processes.iter().any(|p| p == comm),
                "default protected_processes must contain byte-exact {:?}",
                comm
            );
            assert!(
                is_protected(&config, comm),
                "{:?} must be recognized as protected (byte-exact comm)",
                comm
            );
        }

        // The matcher is case-sensitive: a lowercased comm is NOT the same
        // process, so it must NOT be treated as protected by the camelCase
        // entry. (This documents the exact trap; if someone "fixes" the matcher
        // to be case-insensitive, the assertion below flips and CI catches it.)
        assert!(
            !is_protected(&config, "desktopeditors"),
            "lowercase 'desktopeditors' must not be protected by the camelCase entry"
        );

        // An unknown process is not protected.
        assert!(!is_protected(&config, "definitely-not-a-real-process"));
    }
}
