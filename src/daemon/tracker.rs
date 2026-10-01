//! tracker.rs — Process usage learning
//!
//! Observes which processes run and when they actually do work, then
//! classifies them:
//!
//!   daily-use  — seen on enough of the last 7 days that the user clearly
//!                relies on them. These are dynamically excluded from
//!                watchdog notifications (Luna "learns to ignore" them).
//!   important  — learned self-signals, no config: heavy active use (hours
//!                of real work even on irregular days), "never kill" pushback,
//!                or the app being reopened shortly after Luna stopped it.
//!                Important processes are never auto-stopped or suggested.
//!   idle       — running but consuming no CPU across scan cycles.
//!                Idle processes past `auto_stop_after_mins` are stopped
//!                autonomously (announcing intent first); allowlisted ones
//!                stop earlier at `idle_kill_minutes`; everything else only
//!                earns an opt-in suggestion notification.
//!
//! State persists in ~/.local/share/luna/process_stats.json so learning
//! survives reboots. The auto-kill allowlist is a separate plain-text
//! file (~/.local/share/luna/auto_kill.txt) so chat tools can edit it
//! without touching the TOML config.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessStats {
    pub first_seen_epoch: u64,
    pub last_seen_epoch: u64,
    /// Calendar days (YYYY-MM-DD) on which this process was observed
    #[serde(default)]
    pub days_seen: Vec<String>,
    /// Consecutive scan cycles without CPU jiffies advancing
    #[serde(default)]
    pub idle_cycles: u32,
    /// Lifetime observed CPU jiffies
    #[serde(default)]
    pub total_jiffies: u64,
    // ── Importance learning (usage intensity + feedback signals) ──
    /// Total scan cycles this process was observed alive
    #[serde(default)]
    pub observed_cycles: u64,
    /// Scan cycles where the process actually consumed CPU
    #[serde(default)]
    pub active_cycles: u64,
    /// Epoch of the most recent cycle that consumed CPU (recency weighting)
    #[serde(default)]
    pub last_active_epoch: u64,
    /// "Never auto-stop this" — learned (reopen) or explicit (user pushback)
    #[serde(default)]
    pub keep: bool,
    /// Epoch when the daemon last stopped this process (0 = never)
    #[serde(default)]
    pub stopped_at: u64,
    /// App was reopened shortly after being stopped → clearly matters
    #[serde(default)]
    pub reopen_after_stop: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct TrackerData {
    processes: HashMap<String, ProcessStats>,
}

pub struct Tracker {
    path: PathBuf,
    data: TrackerData,
    dirty: bool,
}

fn data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
}

// ── Importance-learning thresholds ────────────────────────────────────────────

/// "Intensive" use: actively worked in across this many scan cycles (≈ hours
/// of real work at typical scan intervals) counts as important by intensity.
pub const INTENSIVE_ACTIVE_CYCLES: u64 = 8;
/// Intensity alone only protects apps used on at least this many distinct days.
const INTENSIVE_MIN_DAYS: usize = 2;
/// An intensively-used app stays important while last used within this window
/// (stops protecting apps abandoned weeks ago).
const RECENCY_WINDOW_SECS: u64 = 21 * 86400;
/// Bringing an app back within this window after Luna stopped it teaches
/// "this one matters".
const REOPEN_WINDOW_SECS: u64 = 36 * 3600;

impl Tracker {
    pub fn load() -> Self {
        let path = data_dir().join("process_stats.json");
        let mut data: TrackerData = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        // Drop entries unseen for a month — stale apps shouldn't haunt us
        let cutoff = now_epoch() - 30 * 86400;
        data.processes.retain(|_, s| s.last_seen_epoch >= cutoff);

        Self { path, data, dirty: false }
    }

    /// Record one observation cycle for `name`. Returns true when the
    /// process consumed CPU since the previous observation.
    pub fn observe(&mut self, name: &str, jiffies_now: u64) -> bool {
        let ts = now_epoch();
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();

        let entry = self.data.processes.entry(name.to_string()).or_default();
        if entry.first_seen_epoch == 0 {
            entry.first_seen_epoch = ts;
        }

        entry.observed_cycles = entry.observed_cycles.saturating_add(1);

        // Did this process do any work since we last looked?
        let advanced = jiffies_now > entry.total_jiffies;
        entry.last_seen_epoch = ts;
        entry.total_jiffies = jiffies_now.max(entry.total_jiffies);

        if !entry.days_seen.contains(&today) {
            entry.days_seen.push(today);
            entry.days_seen.sort();
            // Keep only the trailing month of day markers
            let cutoff_date = (chrono::Local::now() - chrono::Duration::days(31))
                .format("%Y-%m-%d")
                .to_string();
            entry.days_seen.retain(|d| d.as_str() > cutoff_date.as_str());
        }

        if advanced {
            entry.idle_cycles = 0;
            entry.active_cycles = entry.active_cycles.saturating_add(1);
            entry.last_active_epoch = ts;
            // Reopen-after-stop: the user brought this app back shortly after
            // Luna stopped it — that means it matters to them. Learn it.
            let since_stopped = ts.saturating_sub(entry.stopped_at);
            if entry.stopped_at > 0 && since_stopped < REOPEN_WINDOW_SECS {
                if !entry.keep {
                    tracing::info!(
                        "Learned '{}' is important: reopened {}s after being stopped",
                        name,
                        since_stopped
                    );
                }
                entry.keep = true;
                entry.reopen_after_stop = true;
            }
        } else {
            entry.idle_cycles = entry.idle_cycles.saturating_add(1);
        }

        self.dirty = true;
        advanced
    }

    /// Daily-use classification. Returns None while there isn't enough
    /// history yet, Some(true) for daily drivers, Some(false) otherwise.
    pub fn classify_daily_use(&self, name: &str, days_per_week: u32, min_history_days: u32) -> Option<bool> {
        let stats = self.data.processes.get(name)?;
        if (stats.days_seen.len() as u32) < min_history_days {
            return None;
        }
        let week_ago = (chrono::Local::now() - chrono::Duration::days(7))
            .format("%Y-%m-%d")
            .to_string();
        let seen_last_week = stats
            .days_seen
            .iter()
            .filter(|d| d.as_str() > week_ago.as_str())
            .count() as u32;
        Some(seen_last_week >= days_per_week)
    }

    /// Is this process "important" to the user — learned, not configured?
    ///
    /// True when any of these hold:
    ///   - a keep marker was set: user pushback ("never kill X") or the
    ///     reopen-after-stop signal learned via `observe`
    ///   - it has been intensively used: heavy active-CPU time (`active_cycles`)
    ///     spread across several days and used within the last 3 weeks — this
    ///     covers irregular-but-heavy apps that fail the "5 days/week" test
    pub fn is_important(&self, name: &str) -> bool {
        let Some(s) = self.data.processes.get(name) else {
            return false;
        };
        if s.keep {
            return true;
        }
        let ts = now_epoch();
        s.days_seen.len() >= INTENSIVE_MIN_DAYS
            && s.active_cycles >= INTENSIVE_ACTIVE_CYCLES
            && s.last_active_epoch > 0
            && ts.saturating_sub(s.last_active_epoch) < RECENCY_WINDOW_SECS
    }

    /// Persist a keep marker (daemon: reopen learning; tool: user said
    /// "never kill X"). Important processes are never auto-stopped.
    pub fn mark_keep(&mut self, name: &str) {
        let entry = self.data.processes.entry(name.to_string()).or_default();
        entry.keep = true;
        self.dirty = true;
    }

    /// Record that the daemon stopped this process — feeds reopen-after-stop
    /// learning when the user brings it back while `stopped_at` is fresh.
    pub fn mark_stopped(&mut self, name: &str) {
        let entry = self.data.processes.entry(name.to_string()).or_default();
        entry.stopped_at = now_epoch();
        self.dirty = true;
    }

    /// How many consecutive minutes has this process been idle?
    pub fn idle_minutes(&self, name: &str, interval_mins: u64) -> u32 {
        self.data
            .processes
            .get(name)
            .map(|s| s.idle_cycles)
            .unwrap_or(0)
            .saturating_mul(interval_mins.min(u32::MAX as u64) as u32)
    }

    pub fn stats_snapshot(&self) -> &HashMap<String, ProcessStats> {
        &self.data.processes
    }

    pub fn save_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        if let Err(e) = self.save() {
            tracing::warn!("Failed to persist process stats: {}", e);
        } else {
            self.dirty = false;
        }
    }

    fn save(&self) -> Result<()> {
        let dir = data_dir();
        std::fs::create_dir_all(&dir).context("cannot create luna data dir")?;
        let json = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }
}

// ── Auto-kill allowlist (~/.local/share/luna/auto_kill.txt) ───────────────────

pub fn allowlist_path() -> PathBuf {
    data_dir().join("auto_kill.txt")
}

pub fn load_allowlist() -> Vec<String> {
    std::fs::read_to_string(allowlist_path())
        .map(|content| {
            content
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

pub fn allowlist_add(name: &str) -> Result<()> {
    let mut list = load_allowlist();
    if list.iter().any(|n| n == name) {
        return Ok(());
    }
    list.push(name.to_string());
    write_allowlist(&list)
}

pub fn allowlist_remove(name: &str) -> Result<bool> {
    let mut list = load_allowlist();
    let before = list.len();
    list.retain(|n| n != name);
    if list.len() == before {
        return Ok(false);
    }
    write_allowlist(&list)?;
    Ok(true)
}

fn write_allowlist(list: &[String]) -> Result<()> {
    let dir = data_dir();
    std::fs::create_dir_all(&dir)?;
    let mut out = String::from("# Processes Luna may auto-kill when idle\n");
    out.push_str(&list.join("\n"));
    out.push('\n');
    std::fs::write(allowlist_path(), out)?;
    Ok(())
}

/// Persist a keep marker for `name` — called by the `deny_autokill` tool when
/// the user says "never kill X", so the learning survives daemon restarts.
pub fn persist_keep(name: &str) -> Result<()> {
    let mut t = Tracker::load();
    t.mark_keep(name);
    t.save_if_dirty();
    Ok(())
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Tracker {
        Tracker {
            path: PathBuf::from("/tmp/luna-tracker-importance-test.json"),
            data: TrackerData::default(),
            dirty: false,
        }
    }

    #[test]
    fn old_stats_json_without_importance_fields_still_parses() {
        let old = r#"{"processes":{"spotify":{"first_seen_epoch":1,"last_seen_epoch":2,"days_seen":["2026-09-01"],"idle_cycles":3,"total_jiffies":99}}}"#;
        let data: TrackerData = serde_json::from_str(old).unwrap();
        let s = &data.processes["spotify"];
        assert_eq!(s.observed_cycles, 0);
        assert_eq!(s.active_cycles, 0);
        assert_eq!(s.keep, false);
    }

    #[test]
    fn keep_marker_makes_important_forever() {
        let mut t = fresh();
        t.observe("app", 1000); // one active observation, <2 days seen
        assert!(!t.is_important("app"));
        t.mark_keep("app");
        assert!(t.is_important("app"));
    }

    #[test]
    fn intensive_usage_protects_irregular_apps() {
        let mut t = fresh();
        // 10 active observations spread over 2 distinct days
        for i in 0..10u64 {
            t.observe("thesis-app", 1000 + i * 10);
        }
        // force a second day marker
        let s = t.data.processes.get_mut("thesis-app").unwrap();
        s.days_seen.push("2099-01-01".to_string());
        assert!(t.is_important("thesis-app"));
    }

    #[test]
    fn reopen_after_stop_teaches_keep() {
        let mut t = fresh();
        t.observe("chat-app", 500);
        t.mark_stopped("chat-app");
        // user reopens it right away and it does work again
        assert!(t.observe("chat-app", 600));
        let s = t.data.processes.get("chat-app").unwrap();
        assert!(s.keep);
        assert!(s.reopen_after_stop);
        assert!(t.is_important("chat-app"));
    }

    #[test]
    fn idle_usage_never_counts_as_intensive() {
        let mut t = fresh();
        t.observe("idle-runner", 42); // first look only establishes the baseline
        for _ in 0..12 {
            t.observe("idle-runner", 42); // jiffies never advance again
        }
        let s = t.data.processes.get("idle-runner").unwrap();
        assert_eq!(s.active_cycles, 1); // just the baseline look
        assert!(!t.is_important("idle-runner"));
    }

    #[test]
    fn stopped_app_reopened_after_window_is_not_learned() {
        let mut t = fresh();
        t.observe("spotify", 500);
        t.mark_stopped("spotify");
        let s = t.data.processes.get_mut("spotify").unwrap();
        s.stopped_at = now_epoch() - (REOPEN_WINDOW_SECS + 1); // stale stop
        t.observe("spotify", 600);
        let s = t.data.processes.get("spotify").unwrap();
        assert!(!s.keep);
    }
}
