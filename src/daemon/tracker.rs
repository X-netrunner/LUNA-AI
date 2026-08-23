//! tracker.rs — Process usage learning
//!
//! Observes which processes run and when they actually do work, then
//! classifies them:
//!
//!   daily-use  — seen on enough of the last 7 days that the user clearly
//!                relies on them. These are dynamically excluded from
//!                watchdog notifications (Luna "learns to ignore" them).
//!   idle       — running but consuming no CPU across scan cycles.
//!                Allowlisted idle processes are auto-terminated after
//!                `idle_kill_minutes`; everything else only earns an
//!                opt-in suggestion notification.
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

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
