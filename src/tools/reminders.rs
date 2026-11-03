//! tools/reminders.rs — Real scheduled reminders
//!
//! Reminders are stored on disk so they fire even when no chat session is
//! open. Both the daemon and the proactive monitor poll `fire_due()` and
//! deliver via desktop notification.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reminder {
    pub id: u64,
    /// Unix epoch seconds when this should fire
    pub fire_at: u64,
    pub text: String,
}

fn storage_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
        .join("reminders.json")
}

fn load() -> Vec<Reminder> {
    std::fs::read_to_string(storage_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save(reminders: &[Reminder]) -> Result<()> {
    let path = storage_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(reminders)?)?;
    Ok(())
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Schedule a reminder in `minutes` from now. Returns the created entry.
pub fn add_in(minutes: u32, text: &str) -> Result<Reminder> {
    add_at_epoch(now() + minutes as u64 * 60, text)
}

/// Schedule a reminder for the next occurrence of "HH:MM" (local time).
pub fn add_at(hhmm: &str, text: &str) -> Result<Reminder> {
    let parts: Vec<&str> = hhmm.split(':').collect();
    anyhow::ensure!(parts.len() == 2, "time must be HH:MM");
    let h: i64 = parts[0].parse().context("bad hour")?;
    let m: i64 = parts[1].parse().context("bad minute")?;
    anyhow::ensure!((0..24).contains(&h) && (0..60).contains(&m), "time out of range");

    use chrono::{Datelike, TimeZone};
    let local = chrono::Local;
    let today_naive = chrono::Local::now().date_naive();
    let mut target = local
        .with_ymd_and_hms(today_naive.year(), today_naive.month(), today_naive.day(), h as u32, m as u32, 0)
        .single()
        .context("invalid time")?;
    if target.timestamp() <= now() as i64 {
        target += chrono::Duration::days(1); // tomorrow
    }
    add_at_epoch(target.timestamp() as u64, text)
}

fn add_at_epoch(fire_at: u64, text: &str) -> Result<Reminder> {
    anyhow::ensure!(!text.trim().is_empty(), "reminder text is empty");
    let mut all = load();
    let id = now() * 1000 + (all.len() as u64 % 1000);
    let r = Reminder {
        id,
        fire_at,
        text: text.trim().to_string(),
    };
    all.push(r.clone());
    save(&all)?;
    Ok(r)
}

pub fn list() -> Vec<Reminder> {
    let mut all = load();
    all.sort_by_key(|r| r.fire_at);
    all
}

pub fn cancel(id: u64) -> Result<bool> {
    let mut all = load();
    let before = all.len();
    all.retain(|r| r.id != id);
    if all.len() == before {
        return Ok(false);
    }
    save(&all)?;
    Ok(true)
}

/// Remove and return every reminder whose time has come.
/// The caller delivers them (notification), then they are gone.
pub fn fire_due() -> Result<Vec<Reminder>> {
    let t = now();
    let mut all = load();
    let due: Vec<Reminder> = all.iter().filter(|r| r.fire_at <= t).cloned().collect();
    if due.is_empty() {
        return Ok(due);
    }
    all.retain(|r| r.fire_at > t);
    save(&all)?;
    Ok(due)
}

/// How long until the next reminder fires (None = none pending).
pub fn next_in() -> Option<std::time::Duration> {
    list()
        .first()
        .map(|r| r.fire_at.saturating_sub(now()))
        .map(std::time::Duration::from_secs)
}
