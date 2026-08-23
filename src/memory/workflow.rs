//! memory/workflow.rs — Scheduled shell-history learning
//!
//! Once a month (scheduled by the daemon) Luna re-reads the full fish
//! history and distills durable workflow facts into permanent memory:
//! which commands dominate, which packages get managed repeatedly.
//!
//! Facts use stable prefixes ("Shell workflow: ...") so PermanentMemory's
//! near-duplicate guard updates them in place instead of accumulating
//! copies every month.

use crate::memory::permanent::PermanentMemory;
use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;

fn marker_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
        .join("last_workflow_learn")
}

fn history_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/fish/fish_history")
}

/// Run the learner if at least `min_interval_days` days have passed
/// since the last run (or if it has never run). Returns a human summary
/// of what was learned, or None when not due / nothing found.
pub fn learn_if_due(min_interval_days: u32) -> Result<Option<String>> {
    let marker = marker_path();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if min_interval_days == 0 {
        return Ok(None);
    }
    if let Ok(ts) = std::fs::read_to_string(&marker) {
        let last: u64 = ts.trim().parse().unwrap_or(0);
        if now.saturating_sub(last) < min_interval_days as u64 * 86400 {
            return Ok(None);
        }
    }

    let summary = learn_once()?;
    std::fs::write(&marker, now.to_string())?;
    Ok(summary)
}

/// One analysis pass over the whole fish history file.
fn learn_once() -> Result<Option<String>> {
    let content = match std::fs::read_to_string(history_path()) {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };

    // First-token frequency across all history, minus conversational noise
    let mut counts: HashMap<&str, u32> = HashMap::new();
    let mut packages: Vec<&str> = Vec::new();

    for line in content.lines() {
        let Some(cmd) = line.strip_prefix("- cmd:") else {
            continue;
        };
        let cmd = cmd.trim();
        if cmd.is_empty() || cmd.starts_with('#') || cmd.len() > 100 {
            continue;
        }
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        // "sudo pacman ..." is really about pacman — look past sudo
        let key = if tokens[0] == "sudo" && tokens.len() > 1 {
            tokens[1]
        } else {
            tokens[0]
        };
        match key {
            "cd" | "ls" | "clear" | "pwd" | "cat" | "echo" | "grep" | "exit" => continue,
            _ => {}
        }
        *counts.entry(key).or_insert(0) += 1;

        // Package installs: "<sudo> <pacman|paru|yay|pikaur> [-flags with S] pkgs..."
        if let Some(pos) = tokens
            .iter()
            .position(|t| matches!(*t, "pacman" | "paru" | "yay" | "pikaur"))
        {
            let has_install = tokens[pos + 1..]
                .iter()
                .any(|t| t.starts_with('-') && !t.starts_with("--") && t.contains('S'));
            if has_install {
                packages.extend(tokens[pos + 1..].iter().filter(|t| !t.starts_with('-')));
            }
        }
    }

    if counts.is_empty() {
        return Ok(None);
    }

    let mut top: Vec<(&str, u32)> = counts.into_iter().collect();
    top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    let top_commands = top
        .iter()
        .take(10)
        .map(|(cmd, c)| format!("{}({})", cmd, c))
        .collect::<Vec<_>>()
        .join(", ");

    let mut pm = PermanentMemory::load()?;
    let cmds_fact =
        format!("Shell workflow: user's most-used commands are {}", top_commands);
    if let Err(e) = pm.remember(&cmds_fact, "workflow") {
        return Err(e.context("failed to store workflow fact"));
    }

    let mut extra = String::new();
    if !packages.is_empty() {
        packages.sort();
        packages.dedup();
        let pkg_list = packages.iter().take(12).cloned().collect::<Vec<_>>().join(", ");
        let pkg_fact = format!(
            "Shell workflow: packages managed via pacman/AUR helpers include {}",
            pkg_list
        );
        if let Err(e) = pm.remember(&pkg_fact, "workflow") {
            return Err(e.context("failed to store package fact"));
        }
        extra = format!(" + {} known package(s)", packages.len());
    }

    Ok(Some(format!(
        "Learned shell workflow: {} distinct commands analyzed{}",
        top.len(),
        extra
    )))
}

// ── Monthly system indexing ───────────────────────────────────────────────────

fn index_marker_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
        .join("last_index_learn")
}

/// Re-map projects/scripts/configs into permanent memory.
/// Used by the index_system tool and by the daemon's monthly job.
pub async fn run_index_system(sudo_pass: Option<&str>) -> Result<Vec<String>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let commands = vec![
        ("projects", format!("find {} -name '.git' -maxdepth 4 -type d 2>/dev/null | grep -v '.cache' | sed 's/\\/.git//' | head -20", home)),
        ("scripts",  format!("find {} -name '*.sh' -maxdepth 4 2>/dev/null | grep -v '.cache' | head -20", home)),
        ("configs",  "ls ~/.config/ 2>/dev/null | head -30".to_string()),
        ("rust",     format!("find {} -name 'Cargo.toml' -maxdepth 5 2>/dev/null | grep -v '.cache' | sed 's/\\/Cargo.toml//' | head -10", home)),
        ("python",   format!("find {} -name 'pyproject.toml' -maxdepth 5 2>/dev/null | grep -v '.cache' | head -10", home)),
    ];

    let mut pm = PermanentMemory::load()?;
    let mut summary = Vec::new();

    for (key, cmd) in &commands {
        let result = crate::tools::shell::run_command(cmd, sudo_pass).await?;
        let items = result.stdout.trim();
        if !items.is_empty() {
            let fact = format!(
                "System index - {}: {}",
                key,
                items.lines().collect::<Vec<_>>().join(", ")
            );
            pm.remember(&fact, "system").ok();
            summary.push(format!("{}: {} items", key, items.lines().count()));
        }
    }

    Ok(summary)
}

/// Monthly wrapper used by the daemon — no-op until the interval elapses.
pub async fn index_if_due(days: u32, sudo_pass: Option<&str>) -> Result<Option<String>> {
    if days == 0 {
        return Ok(None);
    }
    let marker = index_marker_path();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(ts) = std::fs::read_to_string(&marker) {
        let last: u64 = ts.trim().parse().unwrap_or(0);
        if now.saturating_sub(last) < days as u64 * 86400 {
            return Ok(None);
        }
    }
    let summary = run_index_system(sudo_pass).await?;
    std::fs::write(&marker, now.to_string())?;
    if summary.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!("Re-indexed system: {}", summary.join(", "))))
}
