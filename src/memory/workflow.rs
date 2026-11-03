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
    marker_dir().join("last_workflow_learn")
}

/// Shared directory for Luna's timestamped job markers
/// (~/.local/share/luna). Used by the daemon's marker-gated jobs too.
pub fn marker_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("luna")
}

fn history_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/fish/fish_history")
}

/// Every command line stored in the fish history, cleaned of the YAML
/// prefix, comments and pathological blobs.
fn history_commands() -> Vec<String> {
    let content = match std::fs::read_to_string(history_path()) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    content
        .lines()
        .filter_map(|l| l.strip_prefix("- cmd:").map(str::trim))
        .filter(|c| !c.is_empty() && !c.starts_with('#') && c.len() <= 100)
        .map(str::to_string)
        .collect()
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
    marker_dir().join("last_index_learn")
}

/// Deep system learning pass — home layout + shell history + system map.
///
/// Writes stable-prefix facts into permanent memory so the near-duplicate
/// guard updates them in place rather than stacking copies every month:
///   System index - <projects/scripts/configs/rust/python/...>
///   Shell workflow: user's most-used commands are ...
///   Top directories: ...
///   Most-used files: ...
///
/// Used by the index_system tool ("learn about my system") and the daemon.
pub async fn run_index_system(sudo_pass: Option<&str>) -> Result<Vec<String>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let commands = vec![
        ("projects", format!("find {} -name '.git' -maxdepth 4 -type d 2>/dev/null | grep -v '.cache' | sed 's/\\/.git//' | head -20", home)),
        ("scripts",  format!("find {} -name '*.sh' -maxdepth 4 2>/dev/null | grep -v '.cache' | head -20", home)),
        ("configs",  "ls ~/.config/ 2>/dev/null | head -30".to_string()),
        ("rust",     format!("find {} -name 'Cargo.toml' -maxdepth 5 2>/dev/null | grep -v '.cache' | sed 's/\\/Cargo.toml//' | head -10", home)),
        ("python",   format!("find {} -name 'pyproject.toml' -maxdepth 5 2>/dev/null | grep -v '.cache' | head -10", home)),
        (
            "explicit_packages",
            "pacman -Qqe 2>/dev/null | head -16 | tr '\\n' ',' | sed 's/,$//'".to_string(),
        ),
        (
            "user_services",
            "systemctl --user --type=service --state=running --no-legend --no-pager 2>/dev/null \
             | awk '{print $1}' | head -12 | tr '\\n' ',' | sed 's/,$//'".to_string(),
        ),
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

    // ── Shell history deep-dive ─────────────────────────────────────────
    let hist = history_commands();
    if !hist.is_empty() {
        let (top_cmds, top_dirs, top_files, n) = analyze_history(&hist);

        if !top_cmds.is_empty() {
            let fact = format!(
                "Shell workflow: user's most-used commands are {}",
                top_cmds.join(", ")
            );
            pm.remember(&fact, "workflow").ok();
        }
        if !top_dirs.is_empty() {
            let fact = format!("Top directories: {}", top_dirs.join(", "));
            pm.remember(&fact, "workflow").ok();
            summary.push(format!("dirs: {} learned", top_dirs.len()));
        }
        if !top_files.is_empty() {
            let fact = format!("Most-used files: {}", top_files.join(", "));
            pm.remember(&fact, "workflow").ok();
            summary.push(format!("files: {} learned", top_files.len()));
        }
        summary.push(format!("history: {} commands analyzed", n));
    }

    Ok(summary)
}

/// Editors/openers whose file arguments hint at "files Luna works with a lot".
const EDITORS: &[&str] = &[
    "nvim", "vim", "vi", "code", "codium", "zed", "zeditor", "nano", "subl",
    "kate", "kwrite", "gedit", "helix", "micro",
];

/// Distill most-used commands, most-visited cd dirs, and most-referenced
/// file paths out of the raw fish history. Pure text parsing — fast.
fn analyze_history(hist: &[String]) -> (Vec<String>, Vec<String>, Vec<String>, usize) {
    use std::collections::HashMap;

    let mut cmds: HashMap<&str, u32> = HashMap::new();
    let mut dirs: HashMap<String, u32> = HashMap::new();
    let mut files: HashMap<String, u32> = HashMap::new();

    for c in hist {
        let tokens: Vec<&str> = c.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        // "sudo pacman ..." is really a pacman command
        let key = if tokens[0] == "sudo" && tokens.len() > 1 {
            tokens[1]
        } else {
            tokens[0]
        };
        match key {
            "cd" | "ls" | "clear" | "pwd" | "cat" | "echo" | "grep" | "exit" => {}
            _ => *cmds.entry(key).or_insert(0) += 1,
        }

        if tokens[0] == "cd" && tokens.len() > 1 {
            let d = normalize_path(tokens[1]);
            if !d.is_empty() {
                *dirs.entry(d).or_insert(0) += 1;
            }
        }

        if EDITORS.contains(&tokens[0]) {
            for t in &tokens[1..] {
                if t.starts_with('-') {
                    continue;
                }
                let p = normalize_path(t);
                if p.contains('/') {
                    *files.entry(p).or_insert(0) += 1;
                }
            }
        }
    }

    let rank = |m: &HashMap<String, u32>| -> Vec<String> {
        let mut v: Vec<(&String, &u32)> = m.iter().collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        v.iter()
            .take(10)
            .map(|(p, c)| format!("{}({})", p, c))
            .collect()
    };

    let mut cmds_ranked: Vec<(&str, u32)> = cmds.into_iter().collect();
    cmds_ranked.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    let top_cmds = cmds_ranked
        .iter()
        .take(10)
        .map(|(cmd, c)| format!("{}({})", cmd, c))
        .collect();

    let top_dirs = rank(&dirs);
    let top_files = rank(&files);
    (top_cmds, top_dirs, top_files, hist.len())
}

/// Trim quotes, expand '~' to the real home path, drop trailing slash.
fn normalize_path(raw: &str) -> String {
    let mut t = raw.trim().trim_matches('"').trim_matches('\'').to_string();
    if t.starts_with("~/") || t == "~" {
        if let Some(home) = dirs::home_dir() {
            let home_s = home.to_string_lossy().to_string();
            if t == "~" {
                t = home_s;
            } else {
                t = format!("{}/{}", home_s, &t[2..]);
            }
        }
    }
    let t = t.strip_suffix('/').unwrap_or(&t).to_string();
    if t.is_empty() || t == "/" {
        String::new()
    } else {
        t
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(cmds: &[&str]) -> Vec<String> {
        cmds.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn history_analysis_ranks_commands_dirs_and_files() {
        let h = hist(&[
            "nvim ~/Projects/luna-stable/src/main.rs",
            "nvim ~/Projects/luna-stable/src/main.rs",
            "cd ~/Projects/luna-stable",
            "cd ~/Projects/luna-stable",
            "cd ~/Projects/luna-stable",
            "code ~/.config/hypr/hyprland.conf",
            "git status",
            "git status",
            "sudo pacman -S htop",
            "ls",
            "cd ..",
            "exit",
        ]);
        let (cmds, dirs, files, n) = analyze_history(&h);

        assert_eq!(n, 12);
        assert!(cmds.join(" ").contains("nvim(2)"));
        assert!(cmds.join(" ").contains("git(2)"));
        assert!(!cmds.join(" ").contains("cd(")); // cd filtered from cmd rank
        assert!(!cmds.join(" ").contains("ls("));

        // "~" expands to the real home, so exact match depends on home dir
        let dirs_joined = dirs.join(" ");
        assert!(dirs_joined.contains("Projects/luna-stable(3)"));

        let files_joined = files.join(" ");
        assert!(files_joined.contains("src/main.rs(2)"));
        assert!(files_joined.contains("hyprland.conf(1)"));
    }

    #[test]
    fn history_analysis_ignores_flags_and_short_tokens() {
        let h = hist(&["vim -p a.txt", "zed --wait ./x.rs"]);
        let (_cmds, _dirs, files, _n) = analyze_history(&h);
        let joined = files.join(" ");
        assert!(joined.contains("x.rs(1)"));
        assert!(!joined.contains("-p")); // flags never count as files
    }

    #[test]
    fn history_analysis_normalizes_tilde_and_trailing_slash() {
        let h = hist(&["nvim ~/docs/"]);
        let (_c, _d, files, _n) = analyze_history(&h);
        assert!(files[0].starts_with('/')); // "~/" resolved to an absolute path
        assert!(!files[0].ends_with('/'));  // trailing slash stripped
        assert!(files[0].contains("/docs("));
    }
}
