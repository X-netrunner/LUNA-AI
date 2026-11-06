//! tools/safety.rs — Run and report on the system safety check
//!
//! Delegates to the installed `safety_check.sh` script (light / full /
//! backup modes). The script handles mounting, incremental backup and
//! its own locking. Luna just runs it and reports on the log.

use anyhow::{Context, Result};

fn script_candidates() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Some(home) = dirs::home_dir() {
        v.push(home.join(".local/bin/safety_check"));
    }
    v.push(std::path::PathBuf::from("/usr/local/bin/safety_check"));
    if let Ok(cwd) = std::env::current_dir() {
        v.push(cwd.join("scripts/safety_check.sh"));
    }
    v
}

/// Absolute path to the safety_check script, or None if not installed.
pub fn find_script() -> Option<std::path::PathBuf> {
    script_candidates().into_iter().find(|p| p.exists())
}

fn log_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join("logs/safety_check")
}

/// Every place a safety-check log can be: `~/logs` is a symlink to /var/log/halcyon on a Halcyon install (the check runs as root),
/// and without root the check falls back to the user's own state folder.
fn log_dirs() -> Vec<std::path::PathBuf> {
    let home = dirs::home_dir().unwrap_or_default();
    vec![
        log_dir(),
        std::path::PathBuf::from("/var/log/halcyon/safety_check"),
        home.join(".local/state/halcyon/safety_check"),
    ]
}

fn newest_log() -> Option<std::path::PathBuf> {
    let mut logs: Vec<std::path::PathBuf> = Vec::new();
    for dir in log_dirs() {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            logs.extend(
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("log")),
            );
        }
    }
    logs.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    logs.pop()
}

/// The findings: the newest log's LAST run (a day's log holds every run of that day), so "summarise the safety check logs" is answered
/// from the real text instead of the model hunting for a file. Long runs keep their end (the verdicts come last).
pub fn report() -> String {
    let Some(log) = newest_log() else {
        return "No safety check log found in ~/logs/safety_check/ (also looked in /var/log/halcyon/safety_check and \
                ~/.local/state/halcyon/safety_check). Run the safety check first."
            .into();
    };
    let raw = match std::fs::read_to_string(&log) {
        Ok(r) => r,
        Err(e) => return format!("Could not read {}: {}", log.display(), e),
    };
    let lines: Vec<&str> = raw.lines().collect();
    let start = lines.iter().rposition(|l| l.contains("Safety Check started")).unwrap_or(0);
    let run = lines[start..].join("\n");
    const MAX: usize = 6000;
    let total = run.chars().count();
    let body = if total > MAX {
        let tail: String = run.chars().skip(total - MAX).collect();
        format!("[the start of this run is cut; showing the last {} characters]\n{}", MAX, tail)
    } else {
        run
    };
    format!("Safety check log: {}\n{}", log.display(), body)
}

/// Run the script. Returns a digest of the freshest log so the reply is
/// compact even though the scanner's raw output can be huge.
pub async fn run(mode: &str, sudo_pass: Option<&str>, fallback: bool) -> Result<String> {
    let Some(path) = find_script() else {
        anyhow::bail!(
            "safety_check is not installed (looked in ~/.local/bin and \
             /usr/local/bin). Requested it, but it's missing."
        );
    };

    let mode = match mode { "full" => "full", "backup" => "backup", _ => "light" };
    let fallback_arg = if fallback { " --fallback" } else { "" };

    // Source the script into the same bash that owns the sudo shim — that way
    // every `sudo` inside the script can use the stored password non-interactively.
    // Never log the password.
    let shim = match sudo_pass {
        Some(pass) => {
            let safe = pass.replace('\'', "'\\''");
            format!("sudo() {{ echo '{}' | command sudo -S -p '' \"$@\"; }};", safe)
        }
        None => String::new(),
    };

    let cmd = format!(
        "{} source {} {}{}",
        shim,
        path.display(),
        mode,
        fallback_arg
    );

    let output = crate::tools::shell::run_command(&cmd, sudo_pass).await
        .context("Failed to run safety check")?;
    if output.exit_code != 0 {
        tracing::warn!("safety_check exited {}{}", output.exit_code, output.stderr);
    }

    Ok(summarize_latest().unwrap_or_else(|| {
        "Safety check ran but no log was written yet — check ~/logs/safety_check/.".into()
    }))
}

/// Does a check look like it's running right now?
pub fn is_running() -> bool {
    let lock = log_dir().join(".safety_check.lock");
    if !lock.exists() {
        return false;
    }
    let recent = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .saturating_sub(
            std::fs::metadata(&lock)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );
    recent < 3600
}

fn summarize_latest() -> Option<String> {
    let log = newest_log()?;
    let raw = std::fs::read_to_string(&log).ok()?;
    let mut lines: Vec<&str> = raw.lines().rev().take(12).collect();
    lines.reverse();

    let issues = raw.lines().filter(|l| l.contains("⚠ WARNING") || l.contains("issue(s) found")).count();
    let mut body = String::new();
    for l in lines {
        body.push_str(l);
        body.push('\n');
    }

    let status = if issues == 0 { "all good" } else { "attention needed" };
    Some(format!(
        "Safety check → {}\nLatest log: {}\n{}{}",
        status,
        log.display(),
        body,
        if issues > 0 {
            format!("\n{} issue(s) flagged — see log for details.", issues)
        } else {
            String::new()
        }
    ))
}

pub fn status() -> String {
    if is_running() {
        return "A safety check is running right now — give it time, then ask again.".into();
    }
    summarize_latest().unwrap_or_else(|| {
        "No safety check has run yet — logs will appear in ~/logs/safety_check/.".into()
    })
}