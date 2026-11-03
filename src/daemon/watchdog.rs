//! daemon/watchdog.rs — /proc process scanner
//!
//! Reads process names, RSS and CPU jiffies straight from /proc so no
//! extra dependencies are needed. CPU% is computed as jiffies delta
//! between consecutive scan cycles.

use crate::config::LunaConfig;
use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
pub struct ProcInfo {
    pub pid: u32,
    pub name: String,
    pub rss_mb: u64,
    /// CPU% since the previous scan cycle (None on the first cycle)
    pub cpu_percent: Option<f32>,
    /// Lifetime utime+stime jiffies at this snapshot (for usage learning)
    pub total_jiffies: u64,
}

fn page_size() -> u64 {
    std::process::Command::new("getconf")
        .arg("PAGESIZE")
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(4096)
}

pub async fn scan_processes(
    config: &LunaConfig,
    prev_jiffies: &mut HashMap<u32, (u64, Instant)>,
) -> Result<Vec<ProcInfo>> {
    const CLK_TCK: f64 = 100.0; // kernel CONFIG_HZ — universal on x86_64
    let pgsize = page_size();

    let mut dir = tokio::fs::read_dir("/proc").await?;
    let mut pids: Vec<String> = Vec::new();
    while let Some(entry) = dir.next_entry().await? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.chars().all(|c| c.is_ascii_digit()) {
            pids.push(name.to_string());
        }
    }

    let now = Instant::now();
    let mut out: Vec<ProcInfo> = Vec::new();

    for pid_str in pids {
        let pid: u32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
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

        let total_jiffies: u64 =
            fields[11].parse::<u64>().unwrap_or(0) + fields[12].parse::<u64>().unwrap_or(0);

        let cpu_percent = match prev_jiffies.remove(&pid) {
            Some((prev, at)) => {
                let elapsed = now.duration_since(at).as_secs_f64();
                if elapsed > 0.0 && total_jiffies >= prev {
                    Some(
                        ((total_jiffies - prev) as f64 / (elapsed * CLK_TCK) * 100.0)
                            as f32,
                    )
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
            total_jiffies,
        });
    }

    // Biggest consumers first so the notification reads naturally
    out.sort_by_key(|p| std::cmp::Reverse(p.rss_mb));
    Ok(out)
}
