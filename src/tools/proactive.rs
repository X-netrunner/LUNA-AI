//! tools/proactive.rs — Background proactive monitoring
//!
//! Periodically checks battery, disk space, and pending pacman updates.
//! Fires desktop notifications directly — no LLM call involved, so this
//! is fast, free, and never hallucinates a false alarm.
//!
//! Each warning only fires once until the condition clears, to avoid
//! spamming notifications every check cycle.

use crate::config::LunaConfig;
use std::time::Duration;

/// Spawn the background monitoring task. Call once at startup.
/// No-op if proactive.enabled is false in config.
pub fn spawn(config: &LunaConfig) {
    if !config.proactive.enabled {
        tracing::info!("Proactive monitoring disabled");
        return;
    }

    let interval_secs = config.proactive.check_interval_mins.max(1) * 60;
    let battery_threshold = config.proactive.battery_low_threshold;
    let disk_threshold = config.proactive.disk_full_threshold;
    let check_updates = config.proactive.check_updates;

    tokio::spawn(async move {
        let mut warned_battery = false;
        let mut warned_disk = false;
        let mut last_update_count: Option<u32> = None;

        loop {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            tracing::debug!("Running proactive check cycle");

            // ── Battery ──────────────────────────────────────────────────
            if let Ok(cap) = run("cat /sys/class/power_supply/BAT0/capacity 2>/dev/null").await {
                if let Ok(pct) = cap.trim().parse::<u32>() {
                    let status = run("cat /sys/class/power_supply/BAT0/status 2>/dev/null")
                        .await
                        .unwrap_or_default();
                    let discharging = status.trim().eq_ignore_ascii_case("discharging");

                    if pct <= battery_threshold && discharging {
                        if !warned_battery {
                            notify("Luna", &format!("Battery is at {}% — plug in soon.", pct)).await;
                            warned_battery = true;
                        }
                    } else if pct > battery_threshold + 10 {
                        warned_battery = false;
                    }
                }
            }

            // ── Disk ─────────────────────────────────────────────────────
            if let Ok(out) = run("df -h / | awk 'NR==2 {print $5}' | tr -d '%'").await {
                if let Ok(pct) = out.trim().parse::<u32>() {
                    if pct >= disk_threshold {
                        if !warned_disk {
                            notify("Luna", &format!("Disk is {}% full on /.", pct)).await;
                            warned_disk = true;
                        }
                    } else {
                        warned_disk = false;
                    }
                }
            }

            // ── Pending pacman updates ───────────────────────────────────
            // Requires pacman-contrib (checkupdates). Silently no-ops if absent.
            if check_updates {
                if let Ok(out) = run("checkupdates 2>/dev/null | wc -l").await {
                    if let Ok(n) = out.trim().parse::<u32>() {
                        if n > 0 && last_update_count != Some(n) {
                            notify("Luna", &format!("{} package update(s) available.", n)).await;
                            last_update_count = Some(n);
                        } else if n == 0 {
                            last_update_count = None;
                        }
                    }
                }
            }
        }
    });

    tracing::info!(
        "Proactive monitoring started (every {} min)",
        config.proactive.check_interval_mins
    );
}

async fn run(cmd: &str) -> anyhow::Result<String> {
    let output = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

async fn notify(title: &str, body: &str) {
    let safe_title = title.replace('\'', "'\\''");
    let safe_body = body.replace('\'', "'\\''");
    let cmd = format!("notify-send '{}' '{}'", safe_title, safe_body);
    let _ = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(&cmd)
        .output()
        .await;
}
