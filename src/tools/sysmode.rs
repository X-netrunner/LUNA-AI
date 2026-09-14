//! tools/sysmode.rs — Luna's awareness of the `sysmode` hardening switcher.
//!
//! `sysmode` is a root CLI (typically `/usr/local/bin/sysmode`, or on PATH)
//! that toggles system hardening profiles: `secure` / `cyber` /
//! `stealth` (aka `hacking`) / `lockdown`, plus `status` / `logs` /
//! `reapply` subcommands.  Only `status` and `logs` work without root;
//! switching re-runs the profile via sudo.  This module lets Luna self-test
//! every component the tool manages — recon-deceiver, IDS, cowrie, tcpdump,
//! decoy ports, auditd — and relay results back in plain language.
//!
//! Config: [sysmode] enabled (bool) + bin (default `sysmode`).

use anyhow::{bail, Result};
use crate::config::SysmodeConfig;

const ALLOWED_MODES: &[&str] = &["secure", "cyber", "stealth", "lockdown"];
const DECOY_PORTS: &[u16] = &[21, 22, 23, 80, 445, 3306, 3389, 8080];

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bin_name(cfg: &SysmodeConfig) -> String {
    let b = cfg.bin.trim();
    if b.is_empty() { "sysmode".into() } else { b.to_string() }
}

async fn run(cmd: &str, sudo_pass: Option<&str>) -> Result<String> {
    let out = crate::tools::shell::run_command(cmd, sudo_pass).await?;
    if out.exit_code != 0 {
        bail!(
            "command failed (exit {})\nstdout: {}\nstderr: {}",
            out.exit_code,
            out.stdout.trim(),
            out.stderr.trim()
        );
    }
    Ok(out.stdout.trim().to_string())
}

/// Quick boolean probe — returns true when the command exits 0.
async fn probe(cmd: &str) -> bool {
    matches!(crate::tools::shell::run_command(cmd, None).await, Ok(o) if o.exit_code == 0)
}

// ── Public actions ───────────────────────────────────────────────────────────

pub async fn status(cfg: &SysmodeConfig) -> Result<String> {
    run(&format!("{} status", bin_name(cfg)), None).await
}

pub async fn logs(cfg: &SysmodeConfig) -> Result<String> {
    run(&format!("{} logs", bin_name(cfg)), None).await
}

pub async fn reapply(cfg: &SysmodeConfig, sudo_pass: Option<&str>) -> Result<String> {
    if sudo_pass.is_none() {
        bail!(
            "No sudo password is configured — add [agent] sudo_password to luna.toml so \
             Luna can re-run the sysmode profile as root."
        );
    }
    run(&format!("sudo {} reapply", bin_name(cfg)), sudo_pass).await
}

pub async fn switch_mode(
    cfg: &SysmodeConfig,
    mode: &str,
    hotspot: Option<bool>,
    sudo_pass: Option<&str>,
) -> Result<String> {
    let m = mode.trim().to_lowercase();
    if !ALLOWED_MODES.contains(&m.as_str()) {
        bail!(
            "Unknown sysmode profile '{mode}' — choose one of: secure, cyber, stealth, lockdown"
        );
    }
    if sudo_pass.is_none() {
        bail!(
            "No sudo password is configured — add [agent] sudo_password to luna.toml so \
             Luna can switch the system hardening profile."
        );
    }
    let mut cmd = format!("sudo {} {}", bin_name(cfg), m);
    if m == "stealth" {
        match hotspot {
            Some(true) => cmd.push_str(" --hotspot"),
            Some(false) => cmd.push_str(" --no-hotspot"),
            None => {}
        }
    }
    run(&cmd, sudo_pass).await
}

// ── Health check — the self-aware part ───────────────────────────────────────

pub async fn health_check(cfg: &SysmodeConfig) -> Result<String> {
    let bin = bin_name(cfg);
    let mut out: Vec<String> = Vec::new();

    // 1. Full status output (no root needed)
    match run(&format!("{bin} status"), None).await {
        Ok(s) => {
            out.push("=== sysmode status ===".into());
            out.extend(s.lines().map(|l| format!("  {l}")).collect::<Vec<_>>());
            out.push(String::new());
        }
        Err(_) => {
            return Ok(format!(
                "'{bin}' is not installed or returned an error. If sysmode is installed \
                 elsewhere, set [sysmode] bin in luna.toml. If it isn't installed on this \
                 machine, Luna's sysmode feature has nothing to probe — disable the tool \
                 or install the script (it lives at /usr/local/bin/sysmode on the original \
                 Arch install)."
            ));
        }
    }

    // 2. Component self-test
    out.push("=== Component health check ===".into());

    // Profile from the mode file
    let mode = std::fs::read_to_string("/etc/sysmode.mode")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    out.push(format!("  Active profile (/etc/sysmode.mode): {mode}"));

    let run_mode_relevant = |probe_ok: bool, component: &str| -> String {
        let state = if probe_ok { "running" } else { "stopped" };
        let hint = if !probe_ok {
            match mode.as_str() {
                "stealth" | "hacking" => "",
                _ => " (expected only in stealth/hacking mode)",
            }
        } else {
            ""
        };
        format!("  {component}: {state}{hint}")
    };

    // recon-deceiver honeypot (service or background process)
    let recon_ok = probe(
        "{ systemctl is-active recon-deceiver >/dev/null 2>&1 || \
           pgrep -f recon-deceiver.py >/dev/null 2>&1; }",
    )
    .await;
    out.push(run_mode_relevant(recon_ok, "recon-deceiver honeypot"));

    // IDS log analyst
    let ids_ok = probe(
        "{ systemctl is-active log-analyst >/dev/null 2>&1 || \
           pgrep -f log-analyst.py >/dev/null 2>&1; }",
    )
    .await;
    out.push(run_mode_relevant(ids_ok, "IDS log analyst"));

    // Cowrie SSH honeypot (Docker)
    let cowrie_ok = probe(
        "docker inspect -f '{{.State.Running}}' cowrie 2>/dev/null | grep -qx true 2>/dev/null",
    )
    .await;
    out.push(run_mode_relevant(
        cowrie_ok,
        "Cowrie SSH honeypot",
    ));

    // tcpdump MAC scanner
    let tcpdump_ok = probe(
        "test -f /var/run/sysmode-tcpdump.pid && kill -0 $(cat /var/run/sysmode-tcpdump.pid) 2>/dev/null",
    )
    .await;
    out.push(run_mode_relevant(tcpdump_ok, "tcpdump MAC scanner"));

    // Decoy WiFi interface
    let decoy_wifi = std::fs::read_to_string("/etc/sysmode.decoy-wifi")
        .ok()
        .map(|s| format!("active on {}", s.trim()))
        .unwrap_or_else(|| "inactive".into());
    out.push(format!("  Decoy WiFi hotspot: {decoy_wifi}"));

    // dnscrypt-proxy (encrypted DNS)
    let dns_ok = probe("systemctl is-active dnscrypt-proxy >/dev/null 2>&1").await;
    out.push(format!(
        "  dnscrypt-proxy: {}",
        if dns_ok { "active" } else { "inactive" }
    ));

    // auditd
    let audit_ok = probe("systemctl is-active auditd >/dev/null 2>&1").await;
    out.push(format!(
        "  auditd: {}",
        if audit_ok { "active" } else { "inactive" }
    ));

    // SSH access state
    let ssh_masked = probe("systemctl is-enabled sshd >/dev/null 2>&1 && ! systemctl is-active sshd >/dev/null 2>&1").await;
    let ssh_active = probe("systemctl is-active sshd >/dev/null 2>&1").await;
    let ssh_state = if ssh_active {
        "active"
    } else if ssh_masked {
        "masked (lockdown)"
    } else {
        "stopped"
    };
    out.push(format!("  SSH (sshd): {ssh_state}"));

    // 3. Decoy listening ports
    let ss_out = match run("ss -tlnH 2>/dev/null | awk '{print $4}' | sed -E 's/.*://;s/\\[.*\\]//' | sort -nu", None).await {
        Ok(s) => s,
        Err(_) => String::new(),
    };
    let listening: Vec<u16> = ss_out
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let decoy_listening: Vec<String> = DECOY_PORTS
        .iter()
        .filter(|&&p| listening.contains(&p))
        .map(|p| p.to_string())
        .collect();
    out.push(format!(
        "  Decoy listening ports: {}",
        if decoy_listening.is_empty() {
            "none".to_string()
        } else {
            decoy_listening.join(", ")
        }
    ));

    out.push(String::new());

    // 4. Log freshness — are the honeypot/IDS logs being written to?
    let logs_dir = std::fs::read_to_string("/etc/sysmode.conf")
        .ok()
        .and_then(|c| {
            c.lines()
                .find(|l| l.starts_with("LOGS_DIR="))
                .and_then(|l| l.split('=').nth(1))
                .map(str::trim)
                .map(str::to_string)
        })
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            format!("{}/logs", home)
        });
    let log_files = [
        "recon-attempts.log",
        "ids-alerts.log",
        "attacker-scans.log",
        "attacker-mac-scans.log",
    ];
    out.push("=== Log freshness ===".into());
    for name in &log_files {
        let path = format!("{logs_dir}/{name}");
        let freshness = tokio::fs::metadata(&path)
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .map(|d| {
                let mins = d.as_secs() / 60;
                if mins < 60 {
                    format!("updated {}m ago", mins)
                } else if mins < 1440 {
                    format!("updated {}h ago", mins / 60)
                } else {
                    format!("updated {}d ago", mins / 1440)
                }
            })
            .unwrap_or_else(|| "not found".to_string());
        out.push(format!("  {name}: {freshness}"));
    }

    // Cowrie log
    let cowrie_log = format!("{logs_dir}/cowrie/cowrie.json");
    let cowrie_fresh = tokio::fs::metadata(&cowrie_log)
        .await
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.elapsed().ok())
        .map(|d| format!("{}h ago", d.as_secs() / 3600))
        .unwrap_or_else(|| "not found".into());
    out.push(format!("  cowrie.json: {cowrie_fresh}"));

    out.push(String::new());
    out.push("Everything looks good — ask for 'sysmode logs' for the full attack/IDS report.".into());

    Ok(out.join("\n"))
}
