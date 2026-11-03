//! tools/pkgupdate.rs — guarded package updates.
//!
//! Two-phase by design, because Arch makes unattended upgrades dangerous:
//!   check  — `checkupdates` (uses a throwaway temp DB, changes nothing) to
//!            list what is pending, classify risk (major-version bumps are
//!            flagged as breaking), and print the exact gated command.
//!   apply  — only on explicit approval: a FULL sync upgrade (`yay -Syu` when
//!            an AUR helper is present, else `pacman -Syu`) with a sudo shim
//!            and `--noconfirm`, refusing to proceed on a breaking change
//!            unless the caller explicitly acknowledges it.
//!
//! Why never a bare `pacman -Syu`: that is a *partial* upgrade. Arch warns
//! against it — if a library soname bumps (e.g. libssl.so.3 -> .4) and AUR
//! packages aren't rebuilt, they segfault. A full sync (yay/pacman -Syu over
//! the whole tree, plus AUR rebuild via the helper) is the only safe form.

use anyhow::{Context, Result};

/// One pending package line from `checkupdates`:
/// `name oldver -> newver`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pending {
    pub name: String,
    pub old: String,
    pub new: String,
    pub breaking: bool,
}

/// Pull the leading numeric version component (e.g. "2.0.1-3" -> "2").
/// Returns None if there is no leading number.
fn major(ver: &str) -> Option<u64> {
    let digits: String = ver
        .trim_start_matches(|c: char| !c.is_ascii_digit())
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Classify a single `name old -> new` line from checkupdates.
/// A major-version change (1.x -> 2.x) is treated as breaking.
pub fn parse_line(line: &str) -> Option<Pending> {
    let line = line.trim();
    if line.is_empty() || line.starts_with("::") {
        return None;
    }
    let arrow = line.find("->")?;
    let (lhs, rhs) = line.split_at(arrow);
    let new = rhs["->".len()..].trim().to_string();
    let mut parts = lhs.split_whitespace();
    let name = parts.next()?.to_string();
    let old = parts.collect::<Vec<_>>().join(" ");
    let breaking = match (major(&old), major(&new)) {
        (Some(a), Some(b)) => a != b,
        // If we can't read both versions, don't claim it's safe — err on the
        // side of flagging it so a human looks.
        _ => old != new,
    };
    Some(Pending { name, old, new, breaking })
}

/// Parse a whole `checkupdates` output block.
pub fn parse_checkupdates(out: &str) -> Vec<Pending> {
    out.lines().filter_map(parse_line).collect()
}

/// Is a full-sync AUR helper available (yay / paru)?
async fn has_aur_helper() -> bool {
    for helper in ["yay", "paru"] {
        if which(helper).is_some() {
            return true;
        }
    }
    false
}

/// Minimal PATH lookup (avoid pulling in a dep just for `which`).
fn which(bin: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var("PATH").ok()?;
    for dir in path.split(':') {
        let cand = std::path::Path::new(dir).join(bin);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// sudo shim so every `sudo` inside the command is non-interactive.
fn sudo_shim(sudo_pass: Option<&str>) -> String {
    match sudo_pass {
        Some(p) => {
            let safe = p.replace('\'', "'\\''");
            format!("sudo() {{ echo '{}' | command sudo -S -p '' \"$@\"; }};", safe)
        }
        None => String::new(),
    }
}

/// Phase 1 — inspect pending updates. Changes nothing on the system.
pub async fn check() -> Result<String> {
    let out = crate::tools::shell::run_command("checkupdates 2>/dev/null", None)
        .await
        .context("failed to run checkupdates (is pacman-contrib installed?)")?;

    let pending = parse_checkupdates(&out.stdout);
    if pending.is_empty() {
        return Ok("System is fully up to date — no pending package updates.".into());
    }

    let breaking: Vec<&Pending> = pending.iter().filter(|p| p.breaking).collect();
    let routine = pending.len() - breaking.len();

    let mut msg = String::new();
    msg.push_str(&format!(
        "{} package update(s) pending ({} routine, {} breaking).\n\n",
        pending.len(),
        routine,
        breaking.len()
    ));

    if !breaking.is_empty() {
        msg.push_str("BREAKING (major version bump — may need config migration):\n");
        for p in &breaking {
            msg.push_str(&format!("  {} {} -> {}\n", p.name, p.old, p.new));
        }
        msg.push('\n');
    }

    msg.push_str("Routine:\n");
    for p in pending.iter().filter(|p| !p.breaking) {
        msg.push_str(&format!("  {} {} -> {}\n", p.name, p.old, p.new));
    }

    msg.push_str(&format!(
        "\nTo apply (full sync, AUR rebuilt), run:\n  system_update {{ action: \"apply\"{}}}",
        if breaking.is_empty() {
            String::new()
        } else {
            ", confirm_breaking: true".to_string()
        }
    ));
    if !breaking.is_empty() {
        msg.push_str(
            "\nNote: breaking changes are present. Read the Arch news first, and only \
             confirm if you accept the risk.",
        );
    }
    Ok(msg)
}

/// Phase 2 — apply the gated full-sync upgrade.
///
/// The approval gate is a HUMAN one. There is no in-band approval channel in
/// the ReAct loop, so "the user said yes" cannot be a model-supplied argument —
/// demonstrated the hard way when Luna checked for updates, saw breaking ones,
/// and then called apply with `confirm_breaking: true` on her own initiative.
/// A model asserting "the user approved" is not an approval.
///
/// So: no confirmation argument exists, and installing requires
/// `[updates] allow_apply = true` in luna.toml, which only a human editing
/// the file can set.
pub async fn apply(sudo_pass: Option<&str>, allow_apply: bool) -> Result<String> {
    // Re-check right before applying so a breaking change introduced between
    // the check and the apply is caught here, not silently upgraded.
    let pre = crate::tools::shell::run_command("checkupdates 2>/dev/null", None)
        .await
        .context("failed to run checkupdates before apply")?;
    let pending = parse_checkupdates(&pre.stdout);
    if pending.is_empty() {
        return Ok("Nothing to update — system is already up to date.".into());
    }
    let breaking: Vec<&Pending> = pending.iter().filter(|p| p.breaking).collect();

    // The human gate, checked before anything else runs.
    if !allow_apply {
        return Err(anyhow::anyhow!(
            "Refusing to install: the human apply gate is closed. \
             {} update(s) pending ({} breaking).\n\
             Tell the user: to actually upgrade, they must set \
             [updates] allow_apply = true in ~/.config/luna/luna.toml, restart the \
             daemon, and then ask you again. Do NOT try to work around this — \
             there is no argument you can pass that opens this gate.",
            pending.len(),
            breaking.len()
        ));
    }
    if !breaking.is_empty() {
        let names: Vec<&str> = breaking.iter().map(|p| p.name.as_str()).collect();
        tracing::warn!(
            "pkgupdate: applying WITH breaking updates present ({}), human gate open: {}",
            breaking.len(),
            names.join(", ")
        );
    }

    let shim = sudo_shim(sudo_pass);
    let helper = if has_aur_helper().await {
        if which("yay").is_some() { "yay" } else { "paru" }
    } else {
        "pacman"
    };
    let cmd = format!("{} {} -Syu --noconfirm", shim, helper);
    tracing::info!("pkgupdate: applying full sync via {}", helper);

    let out = crate::tools::shell::run_command(&cmd, sudo_pass)
        .await
        .context("failed to run the system upgrade")?;

    let tail: String = out
        .stdout
        .lines()
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");

    if out.exit_code != 0 {
        return Err(anyhow::anyhow!(
            "{} exited {}. stderr: {}",
            helper,
            out.exit_code,
            out.stderr.trim()
        ));
    }
    Ok(format!(
        "Full system upgrade complete via {} ({} packages).\nLast output:\n{}",
        helper,
        pending.len(),
        tail
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_routine_update() {
        let p = parse_line("firefox 128.0-1 -> 128.1-1").unwrap();
        assert_eq!(p.name, "firefox");
        assert_eq!(p.old, "128.0-1");
        assert_eq!(p.new, "128.1-1");
        assert!(!p.breaking, "same major version is routine");
    }

    #[test]
    fn flags_major_bump_as_breaking() {
        let p = parse_line("python 3.12.1-1 -> 3.13.0-1").unwrap();
        assert!(!p.breaking, "same major (3) is routine");
        let p = parse_line("nginx 1.24.0-1 -> 2.0.0-1").unwrap();
        assert!(p.breaking, "1.x -> 2.x is breaking");
    }

    #[test]
    fn parses_whole_block_and_counts() {
        let out = "linux 6.9.1-1 -> 6.9.2-1\nnginx 1.24.0-1 -> 2.0.0-1\nvim 9.0-1 -> 9.1-1\n";
        let p = parse_checkupdates(out);
        assert_eq!(p.len(), 3);
        assert_eq!(p.iter().filter(|x| x.breaking).count(), 1);
    }

    #[test]
    fn ignores_noise_lines() {
        assert!(parse_line("").is_none());
        assert!(parse_line("==> Checking for updates... [done]").is_none());
        assert!(parse_line("no arrow here").is_none());
    }

    #[test]
    fn unparseable_version_is_flagged_not_assumed_safe() {
        // no leading digits on either side and versions differ -> breaking
        let p = parse_line("weirdpkg a -> b").unwrap();
        assert!(p.breaking);
    }

    // ── The human apply gate ──────────────────────────────────────────────
    // Regression lock for a real incident: asked only to CHECK for updates,
    // Luna checked, saw breaking updates pending, and four seconds later called
    // apply with confirm_breaking=true on her own initiative. Nothing stopped
    // her, because the "approval" was a model-settable argument.
    #[test]
    fn the_apply_gate_is_closed_by_default() {
        // A human must opt in. Default config = no installs, ever.
        let cfg = crate::config::UpdatesConfig::default();
        assert!(cfg.enabled, "the tool should be offered");
        assert!(
            !cfg.allow_apply,
            "the human apply gate must default to CLOSED"
        );
    }

    #[test]
    fn the_tool_schema_offers_no_way_to_confirm_or_apply_approval() {
        // If a model can find a `confirm_breaking` (or any approval) parameter
        // in the schema, it can set it and we are back to self-authorization.
        let defs = crate::tools::tool_definitions();
        let su = defs
            .iter()
            .find(|d| d.function.name == "system_update")
            .expect("system_update must exist");
        let schema = serde_json::to_string(&su.function.parameters).unwrap();
        assert!(
            !schema.contains("confirm_breaking"),
            "the self-authorization parameter must not be offered to the model: {schema}"
        );
        // And no other approval-ish knob sneaks in.
        for banned in ["approve", "force", "yes", "accept_risk"] {
            assert!(
                !schema.contains(banned),
                "'{banned}' must not be an available parameter: {schema}"
            );
        }
        // action=apply must still exist — it just can't bypass the gate.
        assert!(schema.contains("apply"));
    }

    #[tokio::test]
    #[ignore = "runs checkupdates; run explicitly with -- --ignored"]
    async fn apply_refuses_while_the_human_gate_is_closed() {
        // Gate closed -> refuse, and say what the *human* must do.
        let err = apply(None, false)
            .await
            .expect_err("apply must refuse while the gate is closed");
        let msg = err.to_string();
        assert!(msg.contains("human apply gate is closed"), "got: {msg}");
        assert!(msg.contains("allow_apply"), "must name the switch: {msg}");
        // It must not have run anything.
        assert!(!msg.contains("complete"), "must not claim success: {msg}");
    }
}
