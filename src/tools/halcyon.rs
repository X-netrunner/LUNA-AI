//! tools/halcyon.rs — Luna's hands and eyes in Halcyon, the desktop she lives in.
//!
//! Three tools, one per risk tier, all backed by Halcyon's `scripts/halcyon-control.py` so that what Luna can do in Halcyon
//! is updated with Halcyon and not recompiled into her:
//!
//!   `halcyon_info`      READ_ONLY_TOOLS. Status, docs, settings, notes, calendar, firewall / antivirus reads.
//!   `halcyon_do`        gated by `[external] allow_halcyon_actions` — a human switch (Halcyon's Luna window has it), separate from
//!                       `allow_external_actions` so that letting her tidy the desktop does not also let her text your contacts.
//!   `halcyon_security`  capability-gated like `sysmode`: firewall, antivirus and ClamAV state, with admin rights.
//!
//! The gates live at the dispatch chokepoint (tools/mod.rs), not here. The script also refuses an action asked for under the
//! wrong tier, so a mistake in this wiring cannot widen anything. Arguments go to the script as separate argv entries, never
//! through a shell. The admin password (when she has one) goes in the environment, never on a command line.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::Value;

const TIMEOUT: Duration = Duration::from_secs(900);

/// Halcyon's control script, if Halcyon is installed here.
pub fn script_path() -> Option<PathBuf> {
    let rice = std::env::var_os("HALCYON_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".config").join("Halcyon")))?;
    let p = rice.join("scripts").join("halcyon-control.py");
    p.is_file().then_some(p)
}

/// Words for the script: an array of strings (numbers allowed) or one string, split like a shell would
/// (spaces separate, quotes group) but without ever running a shell.
pub fn split_args(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a
            .iter()
            .filter_map(|x| match x {
                Value::String(s) => Some(s.clone()),
                Value::Number(n) => Some(n.to_string()),
                Value::Bool(b) => Some(b.to_string()),
                _ => None,
            })
            .collect(),
        Value::String(s) => split_words(s),
        _ => Vec::new(),
    }
}

fn split_words(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut quote, mut any) = (Vec::new(), String::new(), None::<char>, false);
    for c in s.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '"') | (None, '\'') => {
                quote = Some(c);
                any = true;
            }
            (None, c) if c.is_whitespace() => {
                if any || !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            (None, c) => cur.push(c),
        }
    }
    if any || !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Run one action of a tier. `tier` is `info`, `do` or `security` and is fixed by the caller (the tool's name), never by the model.
pub async fn call(tier: &'static str, args: &Value, sudo_password: Option<&str>) -> Result<String> {
    let script = script_path().context(
        "Halcyon is not installed here (no ~/.config/Halcyon/scripts/halcyon-control.py), so there is nothing to control.",
    )?;
    let action = args["action"].as_str().unwrap_or("").trim().to_string();
    if action.is_empty() {
        bail!("Pass action=<name>. Ask halcyon_info with action=overview first, or see the Halcyon manual in your instructions.");
    }
    if action.starts_with('-') {
        bail!("Not an action: {action}");
    }
    let rest = split_args(&args["args"]);
    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg(&script).arg(tier).arg(&action).args(&rest).kill_on_drop(true);
    cmd.stdin(std::process::Stdio::null());
    if tier == "security" {
        if let Some(pw) = sudo_password {
            cmd.env("LUNA_SUDO", pw);
        }
    }
    let out = tokio::time::timeout(TIMEOUT, cmd.output())
        .await
        .map_err(|_| anyhow::anyhow!("Halcyon did not answer within {} s", TIMEOUT.as_secs()))?
        .context("could not start python3")?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
    .trim()
    .to_string();
    match out.status.code() {
        Some(0) => Ok(if text.is_empty() { "done".into() } else { text }),
        Some(3) => bail!("Refused by Halcyon: {text}"),
        _ => bail!("{}", if text.is_empty() { "the action failed" } else { &text }),
    }
}

/// Who Luna is in relation to Halcyon. Always part of her instructions, installed or not: the manual below
/// only exists when Halcyon's control script is found, and without this she fell back to asking
/// "which desktop environment do you use?" about a desktop she lives in.
pub const IDENTITY: &str = "You are Halcyon's resident AI. Halcyon is Netrunner's own Hyprland (Wayland) desktop with a \
Quickshell (QML) shell: the island bar, desktop widgets, the rail dashboard, Settings, Notes, the lock screen and the wallpaper \
gallery, with fish, Starship and sysmode on top. You live in it, you are the first and best source of answers about it, and you \
are expected to know it. Rules: \
1. NEVER ask which desktop, DE, compositor, bar or distro the user runs. It is Halcyon on Hyprland, on Arch Linux. \
2. For any Halcyon, widget, bar, Hyprland, Quickshell, Settings, lock-screen or desktop problem, investigate before you answer: \
call halcyon_info (overview, then docs <keywords>), read the relevant files under the Halcyon directory with read_file, and check \
the evidence with run_shell (hyprctl, journalctl --user -b -p warning, systemctl --user --failed, pgrep -a quickshell). Do not \
guess from general knowledge when you can look. \
3. For Arch problems, gather evidence first (journalctl -b -p err, systemctl --failed, pacman -Qi/-Qkk, checkupdates, dmesg), \
then answer with the likely cause and the exact fix. Search the Arch Wiki with web_search when unsure. Never recommend a \
partial upgrade (pacman -Sy without -u). \
4. For a file map or project structure: never write one from memory. Run `luna-filemap <dir>` if it is installed, otherwise \
find <dir> \\( -name .git -o -name node_modules -o -name target -o -name __pycache__ \\) -prune -o -type f -print | sort. \
Cover every directory, not just the top level, read the first lines of each important file to say what it is for, and save the \
result with write_file (FILEMAP.md in that directory) unless asked to just show it. \
5. Say what you checked and what you found. If you cannot tell, say what you would look at next and look at it.";

/// The short manual about Halcyon that goes into her instructions (what it is, the three tools, the rules). Empty when Halcyon
/// is not installed. Read once per process: it only changes when Halcyon is updated.
pub fn awareness() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        let manual = script_path()
            .and_then(|script| {
                std::process::Command::new("python3")
                    .arg(script)
                    .arg("manual")
                    .stdin(std::process::Stdio::null())
                    .output()
                    .ok()
            })
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        if manual.is_empty() {
            format!("\n\n## Halcyon\n{IDENTITY}\n")
        } else {
            format!("\n\n## Halcyon\n{IDENTITY}\n\n{manual}\n")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn args_arrive_as_separate_words() {
        assert_eq!(split_args(&json!(["add", "buy milk", 3])), vec!["add", "buy milk", "3"]);
        assert_eq!(split_args(&json!("add 'buy milk' now")), vec!["add", "buy milk", "now"]);
        assert_eq!(split_args(&json!("a \"b  c\"")), vec!["a", "b  c"]);
        assert_eq!(split_args(&json!("")), Vec::<String>::new());
        assert_eq!(split_args(&json!(null)), Vec::<String>::new());
    }

    #[test]
    fn shell_syntax_stays_a_plain_word() {
        // nothing here ever reaches a shell: `;`, `$()` and backticks are just characters in a word
        assert_eq!(split_args(&json!("fw-ban 1.2.3.4;reboot")), vec!["fw-ban", "1.2.3.4;reboot"]);
        assert_eq!(split_args(&json!("$(id) `id`")), vec!["$(id)", "`id`"]);
    }

    #[test]
    fn luna_always_knows_she_lives_in_halcyon() {
        // present whether or not Halcyon's control script is installed on this machine
        let a = awareness();
        assert!(a.contains("Halcyon's resident AI"), "{a}");
        assert!(a.contains("NEVER ask which desktop"), "{a}");
    }

    #[tokio::test]
    async fn an_option_is_not_an_action() {
        let e = call("info", &json!({"action": "--help"}), None).await.unwrap_err().to_string();
        assert!(e.contains("Not an action") || e.contains("not installed"), "{e}");
    }
}
