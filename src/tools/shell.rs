use anyhow::{Context, Result};
use std::process::Command;

pub struct ShellOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub async fn run_command(command: &str, sudo_pass: Option<&str>) -> Result<ShellOutput> {
    let final_cmd = inject_sudo_password(command, sudo_pass);
    tracing::debug!("Running: {}", command); // log original, never log password

    let output = tokio::task::spawn_blocking(move || {
        Command::new("bash")
            .arg("-c")
            .arg(&final_cmd)
            .output()
            .context("Failed to spawn bash")
    })
    .await
    .context("spawn_blocking panicked")??;

    Ok(ShellOutput {
        stdout: truncate(&String::from_utf8_lossy(&output.stdout), 3000),
        stderr: truncate(&String::from_utf8_lossy(&output.stderr), 1000),
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Rewrite sudo commands to pipe password via `sudo -S` so they NEVER open
/// an interactive TTY prompt — which would steal keypresses from Luna's stdin.
///
/// sudo -S reads the password from stdin pipe.
/// sudo -p '' suppresses the "password:" prompt string from appearing in output.
///
/// A bash function shadows `sudo` so EVERY occurrence in the command (e.g.
/// after `&&` or `;`) gets the piped password — otherwise the 2nd sudo would
/// hang waiting for a TTY. `command sudo` bypasses the function for the real call.
pub fn inject_sudo_password(cmd: &str, sudo_pass: Option<&str>) -> String {
    let cmd = cmd.trim();

    // paru/yay escalate privilege themselves — strip the sudo prefix
    if let Some(rest) = cmd.strip_prefix("sudo paru") {
        return format!("paru{}", rest);
    }
    if let Some(rest) = cmd.strip_prefix("sudo yay") {
        return format!("yay{}", rest);
    }

    if !cmd.contains("sudo ") {
        return cmd.to_string();
    }

    match sudo_pass {
        Some(pass) => {
            let safe_pass = pass.replace('\'', "'\\''");
            format!(
                "sudo() {{ echo '{}' | command sudo -S -p '' \"$@\"; }}; {}",
                safe_pass, cmd
            )
        }
        None => {
            // No password stored — strip sudo entirely.
            // Command will fail with a permission error, which is far better
            // than hanging forever waiting for terminal input.
            cmd.replacen("sudo ", "", 1)
        }
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    let t = crate::util::truncate(s, max_chars);
    if t.len() == s.len() {
        t.to_string()
    } else {
        format!("{}... [truncated {} chars]", t, s.chars().count() - max_chars)
    }
}
