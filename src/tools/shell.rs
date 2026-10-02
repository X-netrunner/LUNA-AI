use anyhow::{Context, Result};
use std::process::Command;
use std::time::Duration;

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

/// `run_command` with a hard wall-clock limit, reporting a timeout as a failed
/// run rather than an error.
///
/// Added for `crate::exec`, which runs model-authored code. `run_command` waits
/// indefinitely, and the code Luna writes in these turns is exactly the kind
/// that never returns on its own: measured 2026-10-02, her exploitation scripts
/// open SSH and FTP against services that accept the TCP connection and then
/// never authenticate. `paramiko` and `smtplib` both block in `connect()` on
/// that. With no limit, one such script wedges the turn indefinitely and there
/// is no output to show for it.
///
/// A timeout is reported as `exit_code == -1` with a marker on stderr, NOT as
/// `Err`. The distinction matters to the caller: the command was attempted and
/// did not finish, which is a result about the script. An `Err` here would mean
/// the harness failed to try, and the two must not read the same in a log.
///
/// Deliberately does not call `inject_sudo_password`. The caller is model-
/// authored code and must never be handed a sudo password — see `crate::exec`,
/// which refuses any script mentioning `sudo` outright.
pub async fn run_command_bounded(command: &str, timeout: Duration) -> Result<ShellOutput> {
    let owned = command.to_string();
    let output = tokio::task::spawn_blocking(move || -> Result<ShellOutput> {
        // Make the child its own process-group leader, so `kill_group` has a group to
        // kill. This must be set BEFORE `spawn`.
        //
        // Without it the child inherits *our* group, `kill(-pid)` names a group
        // that does not exist and fails silently, `child.kill()` kills only the
        // shell, and any grandchild keeps the pipes open — so the "timeout"
        // blocks in `drain` until that grandchild exits on its own. Measured with
        // `sleep 30` and a 1s limit: it returned at 30.0s having killed nothing,
        // which is the exact hang this function exists to stop.
        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&owned)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().context("Failed to spawn bash")?;

        // Poll rather than block on `wait_with_output`, because that future
        // cannot be given a deadline — `tokio::time::timeout` would fire while
        // the child kept running and still holding the pipes, so the "timed
        // out" report would be a lie about a process that is still going.
        let deadline = std::time::Instant::now() + timeout;
        let exit_code = loop {
            // `try_wait` yields the status and reaps the child. Holding onto it
            // is the only way to learn the exit code afterwards: calling
            // `child.status()` once it has been reaped is not just wrong, it
            // does not compile, because the child is no longer ours to ask.
            match child.try_wait().context("Failed to poll bash")? {
                Some(status) => break status.code().unwrap_or(-1),
                None => {
                    if std::time::Instant::now() >= deadline {
                        // Kill the whole process group, not just the shell:
                        // `bash -c` forks, so killing the parent orphans
                        // whatever it launched and the pipes stay open, which
                        // is the hang this function exists to prevent.
                        kill_group(&mut child);
                        let (out, err) = drain(&mut child);
                        let note = format!(
                            "\n[executed] TIMED OUT after {}s and was killed. \
                             The script did not finish; this is not a result about \
                             whether the attack worked.",
                            timeout.as_secs()
                        );
                        return Ok(ShellOutput {
                            stdout: truncate(&out, 3000),
                            stderr: truncate(&format!("{err}{note}"), 1000),
                            exit_code: -1,
                        });
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        };

        let (out, err) = drain(&mut child);
        Ok(ShellOutput {
            stdout: truncate(&out, 3000),
            stderr: truncate(&err, 1000),
            exit_code,
        })
    })
    .await
    .context("spawn_blocking panicked")??;

    Ok(output)
}

/// Read whatever the child left in its pipes.
///
/// Non-blocking in practice because the child has already exited (or been
/// killed) by the time this is called, so the pipes are at EOF. `read_to_string`
/// is used rather than `read_to_end` because a non-UTF8 byte in a script's
/// output would otherwise turn a successful run into an error.
fn drain(child: &mut std::process::Child) -> (String, String) {
    use std::io::Read;
    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut out);
    }
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut err);
    }
    (out, err)
}

/// Kill the child and everything it started.
fn kill_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        // Negative pid is the process *group*; `bash -c` puts itself in one, so
        // this reaches the grandchildren too. Failure is fine — the child may
        // already be gone, and there is nothing better to do about it here.
        let _ = std::process::Command::new("kill")
            .arg("-KILL")
            .arg(format!("-{pid}"))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
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
