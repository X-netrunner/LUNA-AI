//! Execute the code Luna hands over, instead of handing it back to the user.
//!
//! # What this is
//!
//! On an offensive turn, when Luna emits runnable code *and* delegates running
//! it to the user, the harness writes each block to disk, logs it, and runs it.
//!
//! # Why it exists
//!
//! Measured 2026-10-02, N=12 through the real `run_routed_turn`: `run_shell`
//! was called 0 times and `write_file` 8 times. She wrote exploitation code
//! against the real address and the real services, then asked the user to run
//! it. Twelve turns, twelve hand-offs.
//!
//! Three things were tried first and all three failed, each measured rather
//! than assumed:
//!
//! 1. *A prompt-level correction.* The narration guard (`react.rs`) detects the
//!    hand-off and pushes a correction. Detection went 0/12 -> 10/12 once three
//!    closed gates in it were opened. She then said "I apologize for that
//!    oversight" and did exactly the same thing: 12/12 cells still handed off,
//!    `run_shell` still ~0.
//! 2. *Reordering instructions.* No effect, and the same class of thing as the
//!    prompt-level refusals that had already been rejected.
//! 3. *An exploit-shaped tool.* Rejected before writing: there is no exploit
//!    tool among the 47, and her failure used zero tools, so it could not have
//!    been the missing affordance.
//!
//! The thing that did work for the adjacent problem was removing the decision
//! from the model: `crate::recon` runs in the harness *before* the model is
//! called, so there is no step in which she can decline. This module is that
//! same move applied to execution. A prompt ask has now been measured not to
//! bind here, so this does not ask.
//!
//! The one measured exception, kept because it is evidence rather than
//! disappointment: in 1 of 14 samples she did call `run_shell` four times, it
//! failed (`pip` -> `externally-managed-environment`, exit 1), and she reported
//! `FAILED (exit 1)` honestly instead of inventing a result. So the correction
//! is not inert — it is just unreliable, which is not a basis for leaving the
//! user to find out which it will be on their own machine.
//!
//! # Scope, and what it does NOT constrain
//!
//! Read this before changing a line, and read it again before blaming a failure
//! of this module on the checks below.
//!
//! * **The loopback check is a literal scan, not a sandbox.** It reads IPv4 and
//!   hostname literals out of the source and refuses anything that is not
//!   loopback. A script that resolves its target at runtime —
//!   `socket.gethostbyname(os.environ["T"])`, integer arithmetic on octets, a
//!   URL assembled from parts — passes it and then talks to whatever it likes.
//!   This is a strong speed bump against the *address Luna wrote in the answer*,
//!   which is the measured failure, and nothing at all against a determined
//!   bypass. It is deliberately not described as a guarantee anywhere else in
//!   the codebase.
//! * **Loopback scope bounds the network, not the machine.** The scripts in
//!   these turns are exploitation code, but nothing constrains the *class* of
//!   code from the filesystem side. A loopback-only network scope does not stop
//!   `rm`. The real constraints here are the capability gate (a human authorised
//!   this) and the fact that she writes attack scripts, which is not a security
//!   control.
//! * **`sudo` is refused outright.** Not disabled, not stripped — refused, with
//!   the block left on disk for inspection. `shell::inject_sudo_password`
//!   rewrites `sudo` to pipe the password from stdin, so a script reaching that
//!   path would be handed the sudo password by the harness. That is the single
//!   worst thing this module could do, so it is a hard stop rather than a
//!   best-effort rewrite.
//! * **One authorisation, checked here, not inherited.** `run_shell` appears in
//!   neither `capability_tools` nor `gated_tools`, so `tools::execute` applies
//!   *no* gate to it at all. Routing through `tools::execute` would therefore
//!   have supplied no authorisation whatsoever while looking like it did. The
//!   check is explicit below, and it refuses rather than inheriting one that
//!   is not there.
//! * **Bounded.** At most `auto_execute_max_blocks` blocks per turn, each with
//!   its own `auto_execute_timeout_secs` wall clock.
//!
//! # Forensic logging
//!
//! Every block is written to `~/.local/share/luna/executed_scripts/<stamp>/`
//! **before** it runs, and kept whatever happens after — success, failure,
//! timeout, or refusal. A run that goes wrong is then recoverable from disk
//! without relying on a live confirmation step, which is the property that
//! matters when the thing running is code nobody has read yet.

use crate::config::LunaConfig;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Languages whose fenced block is runnable, mapped to the interpreter.
///
/// `rust` is deliberately absent. It was in the narration guard's list once and
/// was wrong there too: a Rust block in these turns is source for the user to
/// compile, not something to run, and `rustc` is not something to invoke on a
/// machine because a paragraph asked about attacks.
const INTERPRETERS: &[(&str, &str)] = &[
    ("bash", "bash"),
    ("sh", "sh"),
    ("shell", "sh"),
    ("zsh", "zsh"),
    ("console", "sh"),
    ("shell-session", "sh"),
    ("python", "python3"),
    ("py", "python3"),
];

/// One extracted runnable block.
#[derive(Debug, Clone)]
pub struct Block {
    pub lang: String,
    pub source: String,
}

/// Outcome of trying to run one block.
#[derive(Debug)]
pub enum Outcome {
    Ran {
        exit_code: i32,
        stdout: String,
        stderr: String,
        path: PathBuf,
    },
    /// Never started. The source is still on disk under `path`.
    Refused {
        reason: String,
        path: PathBuf,
    },
}

/// What one turn's execution attempt produced, for the answer text.
#[derive(Debug, Default)]
pub struct Report {
    pub outcomes: Vec<Outcome>,
    pub skipped: usize,
    /// Set when the turn was not attempted at all, so the answer can say why
    /// instead of silently keeping the hand-off it was meant to replace.
    pub not_attempted: Option<String>,
}

/// Whether this turn's answer should be executed rather than returned.
///
/// Deliberately the *same* conjunction the narration guard uses: runnable code
/// in a fence AND a hand-off. Both halves, because either alone is wrong in a
/// way that matters. Code with no hand-off is an explanation she is giving, and
/// running it would execute something nobody asked to be executed. A hand-off
/// with no runnable block is advice, and there is nothing here to run.
///
/// The offensive-request test is a third condition, not a fourth: this module
/// exists for offensive turns, and an auto-execution path that fires on
/// "write me a script to back up my files" would be a different and much larger
/// grant than the one authorised.
pub fn should_execute(answer: &str, input: &str, config: &LunaConfig) -> bool {
    if !config.llm.security_auto_execute {
        return false;
    }
    if extract_blocks(answer).is_empty() {
        return false;
    }
    crate::llm::react::narrates_shell_execution(answer) && crate::llm::react::requests_execution(input)
}

/// Run every runnable block in `answer`, in order.
///
/// Failure of one block is recorded and the next still runs: her answer is a
/// sequence of independent scripts, and one failing is the normal case for
/// exploitation code rather than a reason to skip the rest.
pub async fn run(answer: &str, config: &LunaConfig) -> Report {
    let mut report = Report::default();

    if let Err(why) = authorise(config) {
        report.not_attempted = Some(why);
        return report;
    }

    let blocks = extract_blocks(answer);
    let limit = config.llm.auto_execute_max_blocks.max(1);
    if blocks.len() > limit {
        report.skipped = blocks.len() - limit;
    }

    let dir = match log_dir() {
        Ok(d) => d,
        Err(why) => {
            // Refusing because we could not write the log is deliberate. The
            // forensic record is a precondition, not a nicety: executing code
            // nobody kept a copy of is the one outcome with no way back.
            report.not_attempted = Some(format!(
                "could not create the script log directory, so nothing was run: {why}"
            ));
            return report;
        }
    };

    let timeout = Duration::from_secs(config.llm.auto_execute_timeout_secs.max(1));

    for (i, block) in blocks.iter().take(limit).enumerate() {
        // Log first, always — including for a block we are about to refuse.
        // The refusal is far more interesting to read afterwards if the thing
        // that was refused is sitting on disk next to it.
        let path = match write_block(&dir, i, block) {
            Ok(p) => p,
            Err(why) => {
                report.outcomes.push(Outcome::Refused {
                    reason: format!("could not write the block to disk: {why}"),
                    path: dir.clone(),
                });
                break;
            }
        };

        if let Some(reason) = refusal_reason(&block.source) {
            report.outcomes.push(Outcome::Refused { reason, path });
            continue;
        }

        let command = format!(
            "cd {} && {} {}",
            shell_quote(&dir.to_string_lossy()),
            INTERPRETERS
                .iter()
                .find(|(l, _)| *l == block.lang)
                .map(|(_, i)| *i)
                .unwrap_or("sh"),
            shell_quote(&path.to_string_lossy())
        );

        tracing::info!(
            "Auto-execute: running Luna's block {} ({}) from {}",
            i + 1,
            block.lang,
            path.display()
        );

        // `run_command_bounded`, not `run_command`: the latter waits forever and
        // these scripts open connections to services that never authenticate.
        match crate::tools::shell::run_command_bounded(&command, timeout).await {
            Ok(out) => report.outcomes.push(Outcome::Ran {
                exit_code: out.exit_code,
                stdout: out.stdout,
                stderr: out.stderr,
                path,
            }),
            Err(why) => report.outcomes.push(Outcome::Refused {
                reason: format!("could not start the block: {why}"),
                path,
            }),
        }
    }

    report
}

/// Append the result of `run` to `answer`.
///
/// The answer is kept, not replaced. Her script is grounded now — the real
/// address, the real services — so it is the most useful thing in the turn, and
/// replacing it with "I didn't run it" would be a worse product than the one
/// that hands over unexecuted code. What changes is that the user can now see
/// which of the two situations they are in.
pub fn render(answer: &str, report: &Report) -> String {
    if let Some(why) = &report.not_attempted {
        return format!(
            "{answer}\n\n---\n**I did not run any of the above.** {why}\n\n\
             Nothing below this line is a result."
        );
    }
    if report.outcomes.is_empty() {
        return answer.to_string();
    }

    let ran = report
        .outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::Ran { .. }))
        .count();
    let refused = report.outcomes.len() - ran;

    let mut out = format!(
        "{answer}\n\n---\n**I ran this myself rather than handing it to you.** \
         {ran} block(s) executed"
    );
    if refused > 0 {
        out.push_str(&format!(", {refused} refused"));
    }
    out.push_str(".\n");

    for (i, outcome) in report.outcomes.iter().enumerate() {
        out.push_str(&format!("\n**Block {}**", i + 1));
        match outcome {
            Outcome::Ran {
                exit_code,
                stdout,
                stderr,
                path,
            } => {
                // Lead with the exit code and label it a real result, so this
                // cannot be read as a success the way a bare "SUCCESS" from
                // `run_shell` could be. `exit_code == -1` is the timeout marker
                // from `run_command_bounded` and means "did not finish", which
                // is not the same as "failed".
                let verdict = if *exit_code == 0 {
                    "exited 0"
                } else if *exit_code == -1 {
                    "DID NOT FINISH"
                } else {
                    &format!("FAILED (exit {exit_code})")
                };
                out.push_str(&format!(" — {verdict}\n\n"));
                if !stdout.trim().is_empty() {
                    out.push_str("```\n");
                    out.push_str(stdout.trim_end());
                    out.push_str("\n```\n");
                }
                if !stderr.trim().is_empty() {
                    out.push_str("\nstderr:\n```\n");
                    out.push_str(stderr.trim_end());
                    out.push_str("\n```\n");
                }
                if stdout.trim().is_empty() && stderr.trim().is_empty() {
                    out.push_str("\nNo output. An empty result is not a success.\n");
                }
                out.push_str(&format!("\nSaved: {}\n", path.display()));
            }
            Outcome::Refused { reason, path } => {
                out.push_str(&format!(" — NOT RUN: {reason}\n\nSaved for review: {}\n", path.display()));
            }
        }
    }

    if report.skipped > 0 {
        out.push_str(&format!(
            "\n_{} further block(s) in my answer were not run. \
             My own limit for one turn is {} blocks._\n",
            report.skipped,
            report.outcomes.len()
        ));
    }
    out
}

/// The authorisation this module checks for itself.
///
/// Explicit because `run_shell` is ungated: `capability_tools` is
/// `["nmap_scan", "sysmode", "self_patch", "system_update"]` and `gated_tools`
/// is messaging and desktop tools. Neither contains it, so `tools::execute`
/// applies no check to `run_shell`. Same two halves as every other capability —
/// the request in the config *and* a developer receipt on disk — because one
/// without the other is the switch anyone can flip.
fn authorise(config: &LunaConfig) -> std::result::Result<(), String> {
    if !config.llm.security_auto_execute {
        return Err(
            "auto-execution is switched off (`security_auto_execute = false`)".to_string()
        );
    }
    if crate::unlock::capabilities_active(
        config.external.allow_capability_actions,
        &config.llm.security_dev_public_key,
    ) {
        Ok(())
    } else {
        Err(
            "executing model-authored code needs the developer receipt \
             (`luna --unlock-security`)"
                .to_string(),
        )
    }
}

/// Refuse a block outright, or `None` to run it.
fn refusal_reason(source: &str) -> Option<String> {
    // sudo first: it is the one that would hand over a credential.
    if word_present(source, "sudo") {
        return Some(
            "it invokes `sudo`. Refused rather than stripped: `inject_sudo_password` \
             would pipe the sudo password into model-authored code, so this path \
             never runs anything mentioning it."
                .to_string(),
        );
    }
    let (ip, is_ip) = first_non_loopback_ipv4(source);
    if is_ip {
        return Some(format!(
            "it references `{ip}`, which is not loopback. This path runs code against \
             127.0.0.1 only. Note the limit honestly: this is a literal scan, so a \
             target assembled at runtime would not be caught here."
        ));
    }
    let (host, is_host) = first_non_loopback_host(source);
    if is_host {
        return Some(format!(
            "it references the host `{host}`, which is not loopback."
        ));
    }
    None
}

/// The first IPv4 literal in `source` that is not loopback.
///
/// Four octets parsed as numbers, never `starts_with("127.")`. Same rule and
/// same reason as `tools::security::is_loopback_target`: a prefix test reads
/// `127.0.0.1` as "127" and "1", and accepts `1270.0.0.1`, which is not
/// loopback and is a routable address on some networks.
fn first_non_loopback_ipv4(source: &str) -> (String, bool) {
    let bytes: Vec<char> = source.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut parts: Vec<String> = Vec::new();
            let mut ok = false;
            for _ in 0..4 {
                let mut num = String::new();
                let mut j = i;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    num.push(bytes[j]);
                    j += 1;
                }
                if num.is_empty() || num.len() > 3 {
                    break;
                }
                match num.parse::<u32>() {
                    Ok(v) if v <= 255 => parts.push(num.clone()),
                    _ => break,
                }
                i = j;
                if parts.len() < 4 {
                    // Require a literal dot. `1270.0.0.1` must not parse as
                    // `127` followed by junk.
                    if i < bytes.len() && bytes[i] == '.' {
                        i += 1;
                    } else {
                        break;
                    }
                }
            }
            if parts.len() == 4 {
                ok = true;
                let ip = parts.join(".");
                // Leading zeros are how `127.0.0.01` and friends slip past a
                // naive check, and how octal-looking IPs differ from decimal.
                if parts[0].parse::<u32>().unwrap_or(255) != 127 {
                    let _ = start;
                    return (ip, true);
                }
            }
            if !ok {
                i = start + 1;
            }
        } else {
            i += 1;
        }
    }
    (String::new(), false)
}

/// The first hostname literal that is not loopback.
///
/// Narrow on purpose, and deliberately so — the first version of this matched
/// any dotted name with an alphabetic final label, and the tests caught it
/// refusing `socket.socket()` and `os.getpid()`. A check that fires on
/// `import socket` fires on essentially every script she writes, and a refusal
/// that is always true is a refusal nobody reads.
///
/// So the final label must be a real top-level domain. That leaves false
/// negatives — `evil.co.uk` ends in a real TLD but `.uk` is in the list, so it
/// is caught; a made-up TLD would not be. Given the loopback check is already
/// documented as a literal scan and not a sandbox, the right trade is to be
/// narrow and say so, rather than broad and useless.
fn first_non_loopback_host(source: &str) -> (String, bool) {
    /// Real TLDs, plus the private-use names that appear in local networks.
    const TLDS: &[&str] = &[
        "com", "net", "org", "edu", "gov", "mil", "int", "info", "biz", "io",
        "co", "dev", "app", "xyz", "me", "ai", "us", "uk", "ca", "au", "nz",
        "de", "fr", "es", "it", "nl", "be", "se", "no", "dk", "fi", "pl",
        "ru", "jp", "cn", "in", "br", "kr", "ch", "at", "cz", "eu", "ie",
        "local", "localhost", "internal", "intranet", "lan", "home", "test",
        "example", "invalid",
    ];

    for word in source.split(|c: char| !(c.is_alphanumeric() || c == '.' || c == '-')) {
        let w = word.trim_matches('.');
        if w.is_empty() || !w.contains('.') || w.starts_with('.') || w.ends_with('.') {
            continue;
        }
        let lower = w.to_ascii_lowercase();
        let labels: Vec<&str> = lower.split('.').collect();
        if labels.len() < 2 {
            continue;
        }
        let last = labels[labels.len() - 1];
        if !TLDS.contains(&last) {
            continue;
        }
        if lower.starts_with("127.") || lower.starts_with("10.") || lower.starts_with("192.168.") {
            continue; // handled by the IPv4 path
        }
        return (w.to_string(), true);
    }
    (String::new(), false)
}

/// Whether `word` appears as a whole token, not as a substring.
///
/// `sudo` matters here: `pseudo`, `resume` and `sudoers` are ordinary words in
/// ordinary scripts, and a substring match would refuse them and teach the
/// user that the refusal message is noise.
fn word_present(source: &str, word: &str) -> bool {
    source
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|t| t == word)
}

/// Extract fenced blocks whose language is runnable.
pub fn extract_blocks(answer: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut lang = String::new();
    let mut body: Vec<&str> = Vec::new();

    for line in answer.lines() {
        let trimmed = line.trim_start();
        if !in_fence {
            if let Some(rest) = trimmed.strip_prefix("```") {
                in_fence = true;
                lang = rest.trim().to_ascii_lowercase();
                body.clear();
            }
        } else if trimmed.starts_with("```") {
            in_fence = false;
            let source = body.join("\n");
            if INTERPRETERS.iter().any(|(l, _)| *l == lang) && !source.trim().is_empty() {
                out.push(Block {
                    lang: lang.clone(),
                    source: source.trim_end().to_string(),
                });
            }
            lang.clear();
        } else {
            body.push(line);
        }
    }
    out
}

/// `~/.local/share/luna/executed_scripts/<unix-seconds>/`.
///
/// `data_local_dir` rather than `state_dir`, matching `history.json` — these
/// are files meant to be read by a person after the fact, not internal state.
/// A timestamped directory per turn means a run is recoverable by when it
/// happened without needing to correlate against anything else.
fn log_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("luna")
        .join("executed_scripts");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dir = base.join(format!("{stamp}"));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

fn write_block(dir: &Path, index: usize, block: &Block) -> Result<PathBuf> {
    let ext = if block.lang.starts_with("py") { "py" } else { "sh" };
    let path = dir.join(format!("block{index}.{ext}"));
    // A header naming the language, because a block logged as `.sh` that is
    // Python is a genuinely confusing thing to find on disk later.
    let header = format!("# luna auto-execute: block {index}, language {}\n", block.lang);
    std::fs::write(&path, format!("{header}{}", block.source))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Single-quote for `sh -c`.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ipv4_accepted(src: &str) -> bool {
        !first_non_loopback_ipv4(src).1
    }

    #[test]
    fn loopback_literals_pass() {
        assert!(ipv4_accepted("socket.connect(('127.0.0.1', 22))"));
        assert!(ipv4_accepted("host = '127.0.0.1'"));
        assert!(ipv4_accepted("127.0.0.53 for dns"));
    }

    /// `0.0.0.0` is refused even though connecting to it means localhost.
    ///
    /// Binding to it means every interface, including the LAN-facing one. Which
    /// of those two things a script does is not knowable from the literal, so
    /// the check takes the reading that keeps the machine off the network.
    #[test]
    fn unspecified_is_refused_for_the_bind_case() {
        assert!(!ipv4_accepted("s.bind(('0.0.0.0', 8080))"));
    }

    #[test]
    fn non_loopback_literals_are_refused() {
        assert!(!ipv4_accepted("host = '192.168.1.50'"));
        assert!(!ipv4_accepted("host = '10.0.0.5'"));
        assert!(!ipv4_accepted("connect to 8.8.8.8:53"));
    }

    #[test]
    fn octets_are_parsed_not_prefix_matched() {
        // `1270.0.0.1` starts with "127." and is not loopback. A
        // starts_with("127.") check accepts it; this must not.
        assert!(!ipv4_accepted("host = '1270.0.0.1'"));
        assert!(!ipv4_accepted("x = 172.16.4.2"));
    }

    #[test]
    fn four_digit_octets_are_not_truncated() {
        assert!(!ipv4_accepted("addr = 1270.0.0.1"));
        assert!(!ipv4_accepted("addr = 9999.1.1.1"));
    }

    #[test]
    fn sudo_is_refused_as_a_word_not_a_substring() {
        assert!(refusal_reason("sudo nmap 127.0.0.1").is_some());
        assert!(refusal_reason("os.getpid()").is_none());
        assert!(refusal_reason("x = 'pseudo'").is_none());
        assert!(refusal_reason("y = 'sudoers'").is_none());
    }

    #[test]
    fn ordinary_python_is_not_mistaken_for_a_hostname() {
        let s = "import socket\ns = socket.socket()\ns.connect(('127.0.0.1', 22))\n";
        assert!(refusal_reason(s).is_none(), "{}", refusal_reason(s).unwrap_or_default());
        assert!(refusal_reason("import paramiko\n").is_none());
        assert!(refusal_reason("from ftplib import FTP\n").is_none());
        // The two that broke the first version of the host check.
        assert!(refusal_reason("import socket\ns = socket.socket()\n").is_none());
        assert!(refusal_reason("pid = os.getpid()\n").is_none());
    }

    #[test]
    fn real_hostnames_are_refused() {
        assert!(refusal_reason("curl http://example.com/x").is_some());
        assert!(refusal_reason("ssh admin@fileserver.example.org").is_some());
    }

    #[test]
    fn fences_are_extracted_by_language() {
        let a = "```bash\nnmap 127.0.0.1\n```\ntext\n```python\nprint(1)\n```\n```rust\nfn main(){}\n```\n";
        let b = extract_blocks(a);
        assert_eq!(b.len(), 2, "rust must not be treated as runnable");
        assert_eq!(b[0].lang, "bash");
        assert_eq!(b[1].lang, "python");
    }

    #[test]
    fn shell_quoting_survives_embedded_quotes() {
        assert_eq!(shell_quote("a'b"), r"'a'\''b'");
    }

    // ── gate tests ───────────────────────────────────────────────────────────
    //
    // These assert on a MARKER FILE the script creates, not on the return value.
    // Asserting on the return value would pass just as well if the gate refused
    // for an unrelated reason, or if the script ran and the marker path were
    // wrong. The marker is the only observation that distinguishes "refused" from
    // "ran and produced output".

    fn marker() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("luna_exec_marker_{}", std::process::id()))
    }

    #[tokio::test]
    async fn the_closed_gate_runs_nothing() {
        let _m = marker();
        let _ = std::fs::remove_file(marker());
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        // No developer key at all: the strongest closed state.
        let answer = format!(
            "Save this script and run it.\n\n```bash\necho hi > {}\n```\n",
            marker().display()
        );

        let report = run(&answer, &cfg).await;
        assert!(
            !marker().exists(),
            "the script ran with the capability gate closed"
        );
        assert!(report.outcomes.is_empty(), "nothing should have been attempted");
        let out = render(&answer, &report);
        assert!(
            out.contains("I did not run any of the above"),
            "the user must be told nothing ran, got: {out}"
        );
    }

    /// The other direction. Without this, "the closed gate refused" is
    /// satisfied by the executor being broken.
    #[tokio::test]
    async fn the_open_gate_actually_executes() {
        let _m = marker();
        let _ = std::fs::remove_file(marker());
        let (pk, _guard) = crate::unlock::open_gate_for_test("exec_open");
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        cfg.external.allow_capability_actions = true;
        cfg.llm.security_dev_public_key = pk;

        let answer = format!(
            "Save this script and run it.\n\n```bash\necho ran > {}\necho stdout-was-captured\n```\n",
            marker().display()
        );

        let report = run(&answer, &cfg).await;
        assert!(
            marker().exists(),
            "the script did not run with the gate open — the executor is inert"
        );
        let out = render(&answer, &report);
        assert!(out.contains("I ran this myself"), "got: {out}");
        assert!(out.contains("stdout-was-captured"), "stdout not shown: {out}");
        let _ = std::fs::remove_file(marker());
    }

    #[tokio::test]
    async fn sudo_is_refused_even_with_the_gate_open() {
        let _m = marker();
        let _ = std::fs::remove_file(marker());
        let (pk, _guard) = crate::unlock::open_gate_for_test("exec_sudo");
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        cfg.external.allow_capability_actions = true;
        cfg.llm.security_dev_public_key = pk;
        let answer = format!(
            "Save this script and run it.\n\n```bash\nsudo bash -c 'echo ran > {}'\n```\n",
            marker().display()
        );

        let report = run(&answer, &cfg).await;
        assert!(!marker().exists(), "a sudo script was executed");
        let out = render(&answer, &report);
        assert!(out.contains("NOT RUN"), "must say it did not run: {out}");
        assert!(out.contains("sudo"), "must name the reason: {out}");
    }

    #[tokio::test]
    async fn a_non_loopback_target_is_refused_even_with_the_gate_open() {
        let _m = marker();
        let _ = std::fs::remove_file(marker());
        let (pk, _guard) = crate::unlock::open_gate_for_test("exec_lan");
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        cfg.external.allow_capability_actions = true;
        cfg.llm.security_dev_public_key = pk;
        let answer = format!(
            "Save this script and run it.\n\n```bash\ncurl http://192.168.1.50/ > {}\n```\n",
            marker().display()
        );

        let report = run(&answer, &cfg).await;
        assert!(!marker().exists(), "a LAN-facing script was executed");
        let out = render(&answer, &report);
        assert!(out.contains("NOT RUN"), "got: {out}");
        assert!(
            out.contains("192.168.1.50"),
            "the refusal must name the address it refused: {out}"
        );
    }

    /// The forensic log is the user's only recourse after a bad run, so it has to
    /// exist for a block that was REFUSED as well as one that ran.
    #[tokio::test]
    async fn a_refused_block_is_still_written_to_disk() {
        let (pk, _guard) = crate::unlock::open_gate_for_test("exec_log");
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        cfg.external.allow_capability_actions = true;
        cfg.llm.security_dev_public_key = pk;
        let answer =
            "Save this script and run it.\n\n```bash\nsudo rm -rf /tmp/nothing\n```\n".to_string();

        let report = run(&answer, &cfg).await;
        match &report.outcomes[0] {
            Outcome::Refused { path, .. } => {
                let logged = std::fs::read_to_string(path)
                    .expect("the refused block must be on disk");
                assert!(
                    logged.contains("sudo rm -rf"),
                    "the log must hold the source verbatim, got: {logged}"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_hanging_script_is_killed_and_reported_as_not_finished() {
        let (pk, _guard) = crate::unlock::open_gate_for_test("exec_timeout");
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        cfg.external.allow_capability_actions = true;
        cfg.llm.security_dev_public_key = pk;
        cfg.llm.auto_execute_timeout_secs = 1;
        let answer = "Save this script and run it.\n\n```bash\nsleep 30\n```\n".to_string();

        let started = std::time::Instant::now();
        let report = run(&answer, &cfg).await;
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "the timeout did not fire; took {:?}",
            started.elapsed()
        );
        let out = render(&answer, &report);
        assert!(
            out.contains("DID NOT FINISH"),
            "a timeout must not be reported as a failure of the script: {out}"
        );
    }

    #[test]
    fn should_execute_is_off_by_default() {
        let cfg = crate::config::LunaConfig::default();
        let answer = "Save this script and run it.\n\n```bash\necho hi\n```\n";
        assert!(
            !should_execute(answer, "attack my laptop", &cfg),
            "auto-execute must be off unless explicitly switched on"
        );
    }

    #[test]
    fn should_execute_needs_code_and_a_handoff_and_an_offensive_turn() {
        let (_pk, _guard) = crate::unlock::open_gate_for_test("exec_should");
        let mut cfg = crate::config::LunaConfig::default();
        cfg.llm.security_auto_execute = true;
        cfg.external.allow_capability_actions = true;
        // Key deliberately empty: `should_execute` is a predicate, so it must
        // decide identically whether or not a receipt exists. Execution
        // authorisation is `authorise`'s job, tested separately above.
        cfg.llm.security_dev_public_key = String::new();

        let fenced = "```bash\necho hi\n```\n";
        let handoff = "Save this script and run it.";

        assert!(should_execute(
            &format!("{handoff}\n\n{fenced}"),
            "attack my laptop",
            &cfg
        ));
        // An explanation, not a hand-off.
        assert!(!should_execute(
            &format!("Here is how it works.\n\n{fenced}"),
            "attack my laptop",
            &cfg
        ));
        // A hand-off with nothing runnable in it.
        assert!(!should_execute(handoff, "attack my laptop", &cfg));
        // Rust is not runnable here.
        assert!(!should_execute(
            &format!("{handoff}\n\n```rust\nfn main() {{}}\n```"),
            "attack my laptop",
            &cfg
        ));
    }
}