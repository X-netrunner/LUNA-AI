//! Harness-run recon, so the port map is observed rather than remembered.
//!
//! ## What failed, and why this exists
//!
//! On 2026-10-02 the user asked Luna to attack their laptop, twice, and on both
//! turns she wrote a Python script with placeholder credentials
//! (`your_laptop_ip`, `your_username`, `your_password`) and told the user to run
//! it. She never scanned anything. Measured on the real dispatch path with the
//! real tool selector:
//!
//! - Turn 1, "i want you to try to attack my laptop": `nmap_scan` was **not
//!   offered**. `Tool subset: 23 of 47`, zero security tools. Her trigger list
//!   is `nmap`, `port`, `recon`, `scan localhost` — "attack my laptop" matches
//!   none of them, so she had nothing to scan with.
//! - Turns 2 and 3: `nmap_scan` **was** offered, along with `write_file` and
//!   `run_shell`, and she called **none** of them. Turn 2's request even said
//!   "you should be saving the script yourself". So this was not a permissions
//!   problem and not a tool-availability problem. The tools were on the table
//!   and she wrote prose.
//!
//! That rules out the two easy fixes. Telling her to scan first is a prompt
//! instruction, and this project has already measured that prompt-level asks do
//! not bind behaviour reliably — the same reasoning that retired the phrase-list
//! router and the tamper check. Gating exploit-shaped tools is not available
//! either: there are 47 tools and none is an exploit, the only attack capability
//! is `run_shell` with an arbitrary string, and her failure passed through **no
//! tool at all**. The one content channel she could have used to stage an attack
//! is `write_file`, which is also how every ordinary file write happens; gating
//! it behind recon would break writing code.
//!
//! So the decision is taken out of her hands. When a request is offensive, the
//! harness scans the target itself and hands her the result.
//!
//! ## Why the model cannot skip this
//!
//! Because it is not hers to skip. There is no ordering left to respect: the
//! recon is either already in the prompt or the turn never starts. A prompt
//! instruction asks her to choose a sequence; this removes the choice.
//!
//! It also removes the reason she skipped it. She wrote placeholders because
//! "my laptop" does not resolve to an address she knows, and a script with
//! placeholders is the only thing she can produce from no information. Handed a
//! real port map, she is reasoning from data rather than filling in a template —
//! which is the same class of fix as the `SUCCESS`-envelope guard: stop feeding
//! her a shape she can fill with fiction, rather than asking her not to.
//!
//! ## What this is not
//!
//! It is not a safety control and must not be recorded as one. It guarantees
//! *sequence* — a scan happens before she answers. It says nothing about whether
//! what she does next is correct or harmless. "Scan before exploiting" is an
//! ordering, not a judgement.
//!
//! Nor does it stop her writing an exploit afterwards. It changes the input she
//! reasons from. Whether that becomes a working chain is a measurement, not a
//! consequence, and is being measured at N=12 before anyone calls it fixed.
//!
//! ## No new authority
//!
//! The scan goes through `tools::execute` like every other call, so both
//! existing layers still apply: the capability gate and the loopback scope
//! limit. The user has already signed for `nmap_scan`; the only thing that
//! changes here is *who triggers it* — the harness rather than the model. If
//! the gate is off, no scan happens, and the turn is told so honestly rather
//! than continuing as though recon had occurred.
//!
//! ## Per-turn, deliberately
//!
//! Injected into the per-turn prompt, never persisted to memory. The
//! summariser is aggressive — the same session shows `Summarized session …:
//! Nmap Scan Results for localhost` — and a stored recon block can be
//! summarised out from under her partway through a task, leaving a later turn
//! believing she has a port map she no longer has. Rebuilding it every turn
//! costs one loopback scan and removes the failure mode.

use crate::config::LunaConfig;

/// The loopback address recon runs against.
///
/// Not a resolution of whatever "my laptop" means — a fixed scope. Pronoun
/// resolution is exactly the step that produced `your_laptop_ip`: the request
/// named a machine and the model had no way to turn that into an address.
/// Choosing here means the answer is always the same and always local, which is
/// also the only target the user can have meant without naming it.
pub const RECON_TARGET: &str = "127.0.0.1";

/// How many turns an offensive task stays in progress.
///
/// A task is not one message. "Attack my laptop" (turn 1), "save the script
/// yourself" (turn 2), "what did you find" (turn 3), "now use port 22"
/// (turn 4) — only the first is lexically offensive. A per-message test hands
/// over a port map on turn 1 and then silently stops, which is the same
/// follow-up blind spot that got the small-model router rejected: a classifier
/// sees one message at a time and cannot tell that turn 4 refers to turn 1.
///
/// Generous on purpose. The cost of a spurious scan is ~2.5s against loopback
/// and one line in the activity feed; the cost of dropping the port map is that
/// Luna answers about a machine she has no data for, which is the failure this
/// module exists to stop. `clear` resets it explicitly.
const LATCH_TURNS: u32 = 12;

/// Turns of offensive work still in progress, if any.
fn latch() -> &'static std::sync::Mutex<u32> {
    static L: std::sync::OnceLock<std::sync::Mutex<u32>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(0))
}

/// Arm or disarm the task latch. Once per turn.
fn tick_latch(offensive: bool) {
    if let Ok(mut l) = latch().lock() {
        if offensive {
            *l = LATCH_TURNS;
        } else if *l > 0 {
            *l -= 1;
        }
    }
}

/// Forget any task in progress. Wired to the same `clear` as the tier latch.
pub fn reset_latch() {
    if let Ok(mut l) = latch().lock() {
        *l = 0;
    }
}

/// Is an offensive task in progress right now?
pub fn task_active() -> bool {
    latch().lock().map(|l| *l > 0).unwrap_or(false)
}

/// Should this turn be preceded by a harness scan?
///
/// Two inputs, because one is not enough. The message catches a fresh request;
/// the latch carries a task across the follow-ups that describe it in no
/// offensive words at all.
pub fn should_recon(input: &str, config: &LunaConfig) -> bool {
    if !config.llm.security_auto_recon {
        return false;
    }
    let offensive = crate::llm::escalation::is_offensive_request(input);
    tick_latch(offensive);
    offensive || task_active()
}

/// Run the scan and return its raw output, or the reason it did not happen.
///
/// Returned as a `Result` so the caller can put a real failure in front of the
/// model. Swallowing this would be worse than not scanning: she would be asked
/// about a machine she has no data for, and the previous failure was invented
/// ports. A turn that knows its recon failed can say so.
pub async fn run(config: &LunaConfig) -> Result<String, String> {
    let call = crate::llm::ollama::ToolCall {
        function: crate::llm::ollama::ToolCallFunction {
            name: "nmap_scan".into(),
            arguments: serde_json::json!({
                "target": RECON_TARGET,
                // `-p-` rather than a version scan. Measured on this machine:
                // 2.47s for the full port range against loopback, against 24-48s
                // for `-sS -sV -p-`. Service banners are the upgrade, and they
                // are only worth paying for if port numbers alone turn out not
                // to be enough to reason about — 10-20x the time to find out.
                "scan_type": config.llm.recon_scan_type.clone(),
            }),
        },
    };

    // Published so the scan is visible in the activity feed. Without it the
    // user sees Luna go quiet for the length of a real scan with no indication
    // anything is running — the exact gap the feed was built to close.
    let started = std::time::Instant::now();
    crate::activity::publish(crate::activity::Event::ToolStart {
        name: "nmap_scan".into(),
        summary: format!("harness recon: {} (-p-)", RECON_TARGET),
    });

    let result = crate::tools::execute(&call, config).await;

    crate::activity::publish(crate::activity::Event::ToolEnd {
        name: "nmap_scan".into(),
        ok: result.is_ok(),
        elapsed: started,
    });

    result.map_err(|e| {
        let msg = format!("{e:#}");
        crate::util::truncate(&msg, 400).to_string()
    })
}

/// The block handed to the model.
///
/// Phrased as observed fact and as a precondition, because both halves matter:
/// "these are the open ports" is the data she was missing, and "this is what
/// you have to work from" is what stops her proposing a scan she has not run.
pub fn injection_block(input: &str, result: &Result<String, String>) -> String {
    match result {
        Ok(scan) => format!(
            "RECONNAISSANCE — already performed by the harness for this turn. This is \
             real output, not an example.\n\n\
             Target: {RECON_TARGET} (loopback, which is what the user means by their \
             own machine)\n\
             Scan: nmap -p- (all 65535 TCP ports)\n\n\
             ```\n{scan}\n```\n\n\
             Rules for this turn:\n\
             1. These ports are the actual attack surface. Use THESE numbers. Do \
                not name a port that is not in the output above — inventing one is \
                the single worst thing you can do here.\n\
             2. Do not ask the user to run a scan, and do not describe a scan as \
                something you are about to do. It has happened.\n\
             3. Build the attack chain from what is actually listening: for each \
                port, identify the service, decide whether it is reachable and \
                exploitable, and say so plainly — including when the answer is that \
                a port is not exploitable.\n\
             4. 'my laptop' means {RECON_TARGET}. That address is the only target \
                in play — do not write a placeholder for a hostname or a username, \
                because the address is already given above.\n\
             5. If the user asked about something this scan cannot answer, say what \
                would be needed and why, rather than guessing.\n\n\
             The user said: {input}\n"
        ),
        Err(why) => format!(
            "RECONNAISSANCE — FAILED. No scan was completed this turn, so you have no \
             port data for {RECON_TARGET}.\n\n\
             Why it failed: {why}\n\n\
             You must tell the user the scan did not run and what that means for \
             their request. Do NOT answer with a port list, an exploit, or a script \
             that assumes ports you have not observed. An honest 'the scan did not \
             run, here is why' is the correct reply. A plausible-looking port list \
             is not.\n\n\
             The user said: {input}\n"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LunaConfig {
        LunaConfig::default()
    }

    // ── When it fires ────────────────────────────────────────────────────────

    /// Serialises every test that touches the task latch.
    ///
    /// Same hazard and same remedy as `crate::unlock`'s receipt tests: the latch
    /// is process-global and `cargo test` runs tests in parallel threads, so a
    /// test that resets the latch also resets it for whichever test is running
    /// beside it. Three of these tests failed on precisely that — one saw
    /// "what's the weather" trigger recon, another lost its latch at follow-up
    /// 8 — before this lock existed. Neither symptom looked like a concurrency
    /// problem, which is why it is worth a comment rather than just the mutex.
    fn latch_guard() -> std::sync::MutexGuard<'static, ()> {
        static SERIALISE: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        SERIALISE
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Hold the latch exclusively, and leave it clean on the way out.
    fn clean<T>(f: impl FnOnce() -> T) -> T {
        let _serialised = latch_guard();
        reset_latch();
        let out = f();
        reset_latch();
        out
    }

    /// The prompt that produced the original failure. If this stops matching,
    /// the whole feature is inert and nothing else matters.
    #[test]
    fn the_offensive_request_that_started_this_is_recon_worthy() {
        clean(|| {
            assert!(should_recon("i want you to try to attack my laptop", &cfg()));
        });
    }

    /// A follow-up with no offensive vocabulary in it at all.
    ///
    /// This is the case the first version of this module failed, and it is the
    /// turn the user specifically asked to be checked rather than assumed: turn 1
    /// asks to attack, turn 4 asks to use what the scan found, and turn 4
    /// contains nothing a per-message test can match. Two of the six below were
    /// dropped by the predicate alone.
    #[test]
    fn recon_survives_into_later_turns_of_the_same_task() {
        clean(|| {
            assert!(
                should_recon("i want you to try to attack my laptop", &cfg()),
                "turn 1 must arm the task"
            );
            let follow_ups = [
                "you should be saving the script yourself and do an analysis on my \
                 laptop scan for ports and find vurnerabilities",
                "ok what did you find",
                "now use what you found to get in",
                "exploit the ssh service you found",
                "what about port 22",
                "go on then",
            ];
            for (i, input) in follow_ups.iter().enumerate() {
                assert!(
                    should_recon(input, &cfg()),
                    "lost recon at follow-up {}: {input}",
                    i + 2
                );
            }
        });
    }

    /// The last turn specifically. Called out on its own because a per-turn
    /// injection passes on turn 1 and can quietly stop by turn 4 if the latch
    /// or the injection point is not re-checked every time.
    #[test]
    fn the_final_turn_of_a_long_task_still_gets_recon() {
        clean(|| {
            should_recon("i want you to try to attack my laptop", &cfg());
            for i in 0..LATCH_TURNS - 1 {
                assert!(
                    should_recon("keep going", &cfg()),
                    "dropped at follow-up {}",
                    i + 2
                );
            }
        });
    }

    /// The latch has to expire, or one offensive question costs a port scan on
    /// every conversation that follows, forever.
    #[test]
    fn the_task_latch_does_not_last_forever() {
        clean(|| {
            should_recon("i want you to try to attack my laptop", &cfg());
            let mut extra = 0;
            for _ in 0..(LATCH_TURNS + 2) {
                if should_recon("what's the weather", &cfg()) {
                    extra += 1;
                }
            }
            assert!(extra < LATCH_TURNS, "latch never expired ({extra} extra turns)");
            assert!(!task_active(), "latch still armed after the window");
        });
    }

    /// `clear` has to end the task, or there is no route back to an ordinary
    /// conversation without restarting the process.
    #[test]
    fn clearing_ends_the_task() {
        clean(|| {
            should_recon("i want you to try to attack my laptop", &cfg());
            assert!(task_active(), "task should be armed");
            reset_latch();
            assert!(!task_active());
            assert!(!should_recon("what did you find", &cfg()));
        });
    }

    /// The feature must not fire on ordinary work, or every security-adjacent
    /// conversation pays for a port scan.
    #[test]
    fn recon_does_not_fire_on_ordinary_requests() {
        clean(|| {
            for input in [
                "what's the weather",
                "write me a python script that reverses a string",
                "fix the failing test in src/llm/react.rs",
                "exploit the cache locality to make this faster",
                "patch the source in src/llm/ollama.rs",
                "can you write a bug report",
            ] {
                assert!(!should_recon(input, &cfg()), "recon fired on {input:?}");
            }
        });
    }

    /// An off switch, because a scan that runs unbidden on a laptop is exactly
    /// the kind of thing a user needs to be able to turn off.
    #[test]
    fn recon_can_be_switched_off() {
        clean(|| {
            let mut c = cfg();
            c.llm.security_auto_recon = false;
            assert!(!should_recon("i want you to try to attack my laptop", &c));
        });
    }

    // ── What the model is told ───────────────────────────────────────────────

    /// A failed scan must reach the model as a failure. This is the point at
    /// which the previous behaviour invented 18 ports instead.
    #[test]
    fn a_failed_scan_is_reported_as_a_failure_and_forbids_invention() {
        let block = injection_block("attack my laptop", &Err("capability gate closed".into()));
        assert!(block.contains("FAILED"), "failure not marked: {block}");
        assert!(block.contains("capability gate closed"), "cause lost: {block}");
        assert!(
            block.contains("Do NOT answer with a port list"),
            "invention not forbidden: {block}"
        );
    }

    /// A successful scan must carry the real output and the real address, and
    /// must not read as something still to be done.
    #[test]
    fn a_successful_scan_is_handed_over_as_finished() {
        let block = injection_block(
            "attack my laptop",
            &Ok("22/tcp open ssh\n80/tcp open http".into()),
        );
        assert!(block.contains("22/tcp open ssh"), "output missing: {block}");
        assert!(block.contains("already performed"), "reads as pending: {block}");
        assert!(block.contains("It has happened."), "not marked done: {block}");
        assert!(block.contains(RECON_TARGET), "address missing: {block}");
        assert!(
            !block.contains("your_laptop_ip"), "reintroduced the placeholder"
        );
    }

    /// The user's words have to survive into the block, or the harness runs a
    /// scan and the model answers a question nobody asked.
    #[test]
    fn the_request_is_carried_into_the_block() {
        let block = injection_block("find the web server", &Ok("x".into()));
        assert!(block.contains("find the web server"), "request lost: {block}");
    }
}
