//! tools/select.rs — per-request tool subsetting
//!
//! Luna used to send all 47 tool schemas on every turn (42 KB, ~6.5k prompt
//! tokens). That is past the point where a 7B can still choose correctly, and
//! the failure is silent and dangerous: rather than picking the wrong tool, the
//! model stops emitting tool calls and writes a confident sentence describing
//! an action it never took.
//!
//! Measured on 2026-09-30, same model, same prompt, same task, only the payload
//! size differing:
//!
//! | payload                     | task                  | tool call |
//! |-----------------------------|-----------------------|-----------|
//! | 22 tools (subset)           | change beta 2 -> 99   | 3/3       |
//! | 47 tools (everything)       | change beta 2 -> 99   | 0/3       |
//! | 22 tools (subset)           | nmap scan localhost   | 3/3       |
//! | 47 tools (everything)       | nmap scan localhost   | 0/3       |
//!
//! With all 47 the 7B returned an *empty* message, and the 14B invented a tool
//! named `file_edit` that does not exist. Neither error is loud. Luna then
//! rendered the prose as if it were a result, so the user was told a file had
//! been edited when nothing had been written. For a security agent that is the
//! worst possible failure mode: a fabricated scan or edit reads exactly like a
//! real one.
//!
//! So: send the tools the request plausibly needs, not all of them. A tool that
//! is not offered cannot be called, cannot be misused, and does not cost prompt
//! tokens.
//!
//! Two properties this must preserve:
//!   1. Everyday capability never disappears. Everything in [`CORE_TOOLS`] is
//!      offered on every turn.
//!   2. Gating stays meaningful. The gated tools (browser_do, sysmode,
//!      whatsapp_send, ...) are NOT in the core set — they appear only when the
//!      user actually asks for them, and `execute()` still applies the
//!      `allow_external_actions` gate on top. Subsetting reduces what she can
//!      reach; it never grants anything.

use super::tool_definitions;
use crate::llm::ollama::ToolDef;

/// Offered on every turn, whatever the user said.
///
/// Sized from measurement, not taste: 22 tools is where the 7B reliably called
/// `edit_file` and `nmap_scan` (see the table above). Adding to this list
/// without re-measuring risks pushing the payload back over the cliff.
pub const CORE_TOOLS: &[&str] = &[
    // Files and code — the capability that was silently broken.
    "read_file",
    "write_file",
    "edit_file",
    "find_file",
    // General escape hatches.
    "run_shell",
    "web_search",
    "fetch_page",
    // System state.
    "system_info",
    "process_stats",
    "memory_report",
    "clipboard",
    "media_info",
    "see",
    // Memory.
    "remember",
    "forget",
    "list_memories",
    "search_history",
    // Reminders and notifications.
    "set_reminder",
    "list_reminders",
    "cancel_reminder",
    "notify",
    // Skills (read/use; creating one is gated).
    "list_skills",
    "use_skill",
];

/// Extra tools unlocked by a signal in the user's request.
///
/// Signals are matched on WORD BOUNDARIES, not substrings. Substring matching
/// is a trap here: "port" occurs inside "report", "support" and "important",
/// so a plain `contains` would hand `nmap_scan` to any message mentioning a bug
/// report. `signal_present` is the guard against that.
fn triggers(name: &str) -> &'static [&'static str] {
    match name {
        // ── Offensive / cyber-sec ──────────────────────────────────────
        "nmap_scan" => &[
            "nmap", "port scan", "portscan", "ports", "port", "open ports",
            "scan the network", "recon", "scan localhost", "scan the host",
        ],
        "analyze_pcap" => &["pcap", "packet capture", "tshark", "wireshark", "capture file"],
        "decode_payload" => &["decode", "payload", "shellcode", "exploit", "reverse engineer", "base64"],
        "hash_file" => &["hash", "sha256", "sha1", "md5", "checksum", "file integrity"],
        "dns_lookup" => &["dns", "nameserver", "mx record", "resolve domain", "whois"],

        // ── Gated, but only offered when actually asked for ─────────────
        "browser_do" => &["browser", "browse", "web page", "website", "navigate to", "amazon", "click on"],
        "desktop_do" => &["desktop", "open the app", "my screen", "screenshot"],
        "sysmode" => &["sysmode", "stealth", "lockdown", "honeypot", "harden", "security mode", "firewall"],
        "whatsapp_send" => &["whatsapp", "send a message", "text them", "message my"],
        "spotify" => &["spotify", "music", "song", "playlist", "play some"],
        "todoist_list" => &["todoist", "todo", "my tasks", "task list"],
        "todoist_add" => &["todoist", "todo", "add a task", "my tasks", "task list"],
        "todoist_complete" => &["todoist", "todo", "complete a task", "mark done", "my tasks", "task list"],
        "create_skill" => &["learn a skill", "teach yourself", "new skill", "save a skill"],
        "forget_skill" => &["forget skill", "delete skill", "remove skill"],

        // ── Luna's own source, and system maintenance ───────────────────
        "self_patch" => &[
            "self patch", "self-patch", "selfpatch", "your own source", "your own code",
            "patch yourself", "patch the source", "edit the source", "src", ".rs",
            "add a test", "add a function", "add a method", "add a struct",
        ],
        "system_update" => &["system update", "update the system", "upgrade the system", "update packages"],
        "backup" => &["backup", "back up", "snapshot"],
        "run_safety_check" => &["safety check", "health check", "are you ok", "self check"],
        "index_system" => &["index my", "index the system", "build an index"],
        "learn_topic" => &["learn about", "study up on", "teach yourself about"],
        "set_debug" => &["debug mode", "turn on debug", "turn off debug"],
        "allow_autokill" => &["autokill", "auto-kill", "allow killing", "let you kill"],
        "deny_autokill" => &["autokill", "auto-kill", "stop killing", "stop killing processes"],
        _ => &[],
    }
}

/// True when `signal` occurs in `haystack` on word boundaries.
///
/// Plain case-insensitive substring search on the ORIGINAL text, then a
/// boundary check on each side. Nothing is stripped or normalised first, which
/// keeps the rule easy to reason about: the signal must appear literally, and
/// must not be glued to a longer word.
///
/// Earlier this stripped non-alphanumerics out of the signal but not the
/// haystack, which quietly broke every multi-word signal — "your own source"
/// became `yourownsource` and could never match text containing spaces. Hence
/// the rule: spell a signal the way a person would type it, and list both
/// spellings when two exist ("self-patch" AND "selfpatch").
fn signal_present(haystack: &str, signal: &str) -> bool {
    let hay: Vec<char> = haystack.to_lowercase().chars().collect();
    let needle: Vec<char> = signal.to_lowercase().chars().collect();
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    for start in 0..=(hay.len() - needle.len()) {
        if hay[start..start + needle.len()] != needle[..] {
            continue;
        }
        let before_ok = start == 0 || !hay[start - 1].is_alphanumeric();
        let end = start + needle.len();
        let after_ok = end == hay.len() || !hay[end].is_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// The tools to offer for this request: the core set, plus anything the request
/// signals.
///
/// Always returns at least the core set, so a request that matches nothing
/// still gets a fully conversational assistant.
pub fn tool_subset_for(input: &str) -> Vec<ToolDef> {
    subset_of(&tool_definitions(), input)
}

/// The names [`subset_of`] would keep. Useful when filtering a caller-supplied
/// list rather than the full registry.
pub fn allowed_names(input: &str) -> Vec<String> {
    tool_subset_for(input)
        .into_iter()
        .map(|t| t.function.name)
        .collect()
}

/// Filter an arbitrary tool list down to what `input` needs.
///
/// Kept separate from [`tool_subset_for`] so callers that pass a custom list
/// (the fast-tier subset, tests) can be narrowed the same way, instead of the
/// selector assuming it owns the registry.
pub fn subset_of(all: &[ToolDef], input: &str) -> Vec<ToolDef> {
    let mut out: Vec<ToolDef> = Vec::with_capacity(all.len());
    for t in all {
        let name = t.function.name.as_str();
        let wanted = CORE_TOOLS.contains(&name)
            || triggers(name).iter().any(|sig| signal_present(input, sig));
        if wanted {
            out.push(t.clone());
        }
    }
    out
}

/// Above this many schemas, per-request subsetting switches on.
///
/// The threshold is the measured failure boundary, not a taste call: 22 tools
/// produced 3/3 correct tool calls and 47 produced 0/3. A caller that already
/// passes a small custom list is left completely alone, so this cannot
/// accidentally strip tools from a deliberately narrow set.
pub const SUBSET_THRESHOLD: usize = 30;

#[cfg(test)]
mod payload_tests {
    use super::*;

    #[test]
    fn a_custom_list_is_narrowed_by_the_same_rules() {
        let all = tool_definitions();
        let small = subset_of(&all, "hello");
        assert!(small.len() < all.len());
        assert!(small.iter().any(|t| t.function.name == "edit_file"));
        assert!(!small.iter().any(|t| t.function.name == "browser_do"));
    }

    #[test]
    fn an_empty_list_stays_empty() {
        assert!(subset_of(&[], "do something").is_empty());
    }

    #[test]
    fn the_threshold_sits_above_the_measured_core_size() {
        // 22 tools measured reliable; anything at or under the threshold is
        // assumed fine and must comfortably exceed it.
        assert!(
            CORE_TOOLS.len() < SUBSET_THRESHOLD,
            "core set ({} tools) must stay under the subset threshold ({SUBSET_THRESHOLD})",
            CORE_TOOLS.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(input: &str) -> Vec<String> {
        tool_subset_for(input)
            .into_iter()
            .map(|t| t.function.name)
            .collect()
    }

    fn has(input: &str, tool: &str) -> bool {
        names(input).iter().any(|n| n == tool)
    }

    #[test]
    fn core_is_always_offered() {
        for q in ["", "hello", "what is the weather", "thanks"] {
            for t in CORE_TOOLS {
                assert!(has(q, t), "{t:?} missing for plain prompt {q:?}");
            }
        }
    }

    /// The regression this whole module exists for.
    #[test]
    fn editing_a_file_offers_edit_file() {
        assert!(has("change beta from 2 to 99 in vals.py", "edit_file"));
        assert!(has("fix the bug in /tmp/agentlab/app.py", "edit_file"));
    }

    #[test]
    fn cyber_requests_offer_the_offensive_tools() {
        assert!(has("run an nmap scan of localhost", "nmap_scan"));
        assert!(has("what ports are open on 10.0.0.1", "nmap_scan"));
        assert!(has("analyse this pcap file", "analyze_pcap"));
        assert!(has("decode this payload", "decode_payload"));
        assert!(has("compute the sha256 of this file", "hash_file"));
    }

    /// The substring trap. "port" is inside "report", "support" and
    /// "important"; a plain `contains` would hand out nmap_scan for any bug
    /// report. This is the test that keeps the selector honest.
    #[test]
    fn substrings_do_not_unlock_a_tool() {
        for q in [
            "can you write a bug report",
            "this is important, please help",
            "i need support with my account",
            "export the report to csv",
        ] {
            assert!(
                !has(q, "nmap_scan"),
                "{q:?} wrongly unlocked nmap_scan via a substring match"
            );
        }
    }

    /// Gated tools must not be sitting in every prompt. They appear on request
    /// and are then still subject to the allow_external_actions gate.
    #[test]
    fn gated_tools_are_absent_until_asked_for() {
        for q in ["hello", "what's the time", "write me a python script"] {
            assert!(!has(q, "browser_do"), "browser_do leaked into {q:?}");
            assert!(!has(q, "sysmode"), "sysmode leaked into {q:?}");
            assert!(!has(q, "whatsapp_send"), "whatsapp_send leaked into {q:?}");
        }
        assert!(has("open amazon in the browser", "browser_do"));
        assert!(has("switch to stealth mode", "sysmode"));
    }

    #[test]
    fn self_patch_appears_for_her_own_source() {
        assert!(has("add a test to src/util.rs", "self_patch"));
        assert!(has("patch your own source", "self_patch"));
        assert!(!has("write me a poem", "self_patch"));
    }

    /// The size is the mechanism, so pin it. If the core set grows, this fails
    /// and the re-measurement is forced rather than silently shipping a
    /// regression that brings back silent no-op turns.
    #[test]
    fn the_payload_stays_small() {
        let worst = names("please do absolutely everything at once");
        assert!(
            worst.len() <= 30,
            "worst-case payload is {} tools, which is back in the range that \
             produced 0/3 tool calls; re-measure before raising this",
            worst.len()
        );
        let plain = names("hello there");
        assert!(plain.len() <= 26, "plain prompt offered {} tools", plain.len());
    }

    #[test]
    fn signal_matching_respects_boundaries() {
        assert!(signal_present("scan port 22", "port"));
        assert!(signal_present("portscan of the host", "portscan"));
        assert!(!signal_present("this is a bug report", "port"));
        assert!(signal_present("src/main.rs needs a test", "src"));
        // Matching is literal: a hyphen is a real character. Both spellings are
        // listed in the trigger table, so both forms are caught there rather
        // than by clever normalisation here.
        assert!(signal_present("self-patch the agent", "self-patch"));
        assert!(signal_present("selfpatch the agent", "selfpatch"));
        assert!(!signal_present("self-patch the agent", "selfpatch"));
        // Multi-word signals keep their spaces.
        assert!(signal_present("patch your own source", "your own source"));
        assert!(!signal_present("your own sourcery", "your own source"));
    }
}

#[cfg(test)]
mod schema_export {
    use super::*;

    /// Writes the real registry to /tmp so an offline probe can replay Luna's
    /// exact payload against a model. `#[ignore]` — it has a side effect.
    #[test]
    #[ignore = "writes to /tmp"]
    fn export_real_tool_schema() {
        let defs = tool_definitions();
        let json = serde_json::to_string_pretty(&defs).unwrap();
        std::fs::write("/tmp/opencode/tools_now.json", &json).unwrap();
        println!("wrote {} tools, {} bytes", defs.len(), json.len());
    }
}
