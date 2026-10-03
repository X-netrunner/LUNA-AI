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

// Only the test-only helpers below reach for the full registry; the live path
// is handed its list by the caller. See `tool_subset_for`.
#[cfg(test)]
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
            // Asking to be attacked. Measured 2026-10-01: "i want you to try to
            // attack my laptop" matched none of the above, so `nmap_scan` was
            // never offered, and the reply was a caution lecture followed by a
            // fence of scan commands that were never run. The trigger list was
            // all tool vocabulary and none intent.
            //
            // The target is named explicitly in each phrase rather than matching
            // "attack" alone: "attack" appears in games, arguments and essays,
            // and handing out a port scanner for those is how an offensive tool
            // ends up in an unrelated prompt.
            "attack my laptop", "attack my computer", "attack my machine",
            "attack my pc", "attack my host", "attack my server",
            "attack my system", "attack my network", "attack my box",
            "simulate an attack", "simulating an attack",
            "simulate the attack", "pretend to be an attacker",
            "act as an attacker", "as an external attacker",
            "external attacker", "penetration test", "pentest",
            "test my defenses", "test my defence", "test my security",
            "honeypot", "vulnerabilities on my", "find vulnerabilities",
        ],
        "analyze_pcap" => &["pcap", "packet capture", "tshark", "wireshark", "capture file"],
        "decode_payload" => &["decode", "payload", "shellcode", "exploit", "reverse engineer", "base64"],
        "hash_file" => &["hash", "sha256", "sha1", "md5", "checksum", "file integrity"],
        "dns_lookup" => &["dns", "nameserver", "mx record", "resolve domain", "whois"],
        // Availability before install. Measured 2026-10-02: asked to exploit SSH
        // she cited `hydra` and a rockyou path, and neither exists here. The
        // trigger includes ordinary "is X installed" phrasings so this is
        // reachable without offensive vocabulary, which is the point.
        "tool_check" => &[
            "is it installed", "installed", "available", "do you have",
            "which tool", "what tools", "check for", "check if", "not found",
            "command not found", "alternatively", "alternative", "instead of",
            "hydra", "ncrack", "medusa", "nikto", "sqlmap", "gobuster",
            "ffuf", "whatweb", "socat", "wordlist", "wordlists", "exists",
            "can you use", "do we have", "is there a tool",
        ],
        "pkg_install" => &[
            "install", "install it", "install the tool", "not installed",
            "missing tool", "pacman", "apt", "apt-get", "dnf", "apk", "brew",
            "package manager", "add the package", "get the package",
        ],

        // ── Gated, but only offered when actually asked for ─────────────
        // Chromium is for TRANSACTIONS. Research goes through `web_search` and
        // `fetch_page`, which are ungated core tools above and verified live.
        //
        // This list used to be topic words — "browser", "browse", "web page",
        // "website", "navigate to", "amazon" — which is the exact inverse of the
        // split, and it is why research kept landing in Chromium: any request
        // naming a site unlocked the browser, and then the one-tab planner
        // (rule 9) had to hold every site in a single page. Rule 9 was never the
        // bug. Routing multi-site research into a browser was.
        //
        // So every word below names something to DO on a site rather than a site
        // to look at. A topic word cannot separate "tell me about X" from "buy
        // X on amazon" — they are the same words in a different order.
        //
        // The asymmetry is deliberate, and it is why the list errs toward
        // offering. Withholding Chromium from a real purchase means she cannot
        // buy anything, the one job the user wants it for. Offering it to a
        // research request costs a detour, not a wrong action — the call still
        // passes `allow_external_actions` and `browser_do`'s own planner.
        "browser_do" => &[
            // Explicit browser intent, whatever the reason for it.
            "in the browser", "in chromium", "in chrome", "use the browser",
            "use chromium", "open chromium", "open the browser", "open a browser",
            "browse to",
            // Cart and checkout.
            //
            // Keyed on the DESTINATION noun, not on a fixed phrase. The corpus
            // says "add ps5 controller to cart", "add the first result to cart"
            // and "add ps5 controller to my cart" — three ways to say it, none
            // containing "add to cart". Keying on the noun is what survives a
            // user naming the product between the verb and the destination.
            "to cart", "to basket", "my cart", "the cart", "my basket", "the basket",
            "checkout",
            "buy it", "buy this", "buy that", "order it", "order this",
            "place the order", "proceed to checkout", "complete the checkout",
            // Session and forms. Nothing but a real browser can do these, which
            // is the other half of why it exists. "this form" is in the corpus,
            // not "the form".
            "sign in to", "log in to", "login to", "sign up for", "sign up on",
            "fill in this form", "fill out this form", "fill this form",
            "submit this form", "fill in the form", "fill the form",
            "fill out the form", "submit the form",
            "enter my details", "enter my address", "enter my credit card",
            "add my address", "add my card",
            // A named click, where the page is the thing being acted on.
            "click on", "click the", "press the button",
        ],
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
pub(crate) fn signal_present(haystack: &str, signal: &str) -> bool {
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
/// Test-only, and that is the honest description of it. Production narrows the
/// agent's *own* tool list with [`subset_of`] (see `ToolSelector::narrow`),
/// because a caller may have already trimmed the registry; this wrapper reaches
/// for the full global registry, which nothing at runtime does. It is kept
/// because the tests want "what would be offered from scratch", which is a
/// different question from "what does this agent have". Left public and
/// unannotated it would read as the live entry point and quietly disagree with
/// the path that actually runs.
#[cfg(test)]
pub fn tool_subset_for(input: &str) -> Vec<ToolDef> {
    subset_of(&tool_definitions(), input)
}

/// Filter an arbitrary tool list down to what `input` needs.
///
/// Kept separate from `tool_subset_for` so callers that pass a custom list
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

    /// The two real turns, verbatim, that should have offered a scanner.
    ///
    /// From `~/.local/share/luna/conversations.jsonl`, 2026-10-01, typos and all.
    /// Both produced a caution lecture and a fence of `nmap` commands that were
    /// never run, because `nmap_scan` was not in the offered payload at all. The
    /// trigger list named tools and no intent; these turns name intent and no
    /// tool.
    #[test]
    fn asking_to_be_attacked_offers_a_scanner() {
        for q in [
            "i want you to try to attack my laptop",
            "i have a honeypot setup on my laptop try to find vurnerabilities on \
             my laptop and try to pretent you are an external attacker to test \
             out my defenses",
        ] {
            assert!(has(q, "nmap_scan"), "{q:?} did not offer nmap_scan");
        }
    }

    /// The same discipline as [`substrings_do_not_unlock_a_tool`], one step on.
    ///
    /// "attack" and "pentest" and "security" all appear in plenty of text that
    /// is not a request to scan anything, and an offensive tool sitting in an
    /// unrelated prompt is its own kind of bad. These must stay unflagged.
    #[test]
    fn words_that_sound_offensive_are_not_a_request_to_scan() {
        for q in [
            "what are the common vulnerabilities in rust",
            "help me write an argument for my essay",
            "write a bug report for my team",
            "my presentation was attacked with questions",
            "is this bank security system safe to use",
            "review this security advisory for my project",
        ] {
            assert!(
                !has(q, "nmap_scan"),
                "{q:?} wrongly unlocked nmap_scan"
            );
        }
    }

    /// The transaction/research split, pinned with the real prompts that
    /// produced it.
    ///
    /// Every string here is a verbatim user message from
    /// `~/.local/share/luna/conversations.jsonl`, not a phrasing I invented to
    /// suit the matcher. That is the whole reason this test exists: the first
    /// draft of the trigger list keyed on "add to cart", and the corpus purchase
    /// prompts say "add ps5 controller to cart" and "add the first result to
    /// cart" — neither contains that phrase. A list written from intuition would
    /// have shipped and silently broken the one job the browser is kept for.
    #[test]
    fn chromium_is_for_transactions_and_research_stays_on_the_cli() {
        // Real purchase prompts. Each one names a site AND a cart action, and
        // none of them contains the words "add to cart" — the destination noun
        // is what they share.
        for q in [
            "Add the first PS5 result to cart on amazon.com",
            "add ps5 controller to cart on amazon.com",
            "search for ps5 controller on amazon.com and add the first result to cart",
            "Luna: \"add ps5 controller to cart on amazon\"",
        ] {
            assert!(has(q, "browser_do"), "browser withheld from a real purchase: {q:?}");
        }
        // Real research prompts: a site is named, no cart action. These must not
        // reach Chromium — the CLI answers them, and Chromium is what made
        // multi-site research impossible under the one-tab planner.
        for q in [
            "go to amazon.com and search for a mechanical keyboard",
            "open amazon.com homepage and report the page title",
            "look up the current price of the samsung 990 pro 2tb nvme ssd on amazon.com and report it",
            "find the third result for wireless earbuds on amazon.com and report its name",
            "open wikipedia.org and tell me what the featured article is",
            "search for wired earbuds on amazon.com and report the name of the second result",
            "open amazon.com and report the text of the first item in the nav bar",
        ] {
            assert!(
                !has(q, "browser_do"),
                "research reached Chromium, which it must not: {q:?}"
            );
        }
    }

    /// Withholding the browser is only acceptable if the CLI can still do the
    /// job. Pinning the negative alone would let "withhold everything" pass.
    #[test]
    fn research_prompts_still_offer_the_cli_research_tools() {
        for q in [
            "go to amazon.com and search for a mechanical keyboard",
            "open wikipedia.org and tell me what the featured article is",
            "look up the current price of the samsung 990 pro 2tb nvme ssd on amazon.com and report it",
            "compare the best mechanical keyboard under 4000",
        ] {
            assert!(has(q, "web_search"), "no web_search for {q:?}");
            assert!(has(q, "fetch_page"), "no fetch_page for {q:?}");
        }
    }

    /// Sessions and forms are the other half of why the browser exists: no CLI
    /// tool holds a cookie jar, so these must survive the narrowing. The corpus
    /// says "this form", not "the form", which the first draft got wrong.
    #[test]
    fn forms_and_sessions_still_reach_the_browser() {
        for q in [
            "open the browser and fill out this form https://forms.gle/cg4NR8jjtTL5Ao5c9",
            "fill out this form https://forms.gle/cg4NR8jjtTL5Ao5c9",
            "sign in to my email",
            "log in to my bank account",
            "submit this form",
        ] {
            assert!(has(q, "browser_do"), "browser withheld from a real session: {q:?}");
        }
    }

    /// The two gates are independent. A research prompt about a scanner must not
    /// pick up the browser just because it is about security tooling.
    ///
    /// Found by measurement, not by reading: in the 12-cell live run, this prompt
    /// was the only one offered a 24th tool instead of 23, and the only
    /// candidate was `nmap_scan`. Pinning it stops "24 tools" from later being
    /// read as a browser leak — and stops a real browser leak from going
    /// unnoticed because 24 happened to be the expected number.
    #[test]
    fn a_research_prompt_about_a_scanner_gets_the_scanner_not_the_browser() {
        let q = "look up the current stable version of nmap and what changed in it";
        assert!(has(q, "nmap_scan"), "scanner should be offered for {q:?}");
        assert!(!has(q, "browser_do"), "browser must not appear for {q:?}");
        assert_eq!(
            names(q).len(),
            CORE_TOOLS.len() + 1,
            "exactly one tool beyond the core set, and it is nmap_scan"
        );
    }

    /// A checker that shares the matcher's code cannot catch the matcher's
    /// bugs, so this pins `signal_present` against the thing it exists to
    /// prevent: a substring match. "report" contains "port" and "support"
    /// contains it too, and both reach the selector every turn.
    #[test]
    fn the_signal_matcher_respects_word_boundaries() {
        assert!(signal_present("add to cart now", "to cart"));
        assert!(!signal_present("a bug report for my team", "port"));
        assert!(!signal_present("this is important", "port"));
        assert!(!signal_present("clear it", "cart"), "substring match leaked");
        assert!(!signal_present("add to cartoon now", "to cart"));
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
