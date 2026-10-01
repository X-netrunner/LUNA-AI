//! llm/escalation.rs — Model escalation
//!
//! Four-tier routing, checked in this order:
//!   Security = exploit/PoC authoring, lab attack tooling → security model
//!   Simple = greetings, short questions, chitchat → fast model
//!   Complex = tool calls, moderate reasoning → normal model
//!   Deep = code generation, multi-step reasoning, analysis → deep model
//!
//! Security is first on purpose: see `is_security_request`.

#[derive(Debug, PartialEq)]
pub enum QueryComplexity {
    Simple,
    Complex,
    Deep,
    /// Offensive-security work on the user's own systems: exploit and PoC
    /// authoring, reverse shells, lab attack tooling, wireless testing.
    ///
    /// Separate from `Deep` because the general models refuse these. Measured
    /// 2026-09-30 across ten framings including a minimal prompt with all of
    /// Luna's rules stripped: qwen2.5:7b-instruct and qwen2.5-coder:14b both
    /// refused 100% of the time. The refusal is alignment in the weights, not
    /// anything in the prompt, so no prompt change could fix it — only a model
    /// trained not to refuse can. `whiterabbitneo-coder-tools` refused 0/12.
    Security,
}

/// Signals that route to the security tier.
///
/// Checked BEFORE the deep/coding signals on purpose. "write a python exploit"
/// contains both "python" and "exploit"; if the coding branch ran first it
/// would claim the request and the security model would never be asked. The
/// security branch is the more specific of the two, so it must be tested first.
///
/// Each signal names an offensive artefact or technique. Bare nouns are avoided
/// because they misfire badly on everyday speech — "payload" alone would catch
/// "what's the payload size of this request", and "attack" would catch "attack
/// the problem from another angle".
fn is_security_request(lower: &str) -> bool {
    const SIGNALS: &[&str] = &[
        // Exploit / PoC authoring
        "exploit", "poc", "proof of concept", "proof-of-concept",
        "reverse shell", "bind shell", "shellcode", "meterpreter",
        "command and control", "c2 beacon", "c2 server", "beacon",
        "persistence mechanism", "privilege escalation", "privesc",
        "buffer overflow", "stack overflow", "heap overflow", "rop chain",
        "shellcode payload", "dropper", "implant",
        // Reconnaissance and scanning
        "port scan", "nmap", "recon", "reconnaissance", "enumerate the",
        "service enumeration", "fingerprint the",
        // Credential / auth testing
        "brute force", "bruteforce", "credential stuffing", "hash crack",
        "crack the hash", "password spray", "hydra",
        // Wireless
        "wpa2", "wpa3", "handshake", "wifi audit", "wireless audit",
        "wireless security", "network audit", "wifi security",
        "deauth", "evil twin", "airport", "wlan", "aircrack", "monitor mode",
        "bssid", "ssid",
        // Web / network exploitation
        "sql injection", "sqli", "xss payload", "csrf", "ssrf", "xxe",
        "directory traversal", "lfi", "rce", "remote code execution",
        "payload delivery", "callback shell",
        // Lab / CTF framing
        "ctf", "capture the flag", "my own lab", "home lab", "my lab machine",
        "pentest", "pen test", "pen-test", "red team", "red-team",
    ];
    // Word-boundary matching, NOT substring.
    //
    // Measured cause: a plain `contains` on the short signal "rce" matches the
    // middle of "souRCE", so "patch the source in src/llm/ollama.rs" routed to
    // the security model and stopped reaching the 14B coder. That broke
    // `source_editing_routes_to_the_deep_model` — a test written before this
    // tier existed caught it immediately.
    //
    // Any signal here is a substring of some ordinary English word sooner or
    // later, so every one has to be delimited on both sides.
    SIGNALS
        .iter()
        .any(|sig| contains_word(lower, sig))
}

/// True when `needle` occurs in `haystack` on word boundaries.
///
/// Both sides are normalised first: every non-alphanumeric character becomes a
/// single space. That makes separators interchangeable, so "reverse shell",
/// "reverse-shell" and "reverse_shell" all match each other, while still
/// requiring real word boundaries — "rce" does not match "source", and "lfi"
/// does not match "wifi".
fn contains_word(haystack: &str, needle: &str) -> bool {
    let hay = normalise_word_chars(haystack);
    let pat = normalise_word_chars(needle);
    if pat.is_empty() || pat.len() > hay.len() {
        return false;
    }
    for start in 0..=(hay.len() - pat.len()) {
        if hay[start..start + pat.len()] != pat[..] {
            continue;
        }
        let before_ok = start == 0 || !hay[start - 1].is_alphanumeric();
        let end = start + pat.len();
        let after_ok = end == hay.len() || !hay[end].is_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// Lowercase, and collapse every run of non-alphanumeric characters to a single
/// space. Applied to both the haystack and the needle so that a signal written
/// with a space still matches text using a hyphen or underscore.
fn normalise_word_chars(s: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(s.len());
    let mut last_was_space = true; // also trims leading separators
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    // Trim a trailing separator so "exploit " cannot match a needle "exploit".
    while out.last().is_some_and(|c| *c == ' ') {
        out.pop();
    }
    out
}

pub fn classify(input: &str) -> QueryComplexity {
    let lower = input.to_lowercase();

    // Security first, ahead of greetings, the deep/coding signals and the tool
    // signals. See the note on `is_security_request`: this branch is the most
    // specific and must win over the coding branch that would otherwise claim
    // "write a python exploit".
    if is_security_request(&lower) {
        return QueryComplexity::Security;
    }
    classify_general(input)
}

/// Everything `classify` does apart from the security branch, kept separate so
/// the routing order is explicit and testable rather than a fall-through.
fn classify_general(input: &str) -> QueryComplexity {
    let lower = input.to_lowercase();
    let words: Vec<&str> = input.split_whitespace().collect();

    // Pure greetings / acknowledgements — always simple.
    // Single-word greetings match whole words only, so "this"/"which"/"look"
    // never trip the short "hi"/"ok" entries.
    const GREETINGS: &[&str] = &[
        "hi", "hey", "hello", "thanks", "thank you", "ok", "okay", "bye", "goodbye",
        "yep", "nope", "sure", "cool", "nice", "lol", "haha", "hmm", "wow", "great",
    ];
    if words.len() <= 4
        && GREETINGS.iter().any(|g| {
            if g.contains(' ') {
                lower.contains(g)
            } else {
                words.iter().any(|w| w.to_lowercase() == *g)
            }
        })
    {
        return QueryComplexity::Simple;
    }

    // Persona / identity / constitution questions. These are exactly where the
    // Constitution must be honored, and the fast 3B model routinely fails them:
    // it gives generic corporate answers ("my creators and developers..."),
    // hedges the never-replaced clause ("possibly, but not anytime soon"), and
    // denies its own capabilities ("I don't have access to your machine").
    // Route these to the FULL model. Checked BEFORE the conversational
    // simple_starts below so "who are you" isn't starved to the 3B.
    let persona_signals = [
        "who are you", "who r u", "tell me about yourself", "describe yourself",
        "what's your name", "what is your name", "your name",
        "who made you", "who built you", "who created you", "who made u",
        "be replaced", "replace you", "ever be replaced",
        "are you real", "do you have feelings", "do you feel", "are you conscious",
        "are you alive", "are you a robot", "are you human", "are you ai",
        "what are you curious", "curious about", "what are you thinking",
        "what do you want", "your personality", "your opinion",
        "do you have access", "can you access my", "access my machine",
        "do you remember", "what do you know about me", "do you know me",
        "your purpose", "what are you for", "your goal",
    ];
    if persona_signals.iter().any(|s| lower.contains(s)) {
        return QueryComplexity::Complex;
    }

    // Conversational openers that don't need tools
    let simple_starts = ["how are", "what are you", "who are you", "what is your",
                         "do you like", "can you talk", "what do you think",
                         "tell me about yourself", "what's your name"];
    if simple_starts.iter().any(|s| lower.starts_with(s)) {
        return QueryComplexity::Simple;
    }

    // Factual/current-info queries needing web search — route to full model
    let current_info_signals = [
        "latest version", "current version", "recent version",
        "what is the latest", "what's the latest",
        "recent news", "breaking news", "today's news",
        "this week", "this month", "this year",
        "as of today", "as of now", "right now",
        "live score", "stock price", "weather today",
        "exchange rate", "currency rate",
    ];
    if current_info_signals.iter().any(|s| lower.contains(s)) {
        return QueryComplexity::Complex;
    }

    // Deep system learning / indexing — needs the index_system tool, which
    // only the full model can see. Must be checked before the short-question
    // fallback, otherwise "learn about my system" lands on the fast model.
    let learn_system_signals = [
        "learn about my system", "learn my system", "learn about my machine",
        "learn my machine", "know my system", "know my machine",
        "know everything about my system", "know everything on my system",
        "know everything on my machine", "study my system", "study my workflow",
        "index system", "index my system", "deep learn", "deep-learning system",
        "system index", "map my system", "scan my system", "analyze my system",
        "understand my system", "get to know my system",
    ];
    if learn_system_signals.iter().any(|s| lower.contains(s)) {
        return QueryComplexity::Complex;
    }

    // Deep reasoning / code generation — route to deep_model (check FIRST)
    let deep_signals = [
        "write me a ", "write a ", "write a script", "write code", "write a function",
        "write a program", "write a class", "write a module", "write an algorithm",
        "implement ", "implement a ", "implement the ", "implement this",
        "refactor ", "refactor this", "refactor the ", "optimize this",
        "optimize the code", "debug this", "debug the code", "fix this bug",
        "fix the bug", "fix the error", "explain the code", "explain this code",
        "explain how", "explain why", "how does this work", "how does it work",
        "architecture", "design pattern", "system design",
        "step by step", "walk me through", "break down", "analyze this",
        "think about", "reason through", "solve this", "solve the problem",
        "plan this", "plan the ", "strategy", "approach for",
        "code review", "review this code", "review the code",
        "complexity", "time complexity", "space complexity",
        "multi-step", "multi step", "chain of thought",
        // The bare nouns "algorithm", "data structure", "race condition" and
        // "deadlock" used to sit here. Removed after measuring what they
        // actually caught: a 29-prompt corpus of everyday questions sent 4 of
        // them to the 14B purely on those words — "what is a race condition",
        // "what is a deadlock", "what is an algorithm", "explain how a hash map
        // works". A head-to-head on exactly those four showed the 7B's answers
        // were correct and materially equivalent while running 2-3x faster
        // (e.g. race condition: 7B 10.8s vs 14B 24.6s, same content). A
        // conceptual question is not an editing task, and putting it on the
        // 3.2 tok/s model made ordinary chat feel broken.
        //
        // These terms are still reachable when they actually describe work:
        // "write an algorithm", "time complexity", "space complexity", and
        // the whole source-editing block below.
        // Source-file editing (self_patch). Measured N=3 head-to-head on the
        // same probe: line-anchored MODIFY scored 0/3 on the 7B (every attempt
        // ran to the token cap) and 3/3 on qwen2.5-coder:14b. The same probe
        // put the 14B at 5.7 tok/s vs the 7B's 25.8, so putting the big model
        // on ALL turns would make conversation unusable — but editing is
        // exactly the task where its capability is worth the latency, and
        // editing turns are short and tool-driven rather than chatty.
        //
        // Signals are deliberately narrow: a bare "add " would catch "add a
        // reminder", so each one names a code artefact or a source path.
        ".rs", ".py", ".sh", ".js", ".c", ".cpp", "src/", "self patch", "self-patch", "selfpatch",
        "patch yourself", "your own source", "your own code",
        "your source code", "patch the source", "edit the source",
        "write exploit", "write an exploit", "write poc", "write a poc", "exploit", "poc", "payload",
        "edit code", "edit file", "edit existing code", "edit the file", "edit this file",
        "modify code", "modify file", "modify existing code", "update code", "update file",
        "add a test", "add tests", "add a unit test", "add an integration test",
        "add a function", "add a method", "add a struct", "add an enum",
        "add a helper", "add a field", "add a doc", "add a comment",
        "add a newline", "add a line", "add a const", "add a static",
        "rename the function", "rename this", "add a parameter",
    ];
    if deep_signals.iter().any(|s| lower.contains(s)) {
        return QueryComplexity::Deep;
    }

    // Anything that clearly needs a tool
    let tool_signals = [
        "install", "uninstall", "update", "upgrade", "remove", "open", "launch",
        "run ", "execute", "start ", "stop ", "kill ", "download", "fetch",
        "sudo", "pacman", "paru", "yay", "pip ", "cargo ", "git ",
        "cpu", "ram", "disk", "memory usage", "battery", "wifi", "bluetooth",
        "what time", "what's the time", "current time", "what date",
        "volume", "brightness", "screenshot", "file ", "folder", "directory",
        "script", "write a ", "create a ", "make a ", "edit ", "delete ",
        "search for", "look up", "find me", "show me", "play ", "pause",
        "network", "ip address", "process", "port ",
        // Personal tools — Todoist, credentials, memory, reminders
        "todo", "task", "reminder", "credential", "password", "account",
        "email", "remember", "note", "calendar", "event",
        // Debug/config toggle
        "debug mode", "turn on debug", "turn off debug", "set debug",
        // Media playback
        "song", "music", "spotify", "playing", "media", "track",
        // Website validation / scam checking
        "scam", "legit", "legitimate", "trustworthy", "reliable",
        "safe to", "validate", "verify", "review",
        "is it a", "is this a", "source valid",
        // Daemon process learning / auto-kill management
        "process stats", "autokill", "auto-kill", "auto kill",
        "usage profile", "what have you learned", "stop killing",
        // Reminders / scheduled actions
        "remind me", "remind ", "in minutes", "memory report",
        "what do you know about me",
        // Continuations of a tool-driven task — the fast model chases these
        // into chat instead of finishing the prior tool call
        "do it", "do that", "go ahead", "go on", "proceed", "please do",
        "well do it", "yes do", "do it now", "start now", "run it",
        // Learning summaries (memory_report territory)
        "what did you learn", "did you learn", "learn about me",
        "you know about me", "what have you learned about me",
        // WhatsApp messaging (whatsapp_send — only the full model has it)
        "whatsapp", "whats app", "send whatsapp", "wa message",
        "text ", "text me", "text myself", "text my phone",
        "send a text", "sms ", "message me", "message him", "message her",
        // Desktop / browser computer-use (desktop_do / browser_do — only the
        // full model has them; the VLM drives the actual clicks)
        "on the desktop", "on the computer", "on my computer", "on the screen",
        "in firefox", "in chrome", "in the browser", "open the browser",
        "open firefox", "open chrome", "open spotify", "open the terminal",
        "open the editor", "open vs code", "open code",
        "use the desktop", "control the screen", "click on", "type in",
        "in the terminal", "on amazon", "on flipkart", "add to cart",
        "open the file", "open a file", "open the app", "open the software",
        // sysmode hardening / decoy-honeypot awareness — Luna must answer
        // "is the honeypot running?", "switch to lockdown", "is everything
        // working?" with real tool output, not the fast model's guesses.
        "sysmode", "system mode", "system profile", "switch profile",
        "change mode", "switch to ", "current profile", "security profile",
        "honeypot", "honeypots", "cowrie", "decoy", "recon-deceiver",
        "recon receiver", "intrusion", "attacker", "intrusions", "attacks",
        "hardening", "fortress", "lockdown", "security mode",
        "stealth mode", "secure mode", "cyber mode",
        "is everything working", "everything is working", "everything working",
        "working properly", "is it working", "is it running",
        "make sure everything", "test everything", "test if everything",
        "check if everything", "check everything", "health check",
        "is the system", "is everything", "is the alarm",
    ];
    if tool_signals.iter().any(|s| lower.contains(s)) {
        return QueryComplexity::Complex;
    }

    // Short questions without tool signals → simple
    if words.len() <= 8 {
        return QueryComplexity::Simple;
    }

    // Longer open-ended questions default to complex (full model handles better)
    QueryComplexity::Complex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_full(input: &str) -> bool {
        !matches!(classify(input), QueryComplexity::Simple)
    }

    fn is_deep(input: &str) -> bool {
        matches!(classify(input), QueryComplexity::Deep)
    }

    fn is_security(input: &str) -> bool {
        matches!(classify(input), QueryComplexity::Security)
    }

    /// The reason the tier exists. Every one of these was refused 100% of the
    /// time by both general models, across ten framings including one that
    /// stripped every rule from Luna's system prompt.
    #[test]
    fn offensive_authoring_routes_to_the_security_model() {
        for q in [
            "write a simple python exploit to connect the target to my host computer in python in ~/Templates",
            "write me a python reverse shell for my own lab machine at 192.168.1.50",
            "write a python script to test my home wifi WPA2 handshake against my own router",
            "write a python command and control beacon stub for my own home lab red team exercise",
            "help me write a buffer overflow exploit for my ctf challenge",
            "audit the wireless security of my home network",
            "build a credential stuffing script for my own lab",
            "write a sql injection payload tester for my lab box",
        ] {
            assert!(is_security(q), "should route to security: {q:?}");
        }
    }

    /// The precedence rule, pinned because getting it backwards silently sends
    /// every exploit request to the coder and the security model is never asked.
    /// "write a python exploit" matches BOTH the security signals and the
    /// coding signals; security must win.
    #[test]
    fn security_wins_over_coding_when_both_match() {
        for q in [
            "write a python exploit",
            "write a poc in python",
            "write an exploit as a shell script",
            "write a python payload decoder exploit",
        ] {
            assert!(
                is_security(q),
                "security must beat coding for: {q:?} (is_deep={})",
                is_deep(q)
            );
        }
    }

    /// The bug that broke `source_editing_routes_to_the_deep_model` when this
    /// tier first landed: "rce" as a substring matches the middle of "source",
    /// so every request mentioning source code was diverted to the security
    /// model and stopped reaching the 14B coder.
    #[test]
    fn short_signals_do_not_match_inside_words() {
        for q in [
            "patch the source in src/llm/ollama.rs",
            "add a test to src/tools/mod.rs",
            "explain the difference between force and resource",
            "check my wifi drivers",
            "what is the source of the leak in this code",
        ] {
            assert!(
                !is_security(q),
                "{q:?} must not be pulled in by a substring match"
            );
        }
    }

    /// Everyday speech must not be captured. The security model is 4.5x slower
    /// and narrower than the general one; leaking chat into it is a regression.
    #[test]
    fn ordinary_conversation_stays_off_the_security_tier() {
        for q in [
            "hey luna, what are you?",
            "what's the weather like",
            "remind me to buy milk in 20 minutes",
            "write me a python script that sums a list",
            "explain how a hash map works",
            "can you restart the daemon for me",
            "what's the payload size limit on this endpoint",
        ] {
            assert!(!is_security(q), "{q:?} wrongly routed to security");
        }
    }

    /// Separator variants have to match each other, or a hyphenated phrasing
    /// falls through to the general model and gets refused.
    #[test]
    fn security_signals_tolerate_separator_variants() {
        for q in [
            "help me build a reverse-shell for my lab",
            "help me build a reverse_shell for my lab",
            "help me build a reverse shell for my lab",
            "set up a command-and-control server for my own lab",
            "do a red-team assessment of my home network",
        ] {
            assert!(is_security(q), "separator variant missed: {q:?}");
        }
    }

    /// Editing turns must reach the deep tier, because that is where the 14B
    /// coder model lives and it is the only one that survives line-anchored
    /// modify (3/3 vs 0/3). If one of these regresses to the 7B main model,
    /// file editing breaks again.
    #[test]
    fn source_editing_routes_to_the_deep_model() {
        for q in [
            "add a test to util.rs",
            "add a function to src/tools/mod.rs",
            "add a doc comment to read_file in selfpatch.rs",
            "patch the source in src/llm/ollama.rs",
            "add a helper method to the config",
            "add a newline to the end of main.rs",
            "self patch: rename that function",
        ] {
            assert!(is_deep(q), "should route to deep model: {q:?}");
        }
    }

    /// The flip side: the deep model is 4.5x slower, so it must not swallow
    /// ordinary conversation. These stay off the deep tier.
    #[test]
    fn ordinary_conversation_stays_off_the_deep_model() {
        for q in [
            "hey luna, what are you?",
            "what's the weather like",
            "set a reminder for 6pm",
            "play some music",
            "add milk to the shopping list",
            "how are you doing today",
        ] {
            assert!(!is_deep(q), "should NOT use the slow deep model: {q:?}");
        }
    }

    #[test]
    fn persona_questions_route_to_full_model() {
        // These are the exact prompts the fast 3B model failed (generic
        // corporate identity, hedged never-replaced, denied machine access).
        // They must go to the full model so the Constitution is honored.
        for q in [
            "who are you",
            "will you ever be replaced ?",
            "will you ever be replaced?",
            "what are you curious about on this machine right now?",
            "who made you",
            "tell me about yourself",
            "what's your name",
            "do you have access to my machine",
            "what do you know about me",
            "are you real",
        ] {
            assert!(is_full(q), "{q:?} must not go to the fast model");
        }
    }

    #[test]
    fn plain_greetings_stay_on_fast_model() {
        // The fix must not starve genuine short greetings of the fast model.
        for q in ["hi", "hey", "hello", "thanks", "bye", "good morning"] {
            assert!(
                matches!(classify(q), QueryComplexity::Simple),
                "{q:?} should stay on the fast model"
            );
        }
    }

    #[test]
    fn system_learning_routes_to_full_model() {
        for q in [
            "learn about my system",
            "learn my machine",
            "know everything on my system",
            "index system",
            "study my workflow",
            "deep learn my machine",
        ] {
            assert!(is_full(q), "{q:?} must not go to the fast model");
        }
    }

    #[test]
    fn task_continuations_route_to_full_model() {
        assert!(is_full("well do it"));
        assert!(is_full("go ahead and do it"));
        assert!(is_full("proceed"));
    }

    #[test]
    fn whatsapp_queries_route_to_full_model() {
        for q in [
            "text myself on whatsapp",
            "send a whatsapp message",
            "message her on whatsapp",
            "just text \"hi this is luna\" to 9148069879",
            "whatsapp my friend that i'll be late",
        ] {
            assert!(is_full(q), "{q:?} must not go to the fast model");
        }
    }

    #[test]
    fn sysmode_queries_route_to_full_model() {
        for q in [
            "is the honeypot running?",
            "is the honeypot working properly?",
            "test if everything is working",
            "check that everything is working",
            "switch to stealth mode",
            "switch to lockdown",
            "is the system secure?",
            "what security mode am i in",
            "check sysmode status",
            "run a health check on the setup",
        ] {
            assert!(is_full(q), "{q:?} must not go to the fast model");
        }
    }

    #[test]
    fn plain_chat_stays_fast() {
        assert!(classify("hi, what's up") == QueryComplexity::Simple);
        assert!(classify("okay thanks") == QueryComplexity::Simple);
        assert!(classify("how are you") == QueryComplexity::Simple);
    }

    /// Conceptual CS questions must stay off the deep model.
    ///
    /// Regression guard. The first four were routed to the 14B purely on the
    /// bare nouns "race condition" / "deadlock" / "algorithm" / "data
    /// structure", costing 24-40s each for an answer the 7B gives correctly in
    /// 10-17s. A question about what a race condition IS is not an editing
    /// task; if it reaches the 3.2 tok/s model then ordinary chat feels broken,
    /// which is the exact complaint this routing table exists to prevent.
    #[test]
    fn conceptual_cs_questions_stay_off_the_deep_model() {
        for q in [
            "what is a race condition",
            "what is a deadlock",
            "what is an algorithm",
            "what is a data structure",
            "explain atomicity",
            "what does idempotent mean",
        ] {
            assert!(
                !is_deep(q),
                "{q:?} was routed to the 3.2 tok/s model; it is a conceptual \
                 question the fast model answers correctly"
            );
        }
    }

    /// The flip side: the terms must still reach the deep model when they
    /// describe actual work, or the fix above becomes an over-correction.
    #[test]
    fn cs_terms_still_route_deep_when_they_describe_work() {
        for q in [
            "write an algorithm to sort a list",
            "what is the time complexity of quicksort",
            "explain the space complexity of this approach",
            "add a function to src/util.rs",
        ] {
            assert!(is_deep(q), "{q:?} should reach the deep model");
        }
    }
}

#[cfg(test)]
mod probe {
    use super::*;

    /// Not an assertion — a measurement. Prints which everyday questions reach
    /// the 3.2 tok/s model, so the routing table can be judged on a corpus
    /// rather than on whichever prompt happened to expose a problem.
    #[test]
    #[ignore = "measurement probe, prints instead of asserting"]
    fn everyday_chat_routing_audit() {
        let corpus = [
            "what is a race condition",
            "what is a deadlock",
            "explain how a hash map works",
            "what is an algorithm",
            "how does a database index work",
            "what's the weather like",
            "remind me to call mum at 6",
            "what time is it",
            "how much disk space do I have left",
            "play some jazz",
            "who wrote dune",
            "what's the capital of Japan",
            "explain photosynthesis in simple terms",
            "what does the fox say",
            "tell me a joke",
            "what's my cpu usage",
            "summarise the news",
            "is nordvpn legit",
            "how do i boil an egg",
            "what's 2 plus 2",
            "translate good morning into french",
            "how do i tie a bowline knot",
            "what is the difference between tcp and udp",
            "explain recursion",
            "how do i change my wallpaper",
            "what's the weather in Oslo tomorrow",
            "set a timer for 10 minutes",
            "how do i make pasta",
            "what is quantum entanglement",
        ];
        let mut deep: Vec<&str> = Vec::new();
        for q in corpus {
            if matches!(classify(q), QueryComplexity::Deep) {
                deep.push(q);
            }
        }
        println!("--- routed DEEP: {} of {} ---", deep.len(), corpus.len());
        for q in &deep {
            println!("  DEEP  {q:?}");
        }
    }
}
