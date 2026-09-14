//! llm/escalation.rs — Model escalation
//!
//! Three-tier routing:
//!   Simple = greetings, short questions, chitchat → fast model (qwen3:0.6b)
//!   Complex = tool calls, moderate reasoning → normal model (qwen2.5:7b)
//!   Deep = code generation, multi-step reasoning, analysis → deep model (qwen3:8b)

#[derive(Debug, PartialEq)]
pub enum QueryComplexity {
    Simple,
    Complex,
    Deep,
}

pub fn classify(input: &str) -> QueryComplexity {
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
        "algorithm", "data structure", "race condition", "deadlock",
        "multi-step", "multi step", "chain of thought",
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
    fn plain_chat_stays_fast() {
        assert!(classify("hi, what's up") == QueryComplexity::Simple);
        assert!(classify("okay thanks") == QueryComplexity::Simple);
        assert!(classify("how are you") == QueryComplexity::Simple);
    }
}
