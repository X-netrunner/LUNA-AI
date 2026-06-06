//! llm/escalation.rs — Model escalation
//!
//! Routes queries to a small fast model or the full model based on complexity.
//! Simple = greetings, short questions, chitchat → fast model (e.g. qwen2.5:0.5b)
//! Complex = anything needing tools, code, reasoning → full model (e.g. qwen2.5:7b)

pub enum QueryComplexity {
    Simple,
    Complex,
}

pub fn classify(input: &str) -> QueryComplexity {
    let lower = input.to_lowercase();
    let words: Vec<&str> = input.split_whitespace().collect();

    // Pure greetings / acknowledgements — always simple
    let greetings = ["hi", "hey", "hello", "thanks", "thank you", "ok", "okay",
                     "bye", "goodbye", "yep", "nope", "sure", "cool", "nice",
                     "lol", "haha", "hmm", "wow", "great"];
    if words.len() <= 4 && greetings.iter().any(|g| lower.contains(g)) {
        return QueryComplexity::Simple;
    }

    // Conversational openers that don't need tools
    let simple_starts = ["how are", "what are you", "who are you", "what is your",
                         "do you like", "can you talk", "what do you think",
                         "tell me about yourself", "what's your name"];
    if simple_starts.iter().any(|s| lower.starts_with(s)) {
        return QueryComplexity::Simple;
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
