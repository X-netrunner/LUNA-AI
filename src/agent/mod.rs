//! agent/mod.rs — The main agent loop

use crate::config::{LunaConfig, VoiceMode};
use crate::llm::escalation::{classify, QueryComplexity};
use crate::llm::ollama::OllamaClient;
use crate::llm::react::ReactLoop;
use crate::memory::Memory;
use crate::tts;
use anyhow::Result;
use rustyline::{history::FileHistory, Config as RlConfig, Editor};
use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;

// ── Interactive input (readline) ──────────────────────────────────────────────

/// Shared input-history file so up/down arrows work across sessions.
fn input_history_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("luna").join("input_history"))
}

type RlEditor = Editor<(), FileHistory>;

/// Build a readline editor with persistent history. Falls back to a plain
/// editor without history if the history file can't be used (first run etc.)
/// — input still works either way.
fn make_editor() -> RlEditor {
    let cfg = RlConfig::builder()
        .max_history_size(500)
        .map(|b| b.build())
        .unwrap_or_else(|_| RlConfig::default());
    let mut rl = match RlEditor::with_config(cfg) {
        Ok(rl) => rl,
        Err(_) => Editor::<(), FileHistory>::new().expect("rustyline editor"),
    };
    if let Some(path) = input_history_path() {
        if path.exists() {
            let _ = rl.load_history(&path); // ignore — empty history is fine
        }
    }
    rl
}

fn save_editor_history(rl: &mut RlEditor) {
    if let Some(path) = input_history_path() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = rl.save_history(&path);
    }
}

// ── Shared setup ──────────────────────────────────────────────────────────────


fn build_fast_client(config: &LunaConfig) -> Option<OllamaClient> {
    // Only build if a fast_model is configured
    let model = config.llm.fast_model.as_deref()?;
    Some(
        OllamaClient::new(
            &config.llm.base_url,
            model,
            config.llm.temperature,
            512, // smaller token budget — fast model is for short answers
        )
        .debug(config.logging.level == "debug"),
    )
}

fn build_client(config: &LunaConfig) -> OllamaClient {
    OllamaClient::new(
        &config.llm.base_url,
        &config.llm.model,
        config.llm.temperature,
        config.llm.max_tokens,
    )
    .debug(config.logging.level == "debug")
}

fn build_stt(config: &LunaConfig) -> crate::stt::whisper::WhisperStt {
    crate::stt::whisper::WhisperStt::with_prompt(
        &config.voice.whisper_model.to_string_lossy(),
        // Keep this SHORT and non-conversational — Whisper can hallucinate
        // prompt text back into the transcription on near-silence frames.
        // Just seed it with domain vocabulary and the assistant's name.
        Some("Luna, open, close, run, search, volume, terminal, browser.".into()),
    )
}

/// Load fish shell history and return unique recent commands.
/// These are injected into the system prompt so Luna knows
/// what apps and commands the user actually runs.
fn load_shell_history() -> Vec<String> {
    let history_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/fish/fish_history");

    let Ok(content) = std::fs::read_to_string(&history_path) else {
        tracing::debug!("No fish history found at {:?}", history_path);
        return Vec::new();
    };

    // Fish history format: lines starting with "- cmd: <command>"
    let mut seen = HashSet::new();
    let mut commands: Vec<String> = Vec::new();

    for line in content.lines() {
        if let Some(cmd) = line.strip_prefix("- cmd:") {
            let cmd = cmd.trim().to_string();
            if cmd.is_empty() {
                continue;
            }
            // Skip overly noisy commands
            if cmd.starts_with("cd ")
                || cmd == "ls"
                || cmd == "clear"
                || cmd == "pwd"
                || cmd.starts_with("cat ")
                || cmd.starts_with("echo ")
                || cmd.starts_with("grep ")
                || cmd.starts_with("#")
                || cmd.len() > 100
            // skip long one-liners
            {
                continue;
            }
            if seen.insert(cmd.clone()) {
                commands.push(cmd);
            }
        }
    }

    // Return the 80 most recent unique commands
    // (fish_history is newest-last, so take from the end)
    commands.into_iter().rev().take(30).collect()
}

/// Detect CLI-style flags typed into the chat ("luna --set-key gemini").
/// These are terminal commands; answering them via the LLM just produces
/// confusion, and secrets must never pass through chat (they would be
/// saved to history.json). Returns a canned guidance reply.
fn cli_flag_reply(input: &str) -> Option<String> {
    let t = input.trim();
    let body = t.strip_prefix("luna ").unwrap_or(t);
    if !body.starts_with("--") {
        return None;
    }
    Some(format!(
        "That's a terminal command — run it in your shell instead:\n  luna {}\n\
         I refuse secrets typed in chat: they'd be saved to history.json.",
        body
    ))
}

/// Build an enriched system prompt that includes shell history context.
/// Permanent-memory facts are NOT baked in here — they are recalled
/// per-query by `memory_block_for` so only relevant facts get injected.
fn build_system_prompt(config: &LunaConfig) -> String {
    let history = load_shell_history();
    let history_block = if !history.is_empty() {
        format!(
            "\n[User's shell commands]\n{}\n",
            history
                .iter()
                .take(30)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        )
    } else {
        String::new()
    };

    format!("{}{}", config.agent.system_prompt, history_block)
}

/// Per-turn memory injection: top-k facts semantically similar to the
/// query (RAG-lite), falling back to the full dump when embeddings are
/// unavailable. Covers BOTH models — even simple fast-path queries now
/// see relevant personal facts.
async fn memory_block_for(input: &str, config: &LunaConfig, k: usize) -> String {
    let Ok(pm) = crate::memory::permanent::PermanentMemory::load() else {
        return String::new();
    };
    let facts = pm.all_facts();
    if facts.is_empty() {
        return String::new();
    }
    match crate::memory::recall::relevant_facts(
        &config.llm.base_url,
        &config.llm.embedding_model,
        input,
        facts,
        k,
    )
    .await
    {
        Some(recalled) => crate::memory::recall::format_block(&recalled),
        None => pm.as_prompt_block(),
    }
}

/// Compact prompt for the fast model. The 0.6b model is too small to follow
/// the full ruleset and echoes a long system prompt back as its reply — so it
/// gets a distilled version instead: no tools, no rules, just brevity.
const FAST_PROMPT: &str = "You are Luna. You were built by Netrunner. You run locally \
    on Arch Linux. You are direct, efficient, and have a dry wit. Be brief — answer in \
    1-2 sentences max. Never introduce yourself beyond 'I'm Luna, built by Netrunner'. \
    Never say you were made by a company. Never say you don't have a physical form. \
    If you don't know something specific, say 'I don't know' instead of guessing. \
    If the request needs tools, files, commands, web data, or actions on this machine, \
    reply with exactly: ESCALATE";

#[derive(Debug, PartialEq, Clone, Copy)]
enum RunMode {
    Text,
    Voice,
    Hybrid,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum ControlFlow {
    Continue,
    Exit,
    SwitchToText,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(config: &LunaConfig) -> Result<()> {
    crate::tools::proactive::spawn(config);

    match config.audio.input_mode {
        crate::config::InputMode::WakeWord | crate::config::InputMode::Both => {
            run_hybrid(config).await
        }
        _ => run_text(config).await,
    }
}

/// Run an extended voice session. After the wake word fires once, Luna keeps
/// listening for follow-up commands WITHOUT requiring the wake word again.
/// Session ends on goodbye/stop phrase or ~30s silence timeout.
async fn run_voice_session(
    config: &LunaConfig,
    stt: &crate::stt::whisper::WhisperStt,
    memory: &mut Memory,
    react: &ReactLoop<'_>,
    system_prompt: &str,
    inline_command: Option<String>,
) -> Result<ControlFlow> {
    let mut pending = inline_command;
    let mut turn: u32 = 0;

    loop {
        let input = if let Some(cmd) = pending.take() {
            // Inline command captured together with the wake word
            // ("luna what's the time") — process it directly.
            if turn == 0 {
                println!("  [Inline command captured with wake word]");
            }
            cmd
        } else {
            if turn == 0 {
                println!("  [Wake word detected — listening for command]");
                tts::speak("Yes?", &config.voice.mode, config).await.ok();
            } else {
                println!("  [Session active — say \"that's all\" to end]");
            }

            let wav_path = match crate::audio::capture::record_until_silence(
                config.audio.sample_rate,
                config.audio.vad_silence_ms,
            )
            .await
            {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!("Voice session listen ended: {}", e);
                    if turn > 0 {
                        println!(
                            "  [Session ended — say \"{}\" to wake me again]",
                            config.audio.wake_word
                        );
                    }
                    return Ok(ControlFlow::Continue);
                }
            };

            match stt.transcribe(&wav_path).await {
                Ok(t) => { tokio::fs::remove_file(&wav_path).await.ok(); t }
                Err(e) => {
                    tracing::error!("Transcription failed: {}", e);
                    tokio::fs::remove_file(&wav_path).await.ok();
                    if turn == 0 { return Ok(ControlFlow::Continue); }
                    continue;
                }
            }
        };

        if input.is_empty() || looks_like_artifact(&input) {
            if turn == 0 { return Ok(ControlFlow::Continue); }
            continue;
        }

        turn += 1;
        println!("  You: {}", input);

        let input_lower = input.to_lowercase();
        let input_trim = input_lower.trim();

        match input_trim {
            "exit" | "quit" | "goodbye" | "goodbye luna" => {
                tts::speak("Shutting down.", &config.voice.mode, config).await.ok();
                return Ok(ControlFlow::Exit);
            }
            "clear" | "clear memory" => {
                memory.clear()?;
                tts::speak("Memory cleared.", &config.voice.mode, config).await.ok();
                continue;
            }
            "use text" | "text mode" | "switch to text" => {
                tts::speak("Switching to text mode.", &config.voice.mode, config).await.ok();
                return Ok(ControlFlow::SwitchToText);
            }
            "use voice" | "voice mode" | "switch to voice" => {
                tts::speak("Already in voice mode.", &config.voice.mode, config).await.ok();
                continue;
            }
            "that's all" | "thats all" | "stop listening" | "end session" | "never mind" => {
                tts::speak("Okay.", &config.voice.mode, config).await.ok();
                return Ok(ControlFlow::Continue);
            }
            _ => {}
        }

        print!("  Luna: ");
        io::stdout().flush().ok();

        match react.run(&input, memory, system_prompt).await {
            Ok((response, streamed)) => {
                if !streamed {
                    println!("{}", response);
                }
                if config.voice.mode != VoiceMode::Off {
                    tts::speak(&response, &config.voice.mode, config).await.ok();
                }
            }
            Err(e) => eprintln!("\n  Error: {}", e),
        }
        println!();
        // loop — session stays open for follow-up
    }
}

// ── Text loop ─────────────────────────────────────────────────────────────────

pub async fn run_text(config: &LunaConfig) -> Result<()> {
    tracing::info!("Starting Luna agent (text mode)");

    let client = build_client(config);
    let fast_client = build_fast_client(config);
    let mut memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
    let react = ReactLoop::new(&client, config.agent.max_react_iterations, config.agent.native_tools, config);
    let fast_react = fast_client.as_ref().map(|c| ReactLoop::new(c, config.agent.max_react_iterations, false, config));
    let system_prompt = build_system_prompt(config);

    println!("  Luna — text mode");
    println!(
        "  Model: {}  |  Voice: {:?}",
        config.llm.model, config.voice.mode
    );
    println!("  Type 'exit' to quit, 'clear' to reset memory\n");
    println!("  (↑/↓ cycles input history)\n");

    let mut rl = make_editor();

    loop {
        let line = match rl.readline("You: ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => continue, // ^C → fresh prompt
            Err(rustyline::error::ReadlineError::Eof) => break,           // ^D exits
            Err(e) => {
                tracing::error!("Failed to read input: {}", e);
                break;
            }
        };

        let input = line.trim().to_string();
        if input.is_empty() || looks_like_artifact(&input) {
            continue;
        }
        let _ = rl.add_history_entry(&input);
        save_editor_history(&mut rl);

        match input.to_lowercase().as_str() {
            "exit" | "quit" | "bye" => {
                println!("Luna: Shutting down.");
                break;
            }
            "clear" => {
                memory.clear()?;
                println!("Luna: Memory cleared. Fresh start.");
                continue;
            }
            _ => {}
        }

        if let Some(reply) = cli_flag_reply(&input) {
            println!("Luna: {}", reply);
            continue;
        }

        let debug = config.logging.level == "debug";
        let (mut active_react, mut effective_prompt, mut is_fast): (&ReactLoop, String, bool) =
            match classify(&input) {
                QueryComplexity::Simple if fast_react.is_some() => {
                    tracing::debug!("Simple query — using fast model");
                    (fast_react.as_ref().unwrap(), FAST_PROMPT.to_string(), true)
                }
                _ => (&react, system_prompt.to_string(), false),
            };
        effective_prompt
            .push_str(&memory_block_for(&input, config, if is_fast { 3 } else { 6 }).await);

        // Up to two attempts: a fast-model reply of "ESCALATE" rolls back
        // the exchange and retries once on the full model with tools.
        for attempt in 1..=2 {
            let mem_snapshot = memory.len();
            let tag = if debug {
                if is_fast { "[fast] " } else { "[full] " }
            } else {
                ""
            };
            print!("Luna{}: ", tag);
            io::stdout().flush().ok();

            match active_react.run(&input, &mut memory, &effective_prompt).await {
                Ok((response, streamed)) => {
                    if is_fast && response.trim() == "ESCALATE" && attempt < 2 {
                        tracing::info!("Fast model escalated — re-running on full model");
                        memory.truncate_to(mem_snapshot);
                        active_react = &react;
                        is_fast = false;
                        effective_prompt = system_prompt.to_string();
                        effective_prompt.push_str(&memory_block_for(&input, config, 6).await);
                        continue;
                    }
                    if !streamed {
                        println!("{}", response);
                    }
                    if config.voice.mode != VoiceMode::Off {
                        if let Err(e) = tts::speak(&response, &config.voice.mode, config).await {
                            tracing::warn!("TTS failed: {} — continuing without audio", e);
                        }
                    }
                    break;
                }
                Err(e) => {
                    eprintln!("\nLuna error: {}", e);
                    tracing::error!("Agent error: {:?}", e);
                    break;
                }
            }
        }

        println!();
    }

    Ok(())
}

// ── Hybrid loop ───────────────────────────────────────────────────────────────

async fn run_hybrid(config: &LunaConfig) -> Result<()> {
    tracing::info!("Starting Luna agent (hybrid mode)");

    let client = build_client(config);
    let fast_client = build_fast_client(config);
    let mut memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
    let react = ReactLoop::new(&client, config.agent.max_react_iterations, config.agent.native_tools, config);
    let fast_react = fast_client.as_ref().map(|c| ReactLoop::new(c, config.agent.max_react_iterations, false, config));
    let stt = build_stt(config);
    let system_prompt = build_system_prompt(config);

    println!(
        "\n  Luna — hybrid mode (say \"{}\" or type to interact)",
        config.audio.wake_word
    );
    println!("  Commands: 'use voice', 'use text', 'exit', 'clear'\n");

    let mut mode = RunMode::Hybrid;

    // ── Stdin reader thread → channel ─────────────────────────────────────────
    // Uses rustyline on its own thread so ↑/↓ history works while the main
    // loop handles voice events concurrently. The thread owns the "You: "
    // prompt; the main loop must NOT print its own prompt for typed input.
    let (text_tx, mut text_rx) = tokio::sync::mpsc::channel::<String>(32);
    std::thread::spawn(move || {
        let mut rl = make_editor();
        loop {
            match rl.readline("You: ") {
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        let _ = rl.add_history_entry(&trimmed);
                        save_editor_history(&mut rl);
                    }
                    if text_tx.blocking_send(trimmed).is_err() {
                        break;
                    }
                }
                Err(rustyline::error::ReadlineError::Interrupted) => continue,
                Err(_) => break, // EOF or terminal closed
            }
        }
    });

    // ── Wake word listener task → channel ────────────────────────────────────
    // Runs as a persistent background task — never cancelled, never restarted.
    // Sends the transcribed utterance each time the wake word is detected,
    // so the session can use anything after the wake word as an inline command
    // (e.g. "luna what's the time" works in one breath).
    // The main loop just selects on this channel alongside stdin.
    let (wake_tx, mut wake_rx) = tokio::sync::mpsc::channel::<String>(4);
    let wake_aliases = config.audio.wake_aliases.clone();
    let sample_rate = config.audio.sample_rate;
    let silence_ms = config.audio.vad_silence_ms;
    let stt_for_wake = build_stt(config); // separate STT instance for the background task

    tokio::spawn(async move {
        loop {
            match crate::audio::capture::listen_for_wake_word(
                sample_rate,
                silence_ms,
                &wake_aliases,
                &stt_for_wake,
            )
            .await
            {
                Ok(text) => {
                    if wake_tx.send(text).await.is_err() {
                        break; // main loop exited
                    }
                }
                Err(e) => {
                    tracing::debug!("Wake word listener cycle error: {}", e);
                    // Brief pause to avoid a tight error loop
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                }
            }
        }
    });

    // ── Main event loop ───────────────────────────────────────────────────────
    loop {
        tokio::select! {
            // ── Text input ───────────────────────────────────────────────────
            maybe_line = text_rx.recv() => {
                let line = match maybe_line { Some(l) => l, None => break };
                let input = line.trim().to_string();
                if input.is_empty() { continue; }
                let input_lower = input.to_lowercase();

                match input_lower.as_str() {
                    "exit" | "quit" | "bye" => {
                        println!("Luna: Goodbye.");
                        tts::speak("Goodbye.", &config.voice.mode, config).await.ok();
                        return Ok(());
                    }
                    "clear" => {
                        memory.clear()?;
                        println!("Luna: Memory cleared.");
                        continue;
                    }
                    "use voice" | "voice mode" | "voice" => {
                        mode = RunMode::Voice;
                        println!("  [Voice mode — say \"{}\" to activate]", config.audio.wake_word);
                        continue;
                    }
                    "use text" | "text mode" | "text" => {
                        mode = RunMode::Text;
                        println!("  [Text mode]");
                        continue;
                    }
                    _ => {}
                }

                if looks_like_artifact(&input) { continue; }

                if let Some(reply) = cli_flag_reply(&input) {
                    println!("Luna: {}", reply);
                    continue;
                }

                let debug = config.logging.level == "debug";
                let (mut active_react, mut effective_prompt, mut is_fast): (
                    &ReactLoop,
                    String,
                    bool,
                ) = match classify(&input) {
                    QueryComplexity::Simple if fast_react.is_some() => {
                        (fast_react.as_ref().unwrap(), FAST_PROMPT.to_string(), true)
                    }
                    _ => (&react, system_prompt.to_string(), false),
                };
                effective_prompt
                    .push_str(&memory_block_for(&input, config, if is_fast { 3 } else { 6 }).await);

                for attempt in 1..=2 {
                    let mem_snapshot = memory.len();
                    let tag = if debug {
                        if is_fast { "[fast] " } else { "[full] " }
                    } else {
                        ""
                    };
                    print!("Luna{}: ", tag);
                    io::stdout().flush().ok();

                    match active_react.run(&input, &mut memory, &effective_prompt).await {
                        Ok((response, streamed)) => {
                            if is_fast && response.trim() == "ESCALATE" && attempt < 2 {
                                tracing::info!("Fast model escalated — re-running on full model");
                                memory.truncate_to(mem_snapshot);
                                active_react = &react;
                                is_fast = false;
                                effective_prompt = system_prompt.to_string();
                                effective_prompt
                                    .push_str(&memory_block_for(&input, config, 6).await);
                                continue;
                            }
                            if !streamed {
                                println!("{}", response);
                            }
                            if config.voice.mode != VoiceMode::Off {
                                tts::speak(&response, &config.voice.mode, config).await.ok();
                            }
                            break;
                        }
                        Err(e) => {
                            eprintln!("\nLuna error: {}", e);
                            break;
                        }
                    }
                }
                println!();
            }

            // ── Wake word fired ──────────────────────────────────────────────
            // The background task detected the wake word and sent the full
            // utterance here. Anything after the wake word becomes an inline
            // command; a bare wake word falls back to the "Yes?" prompt.
            // We only act on it when not in pure text mode.
            Some(wake_text) = wake_rx.recv(), if mode != RunMode::Text => {
                let inline = crate::audio::capture::strip_wake_word(
                    &wake_text,
                    &config.audio.wake_aliases,
                )
                .filter(|cmd| !cmd.trim().is_empty());

                match run_voice_session(
                    config, &stt, &mut memory, &react, &system_prompt, inline,
                ).await? {
                    ControlFlow::Exit => return Ok(()),
                    ControlFlow::SwitchToText => {
                        mode = RunMode::Text;
                        println!("  [Switched to text mode]");
                    }
                    ControlFlow::Continue => {}
                }
            }
        }
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn looks_like_artifact(s: &str) -> bool {
    let t = s.trim().to_lowercase();
    // Very short single words that are clearly not commands
    if t.split_whitespace().count() <= 1 && t.len() < 4 {
        return true;
    }
    let hallucinations = [
        "thank you for watching",
        "thanks for watching",
        "see you in the next video",
        "see you later",
        "please subscribe",
        "like and subscribe",
        "and uh",
        "and shadow",
        "and speak",
    ];
    hallucinations.iter().any(|h| t.contains(h))
}
