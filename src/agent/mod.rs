//! agent/mod.rs — The main agent loop

use crate::config::{LunaConfig, VoiceMode};
use crate::llm::escalation::{classify, QueryComplexity};
use crate::llm::ollama::OllamaClient;
use crate::llm::react::ReactLoop;
use crate::memory::Memory;
use crate::tts;
use anyhow::Result;
use std::collections::HashSet;
use std::io::{self, BufRead, Write};

// ── Shared setup ──────────────────────────────────────────────────────────────


fn build_fast_client(config: &LunaConfig) -> Option<OllamaClient> {
    // Only build if a fast_model is configured
    let model = config.llm.fast_model.as_deref()?;
    Some(OllamaClient::new(
        &config.llm.base_url,
        model,
        config.llm.temperature,
        512, // smaller token budget — fast model is for short answers
    ))
}

fn build_client(config: &LunaConfig) -> OllamaClient {
    OllamaClient::new(
        &config.llm.base_url,
        &config.llm.model,
        config.llm.temperature,
        config.llm.max_tokens,
    )
}

fn build_stt(_config: &LunaConfig) -> crate::stt::whisper::WhisperStt {
    let model_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/luna/models/ggml-base.en.bin")
        .to_string_lossy()
        .to_string();

    crate::stt::whisper::WhisperStt::with_prompt(
        &model_path,
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

/// Build an enriched system prompt that includes shell history context.
fn build_system_prompt(config: &LunaConfig) -> String {
    let pm = crate::memory::permanent::PermanentMemory::load().unwrap_or_default();
    let pm_block = pm.as_prompt_block();

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

    format!(
        "{}\n\n{}{}",
        config.agent.system_prompt, pm_block, history_block
    )
}

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
    SwitchToVoice,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(config: &LunaConfig) -> Result<()> {
    run_text(config).await
    //run_hybrid(config).await
}

/// Process a single voice interaction (listen -> transcribe -> react -> speak).
/// Returns ControlFlow for the next step.
async fn process_voice_interaction(
    config: &LunaConfig,
    stt: &crate::stt::whisper::WhisperStt,
    memory: &mut Memory,
    react: &ReactLoop<'_>,
    system_prompt: &str,
) -> Result<ControlFlow> {
    // ── Phase 2: wake word heard — prompt and record command ──────────────
    println!("  [Wake word detected — listening for command]");
    tts::speak("Yes?", &config.voice.mode).await.ok();

    let wav_path = match crate::audio::capture::record_until_silence(
        config.audio.sample_rate,
        config.audio.vad_silence_ms,
    )
    .await
    {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!("No command audio: {}", e);
            return Ok(ControlFlow::Continue);
        }
    };

    let input = match stt.transcribe(&wav_path).await {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("Transcription failed: {}", e);
            tokio::fs::remove_file(&wav_path).await.ok();
            return Ok(ControlFlow::Continue);
        }
    };
    tokio::fs::remove_file(&wav_path).await.ok();

    if input.is_empty() || looks_like_artifact(&input) {
        return Ok(ControlFlow::Continue);
    }

    println!("  You: {}", input);

    let input_lower = input.to_lowercase();
    let input_trim = input_lower.trim();

    match input_trim {
        "exit" | "quit" | "goodbye" | "goodbye luna" => {
            tts::speak("Shutting down.", &config.voice.mode).await.ok();
            return Ok(ControlFlow::Exit);
        }
        "clear" | "clear memory" => {
            memory.clear()?;
            tts::speak("Memory cleared.", &config.voice.mode).await.ok();
            return Ok(ControlFlow::Continue);
        }
        "use text" | "text mode" | "use text mode" | "switch to text" => {
            tts::speak("Switching to text mode.", &config.voice.mode)
                .await
                .ok();
            return Ok(ControlFlow::SwitchToText);
        }
        "use voice" | "voice mode" | "use voice mode" | "switch to voice" => {
            tts::speak("Switching to voice mode.", &config.voice.mode)
                .await
                .ok();
            return Ok(ControlFlow::SwitchToVoice);
        }
        _ => {}
    }

    // ── Phase 3: respond ──────────────────────────────────────────────────
    print!("  Luna: ");
    io::stdout().flush().ok();

    match react.run(&input, memory, system_prompt).await {
        Ok(response) => {
            println!("{}", response);
            if config.voice.mode != VoiceMode::Off {
                tts::speak(&response, &config.voice.mode).await.ok();
            }
        }
        Err(e) => eprintln!("\n  Error: {}", e),
    }
    println!();
    Ok(ControlFlow::Continue)
}

// ── Text loop ─────────────────────────────────────────────────────────────────

pub async fn run_text(config: &LunaConfig) -> Result<()> {
    tracing::info!("Starting Luna agent (text mode)");

    let client = build_client(config);
    let fast_client = build_fast_client(config);
    let mut memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
    let react = ReactLoop::new(&client, config.agent.max_react_iterations, config.agent.native_tools);
    let fast_react = fast_client.as_ref().map(|c| ReactLoop::new(c, config.agent.max_react_iterations, false));
    let system_prompt = build_system_prompt(config);

    println!("  Luna — text mode");
    println!(
        "  Model: {}  |  Voice: {:?}",
        config.llm.model, config.voice.mode
    );
    println!("  Type 'exit' to quit, 'clear' to reset memory\n");

    let stdin = io::stdin();

    loop {
        print!("You: ");
        io::stdout().flush().ok();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("Failed to read input: {}", e);
                break;
            }
        }

        let input = line.trim().to_string();
        if input.is_empty() || looks_like_artifact(&input) {
            continue;
        }

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

        print!("Luna: ");
        io::stdout().flush().ok();

        let (active_react, effective_prompt): (&ReactLoop, String) = match classify(&input) {
            QueryComplexity::Simple if fast_react.is_some() => {
                tracing::debug!("Simple query — using fast model");
                (fast_react.as_ref().unwrap(), format!("{}\n\nBe concise. 1-2 sentences only.", system_prompt))
            }
            _ => {
                (&react, system_prompt.to_string())
            }
        };

        match active_react.run(&input, &mut memory, &effective_prompt).await {
            Ok(response) => {
                println!("{}", response);
                if config.voice.mode != VoiceMode::Off {
                    if let Err(e) = tts::speak(&response, &config.voice.mode).await {
                        tracing::warn!("TTS failed: {} — continuing without audio", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("\nLuna error: {}", e);
                tracing::error!("Agent error: {:?}", e);
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
    let react = ReactLoop::new(&client, config.agent.max_react_iterations, config.agent.native_tools);
    let fast_react = fast_client.as_ref().map(|c| ReactLoop::new(c, config.agent.max_react_iterations, false));
    let stt = build_stt(config);
    let system_prompt = build_system_prompt(config);

    println!(
        "\n  Luna — hybrid mode (say \"{}\" or type to interact)",
        config.audio.wake_word
    );
    println!("  Commands: 'use voice', 'use text', 'exit', 'clear'\n");

    let mut mode = RunMode::Hybrid;

    // ── Stdin reader thread → channel ─────────────────────────────────────────
    let (text_tx, mut text_rx) = tokio::sync::mpsc::channel::<String>(32);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            if stdin.read_line(&mut line).is_ok() {
                if text_tx.blocking_send(line).is_err() {
                    break;
                }
            } else {
                break;
            }
        }
    });

    // ── Wake word listener task → channel ────────────────────────────────────
    // Runs as a persistent background task — never cancelled, never restarted.
    // Sends a () through the channel each time the wake word is detected.
    // The main loop just selects on this channel alongside stdin.
    let (wake_tx, mut wake_rx) = tokio::sync::mpsc::channel::<()>(4);
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
                Ok(()) => {
                    if wake_tx.send(()).await.is_err() {
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
        if mode != RunMode::Voice {
            print!("You: ");
            io::stdout().flush().ok();
        }

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
                        tts::speak("Goodbye.", &config.voice.mode).await.ok();
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

                print!("Luna: ");
                io::stdout().flush().ok();

                // Pick model based on complexity
                let effective_prompt = match classify(&input) {
                    QueryComplexity::Simple => {
                        tracing::debug!("Simple query — using fast path");
                        // For simple queries, add a brevity instruction
                        format!("{}\n\nThis is a simple conversational query. Respond in 1-2 sentences max, no tools needed.", system_prompt)
                    }
                    QueryComplexity::Complex => system_prompt.to_string(),
                };

                match react.run(&input, &mut memory, &effective_prompt).await {
                    Ok(response) => {
                        println!("{}", response);
                        if config.voice.mode != VoiceMode::Off {
                            tts::speak(&response, &config.voice.mode).await.ok();
                        }
                    }
                    Err(e) => eprintln!("\nLuna error: {}", e),
                }
                println!();
            }

            // ── Wake word fired ──────────────────────────────────────────────
            // The background task detected the wake word and sent () here.
            // We only act on it when not in pure text mode.
            _ = wake_rx.recv(), if mode != RunMode::Text => {
                match process_voice_interaction(
                    config, &stt, &mut memory, &react, &system_prompt,
                ).await? {
                    ControlFlow::Exit => return Ok(()),
                    ControlFlow::SwitchToText => {
                        mode = RunMode::Text;
                        println!("  [Switched to text mode]");
                    }
                    ControlFlow::SwitchToVoice => {
                        mode = RunMode::Voice;
                        println!("  [Switched to voice mode]");
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
    if (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('(') && t.ends_with(')')) {
        return true;
    }
    let hallucinations = [
        "thank you for watching",
        "thanks for watching",
        "see you in the next video",
        "see you later",
        "please subscribe",
        "like and subscribe",
    ];
    hallucinations.iter().any(|h| t.contains(h))
}
