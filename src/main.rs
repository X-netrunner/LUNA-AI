//! main.rs — Luna entry point
//!
//! Responsibilities:
//!   1. Parse CLI flags
//!   2. Set up logging
//!   3. Load config
//!   4. Start the agent loop

mod agent;
mod audio;
mod config;
mod llm;
mod memory;
mod stt;
mod tools;
mod tts;

use anyhow::Result;
use rpassword;
use clap::Parser;
use config::{LunaConfig, VoiceMode};
use tracing_subscriber::EnvFilter;

// ── CLI flags ─────────────────────────────────────────────────────────────────
// These can override config file values at runtime.
// Example: `luna --voice jinx` or `luna --no-voice`

#[derive(Parser, Debug)]
#[command(name = "luna", about = "Local AI assistant — fast, personal, yours")]
struct Args {
    /// Override voice mode: basic | jinx | off
    #[arg(long, value_name = "MODE")]
    voice: Option<String>,

    /// Skip voice input, use text mode only (useful for debugging)
    #[arg(long)]
    text_only: bool,

    /// Increase log verbosity (use multiple times: -v, -vv, -vvv)
    #[arg(short, action = clap::ArgAction::Count)]
    verbose: u8,
}

// ── Entry point ───────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Set up logging — LUNA_LOG env var overrides verbosity flag
    // Example: LUNA_LOG=debug luna
    let filter = match args.verbose {
        0 => "luna=info",
        1 => "luna=debug",
        _ => "luna=trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("LUNA_LOG").unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_target(false) // cleaner output — no module path prefix
        .without_time() // skip timestamps for now, add later if needed
        .init();

    tracing::info!("Luna starting up...");

    // Load config — creates defaults if luna.toml doesn't exist yet
    let mut config = LunaConfig::load()?;

    // Apply CLI overrides on top of file config
    if let Some(voice_str) = args.voice {
        config.voice.mode = match voice_str.as_str() {
            "jinx" => VoiceMode::Jinx,
            "off" => VoiceMode::Off,
            _ => VoiceMode::Basic,
        };
        tracing::info!("Voice mode overridden by CLI: {:?}", config.voice.mode);
    }

    // Log active settings so we know exactly what we're running
    tracing::info!("Model:      {}", config.llm.model);
    tracing::info!("Voice mode: {:?}", config.voice.mode);
    tracing::info!("Input mode: {:?}", config.audio.input_mode);

    // ── Hand off to the agent ────────────────────────────────────────────────
    // agent::run() is the main loop — it never returns unless something fails
    // or the user says "exit" / hits Ctrl+C.
    // ── Sudo password ────────────────────────────────────────────────────────
    // Prompt once at startup using /dev/tty so it cannot interfere with
    // Luna's stdin reader — keystrokes go to the password prompt only.
    if config.agent.sudo_password.is_none() {
        if let Ok(pass) = prompt_sudo_password() {
            if !pass.is_empty() {
                config.agent.sudo_password = Some(pass);
                tracing::debug!("Sudo password set for session");
            } else {
                tracing::info!("No sudo password — sudo commands will drop privileges");
            }
        }
    }

        if args.text_only {
        tracing::info!("Text-only mode — voice input disabled");
        agent::run_text(&config).await?;
    } else {
        agent::run(&config).await?;
    }

    tracing::info!("Luna shutting down. Goodbye.");
    Ok(())
}

/// Prompt for sudo password directly from /dev/tty.
/// Using /dev/tty instead of stdin means the input is completely isolated
/// from Luna's async stdin reader — no keystrokes can leak into chat.
fn prompt_sudo_password() -> anyhow::Result<String> {
    use std::io::{self, Write};
    // Open /dev/tty directly — this is the actual terminal even when stdin is piped
    let tty = std::fs::OpenOptions::new().read(true).write(true).open("/dev/tty")?;
    let tty_clone = tty.try_clone()?;
    let mut tty_out = io::BufWriter::new(&tty);
    write!(tty_out, "  [Luna] sudo password (leave blank to skip): ")?;
    tty_out.flush()?;

    // Read without echo using the `rpassword` approach via termios, or fall
    // back to a visible read if that fails — either way it's on /dev/tty
    let pass = rpassword::read_password_with_config(
        rpassword::ConfigBuilder::default()
            .input_reader(io::BufReader::new(tty_clone))
            .build(),
    )
    .unwrap_or_default();
    Ok(pass)
}
