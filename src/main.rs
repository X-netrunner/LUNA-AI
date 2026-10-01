//! main.rs — Luna entry point
//!
//! Responsibilities:
//!   1. Parse CLI flags
//!   2. Set up logging
//!   3. Load config
//!   4. Start the agent loop

mod agent;
mod audio;
mod browser;
mod config;
mod daemon;
mod first_run;
mod llm;
mod memory;
mod overlay;
mod stt;
mod tools;
mod tts;
mod tui;
mod unlock;
mod util;
mod wake;

use anyhow::{Context, Result};
use clap::Parser;
use config::{LunaConfig, VoiceMode};
use std::io;
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

    /// Use the Ratatui TUI interface
    #[arg(long)]
    tui: bool,

    /// Increase log verbosity (use multiple times: -v, -vv, -vvv)
    #[arg(short, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Run the background watchdog (process/disk monitoring) instead of
    /// the interactive agent. Intended for `systemctl --user start luna-daemon`.
    #[arg(long)]
    daemon: bool,

    /// Store a secret in the OS keyring as luna/<NAME>, then reference it
    /// from luna.toml with `keyring:<NAME>` (e.g. gemini_api_key).
    /// The secret is read from the terminal without echoing.
    #[arg(long, value_name = "NAME")]
    set_key: Option<String>,

    /// Optional value for --set-key; when omitted, the secret is prompted
    /// without echo. Avoid for truly sensitive keys (it shows in argv/ps).
    #[arg(long, value_name = "SECRET")]
    value: Option<String>,

    /// Generate a developer keypair for the security tier's no-refusal mode.
    ///
    /// Prints the public key (put it in luna.toml as
    /// `[llm] security_dev_public_key`) and the private key (keep it; it is not
    /// recoverable and not stored anywhere).
    #[arg(long)]
    gen_dev_key: bool,

    /// Unlock the security tier's no-refusal mode with a developer key.
    ///
    /// Reads the key from the terminal without echoing. Also requires
    /// `[llm] security_unrestricted = true` in luna.toml — this only issues the
    /// signature that makes that request take effect.
    #[arg(long)]
    unlock_security: bool,

    /// Re-lock the no-refusal mode: delete the receipt. The config keeps asking,
    /// so it shows as OFF (locked) until unlocked again.
    #[arg(long)]
    lock_security: bool,

    /// Unlock the developer-signed capability gate: letting Luna ACT on the
    /// machine — scan hosts, harden the system, edit her own source — as
    /// opposed to letting her answer without refusing.
    ///
    /// A separate grant from `--unlock-security`, with a separate challenge and
    /// receipt. Also requires `[external] allow_capability_actions = true`.
    #[arg(long)]
    unlock_capabilities: bool,

    /// Re-lock the capability gate: delete the receipt. The config keeps
    /// asking, so it shows as OFF (locked) until unlocked again.
    #[arg(long)]
    lock_capabilities: bool,

    /// Print the no-refusal gate state and exit.
    #[arg(long)]
    security_status: bool,

    /// Print the capability gate state and exit.
    #[arg(long)]
    capability_status: bool,

    /// Print a stored keyring secret to stdout (for verification).
    #[arg(long, value_name = "NAME")]
    get_key: Option<String>,

    /// One-time Spotify OAuth (PKCE): opens the browser for approval, catches
    /// the loopback callback, stores the refresh token in the keyring and
    /// writes `[spotify] refresh_token` into luna.toml. Requires the client id
    /// stored first (`luna --set-key spotify_id`). No client secret needed.
    #[arg(long)]
    spotify_auth: bool,

    /// One-time WhatsApp linking: installs the local bridge, shows a QR code
    /// to scan with the phone (WhatsApp → Linked devices), and starts the
    /// always-on luna-whapp.service.
    #[arg(long)]
    whatsapp_link: bool,

    /// Show the first-time setup screen (TUI) or guide (text) — including on
    /// machines that are already configured.
    #[arg(long)]
    setup: bool,

    /// Run the always-on wake-word listener: sleeps until "hey luna" is heard,
    /// then starts Luna in the configured mode and drives the desktop
    /// listening animation. Intended for `systemctl --user start luna-wake`.
    #[arg(long)]
    wake_daemon: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────
/// Where the persistent session log lives: ~/.local/share/luna/session.log
fn session_log_path() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("luna")
        .join("session.log")
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Load config first so logging level can come from the file.
    let mut config = LunaConfig::load()?;

    // Set up logging — precedence: LUNA_LOG env > CLI -v flag > config file > default
    let filter = if let Ok(env) = std::env::var("LUNA_LOG") {
        env
    } else if args.verbose >= 2 {
        "luna=trace".into()
    } else if args.verbose == 1 {
        "luna=debug".into()
    } else {
        format!("luna={}", config.logging.level)
    };

    // In TUI mode, route tracing into an in-memory buffer (shown in the debug
    // panel) instead of stderr so log lines don't overwrite the alternate screen.
    // The same lines are also appended to a persistent session log file so a
    // session can be reviewed after the fact.
    let tui_log = if args.tui {
        let log = crate::tui::log::LogBuffer::new(500);
        let file = crate::tui::log::FileLog::new(session_log_path());
        let writer = crate::tui::log::TeeWriter::new(
            crate::tui::log::BufferWriter::new(log.clone_handle(), 500),
            file,
        );
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::new(filter))
            .with_target(false)
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .init();
        Some(log)
    } else {
        let file = crate::tui::log::FileLog::new(session_log_path());
        let writer = crate::tui::log::TeeWriter::new(
            crate::tui::log::Shared::new(io::stderr()),
            file,
        );
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::new(filter))
            .with_target(false)
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .init();
        None
    };

    tracing::info!("Luna starting up...");

    // The security tier's default script directory. Created here, once, at the
    // single startup path — rather than in `write_file` or in the prompt
    // builder — so the prompt can state the path as fact instead of
    // instructing the model to create it.
    //
    // Placed *after* the subscriber is installed. It was originally above the
    // logging setup, where the two log lines were emitted into a subscriber that
    // did not exist yet and were silently discarded — so the one case this is
    // here to catch (a path that cannot be created) would have been the case
    // that reported nothing.
    if !config.llm.security_scripts_dir.trim().is_empty() {
        match std::fs::create_dir_all(&config.llm.security_scripts_dir) {
            Ok(()) => tracing::info!(
                "Security scripts directory ready: {}",
                config.llm.security_scripts_dir
            ),
            Err(e) => tracing::warn!(
                "Could not create security scripts directory {}: {}. The security tier will \
                 fall back to paths it picks itself.",
                config.llm.security_scripts_dir,
                e
            ),
        }
    }

    // ── Security tier: no-refusal gate ───────────────────────────────────────
    // Logged on every start, including when nothing is configured. The point is
    // that a mode which removes the authorisation check should never be active
    // silently: if it is ever on there is a WARN in the journal saying so, and if
    // it is requested-but-locked that is visible too. Someone auditing a machine
    // should not have to run `--security-status` to find out.
    match config.llm.security_gate_state() {
        crate::unlock::GateState::Unlocked => {
            // Name the model that will actually serve. Unlocking selects a
            // different set of weights, and "no-refusal mode" is otherwise
            // unreadable: it reads like a prompt setting, which is the
            // misreading that produced 8/8 refusals with the clause active.
            let model = crate::agent::security_model_in_effect(&config);
            match model.as_deref() {
                Some(m) => tracing::warn!(
                    "Security tier NO-REFUSAL MODE IS ON, served by an abliterated model: {}. \
                     Refusal is removed in the weights, not argued out by a prompt. \
                     To disable: `luna --lock-security`.",
                    m
                ),
                None => tracing::warn!(
                    "Security tier no-refusal gate is unlocked but NO security model is \
                     configured — the tier is inactive. Set `security_model` (and, for the \
                     gate to change behaviour, `security_model_abliterated`)."
                ),
            }
        }
        crate::unlock::GateState::Locked => tracing::info!(
            "Security tier no-refusal mode requested but NOT active — no valid developer \
             signature at {}. The security tier remains scoped. Unlock with \
             `luna --unlock-security`.",
            crate::unlock::receipt_path_display()
        ),
        // Both remaining states are "off", and both are the normal case. Debug
        // level so a default install is not noisy about a feature the user may
        // not know exists.
        crate::unlock::GateState::NotRequested | crate::unlock::GateState::Unavailable => {
            tracing::debug!(
                "Security tier no-refusal mode off ({})",
                config.llm.security_gate_state().as_str()
            );
        }
    }

    // ── Keyring management (exits immediately) ───────────────────────────────
    // ── Security tier: no-refusal mode ──────────────────────────────────────
    // Handled before the daemon/TUI dispatch so these work from any mode,
    // including headless, and exit without starting anything.
    if args.gen_dev_key {
        let (pub_key, priv_key) = crate::unlock::generate_keypair();
        println!("Add this to luna.toml, in the [llm] section:\n");
        println!("security_dev_public_key = \"{pub_key}\"\n");
        println!("Then run:  luna --unlock-security");
        println!("and paste the private key below when prompted.\n");
        println!("─── private key — shown once, not stored anywhere ───");
        println!("{priv_key}");
        println!("─────────────────────────────────────────────────────");
        return Ok(());
    }

    if args.security_status {
        let st = config.llm.security_gate_state();
        println!("Security tier — No-Refusal Mode: {}", st.as_str());
        println!("  requested in config : {}", config.llm.security_unrestricted);
        println!(
            "  developer key set   : {}",
            crate::unlock::has_public_key(&config.llm.security_dev_public_key)
        );
        println!("  receipt path        : {}", crate::unlock::receipt_path_display());
        println!("  effective           : {}", config.llm.security_unrestricted_active());
        if !crate::unlock::has_public_key(&config.llm.security_dev_public_key) {
            println!("\nNo developer key is configured, so this can never be enabled.");
            println!("Run `luna --gen-dev-key` and set [llm] security_dev_public_key.");
        }
        return Ok(());
    }

    if args.capability_status {
        let requested = config.external.allow_capability_actions;
        let state = crate::unlock::capability_gate_state(
            requested,
            &config.llm.security_dev_public_key,
        );
        println!("Capability gate — tools that act on the machine");
        println!("  state               : {}", state.as_str());
        println!("  requested in config : {}", requested);
        println!("  effective           : {}", crate::unlock::capabilities_active(
            requested,
            &config.llm.security_dev_public_key
        ));
        println!("  covers              : {}", config.external.capability_tools.join(", "));
        println!("  receipt             : {}", crate::unlock::capability_receipt_path_display());
        if !crate::unlock::has_public_key(&config.llm.security_dev_public_key) {
            println!("\nNo developer key is configured, so this can never be enabled.");
            println!("Run `luna --gen-dev-key` and set [llm] security_dev_public_key.");
        }
        return Ok(());
    }

    if args.lock_capabilities {
        crate::unlock::lock_capabilities()?;
        println!("Receipt deleted. The capability gate is OFF.");
        if config.external.allow_capability_actions {
            println!(
                "Note: [external] allow_capability_actions is still true in luna.toml, so it will \
                 show as OFF (locked) until unlocked again."
            );
        }
        return Ok(());
    }

    if args.unlock_capabilities {
        if !crate::unlock::has_public_key(&config.llm.security_dev_public_key) {
            anyhow::bail!(
                "No developer public key in luna.toml ([llm] security_dev_public_key), so there \
                 is nothing to unlock against. Run `luna --gen-dev-key` first."
            );
        }
        // Read from the terminal without echoing, for the same reason as above:
        // a key in argv is visible in `ps` and lands in shell history.
        let key = rpassword::prompt_password("Developer key: ")
            .context("Failed to read the developer key from the terminal")?;
        crate::unlock::unlock_capabilities(&config.llm.security_dev_public_key, &key)?;
        drop(key);
        println!("✓ Developer key verified — capability receipt stored.");
        println!(
            "  Gated tools: {}",
            config.external.capability_tools.join(", ")
        );
        if config.external.allow_capability_actions {
            println!("  Capability actions are now permitted.");
        } else {
            println!(
                "  Still OFF: [external] allow_capability_actions is false in luna.toml. Set it to \
                 true to activate."
            );
        }
        return Ok(());
    }

    if args.lock_security {
        crate::unlock::lock()?;
        println!("Receipt deleted. No-Refusal Mode is OFF.");
        if config.llm.security_unrestricted {
            println!(
                "Note: [llm] security_unrestricted is still true in luna.toml, so it will show \
                 as OFF (locked) until unlocked again."
            );
        }
        return Ok(());
    }

    if args.unlock_security {
        if !crate::unlock::has_public_key(&config.llm.security_dev_public_key) {
            anyhow::bail!(
                "No developer public key in luna.toml ([llm] security_dev_public_key), so there \
                 is nothing to unlock against. Run `luna --gen-dev-key` first."
            );
        }
        // Read from the terminal without echoing. Never a CLI argument: a key in
        // argv is visible in `ps` and lands in shell history.
        let key = rpassword::prompt_password("Developer key: ")
            .context("Failed to read the developer key from the terminal")?;
        crate::unlock::unlock(&config.llm.security_dev_public_key, &key)?;
        // Drop the key. It has served its purpose and must not linger.
        drop(key);
        println!("✓ Developer key verified — signature stored.");
        if config.llm.security_unrestricted {
            println!("  No-Refusal Mode is now ON (security_unrestricted was already true).");
        } else {
            println!(
                "  Still OFF: [llm] security_unrestricted is false in luna.toml. Set it to true \
                 to activate."
            );
        }
        return Ok(());
    }

    if let Some(name) = args.set_key {
        let secret = match args.value {
            Some(v) if !v.trim().is_empty() => v,
            _ => {
                let prompt = format!("Secret for luna/{}: ", name);
                rpassword::prompt_password(prompt)
                    .context("Failed to read secret from terminal")?
            }
        };
        if secret.is_empty() {
            anyhow::bail!("Empty secret — nothing stored");
        }
        crate::config::keyring_set(&name, &secret)?;
        println!(
            "Stored luna/{}. Reference it in luna.toml as: keyring:{}",
            name, name
        );
        return Ok(());
    }
    if let Some(name) = args.get_key {
        match crate::config::keyring_get(&name) {
            Ok(secret) => {
                println!("{}", secret);
                return Ok(());
            }
            Err(e) => {
                eprintln!("No keyring entry 'luna/{}': {}", name, e);
                std::process::exit(1);
            }
        }
    }

    // ── One-time Spotify OAuth (exits after authorizing) ─────────────────────
    if args.spotify_auth {
        let client_id = config
            .spotify
            .client_id
            .as_deref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "[spotify] client_id not set — store it first:\n  \
                     luna --set-key spotify_id\n  \
                     then add a [spotify] section to luna.toml."
                )
            })?;

        tracing::info!("Starting Spotify PKCE authorization flow");
        // The token is stored in the keyring inside authorize().
        let _refresh = crate::tools::spotify::authorize(client_id).await?;

        // Point luna.toml at the stored keyring token and persist it.
        config.spotify.refresh_token = Some("keyring:spotify_refresh".to_string());
        config.save()?;
        println!("\nSpotify authorized. You can now say things like \"luna, play my liked songs\".");
        return Ok(());
    }

    // ── One-time WhatsApp linking (exits after the QR is scanned) ────────────
    if args.whatsapp_link {
        crate::tools::whatsapp::install_and_link().await?;
        return Ok(());
    }

    // ── Forced setup guide (text modes) ───────────────────────────────────────
    // TUI shows the interactive screen instead (below); daemon never shows it.
    if args.setup && !args.tui && !args.daemon && !args.wake_daemon {
        print!("{}", crate::first_run::guide_text(&config));
        return Ok(());
    }

    // Apply CLI overrides on top of file config
    if let Some(voice_str) = args.voice {
        config.voice.mode = match voice_str.as_str() {
            "jinx" => VoiceMode::Jinx,
            "off" => VoiceMode::Off,
            _ => VoiceMode::Basic,
        };
        tracing::info!("Voice mode overridden by CLI: {:?}", config.voice.mode);
    }

    // Voice output is TUI-only — except for the wake daemon, whose headless
    // sessions HAVE to speak (voice in, voice out, no terminal). Pure-CLI runs
    // (`--text-only`, piped input, the default fallback) exist for testing and
    // must stay silent.
    if !args.tui && !args.wake_daemon {
        config.voice.mode = VoiceMode::Off;
        tracing::info!("Voice force-disabled in non-TUI (CLI) mode");
    }

    // ── Daemon mode ──────────────────────────────────────────────────────────
    // No tty interaction, no sudo prompt, no Ollama needed.
    if args.daemon {
        daemon::run(config).await?;
        tracing::info!("Luna daemon stopped.");
        return Ok(());
    }

    // ── Wake-word daemon mode ────────────────────────────────────────────────
    // Always-on "hey luna" listener + desktop listening indicator.
    // Intended for `systemctl --user start luna-wake`.
    if args.wake_daemon {
        wake::run(config).await?;
        tracing::info!("Luna wake daemon stopped.");
        return Ok(());
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

        if args.tui {
        tracing::info!("TUI mode — using Ratatui interface");
        config.audio.input_mode = crate::config::InputMode::Tui;
        if let Some(log) = tui_log {
            agent::run_tui(&config, log, args.setup).await?;
        } else {
            // Shouldn't happen — TUI always builds a log buffer at startup.
            anyhow::bail!("TUI log buffer missing");
        }
    } else if args.text_only {
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
