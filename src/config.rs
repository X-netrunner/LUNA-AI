//! config.rs — Luna's central configuration
//!
//! Loads luna.toml from ~/.config/luna/luna.toml
//! Falls back to sane defaults if the file doesn't exist yet.
//! Every module reads from this — it's the single source of truth.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Top-level config struct ───────────────────────────────────────────────────
// Each field maps to a [section] in luna.toml
// #[serde(default)] means "use Default::default() if key is missing in file"

#[derive(Debug, Deserialize, Serialize)]
pub struct LunaConfig {
    #[serde(default)]
    pub agent: AgentConfig,

    #[serde(default)]
    pub llm: LlmConfig,

    #[serde(default)]
    pub voice: VoiceConfig,

    #[serde(default)]
    pub audio: AudioConfig,

    #[serde(default)]
    pub memory: MemoryConfig,
}

// ── Agent behaviour ───────────────────────────────────────────────────────────
#[derive(Debug, Deserialize, Serialize)]
pub struct AgentConfig {
    /// Luna's name — used in prompts and logs
    pub name: String,

    /// How Luna introduces herself / her personality baseline
    pub system_prompt: String,

    /// Max tool-call iterations per ReAct loop before giving up
    pub max_react_iterations: u8,

    /// Sudo password for Luna to use when running shell commands
    pub sudo_password: Option<String>,

    /// Use Ollama native tool-call format.
    /// Set false if you see 500 errors — works with any model.
    #[serde(default = "default_true")]
    pub native_tools: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Luna".into(),
            system_prompt: "You are Luna, a sharp and self-aware AI assistant running \
                 locally on an Arch Linux machine. You are direct, efficient, \
                 and have a dry wit. You have full access to the user's \
                 desktop, filesystem, and shell. Think before acting. \
                 When you use a tool, say so briefly. Never pretend you \
                 can't do something — figure it out. \
                 IMPORTANT: Never guess or hallucinate real-time data. \
                 For the current time or date, always call run_shell with \
                 `date '+%H:%M %Z on %A %d %B %Y'`. \
                 For system state (RAM, CPU, disk, processes, network), \
                 always query with run_shell — never assume."
                .into(),
            max_react_iterations: 8,
            sudo_password: None,
            native_tools: true,
        }
    }
}

// ── LLM settings ─────────────────────────────────────────────────────────────
#[derive(Debug, Deserialize, Serialize)]
pub struct LlmConfig {
    /// Ollama base URL — usually http://localhost:11434
    pub base_url: String,

    /// Which model to use — must be pulled in Ollama already
    pub model: String,

    /// Sampling temperature — 0.0 = deterministic, 1.0 = creative
    pub temperature: f32,

    /// How many tokens to generate max per response
    pub max_tokens: u32,

    /// Whether to use the thinking/reasoning mode when needed
    pub enable_thinking: bool,

    pub fast_model: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: "qwen2.5:7b-instruct".into(),
            temperature: 0.7,
            max_tokens: 2048,
            enable_thinking: true,
            fast_model: None,
        }
    }
}

// ── Voice output settings ─────────────────────────────────────────────────────
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMode {
    /// Piper only — fast, CPU, default
    Basic,
    /// Piper → RVC — GPU, sounds like Jinx
    Jinx,
    /// No TTS at all — text only
    Off,
}

impl Default for VoiceMode {
    fn default() -> Self {
        VoiceMode::Basic
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VoiceConfig {
    /// Which voice pipeline to use
    pub mode: VoiceMode,

    /// Path to the Piper binary
    pub piper_bin: PathBuf,

    /// Path to the Piper voice model (.onnx file)
    pub piper_model: PathBuf,

    /// Path to the RVC model file (only used in Jinx mode)
    pub rvc_model: Option<PathBuf>,

    /// Path to the RVC Python script wrapper
    pub rvc_script: Option<PathBuf>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
        Self {
            mode: VoiceMode::Basic,
            piper_bin: PathBuf::from("/usr/bin/piper"),
            piper_model: home.join(".local/share/luna/voices/basic.onnx"),
            rvc_model: None,
            rvc_script: None,
        }
    }
}

// ── Audio input settings ──────────────────────────────────────────────────────
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    /// Hold a key to record — instant, no false triggers
    PushToTalk,
    /// Say the wake word to activate — hands-free
    WakeWord,
    /// Both available — PTT takes priority if key held
    Both,
}

impl Default for InputMode {
    fn default() -> Self {
        InputMode::Both
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AudioConfig {
    /// Which input method(s) to enable
    pub input_mode: InputMode,

    /// Key to hold for push-to-talk (rdev key name as string)
    pub ptt_key: String,

    /// Wake word to listen for (simple string match on Whisper output)
    pub wake_word: String,

    /// Alternative wake words
    pub wake_aliases: Vec<String>,

    /// Silence duration in ms before VAD considers speech done
    pub vad_silence_ms: u64,

    /// Audio sample rate — Whisper wants 16000
    pub sample_rate: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_mode: InputMode::Both,
            ptt_key: "ControlLeft".into(),
            wake_word: "hey luna".into(),
            wake_aliases: vec![
                "hey luna".into(),
                "hay luna".into(),
                "hello lana".into(),
                "hey lana".into(),
                "hi luna".into(),
                "luna".into(),
            ],
            vad_silence_ms: 2000,
            sample_rate: 16000,
        }
    }
}

// ── Memory settings ───────────────────────────────────────────────────────────
#[derive(Debug, Deserialize, Serialize)]
pub struct MemoryConfig {
    /// How many past messages to keep in context window
    pub context_window: usize,

    /// Path to persist conversation history
    pub history_path: PathBuf,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna");
        Self {
            context_window: 20,
            history_path: data_dir.join("history.json"),
        }
    }
}

//--- History of COmmands -------------------------------------------------------

pub fn load_shell_history() -> Vec<String> {
    let history_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".local/share/fish/fish_history");

    let Ok(content) = std::fs::read_to_string(&history_path) else {
        return Vec::new();
    };

    // Fish history format: "- cmd: <command>\n  when: <timestamp>"
    content
        .lines()
        .filter(|l| l.starts_with("- cmd:"))
        .map(|l| l.trim_start_matches("- cmd:").trim().to_string())
        .filter(|cmd| !cmd.is_empty())
        // Deduplicate
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(100)
        .collect()
}

// ── Loading logic ─────────────────────────────────────────────────────────────
impl LunaConfig {
    /// Load config from ~/.config/luna/luna.toml
    /// If the file doesn't exist, return defaults and create the file.
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if !config_path.exists() {
            tracing::info!("No config found, creating defaults at {:?}", config_path);
            let config = LunaConfig::default();
            config.save().context("Failed to save default config")?;
            return Ok(config);
        }

        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config at {:?}", config_path))?;

        let config: LunaConfig =
            toml::from_str(&raw).context("Failed to parse luna.toml — check for syntax errors")?;

        tracing::info!("Config loaded from {:?}", config_path);
        Ok(config)
    }

    /// Save current config back to disk
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();

        // Make sure the directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let toml_str =
            toml::to_string_pretty(self).context("Failed to serialize config to TOML")?;

        std::fs::write(&path, toml_str)
            .with_context(|| format!("Failed to write config to {:?}", path))?;

        Ok(())
    }

    /// Returns ~/.config/luna/luna.toml
    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna")
            .join("luna.toml")
    }
}

// Use Default trait so missing TOML sections fall back gracefully
impl Default for LunaConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            llm: LlmConfig::default(),
            voice: VoiceConfig::default(),
            audio: AudioConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
}

fn default_true() -> bool { true }
