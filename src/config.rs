//! config.rs — Luna's central configuration
//!
//! Loads luna.toml from ~/.config/luna/luna.toml
//! Falls back to sane defaults if the file doesn't exist yet.
//! Every module reads from this — it's the single source of truth.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Top-level config struct ───────────────────────────────────────────────────

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

    #[serde(default)]
    pub todoist: TodoistConfig,

    #[serde(default)]
    pub proactive: ProactiveConfig,
}

// ── Agent behaviour ───────────────────────────────────────────────────────────
#[derive(Debug, Deserialize, Serialize)]
pub struct AgentConfig {
    pub name: String,
    pub system_prompt: String,
    pub max_react_iterations: u8,
    pub sudo_password: Option<String>,
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
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
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
    Basic,
    Jinx,
    Off,
}

impl Default for VoiceMode {
    fn default() -> Self {
        VoiceMode::Basic
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VoiceConfig {
    pub mode: VoiceMode,
    pub piper_bin: PathBuf,
    pub piper_model: PathBuf,
    pub rvc_model: Option<PathBuf>,
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
    PushToTalk,
    WakeWord,
    Both,
}

impl Default for InputMode {
    fn default() -> Self {
        InputMode::Both
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AudioConfig {
    pub input_mode: InputMode,
    pub ptt_key: String,
    pub wake_word: String,
    pub wake_aliases: Vec<String>,
    pub vad_silence_ms: u64,
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
                "hello luna".into(),
                "hello lana".into(),
                "hey lana".into(),
                "hi luna".into(),
                "hi lana".into(),
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
    pub context_window: usize,
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

// ── Todoist integration ───────────────────────────────────────────────────────
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct TodoistConfig {
    /// Todoist API token — get yours at todoist.com/app/settings/integrations
    /// Leave unset to disable Todoist tools.
    pub api_token: Option<String>,
}

// ── Proactive background monitoring ───────────────────────────────────────────
#[derive(Debug, Deserialize, Serialize)]
pub struct ProactiveConfig {
    /// Master switch — set false to disable all background checks
    pub enabled: bool,
    /// How often to check, in minutes
    pub check_interval_mins: u64,
    /// Notify when battery drops to/below this percent while discharging
    pub battery_low_threshold: u32,
    /// Notify when disk usage on / reaches this percent
    pub disk_full_threshold: u32,
    /// Notify when pacman updates are available (requires pacman-contrib)
    pub check_updates: bool,
}

impl Default for ProactiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_mins: 15,
            battery_low_threshold: 20,
            disk_full_threshold: 90,
            check_updates: true,
        }
    }
}

// ── Loading logic ─────────────────────────────────────────────────────────────
impl LunaConfig {
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

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let toml_str =
            toml::to_string_pretty(self).context("Failed to serialize config to TOML")?;

        std::fs::write(&path, toml_str)
            .with_context(|| format!("Failed to write config to {:?}", path))?;

        Ok(())
    }

    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna")
            .join("luna.toml")
    }
}

impl Default for LunaConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            llm: LlmConfig::default(),
            voice: VoiceConfig::default(),
            audio: AudioConfig::default(),
            memory: MemoryConfig::default(),
            todoist: TodoistConfig::default(),
            proactive: ProactiveConfig::default(),
        }
    }
}

fn default_true() -> bool {
    true
}
