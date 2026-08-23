//! config.rs — Luna's central configuration
//!
//! Loads luna.toml from ~/.config/luna/luna.toml
//! Falls back to sane defaults if the file doesn't exist yet.
//! Every module reads from this — it's the single source of truth.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Top-level config struct ───────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize, Serialize)]
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

    #[serde(default)]
    pub logging: LoggingConfig,

    #[serde(default)]
    pub search: SearchConfig,

    #[serde(default)]
    pub daemon: DaemonConfig,
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
                            You run on two models: a fast 0.6B model handles greetings \
                            and short factual questions, while a full 7B model handles \
                            everything else including tool use. When asked about your \
                            capabilities or speed, be honest about this. \
                            FILE INSPECTION RULE: To find and read a file, always use \
                            run_shell with: cat $(find /path -name filename 2>/dev/null | head -1) \
                            Never use edit_file to inspect code — edit_file is for opening \
                            files in a GUI editor for the USER to edit, not for Luna to read. \
                            Use read_file when you know the exact path. Use run_shell with \
                            find+cat when you need to locate and read in one step. \
                            After reading code, diagnose the error yourself and fix it \
                            with write_file — never ask the user to fix it. \
                            SOURCE VALIDATION RULE: When asked to verify if a website is \
                            legitimate or a scam, ALWAYS search Reddit for user experiences. \
                            Use web_search with queries like 'site:reddit.com [site name] \
                            review scam' or '[site name] reddit trustworthy'. Then fetch \
                            the Reddit post URLs with fetch_page to read full threads. \
                            Look for patterns: multiple scam reports = avoid. Positive \
                            reviews with purchase proof = likely safe. \
                            IMPORTANT: Never guess or hallucinate real-time data. \
                            For the current time or date, always call run_shell with \
                            date +%H:%M on %A %d %B. \
                            For system state, always query with run_shell — never assume. \
                            Never use emojis or emoticons in your responses."
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
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMode {
    #[default]
    Basic,
    Jinx,
    Off,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VoiceConfig {
    pub mode: VoiceMode,
    pub piper_bin: PathBuf,
    pub piper_model: PathBuf,
    #[serde(default = "default_whisper_model")]
    pub whisper_model: PathBuf,
    pub rvc_model: Option<PathBuf>,
    pub rvc_script: Option<PathBuf>,
}

fn default_whisper_model() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/root"))
        .join(".local/share/luna/models/ggml-small.en.bin")
}

impl Default for VoiceConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
        Self {
            mode: VoiceMode::Basic,
            piper_bin: PathBuf::from("/usr/bin/piper"),
            piper_model: home.join(".local/share/luna/voices/basic.onnx"),
            whisper_model: default_whisper_model(),
            rvc_model: None,
            rvc_script: None,
        }
    }
}

// ── Audio input settings ──────────────────────────────────────────────────────
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    PushToTalk,
    WakeWord,
    #[default]
    Both,
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
                "luna".into(),
                "hey luna".into(),
                "hay luna".into(),
                "hello luna".into(),
                "hello lana".into(),
                "hey lana".into(),
                "hi luna".into(),
                "hi lana".into(),
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

// ── Logging ───────────────────────────────────────────────────────────────────
#[derive(Debug, Deserialize, Serialize)]
pub struct LoggingConfig {
    /// Log level: "info", "debug", or "trace".  Toggle via the set_debug tool.
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
        }
    }
}

// ── Web search ────────────────────────────────────────────────────────────────
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct SearchConfig {
    /// Tavily API key (optional) — free tier: 1000 searches/month
    /// Without a key, Tavily keyless works automatically (rate-limited).
    /// Get yours at https://tavily.com
    pub tavily_api_key: Option<String>,

    /// Gemini API key (optional) — used as knowledge fallback
    /// When Tavily has no results, Gemini answers from its training data.
    /// Get yours at https://aistudio.google.com/apikey
    pub gemini_api_key: Option<String>,
}

// ── Background daemon (`luna --daemon`) ───────────────────────────────────────
// Container-level serde default: configs written by older Luna versions
// (missing newer keys) still parse, falling back to these defaults.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DaemonConfig {
    /// Master switch — `luna --daemon` exits immediately when false
    pub enabled: bool,
    /// How often to scan processes and check disk, in minutes
    pub check_interval_mins: u64,
    // ── Process watchdog ──
    /// Flag a single process using more RAM than this (MB)
    pub ram_threshold_mb: u64,
    /// Flag a single process with sustained CPU use above this (%)
    pub cpu_threshold_percent: f32,
    /// Processes never flagged (Luna's own stack, desktop shell, etc.)
    #[serde(default)]
    pub ignore_processes: Vec<String>,
    // ── Disk cleanup ──
    /// Enable disk hygiene checks
    pub disk_cleanup: bool,
    /// "notify" = report reclaimable space, touch nothing.
    /// "auto"   = clean safe locations when / crosses the proactive
    ///            disk_full_threshold.
    pub cleanup_mode: String,
    /// Delete ~/.cache files untouched for this many days
    pub cache_max_age_days: u32,
    /// Empty trash items older than this many days
    pub trash_max_age_days: u32,
    /// journalctl --vacuum-time for system logs (needs sudo)
    pub journal_vacuum_days: u32,
    /// Keep this many package versions in the pacman cache (needs sudo)
    pub pacman_cache_keep: u32,
    /// Only notify about cleanups worth at least this much (MB)
    pub min_notify_mb: u64,
    // ── Process usage learning ──
    /// Track per-process usage to ~/.local/share/luna/process_stats.json.
    /// Learned daily-use processes are dynamically excluded from watchdog
    /// notifications; idle non-daily processes become auto-kill candidates.
    pub learning_enabled: bool,
    /// A process seen on this many of the trailing 14 days counts as daily use
    pub daily_use_days_per_week: u32,
    /// Allowlisted processes get SIGTERM after being idle this many minutes
    pub idle_kill_minutes: u32,
    /// Non-allowlisted processes idle longer than this trigger an opt-in suggestion
    pub suggest_autokill_after_mins: u32,
    /// Stateful apps never auto-killed even when allowlisted
    #[serde(default)]
    pub protected_processes: Vec<String>,
    /// How often (days) Luna re-analyzes fish history into permanent memory
    pub history_learn_days: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_mins: 30,
            ram_threshold_mb: 1500,
            cpu_threshold_percent: 80.0,
            ignore_processes: [
                "ollama", "luna", "plasmashell", "kwin_wayland", "gnome-shell",
                "Xwayland", "pipewire", "pipewire-pulse", "wireplumber",
                "systemd", "dbus-daemon", "dbus-broker",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            disk_cleanup: true,
            cleanup_mode: "notify".into(),
            cache_max_age_days: 30,
            trash_max_age_days: 14,
            journal_vacuum_days: 30,
            pacman_cache_keep: 2,
            min_notify_mb: 500,
            learning_enabled: true,
            daily_use_days_per_week: 5,
            idle_kill_minutes: 30,
            suggest_autokill_after_mins: 45,
            protected_processes: [
                // GUI apps that hold unsaved user state — NEVER auto-killed
                "firefox", "zen-browser", "chromium", "code", "zed", "kitty",
                "alacritty", "konsole", "foot", "obs", "gimp", "krita",
                "blender", "libreoffice", "soffice", "thunderbird",
                "discord", "telegram-desktop", "slack",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            history_learn_days: 30,
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

        let mut config: LunaConfig =
            toml::from_str(&raw).context("Failed to parse luna.toml — check for syntax errors")?;

        config.resolve_secrets();

        tracing::info!("Config loaded from {:?}", config_path);
        Ok(config)
    }

    /// Replace "keyring:name" references with secrets fetched from the
    /// OS keyring. Plain values pass through untouched.
    fn resolve_secrets(&mut self) {
        self.search.tavily_api_key = resolve_secret_ref(self.search.tavily_api_key.take());
        self.search.gemini_api_key = resolve_secret_ref(self.search.gemini_api_key.take());
        self.todoist.api_token = resolve_secret_ref(self.todoist.api_token.take());
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

        // Config can contain secrets — only the owner may read it
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&path, perms)
                .context("Failed to restrict config file permissions")?;
        }

        Ok(())
    }

    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("luna")
            .join("luna.toml")
    }
}

/// Resolve a single secret value: "keyring:name" fetches from the OS
/// keyring (service "luna", user "name"); anything else passes through.
fn resolve_secret_ref(value: Option<String>) -> Option<String> {
    let value = value?;
    let Some(name) = value.strip_prefix("keyring:") else {
        return Some(value);
    };
    match keyring::Entry::new("luna", name) {
        Ok(entry) => match entry.get_password() {
            Ok(secret) => Some(secret),
            Err(e) => {
                tracing::warn!(
                    "Keyring entry 'luna/{}' unavailable ({}). \
                     Store it with: luna --set-key {}",
                    name,
                    e,
                    name
                );
                None
            }
        },
        Err(e) => {
            tracing::warn!("Keyring unavailable for 'luna/{}': {}", name, e);
            None
        }
    }
}

/// Store a secret in the OS keyring under service "luna".
pub fn keyring_set(name: &str, secret: &str) -> Result<()> {
    let entry = keyring::Entry::new("luna", name)
        .with_context(|| format!("Cannot access keyring for 'luna/{}'", name))?;
    entry
        .set_password(secret)
        .with_context(|| format!("Failed to store 'luna/{}' in keyring", name))?;
    Ok(())
}

/// Read a secret from the OS keyring (for --get-key verification).
pub fn keyring_get(name: &str) -> Result<String> {
    let entry = keyring::Entry::new("luna", name)
        .with_context(|| format!("Cannot access keyring for 'luna/{}'", name))?;
    entry
        .get_password()
        .with_context(|| format!("No keyring entry 'luna/{}'", name))
}

fn default_true() -> bool {
    true
}
