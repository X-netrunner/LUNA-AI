//! tui/app.rs — TUI application state and event loop

use crate::config::{LunaConfig, VoiceMode};
use crate::memory::Memory;
use crate::tui::log::LogBuffer;
use crate::tui::onboarding::Onboarding;
use crate::tui::widgets::{ChatHistory, ConfigEditorModal, DebugPanel, InputLine, SettingsMenuModal, ShortcutsBar, ShortcutsModal, SlashCommandMenu, StatusBar};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Current time as milliseconds since the Unix epoch.
fn millis_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

enum AppEvent {
    Input(Event),
    Response(Msg, Duration, Memory, String),
    Metrics,
    Voice(String),
    OnboardingLog(String),
    SpotifyAuthed,
}

#[derive(Debug, Clone)]
pub struct Msg {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
}

/// Shared live metrics written by a background poll task and read by the UI.
#[derive(Default)]
struct Metrics {
    gpu: AtomicU64,
    cpu: AtomicU64,
}

/// Which panel Up/Down/PgUp/PgDn scroll. Tab switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Chat,
    Debug,
}

/// While `key_prompt` is Some, the chat input is replaced by a masked prompt
/// asking for the secret for a `--set-key <name>` typed in chat.
struct KeyPrompt {
    name: String,
    secret: String,
}


/// State for the interactive luna.toml editor modal.
pub struct ConfigState {
    pub path: std::path::PathBuf,
    pub lines: Vec<String>,
    pub cursor_row: usize,
    pub cursor_col: usize,
    pub scroll: usize,
    pub status: String,
    pub modified: bool,
}

impl ConfigState {
    pub fn load() -> Self {
        let path = crate::config::LunaConfig::config_path();
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let lines: Vec<String> = if text.is_empty() {
            vec![String::new()]
        } else {
            text.lines().map(String::from).collect()
        };
        Self {
            path,
            lines,
            cursor_row: 0,
            cursor_col: 0,
            scroll: 0,
            status: String::from("[Ctrl+S] Save · [Ctrl+E] External Editor · [Esc] Exit"),
            modified: false,
        }
    }

    pub fn to_string(&self) -> String {
        self.lines.join("\n")
    }

    pub fn save(&mut self, config: &mut crate::config::LunaConfig) -> bool {
        let content = self.to_string();
        if let Err(e) = toml::from_str::<serde_json::Value>(&content) {
            self.status = format!("✗ TOML Syntax Error: {e}");
            return false;
        }
        if let Err(e) = std::fs::write(&self.path, &content) {
            self.status = format!("✗ File Save Failed: {e}");
            return false;
        }
        match crate::config::LunaConfig::load() {
            Ok(new_cfg) => {
                *config = new_cfg;
                self.modified = false;
                self.status = String::from("✓ Saved & reloaded luna.toml successfully!");
                true
            }
            Err(e) => {
                self.status = format!("✗ Config Reload Error: {e}");
                false
            }
        }
    }

    pub fn insert_char(&mut self, c: char) {
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        let line = &mut self.lines[self.cursor_row];
        let col = self.cursor_col.min(line.len());
        line.insert(col, c);
        self.cursor_col = col + 1;
        self.modified = true;
    }

    pub fn backspace(&mut self) {
        if self.lines.is_empty() {
            return;
        }
        if self.cursor_col > 0 {
            let line = &mut self.lines[self.cursor_row];
            let col = self.cursor_col.min(line.len());
            line.remove(col - 1);
            self.cursor_col = col - 1;
            self.modified = true;
        } else if self.cursor_row > 0 {
            let current = self.lines.remove(self.cursor_row);
            self.cursor_row -= 1;
            let prev_len = self.lines[self.cursor_row].len();
            self.lines[self.cursor_row].push_str(&current);
            self.cursor_col = prev_len;
            self.modified = true;
        }
    }

    pub fn newline(&mut self) {
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        let line = &self.lines[self.cursor_row];
        let col = self.cursor_col.min(line.len());
        let remainder = line[col..].to_string();
        self.lines[self.cursor_row].truncate(col);
        self.cursor_row += 1;
        self.lines.insert(self.cursor_row, remainder);
        self.cursor_col = 0;
        self.modified = true;
    }

    pub fn move_up(&mut self) {
        if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.cursor_col = self.cursor_col.min(self.lines[self.cursor_row].len());
        }
    }

    pub fn move_down(&mut self) {
        if self.cursor_row + 1 < self.lines.len() {
            self.cursor_row += 1;
            self.cursor_col = self.cursor_col.min(self.lines[self.cursor_row].len());
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].len();
        }
    }

    pub fn move_right(&mut self) {
        let line_len = self.lines.get(self.cursor_row).map(|l| l.len()).unwrap_or(0);
        if self.cursor_col < line_len {
            self.cursor_col += 1;
        } else if self.cursor_row + 1 < self.lines.len() {
            self.cursor_row += 1;
            self.cursor_col = 0;
        }
    }

    pub fn delete(&mut self) {
        if self.lines.is_empty() {
            return;
        }
        let line_len = self.lines[self.cursor_row].len();
        if self.cursor_col < line_len {
            self.lines[self.cursor_row].remove(self.cursor_col);
            self.modified = true;
        } else if self.cursor_row + 1 < self.lines.len() {
            let next_line = self.lines.remove(self.cursor_row + 1);
            self.lines[self.cursor_row].push_str(&next_line);
            self.modified = true;
        }
    }

    pub fn home(&mut self) {
        self.cursor_col = 0;
    }

    pub fn end(&mut self) {
        if let Some(line) = self.lines.get(self.cursor_row) {
            self.cursor_col = line.len();
        }
    }

    pub fn page_up(&mut self, amount: usize) {
        self.cursor_row = self.cursor_row.saturating_sub(amount);
        self.cursor_col = self.cursor_col.min(self.lines.get(self.cursor_row).map(|l| l.len()).unwrap_or(0));
    }

    pub fn page_down(&mut self, amount: usize) {
        if !self.lines.is_empty() {
            self.cursor_row = (self.cursor_row + amount).min(self.lines.len() - 1);
            self.cursor_col = self.cursor_col.min(self.lines.get(self.cursor_row).map(|l| l.len()).unwrap_or(0));
        }
    }

    pub fn insert_tab(&mut self) {
        for _ in 0..2 {
            self.insert_char(' ');
        }
    }
}

pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/settings", "Open interactive Settings & Toggle Controls"),
    ("/toml", "Open raw luna.toml file editor modal"),
    ("/clear", "Clear conversation history & reset memory"),
    ("/voice", "Toggle hands-free voice mode (on/off)"),
    ("/shortcuts", "Toggle bottom shortcuts keybindings bar"),
    ("/debug", "Run full system diagnostic & write to Debug Panel"),
    ("/test-voice", "Test Piper TTS & RVC voice subsystem"),
    ("/test-model", "Test LLM Ollama model connectivity"),
    ("/test-stt", "Test Whisper STT audio device & model"),
    ("/test-memory", "Test Memory & vector embeddings"),
    ("/log", "Write custom log line to Debug Panel"),
    ("/setup", "Launch first-run setup wizard"),
    ("/help", "Toggle keyboard shortcuts help dialog"),
    ("/exit", "Exit Luna TUI session"),
];

#[derive(Clone, Debug)]
pub enum SettingType {
    Toggle(bool),
    Value(String),
}

#[derive(Clone, Debug)]
pub struct SettingItem {
    pub key: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub setting_type: SettingType,
}

#[derive(Clone, Debug)]
pub struct SettingCategory {
    pub name: &'static str,
    pub items: Vec<SettingItem>,
}

pub struct SettingsMenuState {
    pub config: LunaConfig,
    pub categories: Vec<SettingCategory>,
    pub cat_index: usize,
    pub item_index: usize,
    pub status: String,
    pub modified: bool,
}

impl SettingsMenuState {
    pub fn load(config: LunaConfig) -> Self {
        let categories = vec![
            SettingCategory {
                name: "🤖 AI & Models",
                items: vec![
                    SettingItem {
                        key: "llm.enable_thinking",
                        label: "Thinking Mode",
                        description: "Enable step-by-step reasoning blocks before answering",
                        setting_type: SettingType::Toggle(config.llm.enable_thinking),
                    },
                    SettingItem {
                        key: "agent.native_tools",
                        label: "Native Tools",
                        description: "Use native Ollama tool schema dispatch",
                        setting_type: SettingType::Toggle(config.agent.native_tools),
                    },
                    SettingItem {
                        key: "llm.model",
                        label: "Primary Model",
                        description: "Ollama model name",
                        setting_type: SettingType::Value(config.llm.model.clone()),
                    },
                ],
            },
            SettingCategory {
                name: "🎙️ Voice & Audio",
                items: vec![
                    SettingItem {
                        key: "wake.enabled",
                        label: "Wake Daemon",
                        description: "Background listener for 'hey luna' wake phrase",
                        setting_type: SettingType::Toggle(config.wake.enabled),
                    },
                    SettingItem {
                        key: "wake.overlay_enabled",
                        label: "Desktop Overlay",
                        description: "Draw Wayland desktop listening animation on activation",
                        setting_type: SettingType::Toggle(config.wake.overlay_enabled),
                    },
                    SettingItem {
                        key: "voice.mode",
                        label: "Voice Mode",
                        description: "Speech output engine (basic, jinx, off)",
                        setting_type: SettingType::Value(format!("{:?}", config.voice.mode)),
                    },
                ],
            },
            SettingCategory {
                name: "⚡ System & Watchdog",
                items: vec![
                    SettingItem {
                        key: "proactive.enabled",
                        label: "Proactive Monitor",
                        description: "Background system health, battery, & disk checks",
                        setting_type: SettingType::Toggle(config.proactive.enabled),
                    },
                    SettingItem {
                        key: "proactive.check_updates",
                        label: "Update Watcher",
                        description: "Notify when Arch Linux pacman updates are pending",
                        setting_type: SettingType::Toggle(config.proactive.check_updates),
                    },
                    SettingItem {
                        key: "daemon.enabled",
                        label: "Process Watchdog",
                        description: "Monitor high RAM/CPU usage and idle processes",
                        setting_type: SettingType::Toggle(config.daemon.enabled),
                    },
                    SettingItem {
                        key: "daemon.learning_enabled",
                        label: "Usage Learning",
                        description: "Learn daily process habits to avoid false-positive kills",
                        setting_type: SettingType::Toggle(config.daemon.learning_enabled),
                    },
                    SettingItem {
                        key: "daemon.disk_cleanup",
                        label: "Disk Hygiene",
                        description: "Clean stale ~/.cache and trash when disk gets low",
                        setting_type: SettingType::Toggle(config.daemon.disk_cleanup),
                    },
                ],
            },
            SettingCategory {
                name: "🛡️ Safety Gates",
                items: vec![
                    SettingItem {
                        key: "external.allow_external_actions",
                        label: "External Actions Gate",
                        description: "HUMAN SWITCH: Allow actions that send messages or alter external state",
                        setting_type: SettingType::Toggle(config.external.allow_external_actions),
                    },
                    SettingItem {
                        key: "updates.allow_apply",
                        label: "Package Upgrade Gate",
                        description: "HUMAN SWITCH: Allow Luna to run pacman upgrade",
                        setting_type: SettingType::Toggle(config.updates.allow_apply),
                    },
                    SettingItem {
                        key: "selfpatch.enabled",
                        label: "Self-Patch Gate",
                        description: "Allow Luna to propose verified source self-modifications",
                        setting_type: SettingType::Toggle(config.selfpatch.enabled),
                    },
                ],
            },
            SettingCategory {
                name: "🌐 Integrations",
                items: vec![
                    SettingItem {
                        key: "vision.enabled",
                        label: "Vision ('Eyes')",
                        description: "Enable desktop screenshot & browser page visual understanding",
                        setting_type: SettingType::Toggle(config.vision.enabled),
                    },
                    SettingItem {
                        key: "browser.enabled",
                        label: "Browser Automation",
                        description: "Drive Chromium browser CDP tasks",
                        setting_type: SettingType::Toggle(config.browser.enabled),
                    },
                    SettingItem {
                        key: "desktop.enabled",
                        label: "Desktop Automation",
                        description: "Drive desktop GUI apps via ydotool & grim",
                        setting_type: SettingType::Toggle(config.desktop.enabled),
                    },
                    SettingItem {
                        key: "sysmode.enabled",
                        label: "Sysmode IDS",
                        description: "Hardening profile switcher & intrusion deception",
                        setting_type: SettingType::Toggle(config.sysmode.enabled),
                    },
                    SettingItem {
                        key: "whatsapp.enabled",
                        label: "WhatsApp Bridge",
                        description: "Enable WhatsApp message sending & reading tools",
                        setting_type: SettingType::Toggle(config.whatsapp.enabled),
                    },
                ],
            },
        ];

        Self {
            config,
            categories,
            cat_index: 0,
            item_index: 0,
            status: String::from("[Space/Enter] Toggle · [S] Save & Auto-Restart Daemon · [T] Raw TOML · [Esc] Close"),
            modified: false,
        }
    }

    pub fn toggle_selected(&mut self) {
        if let Some(cat) = self.categories.get_mut(self.cat_index) {
            if let Some(item) = cat.items.get_mut(self.item_index) {
                match &mut item.setting_type {
                    SettingType::Toggle(ref mut b) => {
                        *b = !*b;
                        let val = *b;
                        match item.key {
                            "llm.enable_thinking" => self.config.llm.enable_thinking = val,
                            "agent.native_tools" => self.config.agent.native_tools = val,
                            "wake.enabled" => self.config.wake.enabled = val,
                            "wake.overlay_enabled" => self.config.wake.overlay_enabled = val,
                            "proactive.enabled" => self.config.proactive.enabled = val,
                            "proactive.check_updates" => self.config.proactive.check_updates = val,
                            "daemon.enabled" => self.config.daemon.enabled = val,
                            "daemon.learning_enabled" => self.config.daemon.learning_enabled = val,
                            "daemon.disk_cleanup" => self.config.daemon.disk_cleanup = val,
                            "external.allow_external_actions" => self.config.external.allow_external_actions = val,
                            "updates.allow_apply" => self.config.updates.allow_apply = val,
                            "selfpatch.enabled" => self.config.selfpatch.enabled = val,
                            "vision.enabled" => self.config.vision.enabled = val,
                            "browser.enabled" => self.config.browser.enabled = val,
                            "desktop.enabled" => self.config.desktop.enabled = val,
                            "sysmode.enabled" => self.config.sysmode.enabled = val,
                            "whatsapp.enabled" => self.config.whatsapp.enabled = val,
                            _ => {}
                        }
                        self.modified = true;
                        self.status = format!("✓ Toggled {} to {} [Press S to save]", item.label, if val { "ON" } else { "OFF" });
                    }
                    SettingType::Value(ref mut v) => {
                        if item.key == "voice.mode" {
                            let (next_mode, next_str) = match self.config.voice.mode {
                                VoiceMode::Basic => (VoiceMode::Jinx, "Jinx"),
                                VoiceMode::Jinx => (VoiceMode::Off, "Off"),
                                VoiceMode::Off => (VoiceMode::Basic, "Basic"),
                            };
                            self.config.voice.mode = next_mode;
                            *v = next_str.to_string();
                            self.modified = true;
                            self.status = format!("✓ Switched Voice Mode to {} [Press S to save]", next_str);
                        }
                    }
                }
            }
        }
    }

    pub fn save(&mut self, app_config: &mut LunaConfig) -> bool {
        match self.config.save() {
            Ok(()) => {
                *app_config = self.config.clone();
                self.modified = false;
                self.status = String::from("✓ Saved luna.toml! Daemon auto-restarted.");
                true
            }
            Err(e) => {
                self.status = format!("✗ Save Failed: {e}");
                false
            }
        }
    }

    pub fn next_category(&mut self) {
        if !self.categories.is_empty() {
            self.cat_index = (self.cat_index + 1) % self.categories.len();
            self.item_index = 0;
        }
    }

    pub fn prev_category(&mut self) {
        if !self.categories.is_empty() {
            if self.cat_index == 0 {
                self.cat_index = self.categories.len() - 1;
            } else {
                self.cat_index -= 1;
            }
            self.item_index = 0;
        }
    }

    pub fn next_item(&mut self) {
        if let Some(cat) = self.categories.get(self.cat_index) {
            if !cat.items.is_empty() {
                self.item_index = (self.item_index + 1) % cat.items.len();
            }
        }
    }

    pub fn prev_item(&mut self) {
        if let Some(cat) = self.categories.get(self.cat_index) {
            if !cat.items.is_empty() {
                if self.item_index == 0 {
                    self.item_index = cat.items.len() - 1;
                } else {
                    self.item_index -= 1;
                }
            }
        }
    }
}

pub struct TuiApp {
    config: LunaConfig,
    memory: Memory,
    messages: Vec<Msg>,
    input: String,
    cursor_pos: usize,
    /// Lines scrolled up from bottom (0 = newest). Shared by chat area.
    chat_scroll: usize,
    /// Lines scrolled up from bottom (0 = newest). Right-hand debug panel.
    debug_scroll: usize,
    /// Which panel the arrow keys scroll.
    focus: Focus,
    status: String,
    model_name: String,
    metrics: Arc<Metrics>,
    last_latency: Duration,
    should_quit: bool,
    thinking: bool,
    tx: mpsc::UnboundedSender<AppEvent>,
    rx: mpsc::UnboundedReceiver<AppEvent>,
    logs: LogBuffer,
    /// Unix-millis deadline until which the wake listener skips wake-word
    /// detection and accepts any speech (0 = no active window).
    window_deadline: Arc<AtomicU64>,
    /// true = hands-free "voice mode" is active (TUI keeps accepting any
    /// speech without the wake word until told to stop or idle runs out).
    handsfree_active: Arc<AtomicBool>,
    /// Unix-millis of the last hands-free utterance, for the idle auto-return.
    last_voice_activity: Arc<AtomicU64>,
    /// First-run setup screen (active until the user finishes/dismisses it).
    onboarding: Option<Onboarding>,
    /// Active masked prompt for a chat-typed `--set-key <name>` (None = none).
    key_prompt: Option<KeyPrompt>,
    /// Toggleable shortcuts help modal.
    show_help: bool,
    /// Toggleable bottom keybindings shortcuts bar.
    show_shortcuts_bar: bool,
    /// Active interactive luna.toml editor modal (None = none).
    config_editor: Option<ConfigState>,
    /// Active interactive Settings & Controls menu (None = none).
    settings_menu: Option<SettingsMenuState>,
    /// Index in slash command autocomplete popup.
    slash_selected: usize,
}

impl TuiApp {
    pub fn with_setup(config: LunaConfig, logs: LogBuffer, force_setup: bool) -> Result<Self> {
        let onboarding = if force_setup || crate::first_run::needs_onboarding(&config) {
            Some(Onboarding::new())
        } else {
            None
        };
        let memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
        let (tx, rx) = mpsc::unbounded_channel();

        let model_name = config.llm.model.clone();
        let thinking = config.llm.enable_thinking;

        Ok(Self {
            config,
            memory,
            messages: Vec::new(),
            input: String::new(),
            cursor_pos: 0,
            chat_scroll: 0,
            debug_scroll: 0,
            focus: Focus::Chat,
            status: String::from("Ready"),
            model_name,
            metrics: Arc::new(Metrics::default()),
            last_latency: Duration::ZERO,
            should_quit: false,
            thinking,
            tx,
            rx,
            logs,
            window_deadline: Arc::new(AtomicU64::new(0)),
            handsfree_active: Arc::new(AtomicBool::new(false)),
            last_voice_activity: Arc::new(AtomicU64::new(millis_now())),
            onboarding,
            key_prompt: None,
            show_help: false,
            show_shortcuts_bar: false,
            config_editor: None,
            settings_menu: None,
            slash_selected: 0,
        })
    }

    fn slash_matches(&self) -> Vec<(&'static str, &'static str)> {
        if !self.input.starts_with('/') {
            return Vec::new();
        }
        let query = self.input.to_lowercase();
        SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|(cmd, _)| cmd.starts_with(&query))
            .collect()
    }

    /// Inject a synthetic voice utterance at startup (the wake daemon's
    /// pending one-breath command). Sent before the event loop runs, so it is
    /// the first thing the app processes.
    pub fn enqueue_voice(&self, text: String) {
        let _ = self.tx.send(AppEvent::Voice(text));
    }

    pub fn run_voice_debug(&mut self) {
        tracing::info!("── [DEBUG VOICE DIAGNOSTIC] ──");
        tracing::info!("Voice Mode: {:?}", self.config.voice.mode);
        tracing::info!("Piper Bin: {:?}", self.config.voice.piper_bin);
        tracing::info!("Piper Model: {:?}", self.config.voice.piper_model);
        tracing::info!("RVC Script: {:?}", self.config.voice.rvc_script);
        tracing::info!("RVC Model: {:?}", self.config.voice.rvc_model);

        let home = dirs::home_dir().unwrap_or_default();
        let rvc_env_py = home.join(".local/share/luna/rvc_env/bin/python3");
        let tts_env_py = home.join(".local/share/luna/tts_env/bin/python3");
        let sys_py = std::path::Path::new("/usr/bin/python3");

        tracing::info!("Python env check -> rvc_env exists: {}", rvc_env_py.exists());
        tracing::info!("Python env check -> tts_env exists: {}", tts_env_py.exists());
        tracing::info!("Python env check -> /usr/bin/python3 exists: {}", sys_py.exists());

        let active_py = if rvc_env_py.exists() {
            rvc_env_py.display().to_string()
        } else if tts_env_py.exists() {
            tts_env_py.display().to_string()
        } else if sys_py.exists() {
            sys_py.display().to_string()
        } else {
            "python3 (PATH)".to_string()
        };
        tracing::info!("Active RVC Python resolver chosen: {}", active_py);

        let piper_ok = self.config.voice.piper_bin.exists();
        let piper_model_ok = self.config.voice.piper_model.exists();
        let rvc_script_ok = self.config.voice.rvc_script.as_ref().map(|p| p.exists()).unwrap_or(false);
        let rvc_model_ok = self.config.voice.rvc_model.as_ref().map(|p| p.exists()).unwrap_or(false);

        tracing::info!("Component files check -> Piper Bin: {}", if piper_ok { "OK" } else { "MISSING" });
        tracing::info!("Component files check -> Piper Model: {}", if piper_model_ok { "OK" } else { "MISSING" });
        tracing::info!("Component files check -> RVC Script: {}", if rvc_script_ok { "OK" } else { "NOT FOUND" });
        tracing::info!("Component files check -> RVC Model: {}", if rvc_model_ok { "OK" } else { "NOT FOUND" });

        let summary = format!(
            "Voice Subsystem Diagnostic:\n\
             • Mode: {:?}\n\
             • Active RVC Python: {}\n\
             • Piper Bin: {}\n\
             • Piper Model: {}\n\
             • RVC Script: {}\n\
             • RVC Model: {}\n\
             (Step-by-step logs written to Debug Panel on right)",
            self.config.voice.mode,
            active_py,
            if piper_ok { "OK" } else { "Missing" },
            if piper_model_ok { "OK" } else { "Missing" },
            if rvc_script_ok { "OK" } else { "Missing" },
            if rvc_model_ok { "OK" } else { "Missing" }
        );

        self.messages.push(Msg {
            role: "assistant".into(),
            content: summary,
            thinking: None,
        });
    }

    pub fn run_model_debug(&mut self) {
        tracing::info!("── [DEBUG MODEL DIAGNOSTIC] ──");
        tracing::info!("Configured Main Model: {}", self.config.llm.model);
        tracing::info!("Fast Model: {:?}", self.config.llm.fast_model);
        tracing::info!("Deep Model: {:?}", self.config.llm.deep_model);
        tracing::info!("Host Endpoint: {}", self.config.llm.base_url);
        tracing::info!("Context Window: {}", self.config.llm.num_ctx);

        let summary = format!(
            "LLM Model Subsystem Diagnostic:\n\
             • Main Model: {}\n\
             • Fast Model: {}\n\
             • Deep Model: {}\n\
             • Host Endpoint: {}\n\
             (Step-by-step logs written to Debug Panel on right)",
            self.config.llm.model,
            self.config.llm.fast_model.as_deref().unwrap_or("none"),
            self.config.llm.deep_model.as_deref().unwrap_or("none"),
            self.config.llm.base_url
        );

        self.messages.push(Msg {
            role: "assistant".into(),
            content: summary,
            thinking: None,
        });
    }

    pub fn run_stt_debug(&mut self) {
        tracing::info!("── [DEBUG STT / WHISPER DIAGNOSTIC] ──");
        tracing::info!("Input Mode: {:?}", self.config.audio.input_mode);
        tracing::info!("Whisper Model Path: {:?}", self.config.voice.whisper_model);
        let model_exists = self.config.voice.whisper_model.exists();
        tracing::info!("Whisper Model File Exists: {}", model_exists);

        let summary = format!(
            "STT / Whisper Subsystem Diagnostic:\n\
             • Input Mode: {:?}\n\
             • Whisper Model: {} ({})\n\
             (Step-by-step logs written to Debug Panel on right)",
            self.config.audio.input_mode,
            if model_exists { "Found" } else { "Missing" },
            self.config.voice.whisper_model.display()
        );

        self.messages.push(Msg {
            role: "assistant".into(),
            content: summary,
            thinking: None,
        });
    }

    pub fn run_memory_debug(&mut self) {
        tracing::info!("── [DEBUG MEMORY DIAGNOSTIC] ──");
        tracing::info!("Context Window Cap: {}", self.config.memory.context_window);
        tracing::info!("History Path: {:?}", self.config.memory.history_path);
        tracing::info!("Active Chat UI Messages: {}", self.messages.len());

        let perm_path = dirs::home_dir().unwrap_or_default().join(".local/share/luna/permanent_memory.json");
        let perm_exists = perm_path.exists();
        tracing::info!("Permanent Memory File Exists: {}", perm_exists);

        let summary = format!(
            "Memory Subsystem Diagnostic:\n\
             • Context Window: {} msgs\n\
             • History Path: {:?}\n\
             • Active UI Messages: {}\n\
             • Permanent Memory File: {}\n\
             (Step-by-step logs written to Debug Panel on right)",
            self.config.memory.context_window,
            self.config.memory.history_path,
            self.messages.len(),
            if perm_exists { "Found" } else { "Not created yet" }
        );

        self.messages.push(Msg {
            role: "assistant".into(),
            content: summary,
            thinking: None,
        });
    }

    pub fn run_all_debug(&mut self) {
        tracing::info!("════════════════════════════════════════════════");
        tracing::info!("       SYSTEM-WIDE FULL DIAGNOSTIC RUN          ");
        tracing::info!("════════════════════════════════════════════════");
        self.run_voice_debug();
        self.run_model_debug();
        self.run_stt_debug();
        self.run_memory_debug();
        tracing::info!("════════════════════════════════════════════════");
        tracing::info!("          DIAGNOSTIC RUN COMPLETED              ");
        tracing::info!("════════════════════════════════════════════════");
    }

    pub async fn run(mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        // Event reader task (keyboard/resize)
        let tx = self.tx.clone();
        tokio::spawn(async move {
            loop {
                if event::poll(Duration::from_millis(50)).unwrap_or(false) {
                    if let Ok(ev) = event::read() {
                        if tx.send(AppEvent::Input(ev)).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        // Metrics poller task (GPU + CPU every 2s)
        let metrics = self.metrics.clone();
        let tx_metrics = self.tx.clone();
        tokio::spawn(async move {
            loop {
                if let Some(g) = gpu_utilization().await {
                    metrics.gpu.store(g, Ordering::Relaxed);
                }
                if let Some(c) = cpu_utilization().await {
                    metrics.cpu.store(c, Ordering::Relaxed);
                }
                if tx_metrics.send(AppEvent::Metrics).is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });

        // Voice wake-word listener (if voice enabled)
        if self.config.voice.mode != VoiceMode::Off {
            let tx_voice = self.tx.clone();
            let stt = crate::stt::whisper::WhisperStt::with_prompt(
                &self.config.voice.whisper_model.to_string_lossy(),
                Some("Luna, open, close, run, search, volume, terminal, browser.".into()),
            );
            let aliases = self.config.audio.wake_aliases.clone();
            let sample_rate = self.config.audio.sample_rate;
            let silence_ms = self.config.audio.vad_silence_ms;
            let window_deadline = Arc::clone(&self.window_deadline);
            let window_secs = self.config.audio.conversation_window_secs;
            let window_active = window_secs > 0;
            let handsfree_active = Arc::clone(&self.handsfree_active);
            tokio::spawn(async move {
                loop {
                    // Hands-free "voice mode": accept any speech, no wake word.
                    if handsfree_active.load(Ordering::Relaxed) {
                        match crate::audio::capture::listen_continuous(
                            sample_rate,
                            silence_ms,
                            &stt,
                        )
                        .await
                        {
                            Ok(text) if !text.trim().is_empty() => {
                                crate::overlay::signal("listening");
                                if tx_voice.send(AppEvent::Voice(text)).is_err() {
                                    break;
                                }
                            }
                            Ok(_) => continue,
                            Err(e) => {
                                tracing::debug!("Hands-free listener error: {}", e);
                                tokio::time::sleep(Duration::from_millis(200)).await;
                            }
                        }
                        continue;
                    }

                    // If a conversation window is open (user recently got a
                    // voice response), accept any speech without the wake word.
                    if window_active
                        && window_deadline.load(Ordering::Relaxed) > 0
                        && millis_now() < window_deadline.load(Ordering::Relaxed)
                    {
                        match crate::audio::capture::listen_continuous(
                            sample_rate,
                            silence_ms,
                            &stt,
                        )
                        .await
                        {
                            Ok(text) if !text.trim().is_empty() => {
                                crate::overlay::signal("listening");
                                if tx_voice.send(AppEvent::Voice(text)).is_err() {
                                    break;
                                }
                            }
                            Ok(_) => {
                                // silence timeout / no speech — refresh deadline check
                                continue;
                            }
                            Err(e) => {
                                tracing::debug!("Conversation window listener error: {}", e);
                                tokio::time::sleep(Duration::from_millis(200)).await;
                            }
                        }
                        continue;
                    }

                    match crate::audio::capture::listen_for_wake_word(
                        sample_rate,
                        silence_ms,
                        &aliases,
                        &stt,
                    )
                    .await
                    {
                        Ok(text) => {
                            crate::overlay::signal("listening");
                            if tx_voice.send(AppEvent::Voice(text)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Wake word listener error: {}", e);
                            tokio::time::sleep(Duration::from_millis(200)).await;
                        }
                    }
                }
            });
        }

        // Initial draw
        terminal.draw(|f| self.ui(f))?;

        // Main loop
        while !self.should_quit {
            while let Ok(ev) = self.rx.try_recv() {
                self.handle_app_event(ev);
            }

            self.check_voice_mode_idle();

            terminal.draw(|f| self.ui(f))?;
            tokio::time::sleep(Duration::from_millis(16)).await;
        }

        // Cleanup
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        Ok(())
    }

    fn handle_app_event(&mut self, ev: AppEvent) {
        match ev {
            AppEvent::Input(Event::Key(key)) => self.handle_key(key),
            AppEvent::Input(Event::Resize(_, _)) => {}
            AppEvent::Input(_) => {}
            AppEvent::Metrics => {}
            AppEvent::Response(msg, latency, memory, model) => {
                self.messages.push(msg);
                self.memory = memory;
                self.last_latency = latency;
                self.model_name = model;
                self.chat_scroll = 0;
                self.status = String::from("Ready");
                // Reply delivered (and spoken) — clear the thinking pulse.
                crate::overlay::signal("idle");
                // After a voice reply, keep the wake-word window open so the
                // user can keep talking without repeating "luna". Only extends
                // an already-active window — a typed turn shouldn't open one.
                if self.window_deadline.load(Ordering::Relaxed) > 0 {
                    self.refresh_conversation_window();
                }
            }
            AppEvent::Voice(text) => {
                let raw = text.trim();
                if raw.is_empty() {
                    return;
                }
                // Strip leading wake word so "Luna, what's the time" shows as
                // just "what's the time" and the command-matching functions
                // get clean input.
                let stripped = crate::audio::capture::strip_wake_word(
                    raw,
                    &self.config.audio.wake_aliases,
                )
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| raw.to_string());
                let lower = stripped.trim().to_lowercase();

                // ── Hands-free voice mode active ─────────────────────────
                if self.handsfree_active.load(Ordering::Relaxed) {
                    // Exit phrase → leave voice mode immediately
                    if crate::agent::voice_mode_exit_match(&lower) {
                        self.handsfree_active.store(false, Ordering::Relaxed);
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: "Voice mode off.".into(),
                            thinking: None,
                        });
                        self.speak_announce("Voice mode off.");
                        return;
                    }
                    // Normal hands-free question: answer it and reset idle timer
                    self.last_voice_activity.store(millis_now(), Ordering::Relaxed);
                    self.add_user_message(&stripped);
                    crate::overlay::signal("thinking");
                    self.submit(stripped);
                    return;
                }

                // ── Hands-free OFF: did this wake utterance toggle voice mode on?
                if crate::agent::wake_toggles_voice_mode(&lower) {
                    self.handsfree_active.store(true, Ordering::Relaxed);
                    self.last_voice_activity.store(millis_now(), Ordering::Relaxed);
                    let wake = self.config.audio.wake_word.clone();
                    let msg = format!(
                        "[Voice mode ON — hands-free. Say \"{wake} voice mode off\" to turn it off]"
                    );
                    self.messages.push(Msg {
                        role: "assistant".into(),
                        content: msg,
                        thinking: None,
                    });
                    self.speak_announce("Voice mode on.");
                    return;
                }

                // ── Normal wake utterance: answer and extend conversation window ──
                self.refresh_conversation_window();
                self.add_user_message(&stripped);
                crate::overlay::signal("thinking");
                self.submit(stripped);
            }
            AppEvent::OnboardingLog(line) => self.app_emit(&line),
            AppEvent::SpotifyAuthed => {
                // The refresh token lives in the keyring; point config at it.
                self.config.spotify.refresh_token =
                    Some("keyring:spotify_refresh".to_string());
                match self.config.save() {
                    Ok(()) => self.app_emit("✓ Spotify authorized. Say \"play my liked songs\" anytime."),
                    Err(e) => self.app_emit(&format!("✗ {e}")),
                }
            }
        }
    }

    /// Show a setup/progress line: in the onboarding panel when it's open,
    /// otherwise as a chat message.
    fn app_emit(&mut self, line: &str) {
        if self.onboarding.as_ref().map(Onboarding::is_active).unwrap_or(false) {
            if let Some(onb) = &mut self.onboarding {
                onb.log(line);
            }
        } else {
            self.messages.push(Msg {
                role: "assistant".into(),
                content: line.to_string(),
                thinking: None,
            });
            self.chat_scroll = 0;
        }
    }


    /// interaction so the user can keep talking without repeating "luna".
    fn refresh_conversation_window(&mut self) {
        let secs = self.config.audio.conversation_window_secs;
        if secs > 0 {
            self.window_deadline.store(millis_now() + secs * 1000, Ordering::Relaxed);
        }
    }

    /// Auto-exit hands-free voice mode after `voice_mode_idle_mins` of silence
    /// (0 = never). Runs from the main draw loop.
    fn check_voice_mode_idle(&mut self) {
        if !self.handsfree_active.load(Ordering::Relaxed) {
            return;
        }
        let idle_mins = self.config.audio.voice_mode_idle_mins;
        if idle_mins == 0 {
            return;
        }
        let last = self.last_voice_activity.load(Ordering::Relaxed);
        let now = millis_now();
        if now.saturating_sub(last) < idle_mins * 60 * 1000 {
            return;
        }
        self.handsfree_active.store(false, Ordering::Relaxed);
        let wake = self.config.audio.wake_word.clone();
        let msg = format!(
            "[Voice mode ended — {} min of inactivity. Say \"{wake}\" to wake me.]",
            idle_mins
        );
        self.messages.push(Msg {
            role: "assistant".into(),
            content: msg,
            thinking: None,
        });
        self.speak_announce("Voice mode ended due to inactivity.");
    }

    /// Fire-and-forget spoken announcement of a mode change, mirroring the
    /// hybrid loop's TTS of "Voice mode on/off.".
    fn speak_announce(&self, phrase: &str) {
        if self.config.voice.mode != VoiceMode::Off {
            let mode = self.config.voice.mode.clone();
            let cfg = self.config.clone();
            let phrase = phrase.to_string();
            tokio::spawn(async move {
                let _ = crate::tts::speak(&phrase, &mode, &cfg).await;
            });
        }
    }

    fn delete_word_backwards(&mut self) {
        if self.cursor_pos == 0 {
            return;
        }
        let before = &self.input[..self.cursor_pos];
        let trimmed = before.trim_end();
        let new_pos = match trimmed.rfind(' ') {
            Some(idx) => idx + 1,
            None => 0,
        };
        self.input.drain(new_pos..self.cursor_pos);
        self.cursor_pos = new_pos;
    }

    fn handle_key(&mut self, key: KeyEvent) {
        // First-run setup screen steals the keyboard until dismissed.
        if self.onboarding.as_ref().map(Onboarding::is_active).unwrap_or(false) {
            self.handle_onboarding_key(key);
            return;
        }

        // Masked --set-key prompt steals the keyboard while it's open.
        if self.key_prompt.is_some() {
            self.handle_key_prompt(key);
            return;
        }

        // Settings Menu modal steals the keyboard while open.
        if let Some(sm) = &mut self.settings_menu {
            match (key.modifiers, key.code) {
                (_, KeyCode::Esc) => {
                    self.settings_menu = None;
                }
                (_, KeyCode::Char('s')) | (_, KeyCode::Char('S')) => {
                    sm.save(&mut self.config);
                }
                (_, KeyCode::Char('t')) | (_, KeyCode::Char('T')) => {
                    self.settings_menu = None;
                    self.config_editor = Some(ConfigState::load());
                }
                (_, KeyCode::Tab) | (_, KeyCode::Right) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    sm.next_category();
                }
                (_, KeyCode::Left) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    sm.prev_category();
                }
                (_, KeyCode::Tab) => {
                    sm.next_category();
                }
                (_, KeyCode::Up) => sm.prev_item(),
                (_, KeyCode::Down) => sm.next_item(),
                (_, KeyCode::Char(' ')) | (_, KeyCode::Enter) => {
                    sm.toggle_selected();
                }
                _ => {}
            }
            return;
        }

        // Config Editor modal steals the keyboard while open.
        if let Some(ed) = &mut self.config_editor {
            match (key.modifiers, key.code) {
                (_, KeyCode::Esc) => {
                    self.config_editor = None;
                }
                (_, KeyCode::Char('s')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    ed.save(&mut self.config);
                }
                (_, KeyCode::Char('e')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let path = ed.path.clone();
                    let _ = disable_raw_mode();
                    let _ = execute!(io::stdout(), LeaveAlternateScreen);
                    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".into());
                    let _ = std::process::Command::new(editor).arg(&path).status();
                    let _ = execute!(io::stdout(), EnterAlternateScreen);
                    let _ = enable_raw_mode();
                    *ed = ConfigState::load();
                    if let Ok(new_cfg) = crate::config::LunaConfig::load() {
                        self.config = new_cfg;
                    }
                }
                (_, KeyCode::Up) => ed.move_up(),
                (_, KeyCode::Down) => ed.move_down(),
                (_, KeyCode::Left) => ed.move_left(),
                (_, KeyCode::Right) => ed.move_right(),
                (_, KeyCode::PageUp) => ed.page_up(15),
                (_, KeyCode::PageDown) => ed.page_down(15),
                (_, KeyCode::Home) => ed.home(),
                (_, KeyCode::End) => ed.end(),
                (_, KeyCode::Enter) => ed.newline(),
                (_, KeyCode::Backspace) => ed.backspace(),
                (_, KeyCode::Delete) => ed.delete(),
                (_, KeyCode::Tab) => ed.insert_tab(),
                (_, KeyCode::Char(c)) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    ed.insert_char(c);
                }
                _ => {}
            }
            return;
        }

        // Slash command autocomplete navigation while typing '/'
        let slash_opts = self.slash_matches();
        if !slash_opts.is_empty() {
            match key.code {
                KeyCode::Up => {
                    if self.slash_selected == 0 {
                        self.slash_selected = slash_opts.len().saturating_sub(1);
                    } else {
                        self.slash_selected -= 1;
                    }
                    return;
                }
                KeyCode::Down => {
                    self.slash_selected = (self.slash_selected + 1) % slash_opts.len();
                    return;
                }
                KeyCode::Tab => {
                    if let Some((cmd, _)) = slash_opts.get(self.slash_selected) {
                        self.input = cmd.to_string();
                        self.cursor_pos = self.input.len();
                    }
                    return;
                }
                _ => {}
            }
        }

        match (key.modifiers, key.code) {
            (_, KeyCode::Char('c')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.input.is_empty() {
                    self.should_quit = true;
                } else {
                    self.input.clear();
                    self.cursor_pos = 0;
                }
            }
            (_, KeyCode::Char('d')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.input.is_empty() {
                    self.should_quit = true;
                }
            }
            (_, KeyCode::Char('u')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.clear();
                self.cursor_pos = 0;
            }
            (_, KeyCode::Char('w')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_word_backwards();
            }
            (_, KeyCode::Char('a')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_pos = 0;
            }
            (_, KeyCode::Char('e')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_pos = self.input.len();
            }
            (_, KeyCode::Char('b')) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.show_shortcuts_bar = !self.show_shortcuts_bar;
            }
            (_, KeyCode::F(1)) => {
                self.show_help = !self.show_help;
            }
            (_, KeyCode::F(2)) => {
                self.settings_menu = Some(SettingsMenuState::load(self.config.clone()));
            }
            (_, KeyCode::Char('?')) if self.input.is_empty() => {
                self.show_help = !self.show_help;
            }
            (_, KeyCode::Esc) => {
                if self.show_help {
                    self.show_help = false;
                } else {
                    self.chat_scroll = 0;
                    self.debug_scroll = 0;
                    self.focus = Focus::Chat;
                }
            }
            (_, KeyCode::Enter) => {
                if !self.input.trim().is_empty() {
                    let user_input = self.input.clone();
                    self.input.clear();
                    self.cursor_pos = 0;
                    let trimmed = user_input.trim();
                    let lower = trimmed.to_lowercase();
                    if lower == "/config" || lower == "/settings" || lower == ":config" || lower == ":settings" {
                        self.settings_menu = Some(SettingsMenuState::load(self.config.clone()));
                        return;
                    }
                    if lower == "/toml" || lower == ":toml" || lower == "/edit" || lower == ":edit" {
                        self.config_editor = Some(ConfigState::load());
                        return;
                    }
                    if lower == "/clear" {
                        let _ = self.memory.clear();
                        self.messages.clear();
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: "Memory and chat history cleared.".into(),
                            thinking: None,
                        });
                        self.chat_scroll = 0;
                        return;
                    }
                    if lower == "/voice" {
                        let active = !self.handsfree_active.load(Ordering::Relaxed);
                        self.handsfree_active.store(active, Ordering::Relaxed);
                        let msg = if active {
                            "[Voice mode ON — hands-free. Say \"voice mode off\" to exit]"
                        } else {
                            "[Voice mode OFF]"
                        };
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: msg.into(),
                            thinking: None,
                        });
                        self.speak_announce(msg);
                        return;
                    }
                    if lower == "/shortcuts" || lower == "/bar" {
                        self.show_shortcuts_bar = !self.show_shortcuts_bar;
                        let msg = if self.show_shortcuts_bar {
                            "[Bottom shortcuts bar ON — press Ctrl+B or /shortcuts to hide]"
                        } else {
                            "[Bottom shortcuts bar OFF — press F1 for help anytime]"
                        };
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: msg.into(),
                            thinking: None,
                        });
                        return;
                    }
                    if lower.starts_with("/log ") || lower.starts_with("/debug log ") {
                        let msg_text = if lower.starts_with("/log ") {
                            trimmed[5..].trim()
                        } else {
                            trimmed[11..].trim()
                        };
                        tracing::info!("[DEBUG USER] {}", msg_text);
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: format!("[Debug log written to Debug Panel: \"{}\"]", msg_text),
                            thinking: None,
                        });
                        return;
                    }
                    if lower == "/test-voice" || lower == "/debug voice" {
                        self.run_voice_debug();
                        return;
                    }
                    if lower == "/test-model" || lower == "/debug model" {
                        self.run_model_debug();
                        return;
                    }
                    if lower == "/test-stt" || lower == "/debug stt" {
                        self.run_stt_debug();
                        return;
                    }
                    if lower == "/test-memory" || lower == "/debug memory" {
                        self.run_memory_debug();
                        return;
                    }
                    if lower == "/debug" || lower == "/debug help" || lower == "/debug status" || lower == "/debug all" {
                        self.run_all_debug();
                        return;
                    }
                    if lower == "/setup" {
                        self.onboarding = Some(Onboarding::new());
                        return;
                    }
                    if lower == "/help" {
                        self.show_help = !self.show_help;
                        return;
                    }
                    if lower == "/exit" || lower == "/quit" {
                        self.should_quit = true;
                        return;
                    }
                    self.messages.push(Msg {
                        role: "user".into(),
                        content: user_input.clone(),
                        thinking: None,
                    });
                    self.chat_scroll = 0;
                    self.submit(user_input);
                }
            }
            (_, KeyCode::Backspace) => {
                if self.cursor_pos > 0 {
                    self.input.remove(self.cursor_pos - 1);
                    self.cursor_pos -= 1;
                }
            }
            (_, KeyCode::Delete) => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            (_, KeyCode::Left) => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                }
            }
            (_, KeyCode::Right) => {
                if self.cursor_pos < self.input.len() {
                    self.cursor_pos += 1;
                }
            }
            (_, KeyCode::Home) => self.cursor_pos = 0,
            (_, KeyCode::End) => self.cursor_pos = self.input.len(),
            (_, KeyCode::Tab) => {
                self.focus = match self.focus {
                    Focus::Chat => Focus::Debug,
                    Focus::Debug => Focus::Chat,
                };
            }
            (_, KeyCode::Up) => match self.focus {
                Focus::Chat => self.chat_scroll += 1,
                Focus::Debug => self.debug_scroll += 1,
            },
            (_, KeyCode::Down) => match self.focus {
                Focus::Chat => self.chat_scroll = self.chat_scroll.saturating_sub(1),
                Focus::Debug => {
                    self.debug_scroll = self.debug_scroll.saturating_sub(1);
                }
            },
            (_, KeyCode::PageUp) => match self.focus {
                Focus::Chat => self.chat_scroll = self.chat_scroll.saturating_add(10),
                Focus::Debug => self.debug_scroll = self.debug_scroll.saturating_add(10),
            },
            (_, KeyCode::PageDown) => match self.focus {
                Focus::Chat => self.chat_scroll = self.chat_scroll.saturating_sub(10),
                Focus::Debug => self.debug_scroll = self.debug_scroll.saturating_sub(10),
            },
            (_, KeyCode::Char(c)) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            _ => {}
        }
    }

    fn add_user_message(&mut self, text: &str) {
        self.messages.push(Msg {
            role: "user".into(),
            content: text.into(),
            thinking: None,
        });
        self.chat_scroll = 0;
    }

    /// Keys while the first-run setup screen is active. While editing a secret,
    /// every keystroke goes to the hidden buffer; otherwise Up/Down/Enter/Finish.
    fn handle_onboarding_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('d') => self.should_quit = true,
                _ => {}
            }
            return;
        }

        let editing = self
            .onboarding
            .as_ref()
            .and_then(|o| o.editing.map(|i| i == o.cursor))
            .unwrap_or(false);

        if editing {
            match key.code {
                KeyCode::Enter => self.commit_secret(),
                KeyCode::Esc => {
                    if let Some(onb) = &mut self.onboarding {
                        onb.cancel_secret();
                    }
                }
                KeyCode::Backspace => {
                    if let Some(onb) = &mut self.onboarding {
                        onb.pop_char();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(onb) = &mut self.onboarding {
                        onb.append_char(c);
                    }
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Up => {
                if let Some(onb) = &mut self.onboarding {
                    onb.cursor_up();
                }
            }
            KeyCode::Down => {
                if let Some(onb) = &mut self.onboarding {
                    let n = onb.actions(&self.config).len();
                    onb.cursor_down(n);
                }
            }
            KeyCode::Enter => self.activate_onboarding_row(),
            KeyCode::Esc => self.finish_onboarding(),
            KeyCode::Char('q') => self.finish_onboarding(),
            _ => {}
        }
    }

    fn activate_onboarding_row(&mut self) {
        let (idx, id) = {
            let Some(onb) = &mut self.onboarding else { return };
            let rows = onb.actions(&self.config);
            let idx = onb.cursor.min(rows.len().saturating_sub(1));
            (idx, rows.get(idx).map(|r| r.id))
        };
        let Some(id) = id else { return };

        match id {
            crate::tui::onboarding::Action::SetTavily
            | crate::tui::onboarding::Action::SetGemini
            | crate::tui::onboarding::Action::SetTodoist
            | crate::tui::onboarding::Action::SetSpotifyId => {
                if let Some(onb) = &mut self.onboarding {
                    onb.begin_secret(idx);
                }
            }
            crate::tui::onboarding::Action::SpotifyAuth => self.start_spotify_auth(),
            crate::tui::onboarding::Action::Finish => self.finish_onboarding(),
        }
    }

    /// Save an entered secret (onboarding flow) to the keyring and config.
    fn commit_secret(&mut self) {
        let (keyring_name, label) = {
            let Some(onb) = &mut self.onboarding else { return };
            let rows = onb.actions(&self.config);
            let Some(row) = rows.get(onb.cursor) else {
                onb.cancel_secret();
                return;
            };
            let Some(name) = crate::tui::onboarding::action_keyring_name(row.id) else {
                onb.cancel_secret();
                return;
            };
            (name.to_string(), row.label.clone())
        };

        // Pull the secret out BEFORE touching config so the borrow is clean.
        let secret = self
            .onboarding
            .as_mut()
            .map(|o| o.take_secret())
            .unwrap_or_default();
        self.store_secret(&keyring_name, secret)
            .map(|_| {
                if let Some(onb) = &mut self.onboarding {
                    onb.log(&format!("✓ stored {}", label));
                }
            })
            .unwrap_or_else(|e| {
                if let Some(onb) = &mut self.onboarding {
                    onb.log(&format!("✗ {}: {}", label, e));
                }
            });
    }

    /// Keys while a chat-typed `--set-key <name>` masked prompt is open.
    fn handle_key_prompt(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('d') => self.should_quit = true,
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Enter => {
                let KeyPrompt { name, secret } = self.key_prompt.take().unwrap();
                let name2 = name.clone();
                match self.store_secret(&name, secret) {
                    Ok(()) => {
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: format!(
                                "[✓ Stored luna/{0} — referenced as keyring:{0} in luna.toml]",
                                name2
                            ),
                            thinking: None,
                        });
                        self.status = String::from("Ready");
                    }
                    Err(e) => {
                        self.messages.push(Msg {
                            role: "assistant".into(),
                            content: format!("[✗ Could not store {}: {}]", name2, e),
                            thinking: None,
                        });
                    }
                }
                self.chat_scroll = 0;
            }
            KeyCode::Esc => {
                self.key_prompt = None;
                self.messages.push(Msg {
                    role: "assistant".into(),
                    content: "[Cancelled — nothing stored]".into(),
                    thinking: None,
                });
                self.status = String::from("Ready");
            }
            KeyCode::Backspace => {
                if let Some(kp) = &mut self.key_prompt {
                    kp.secret.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(kp) = &mut self.key_prompt {
                    kp.secret.push(c);
                }
            }
            _ => {}
        }
    }

    /// Keyring-set + config-ref wiring shared by the onboarding screen and the
    /// chat `--set-key` prompt. Known names update the matching config leaf and
    /// persist; unknown names are stored in the keyring alone (and can be
    /// referenced manually later).
    fn store_secret(&mut self, name: &str, secret: String) -> anyhow::Result<()> {
        if secret.is_empty() {
            anyhow::bail!("empty secret — nothing stored");
        }
        crate::config::keyring_set(name, &secret)?;
        match name {
            "tavily" => self.config.search.tavily_api_key = Some(format!("keyring:{}", name)),
            "gemini" => self.config.search.gemini_api_key = Some(format!("keyring:{}", name)),
            "todoist" => self.config.todoist.api_token = Some(format!("keyring:{}", name)),
            "spotify_id" => self.config.spotify.client_id = Some(format!("keyring:{}", name)),
            _ => {}
        }
        self.config.save()?;
        Ok(())
    }

    /// Kick off the one-time Spotify PKCE authorization in the background,
    /// streaming its progress into the onboarding panel.
    fn start_spotify_auth(&mut self) {
        let id = match &self.config.spotify.client_id {
            Some(a) if !a.trim().is_empty() => a.clone(),
            _ => {
                self.app_emit(
                    "Store your Spotify client id first with `--set-key spotify_id`.",
                );
                return;
            }
        };
        self.app_emit("Starting Spotify authorization — a browser tab will open.");
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let result =
                crate::tools::spotify::authorize_with(&id, |line| {
                    let _ = tx.send(AppEvent::OnboardingLog(format!("  {line}")));
                })
                .await;
            match result {
                Ok(_) => {
                    let _ = tx.send(AppEvent::SpotifyAuthed);
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::OnboardingLog(format!("✗ {e}")));
                }
            }
        });
    }

    fn finish_onboarding(&mut self) {
        let _ = crate::first_run::mark_done();
        self.onboarding = None;
        self.status = String::from("Ready");
    }

    /// Spawn the routed async turn (fast/deep/full) — the UI stays responsive.
    /// The status bar shows the model that actually answered.
    fn submit(&mut self, user_input: String) {
        let trimmed = user_input.trim().to_string();
        // Strip a leading wake word ("luna voice mode") for clean display and
        // so the same command matchers used by the hybrid loop apply here.
        let stripped = crate::audio::capture::strip_wake_word(
            &trimmed,
            &self.config.audio.wake_aliases,
        )
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| trimmed.clone());
        let lower = stripped.trim().to_lowercase();

        // ── CLI-style flags typed into chat are handled here, exactly like
        // ── their shell counterparts (luna --set-key / --spotify-auth / --setup).
        let tokens: Vec<&str> = stripped.split_whitespace().collect();
        if !tokens.is_empty() {
            match tokens[0] {
                "--set-key" => {
                    match tokens.get(1) {
                        Some(name) if tokens.len() == 2 => {
                            self.messages.push(Msg {
                                role: "user".into(),
                                content: stripped.clone(),
                                thinking: None,
                            });
                            self.messages.push(Msg {
                                role: "assistant".into(),
                                content: format!(
                                    "Paste the secret for `{}` (typed hidden) [Enter] save  [Esc] cancel",
                                    name
                                ),
                                thinking: None,
                            });
                            self.key_prompt = Some(KeyPrompt {
                                name: name.to_string(),
                                secret: String::new(),
                            });
                            self.status = String::from("Entering secret…");
                            self.chat_scroll = 0;
                        }
                        _ => {
                            self.messages.push(Msg {
                                role: "assistant".into(),
                                content:
                                    "[Usage: --set-key <name> — e.g. `--set-key tavily`]".into(),
                                thinking: None,
                            });
                            self.chat_scroll = 0;
                        }
                    }
                    return;
                }
                "--spotify-auth" => {
                    self.messages.push(Msg {
                        role: "user".into(),
                        content: stripped.clone(),
                        thinking: None,
                    });
                    self.start_spotify_auth();
                    self.chat_scroll = 0;
                    return;
                }
                "--setup" => {
                    self.messages.push(Msg {
                        role: "user".into(),
                        content: stripped.clone(),
                        thinking: None,
                    });
                    self.onboarding = Some(Onboarding::new());
                    self.status = String::from("Setup screen");
                    self.chat_scroll = 0;
                    return;
                }
                _ => {}
            }
        }

        // ── Hands-free voice mode toggles (self-aware built-in commands) ──
        if crate::agent::voice_mode_enter_match(&lower)
            && !crate::agent::voice_mode_exit_match(&lower)
        {
            if self.handsfree_active.load(Ordering::Relaxed) {
                self.messages.push(Msg {
                    role: "assistant".into(),
                    content: "Already in voice mode.".into(),
                    thinking: None,
                });
                return;
            }
            self.handsfree_active.store(true, Ordering::Relaxed);
            self.last_voice_activity.store(millis_now(), Ordering::Relaxed);
            let wake = self.config.audio.wake_word.clone();
            let msg = format!(
                "[Voice mode ON — hands-free. Say \"{wake} voice mode off\" to turn it off]"
            );
            self.messages.push(Msg {
                role: "assistant".into(),
                content: msg,
                thinking: None,
            });
            self.speak_announce("Voice mode on.");
            return;
        }
        if crate::agent::voice_mode_exit_match(&lower) {
            if self.handsfree_active.load(Ordering::Relaxed) {
                self.handsfree_active.store(false, Ordering::Relaxed);
                self.messages.push(Msg {
                    role: "assistant".into(),
                    content: "Voice mode off.".into(),
                    thinking: None,
                });
                self.speak_announce("Voice mode off.");
            }
            return;
        }

        match lower.as_str() {
            "clear" => {
                let _ = self.memory.clear();
                self.messages.clear();
                self.messages.push(Msg {
                    role: "assistant".into(),
                    content: "Memory and chat history cleared.".into(),
                    thinking: None,
                });
                self.chat_scroll = 0;
                return;
            }
            "test voice" | "debug voice" => {
                self.run_voice_debug();
                return;
            }
            "test model" | "debug model" => {
                self.run_model_debug();
                return;
            }
            "test stt" | "debug stt" => {
                self.run_stt_debug();
                return;
            }
            "test memory" | "debug memory" => {
                self.run_memory_debug();
                return;
            }
            "debug" | "test debug" | "debug all" => {
                self.run_all_debug();
                return;
            }
            "exit" | "quit" | "bye" => {
                self.should_quit = true;
                return;
            }
            _ => {}
        }
        self.status = String::from("Thinking...");
        let config = self.config.clone();
        let mut memory = self.memory.clone();
        let response_tx = self.tx.clone();
        let started = Instant::now();

        tokio::spawn(async move {
            let result = crate::agent::run_routed_turn(&stripped, &mut memory, &config).await;
            let latency = started.elapsed();
            match result {
                Ok(outcome) => {
                    if config.voice.mode != crate::config::VoiceMode::Off {
                        if let Err(e) = crate::tts::speak(&outcome.text, &config.voice.mode, &config).await
                        {
                            tracing::warn!("TTS error: {}", e);
                        }
                    }
                    let msg = Msg {
                        role: "assistant".into(),
                        content: outcome.text,
                        thinking: None,
                    };
                    let _ = response_tx.send(AppEvent::Response(
                        msg,
                        latency,
                        memory,
                        outcome.model,
                    ));
                }
                Err(e) => {
                    let msg = Msg {
                        role: "assistant".into(),
                        content: format!("Error: {}", e),
                        thinking: None,
                    };
                    let _ = response_tx.send(AppEvent::Response(
                        msg,
                        latency,
                        memory,
                        config.llm.model.clone(),
                    ));
                }
            }
        });
    }

    fn ui(&mut self, f: &mut Frame) {
        let size = f.area();

        // First-run setup screen replaces the whole layout until finished.
        if self
            .onboarding
            .as_ref()
            .map(Onboarding::is_active)
            .unwrap_or(false)
        {
            let (onb, cfg) = (self.onboarding.as_ref().unwrap(), &self.config);
            crate::tui::onboarding::render(onb, f, cfg, size);
            return;
        }

        // ── Layout: status on top, then chat+debug side by side, input below, shortcuts bar at bottom (if enabled)
        let outer = if self.show_shortcuts_bar {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Status bar
                    Constraint::Min(8),     // Chat + debug
                    Constraint::Length(3),  // Input line
                    Constraint::Length(1),  // Shortcuts bar
                ])
                .split(size)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Status bar
                    Constraint::Min(9),     // Chat + debug
                    Constraint::Length(3),  // Input line
                ])
                .split(size)
        };

        // Horizontal split: chat takes 60%, debug 40%
        let mid = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(outer[1]);

        let gpu = self.metrics.gpu.load(Ordering::Relaxed);
        let cpu = self.metrics.cpu.load(Ordering::Relaxed);
        let gpu_str = if gpu == 0 { "N/A".to_string() } else { format!("{}%", gpu) };
        let cpu_str = if cpu == 0 { "N/A".to_string() } else { format!("{}%", cpu) };

        // Status bar
        let status = StatusBar::new(
            &self.model_name,
            &gpu_str,
            &cpu_str,
            self.last_latency,
            &self.status,
            self.thinking,
        );
        f.render_widget(status, outer[0]);

        // Chat history
        let chat = ChatHistory::new(&self.messages, self.chat_scroll, self.focus == Focus::Chat);
        f.render_widget(chat, mid[0]);

        // Debug panel (right side)
        let log_lines = self.logs.lines();
        let focused = self.focus == Focus::Debug;
        let debug = DebugPanel::new(&log_lines, self.debug_scroll, focused);
        f.render_widget(debug, mid[1]);

        // Input line
        let input = InputLine::new(&self.input);
        f.render_widget(input, outer[2]);

        // Shortcuts bar (rendered only when visible)
        if self.show_shortcuts_bar {
            f.render_widget(ShortcutsBar, outer[3]);
        }

        // Masked --set-key prompt overlay (hides the secret while typing).
        if let Some(kp) = &self.key_prompt {
            let pop_w = (size.width.saturating_mul(3) / 4).clamp(30, 64);
            let pop_h = 6;
            let pop = Rect::new(
                size.x + size.width.saturating_sub(pop_w) / 2,
                outer[0].y + outer[1].height.saturating_sub(pop_h) / 2,
                pop_w,
                pop_h,
            );
            f.render_widget(Clear, pop);
            let title = format!(" --set-key {} — secret hidden ", kp.name);
            let masked = if kp.secret.is_empty() {
                Span::from("Type the secret (hidden)…")
            } else {
                Span::styled(
                    "•".repeat(kp.secret.len()),
                    Style::default().fg(Color::Cyan),
                )
            };
            let paragraph = Paragraph::new(Vec::from([
                Line::from(Span::styled(
                    title,
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(masked),
                Line::from(" "),
                Line::from(format!(
                    "{} chars · [Enter] save · [Esc] cancel",
                    kp.secret.len()
                )),
            ]))
            .block(Block::default().borders(Borders::ALL))
            .wrap(Wrap { trim: false });
            f.render_widget(paragraph, pop);
        }

        // Shortcuts help modal overlay (toggle with ? or F1)
        if self.show_help {
            let pop_w = (size.width.saturating_mul(4) / 5).clamp(40, 78);
            let pop_h = 24.min(size.height.saturating_sub(2));
            let pop = Rect::new(
                size.x + size.width.saturating_sub(pop_w) / 2,
                size.y + size.height.saturating_sub(pop_h) / 2,
                pop_w,
                pop_h,
            );
            f.render_widget(Clear, pop);
            f.render_widget(ShortcutsModal, pop);
        }

        // Slash Command Autocomplete Popover (when typing '/')
        let slash_opts = self.slash_matches();
        if !slash_opts.is_empty() {
            let pop_h = ((slash_opts.len() + 2) as u16).min(10);
            let pop_w = outer[2].width.min(68);
            let pop_y = outer[2].y.saturating_sub(pop_h);
            let pop = Rect::new(outer[2].x, pop_y, pop_w, pop_h);
            f.render_widget(Clear, pop);
            f.render_widget(
                SlashCommandMenu {
                    options: &slash_opts,
                    selected: self.slash_selected,
                },
                pop,
            );
        }

        // Interactive Settings & Menu Controls modal overlay (F2 or /config)
        if let Some(sm) = &self.settings_menu {
            let pop_w = (size.width.saturating_mul(9) / 10).clamp(50, 130);
            let pop_h = (size.height.saturating_mul(9) / 10).clamp(16, 40);
            let pop = Rect::new(
                size.x + size.width.saturating_sub(pop_w) / 2,
                size.y + size.height.saturating_sub(pop_h) / 2,
                pop_w,
                pop_h,
            );
            f.render_widget(Clear, pop);
            f.render_widget(SettingsMenuModal { state: sm }, pop);
        }

        // Interactive luna.toml editor modal overlay (toggle with F2 or :config / /config)
        if let Some(ed) = &self.config_editor {
            let pop_w = (size.width.saturating_mul(9) / 10).clamp(50, 130);
            let pop_h = (size.height.saturating_mul(9) / 10).clamp(16, 40);
            let pop = Rect::new(
                size.x + size.width.saturating_sub(pop_w) / 2,
                size.y + size.height.saturating_sub(pop_h) / 2,
                pop_w,
                pop_h,
            );
            f.render_widget(Clear, pop);
            f.render_widget(ConfigEditorModal { state: ed }, pop);
            if let Some(cursor_area) = ConfigEditorModal::cursor_area(pop, ed) {
                f.set_cursor_position(cursor_area);
            }
        }

 // Cursor
        if self.config_editor.is_none() && self.settings_menu.is_none() {
            if let Some(cursor_area) = InputLine::cursor_area(outer[2], self.cursor_pos, &self.input) {
                f.set_cursor_position(cursor_area);
            }
        }
    }
}

/// Query NVIDIA GPU utilization (%) via nvidia-smi. Returns None if unavailable.
async fn gpu_utilization() -> Option<u64> {
    let out = tokio::process::Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().split(',').next()?.trim().parse::<u64>().ok()
}

/// Approximate CPU utilization (%) by sampling /proc/stat twice.
async fn cpu_utilization() -> Option<u64> {
    let (idle1, total1) = read_proc_stat()?;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (idle2, total2) = read_proc_stat()?;
    let id_delta = idle2.saturating_sub(idle1);
    let total_delta = total2.saturating_sub(total1);
    if total_delta == 0 {
        return None;
    }
    Some(100u64.saturating_sub(id_delta.saturating_mul(100) / total_delta))
}

/// Read aggregate CPU times from /proc/stat. Returns (idle_ticks, total_ticks).
fn read_proc_stat() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    let mut fields = line.split_whitespace();
    // "cpu" header is fields[0]; values begin at fields[1]
    fields.next()?;
    let vals: Vec<u64> = fields.take(8).map(|f| f.parse().unwrap_or(0)).collect();
    let idle = vals.get(3).copied().unwrap_or(0) + vals.get(4).copied().unwrap_or(0);
    let total: u64 = vals.iter().sum();
    Some((idle, total))
}
