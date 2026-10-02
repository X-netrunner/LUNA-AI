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
    /// A live progress line for the turn in flight — "step 2", "⚙ nmap_scan…",
    /// "✓ run_shell (2.5s)". Carries a rendered string rather than the
    /// `activity::Event` itself, so the status-bar formatting lives with the
    /// other UI code instead of in the event plumbing.
    Activity(String),
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

/// Modal for pasting a developer key to unlock the security tier's no-refusal
/// mode.
///
/// A separate modal from `key_prompt` because the two are different things: that
/// one stores a secret in the OS keyring, this one consumes a signing key to
/// produce a receipt and then throws the key away. Nothing typed here is stored,
/// and the input is masked because a developer pasting a key into a shared
/// screen would otherwise put it in their scrollback.
struct DevKeyPrompt {
    input: String,
    error: Option<String>,
    /// Set when the config has no public key, where no key could ever work and
    /// the modal should say so instead of collecting input.
    unavailable_reason: Option<String>,
    /// Which switch opened this prompt.
    ///
    /// There is one signature, so one key unlocks both switches — but it must
    /// still know which one the user was trying to turn on, because verifying
    /// the key only authorises the gate; the switch itself is a separate config
    /// flag, and a prompt that forgot which flag to set would leave the user
    /// staring at an unchanged menu.
    pending: PendingSwitch,
}

/// Which config switch a `DevKeyPrompt` was opened for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PendingSwitch {
    /// `llm.security_unrestricted` — answer without refusing.
    NoRefusal,
    /// `external.allow_capability_actions` — act on the machine.
    Capability,
}

impl PendingSwitch {
    fn label(self) -> &'static str {
        match self {
            PendingSwitch::NoRefusal => "No-Refusal Mode",
            PendingSwitch::Capability => "Capability Actions",
        }
    }
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
                    SettingItem {
                        key: "llm.security_unrestricted",
                        label: "Security Tier: No Refusal",
                        description: "DEVELOPER KEY REQUIRED. Removes authorisation checks from \
                                      the offensive-security tier. Shows the real gate state — the \
                                      switch is only ON when a valid developer signature is on \
                                      disk, not merely when this is ticked.",
                        setting_type: SettingType::Value(
                            config.llm.security_gate_state().as_str().to_string(),
                        ),
                    },
                    SettingItem {
                        key: "external.allow_capability_actions",
                        label: "Capability Actions",
                        description: "DEVELOPER KEY REQUIRED — the same key as No Refusal. Lets \
                                      Luna act on the machine: scan hosts, rewrite firewall and \
                                      sysctl state, or edit her own source. Show the real gate \
                                      state, not the raw switch, so a ticked box with no valid \
                                      signature on disk cannot read as ON.",
                        setting_type: SettingType::Value(
                            crate::unlock::gate_state(
                                config.external.allow_capability_actions,
                                &config.llm.security_dev_public_key,
                            )
                            .as_str()
                            .to_string(),
                        ),
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

    /// Space/Enter on the item under the cursor.
    ///
    /// The signed switches are not toggles. Flipping the config boolean alone
    /// would put Luna in a state where the config says ON and the tool is still
    /// refused, which is exactly the confusion the gate exists to prevent. So
    /// they report the real state and, if it is off, say what would unlock it —
    /// resolved before the mutable borrow of `categories` below.
    pub fn toggle_selected(&mut self) {
        if let Some(key) = self.signed_item_key() {
            self.describe_signed_gate(key);
            return;
        }
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

    /// Is the cursor on one of the two signed switches?
    pub fn is_on_security_item(&self) -> bool {
        self.signed_item_key().is_some()
    }

    /// The signed switch under the cursor, if any.
    pub fn signed_item_key(&self) -> Option<&'static str> {
        self.categories
            .get(self.cat_index)
            .and_then(|c| c.items.get(self.item_index))
            .and_then(|i| match i.key {
                "llm.security_unrestricted" => Some("llm.security_unrestricted"),
                "external.allow_capability_actions" => Some("external.allow_capability_actions"),
                _ => None,
            })
    }

    /// The effective gate state of either signed switch.
    fn gate_state_of(&self, key: &str) -> crate::unlock::GateState {
        if key == "external.allow_capability_actions" {
            crate::unlock::gate_state(
                self.config.external.allow_capability_actions,
                &self.config.llm.security_dev_public_key,
            )
        } else {
            self.config.llm.security_gate_state()
        }
    }

    /// Report the true state of whichever signed switch is under the cursor.
    ///
    /// Deliberately does not flip the config boolean. The switch is only ON when a
    /// valid signature is on disk, so a tick mark that could mean "asked for" is
    /// a lie. This says which of the four states it is in and, for the two
    /// unlockable ones, what the user has to do.
    fn describe_signed_gate(&mut self, key: &'static str) {
        use crate::unlock::GateState;
        let st = self.gate_state_of(key);
        let cap = key == "external.allow_capability_actions";
        let label = if cap { "Capability Actions" } else { "No-Refusal Mode" };
        let other = if cap {
            "It shares the developer key with No-Refusal Mode, but has its own switch: Luna may \
             still scan hosts, harden the system, or edit her own source."
        } else {
            "It shares the developer key with Capability Actions, but has its own switch: Luna \
             may still refuse to scan hosts, harden the system, or edit her own source."
        };
        self.status = match st {
            GateState::Unlocked => format!(
                "🔓 {label}: ON. {} [S] to save · press again to LOCK",
                if cap {
                    "Luna may act on the machine."
                } else {
                    "The security tier will not evaluate authorisation and will not refuse."
                }
            ),
            GateState::Locked => format!(
                "🔒 {label}: OFF — requested but not unlocked (no valid developer signature at {}). \
                 Unlock it with: luna --unlock-security, or press T and set the switch to false to \
                 drop the request. {other}",
                crate::unlock::receipt_path_display()
            ),
            GateState::NotRequested => format!(
                "🔒 {label}: OFF. To enable: run `luna --gen-dev-key`, put the public key in \
                 luna.toml as [llm] security_dev_public_key, then run `luna --unlock-security` with \
                 the private key."
            ),
            GateState::Unavailable => format!(
                "⛔ {label}: permanently unavailable — no developer public key is configured, so \
                 no key can ever unlock it. Run `luna --gen-dev-key` and set [llm] \
                 security_dev_public_key in luna.toml first."
            ),
        };
    }

    /// Enable or disable one of the two signed switches.
    ///
    /// Returns `true` when the caller should open the key prompt, because a key
    /// is needed and one was supplied by nobody yet.
    ///
    /// The asymmetry from the two-gate version: turning a switch OFF no longer
    /// deletes the receipt, because the receipt is shared and deleting it would
    /// silently revoke the *other* switch too. It doesn't need to — a switch that
    /// is false reports `NotRequested` and is inactive regardless of what is on
    /// disk, so "off" is genuinely off. What the receipt would still buy is a
    /// re-enable without re-entering the key, so the status line says so and
    /// points at `--lock-security` for a real revocation.
    pub fn set_signed_switch(&mut self, key: &'static str, want: bool) -> bool {
        use crate::unlock::GateState;
        let cap = key == "external.allow_capability_actions";
        let label = if cap { "Capability Actions" } else { "No-Refusal Mode" };

        if cap {
            self.config.external.allow_capability_actions = want;
        } else {
            self.config.llm.security_unrestricted = want;
        }

        if !want {
            self.refresh_signed_items();
            self.modified = true;
            let other_is_on = if cap {
                self.config.llm.security_unrestricted_active()
            } else {
                crate::unlock::capabilities_active(
                    self.config.external.allow_capability_actions,
                    &self.config.llm.security_dev_public_key,
                )
            };
            self.status = if other_is_on {
                format!(
                    "🔒 {label}: OFF. The developer receipt was NOT deleted — \
                     {} is still on and shares it. Use `luna --lock-security` to revoke the key \
                     for real. [Press S to save]",
                    if cap { "No-Refusal Mode" } else { "Capability Actions" }
                )
            } else {
                format!(
                    "🔒 {label}: OFF. [Press S to save · `luna --lock-security` revokes the key \
                     for real]"
                )
            };
            return false;
        }

        match self.gate_state_of(key) {
            GateState::Unlocked => {
                self.refresh_signed_items();
                self.modified = true;
                self.status =
                    format!("🔓 {label}: ON (developer key verified). [Press S to save]");
                false
            }
            GateState::Unavailable => {
                // The request stays set. Reverting it would make the state
                // `NotRequested`, which reads as "the user never asked" and hides
                // the actual problem — that they asked and there is no key. The
                // effective state is still off either way, so nothing is enabled
                // by leaving it; and the status line explains the remedy.
                self.refresh_signed_items();
                self.describe_signed_gate(key);
                false
            }
            _ => {
                // Locked, or NotRequested-with-a-key. Either way a key is needed.
                // Keep the request set so that a successful unlock takes effect
                // without a second toggle.
                true
            }
        }
    }

    /// Enable or disable the no-refusal mode, subject to the gate.
    pub fn set_security_unrestricted(&mut self, want: bool) -> bool {
        self.set_signed_switch("llm.security_unrestricted", want)
    }

    /// Re-read both signed gates and mirror them onto the menu items.
    fn refresh_signed_items(&mut self) {
        let no_refusal = self.config.llm.security_gate_state().as_str().to_string();
        let cap = crate::unlock::gate_state(
            self.config.external.allow_capability_actions,
            &self.config.llm.security_dev_public_key,
        )
        .as_str()
        .to_string();
        for cat in self.categories.iter_mut() {
            for item in cat.items.iter_mut() {
                match item.key {
                    "llm.security_unrestricted" => {
                        item.setting_type = SettingType::Value(no_refusal.clone())
                    }
                    "external.allow_capability_actions" => {
                        item.setting_type = SettingType::Value(cap.clone())
                    }
                    _ => {}
                }
            }
        }
    }

    /// Re-read the gate and mirror it onto the menu item.
    fn refresh_security_item(&mut self) {
        self.refresh_signed_items();
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
    /// Modal asking for a developer key to unlock the no-refusal mode. Takes
    /// priority over the settings menu while open.
    dev_key_prompt: Option<DevKeyPrompt>,
    /// Index in slash command autocomplete popup.
    slash_selected: usize,
    /// True while a turn is in flight, so live `Activity` lines are shown.
    ///
    /// Without this the status bar would be overwritten by progress from a turn
    /// that has already finished — the subscription is dropped when the spawned
    /// task ends, but an event can already be queued behind the `Response` that
    /// ends the turn, and it would then stomp "Ready".
    pending_turn: bool,
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
            pending_turn: false,
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
            dev_key_prompt: None,
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
                self.pending_turn = false;
                // Reply delivered (and spoken) — clear the thinking pulse.
                crate::overlay::signal("idle");
                // After a voice reply, keep the wake-word window open so the
                // user can keep talking without repeating "luna". Only extends
                // an already-active window — a typed turn shouldn't open one.
                if self.window_deadline.load(Ordering::Relaxed) > 0 {
                    self.refresh_conversation_window();
                }
            }
            // Progress replaces the static "Thinking..." rather than appending to
            // the chat, so a turn doing real work stays legible: the newest line
            // is the current one, and the reply lands underneath it.
            AppEvent::Activity(line) => {
                if self.pending_turn {
                    self.status = line;
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

    /// The one place the signed switches are driven from the UI.
    ///
    /// Turning one ON requires a developer key, so this either opens the key
    /// prompt or explains why it cannot. Turning one OFF just drops the config
    /// request; the shared receipt stays, because revoking it here would revoke
    /// the other switch without being asked.
    fn press_security_item(&mut self) {
        // Already unlocked → this press turns it OFF. Requested but locked →
        // this press is a request for the key, so keep the request set and open
        // the prompt rather than dropping it.
        let (key, want) = match self.settings_menu.as_mut() {
            Some(sm) => {
                let Some(key) = sm.signed_item_key() else { return };
                let want = sm.gate_state_of(key) != crate::unlock::GateState::Unlocked;
                (key, want)
            }
            None => return,
        };
        let Some(sm) = self.settings_menu.as_mut() else {
            return;
        };
        if sm.set_signed_switch(key, want) {
            // A key is needed. When none is configured, say so in the modal
            // instead of collecting a key that could never work.
            let unavailable_reason = if !crate::unlock::has_public_key(
                &self.config.llm.security_dev_public_key,
            ) {
                Some(
                    "No developer public key is configured.\n\n\
                     Run:  luna --gen-dev-key\n\
                     Put the public key in luna.toml as [llm] security_dev_public_key\n\
                     Then come back and press Space again."
                        .to_string(),
                )
            } else {
                None
            };
            self.dev_key_prompt = Some(DevKeyPrompt {
                input: String::new(),
                error: None,
                unavailable_reason,
                pending: if key == "external.allow_capability_actions" {
                    PendingSwitch::Capability
                } else {
                    PendingSwitch::NoRefusal
                },
            });
        } else {
            self.dev_key_prompt = None;
        }
    }

    /// Open (or extend) the wake-word-free conversation window on any voice
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

        // Developer key modal takes priority over everything below it.
        if self.dev_key_prompt.is_some() {
            self.handle_dev_key_prompt(key);
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
                    if sm.is_on_security_item() {
                        self.press_security_item();
                    } else {
                        sm.toggle_selected();
                    }
                }
                // An explicit, discoverable route to the same place. The security
                // item is a Value, not a Toggle, so Space is ambiguous — and a
                // user who wants this should not have to guess which key opens it.
                (_, KeyCode::Char('u')) | (_, KeyCode::Char('U')) => {
                    if sm.is_on_security_item() {
                        self.press_security_item();
                    }
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
    /// Key handling for the developer-key modal.
    ///
    /// The key is consumed on Enter and immediately dropped: it is never stored,
    /// never logged, and never written. What lands on disk is a signature.
    fn handle_dev_key_prompt(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            if matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d')) {
                self.should_quit = true;
            }
            return;
        }
        match key.code {
            KeyCode::Esc => {
                self.dev_key_prompt = None;
            }
            KeyCode::Backspace => {
                if let Some(p) = self.dev_key_prompt.as_mut() {
                    p.input.pop();
                    p.error = None;
                }
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(p) = self.dev_key_prompt.as_mut() {
                    p.input.clear();
                    p.error = None;
                }
            }
            KeyCode::Enter => {
                let Some(p) = self.dev_key_prompt.take() else {
                    return;
                };
                let cfg = &self.config;
                match crate::unlock::unlock(&cfg.llm.security_dev_public_key, &p.input) {
                    Ok(()) => {
                        // Verified. Now set the request the user actually asked for
                        // and mirror the real state onto both items, so the user
                        // sees the effect rather than having to infer it.
                        if let Some(sm) = self.settings_menu.as_mut() {
                            match p.pending {
                                PendingSwitch::NoRefusal => {
                                    sm.config.llm.security_unrestricted = true
                                }
                                PendingSwitch::Capability => {
                                    sm.config.external.allow_capability_actions = true
                                }
                            }
                            sm.refresh_security_item();
                            sm.modified = true;
                            sm.status = format!(
                                "🔓 Developer key verified — {} ON. [Press S to save]",
                                p.pending.label()
                            );
                        }
                        // `input` is dropped here at the end of scope.
                        self.status = String::from("Developer key accepted.");
                    }
                    Err(e) => {
                        // Keep the modal open with the error. A failed attempt
                        // must leave the feature off, and the user needs to see
                        // why rather than having the dialog vanish.
                        self.dev_key_prompt = Some(DevKeyPrompt {
                            input: p.input,
                            error: Some(e.to_string()),
                            unavailable_reason: p.unavailable_reason,
                            pending: p.pending,
                        });
                    }
                }
            }
            KeyCode::Char(c) => {
                if let Some(p) = self.dev_key_prompt.as_mut() {
                    p.input.push(c);
                    p.error = None;
                }
            }
            _ => {}
        }
    }

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
        self.pending_turn = true;
        let config = self.config.clone();
        let mut memory = self.memory.clone();
        let response_tx = self.tx.clone();
        let started = Instant::now();

        tokio::spawn(async move {
            // Live tool progress in the status bar.
            //
            // The TUI previously showed "Thinking..." for the entire turn, which
            // covered the 16–37s generations AND the tool calls — so a turn that
            // was running `sudo pacman -Syu` looked identical to one that was
            // generating a sentence. The subscription lives inside the spawned
            // task, so it is dropped when the turn ends and cannot outlive it or
            // leak into the next one.
            //
            // `activity_tx` is a separate clone moved into the 'static closure.
            // Borrowing `response_tx` cannot work: subscribers outlive this task
            // (they unregister on drop, and the compiler cannot prove that), so
            // the closure must own what it sends on.
            let activity_tx = response_tx.clone();
            let _activity = crate::activity::subscribe(move |ev| {
                let line = match ev {
                    crate::activity::Event::Iteration { n } => {
                        format!("Thinking… (step {n})")
                    }
                    crate::activity::Event::ToolStart { name, summary } => {
                        if summary.is_empty() || summary == name {
                            format!("⚙ {name}…")
                        } else {
                            format!("⚙ {name}: {summary}")
                        }
                    }
                    crate::activity::Event::ToolEnd { name, ok, elapsed } => format!(
                        "{} {name} ({:.1}s)",
                        if *ok { "✓" } else { "✗" },
                        elapsed.elapsed().as_secs_f64()
                    ),
                };
                let _ = activity_tx.send(AppEvent::Activity(line));
            });

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

        // Developer key modal — drawn last so it sits above the settings menu it
        // was opened from.
        if let Some(dk) = &self.dev_key_prompt {
            let pop_w = (size.width.saturating_mul(3) / 4).clamp(48, 96);
            let pop_h = 14.min(size.height.saturating_sub(2));
            let pop = Rect::new(
                size.x + size.width.saturating_sub(pop_w) / 2,
                size.y + size.height.saturating_sub(pop_h) / 2,
                pop_w,
                pop_h,
            );
            f.render_widget(Clear, pop);

            let outer_p = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),
                    Constraint::Min(1),
                    Constraint::Length(4),
                    Constraint::Length(1),
                ])
                .split(pop);

            let title = Paragraph::new("🔑 Developer Key — Security Tier: No Refusal")
                .style(Style::default().add_modifier(Modifier::BOLD));
            f.render_widget(title, outer_p[0]);

            let shown = if dk.input.is_empty() {
                " ".repeat(16)
            } else {
                // Masked. A 64-character key is short, but a paste of the wrong
                // thing could be anything, so the field is capped and the tail
                // is shown to confirm what arrived.
                let masked: String = "•".repeat(dk.input.len().min(24));
                if dk.input.len() > 24 {
                    format!("{masked}… ({} chars)", dk.input.len())
                } else {
                    masked
                }
            };
            let field = Paragraph::new(format!("Key: {shown}")).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(if dk.input.is_empty() {
                        " paste developer key "
                    } else {
                        " "
                    }),
            );
            f.render_widget(field, outer_p[1]);

            let msg = if let Some(reason) = &dk.unavailable_reason {
                Paragraph::new(reason.clone()).wrap(Wrap { trim: true })
            } else if let Some(err) = &dk.error {
                Paragraph::new(format!("✗ {err}")).wrap(Wrap { trim: true })
            } else {
                Paragraph::new(
                    "The key is used once to sign a challenge, then discarded. Only the \
                     signature is stored. Nothing typed here is saved or logged.",
                )
                .wrap(Wrap { trim: true })
            };
            f.render_widget(msg, outer_p[2]);

            let hint = Paragraph::new("[Enter] verify  [Esc] cancel  [Ctrl+U] clear")
                .style(Style::default().add_modifier(Modifier::DIM));
            f.render_widget(hint, outer_p[3]);
        }

        // Cursor
        if self.dev_key_prompt.is_none() && self.config_editor.is_none() && self.settings_menu.is_none()
        {
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
#[cfg(test)]
mod security_gate_tests {
    use super::*;

    /// A cursor on the security item is recognised as such, so Space reaches the
    /// gate instead of flipping a boolean.
    ///
    /// The category list is positional, so this pins that the item lives where
    /// the tests think it does — otherwise a test that "navigated" to the wrong
    /// row would pass while asserting nothing.
    #[test]
    fn the_security_item_is_findable_and_is_not_a_toggle() {
        let mut sm = SettingsMenuState::load(crate::config::LunaConfig::default());
        assert!(
            !sm.is_on_security_item(),
            "the security item must not be selected by default"
        );

        let mut found = false;
        for _ in 0..sm.categories.len() {
            for _ in 0..20 {
                if sm.is_on_security_item() {
                    found = true;
                    break;
                }
                sm.next_item();
            }
            if found {
                break;
            }
            sm.next_category();
        }
        assert!(found, "no security item exists in the settings menu");

        // It must render as a state, not a checkbox. A tick that could mean
        // "requested" while the model is still scoped is the confusion the gate
        // exists to prevent.
        let cat = &sm.categories[sm.cat_index];
        let item = &cat.items[sm.item_index];
        assert!(
            matches!(item.setting_type, SettingType::Value(_)),
            "the security item must show state, not a boolean toggle"
        );
    }

    /// Enabling without a key must not enable anything, and must say what the
    /// user has to do about it.
    ///
    /// The request flag is deliberately left set. Reverting it would report
    /// `NotRequested` — indistinguishable from "never asked" — and bury the
    /// actual problem, which is that the request is unsatisfiable. The effective
    /// state is off regardless, so nothing is enabled by leaving it.
    #[test]
    fn enabling_without_a_key_enables_nothing_and_explains_itself() {
        let mut sm = SettingsMenuState::load(crate::config::LunaConfig::default());
        let needs_prompt = sm.set_security_unrestricted(true);

        assert!(
            !needs_prompt,
            "opening a key prompt with no configured key would collect input that \
             can never work"
        );
        assert!(
            !sm.config.llm.security_unrestricted_active(),
            "the feature was enabled with no developer key"
        );
        assert_eq!(
            sm.config.llm.security_gate_state(),
            crate::unlock::GateState::Unavailable,
            "the state must show that no key is configured, not 'never asked'"
        );
        assert!(
            sm.status.contains("gen-dev-key"),
            "unhelpful status: {}",
            sm.status
        );
    }

    /// The capability switch exists in the menu, reports its real state, and is
    /// not a plain toggle.
    ///
    /// The TUI was the one place the capability switch never got exposed, so the
    /// only way to turn it on was hand-editing the config — exactly the path
    /// that, combined with the boolean gate firing first, produced a confusing
    /// refusal. Three properties asserted, because each fails differently:
    /// present, honest, and non-toggleable.
    #[test]
    fn the_capability_switch_is_present_honest_and_not_a_plain_toggle() {
        use crate::unlock;
        let _receipt = unlock::test_receipt(unlock::scratch_receipt("tui_cap"));

        let (pk, sk) = unlock::generate_keypair();
        let mut sm = SettingsMenuState::load(crate::config::LunaConfig::default());
        sm.config.llm.security_dev_public_key = pk.clone();

        // Present.
        assert!(
            sm.categories
                .iter()
                .flat_map(|c| c.items.iter())
                .any(|i| i.key == "external.allow_capability_actions"),
            "the capability switch must be reachable from the menu"
        );

        // Honest before anything is requested: NOT "ON" off the back of the key.
        let mut item = sm
            .categories
            .iter()
            .flat_map(|c| c.items.iter())
            .find(|i| i.key == "external.allow_capability_actions")
            .map(|i| i.setting_type.clone())
            .expect("just asserted");
        assert!(
            matches!(&item, crate::tui::app::SettingType::Value(v) if v == "OFF"),
            "must show the gate state, not a boolean: {item:?}"
        );

        // Not toggleable: Space/Enter must not flip the config boolean, because
        // that would produce "config says ON, tool still refused".
        sm.config.external.allow_capability_actions = false;
        for (ci, cat) in sm.categories.iter().enumerate() {
            if let Some(ii) = cat
                .items
                .iter()
                .position(|i| i.key == "external.allow_capability_actions")
            {
                sm.cat_index = ci;
                sm.item_index = ii;
            }
        }
        sm.modified = false;
        assert_eq!(sm.signed_item_key(), Some("external.allow_capability_actions"));
        sm.toggle_selected();
        assert!(
            !sm.config.external.allow_capability_actions,
            "Space on the capability switch must report, not flip"
        );
        assert!(!sm.modified, "reporting a state is not an edit");

        // Real unlock: requested + signed → ON.
        unlock::unlock(&pk, &sk).expect("unlock must work in a test");
        assert!(!sm.set_signed_switch("external.allow_capability_actions", true));
        assert!(
            unlock::capabilities_active(sm.config.external.allow_capability_actions, &pk),
            "after a verified unlock the capability switch must actually be on"
        );

        // And the same key covers the other switch, which is the whole point of
        // one signature: it authorises whichever switches are requested, and only
        // those. Here the other switch was never requested, so it stays off.
        assert!(!sm.set_signed_switch("llm.security_unrestricted", true));
        assert!(
            sm.config.llm.security_unrestricted_active(),
            "a verified key should authorise the second switch too"
        );
        assert!(
            sm.config.external.allow_capability_actions,
            "the independent switch must not have been dropped"
        );

        // Revoking the one key closes both, which is the cost of sharing it.
        unlock::lock().expect("locking must succeed");
        assert!(!sm.config.llm.security_unrestricted_active());
        assert!(
            !unlock::capabilities_active(sm.config.external.allow_capability_actions, &pk),
            "an explicit lock must revoke both switches"
        );
    }

    /// With a key configured but not unlocked, the request is kept — so that a
    /// successful unlock takes effect without a second press — but the effective
    /// state must stay off.
    #[test]
    fn a_configured_but_locked_key_keeps_the_request_and_stays_off() {
        let mut sm = SettingsMenuState::load(crate::config::LunaConfig::default());
        sm.config.llm.security_dev_public_key = "22".repeat(32);

        let needs_prompt = sm.set_security_unrestricted(true);
        assert!(needs_prompt, "a key prompt is needed when locked");
        assert!(sm.config.llm.security_unrestricted, "the request should persist");
        assert!(
            !sm.config.llm.security_unrestricted_active(),
            "locked, so the effective state must be off"
        );
        assert_eq!(
            sm.config.llm.security_gate_state(),
            crate::unlock::GateState::Locked
        );
    }

    /// Turning a signed switch off drops the request, not the shared receipt.
    ///
    /// This is the one behaviour that genuinely changed when the two gates were
    /// collapsed into one key. The old code deleted the receipt here, which was
    /// safe when each gate had its own — but with one receipt, turning off
    /// No-Refusal would silently revoke Capability Actions too, without the user
    /// asking for that. So the request is dropped and the receipt stays, and the
    /// status line says so and points at the real revocation.
    ///
    /// The property that makes off still *off*: a switch set to false reports
    /// `NotRequested` and is inactive no matter what is on disk. Asserted below.
    ///
    /// Uses the shared test receipt guard, so the real `~/.local/state` is never
    /// written — otherwise this test would leave the real feature unlocked.
    #[test]
    fn turning_one_off_drops_the_request_but_keeps_the_shared_receipt() {
        use crate::unlock;
        let _receipt = unlock::test_receipt(unlock::scratch_receipt("tui_off"));

        let (pk, sk) = unlock::generate_keypair();
        unlock::unlock(&pk, &sk).expect("unlock must work in a test");

        let mut sm = SettingsMenuState::load(crate::config::LunaConfig::default());
        sm.config.llm.security_dev_public_key = pk.clone();
        sm.config.llm.security_unrestricted = true;
        sm.config.external.allow_capability_actions = true;

        // Precondition: genuinely on, so the assertions afterwards mean something.
        assert!(
            sm.config.llm.security_unrestricted_active(),
            "test setup failed — never reached the ON state, so turning it off proves nothing"
        );
        assert!(
            unlock::capabilities_active(sm.config.external.allow_capability_actions, &pk),
            "test setup failed — capability switch never came on"
        );

        let needs_prompt = sm.set_security_unrestricted(false);
        assert!(!needs_prompt, "turning off must never need a key");
        assert!(!sm.config.llm.security_unrestricted);
        assert!(
            !sm.config.llm.security_unrestricted_active(),
            "still on after turning off"
        );
        // The switch really is off regardless of the surviving receipt.
        assert_eq!(
            unlock::gate_state(false, &pk),
            unlock::GateState::NotRequested,
            "a false switch must report NotRequested, whatever is on disk"
        );
        // And the user is told the receipt survived, and how to really revoke.
        assert!(
            sm.status.contains("NOT deleted") && sm.status.contains("--lock-security"),
            "status must not imply the key was revoked: {}",
            sm.status
        );

        // The receipt survives, so the OTHER switch is untouched — the reason
        // this test exists.
        assert!(
            unlock::is_unlocked(&pk),
            "the receipt must survive so the other switch keeps working"
        );
        assert!(
            sm.config.llm.security_unrestricted_active()
                || unlock::capabilities_active(sm.config.external.allow_capability_actions, &pk),
            "turning one switch off must not have disabled both"
        );

        // The real revocation is explicit and affects both.
        unlock::lock().expect("locking must succeed");
        assert!(
            !unlock::capabilities_active(true, &pk),
            "an explicit lock must revoke the other switch too"
        );
    }

    /// The UI must never be the thing that decides. If a receipt exists and
    /// verifies, the state is ON — but only if the config also asks for it.
    ///
    /// Both halves matter. A receipt alone must not silently re-enable the
    /// feature, and a request alone must not enable it either.
    #[test]
    fn the_effective_state_follows_the_receipt_not_the_boolean() {
        use crate::unlock;
        let _receipt = unlock::test_receipt(unlock::scratch_receipt("tui_effective"));
        let (pk, sk) = unlock::generate_keypair();
        unlock::unlock(&pk, &sk).unwrap();

        // Requested + unlocked → on.
        let mut cfg = crate::config::LlmConfig::default();
        cfg.security_unrestricted = true;
        cfg.security_dev_public_key = pk.clone();
        assert!(cfg.security_unrestricted_active());
        assert_eq!(cfg.security_gate_state(), unlock::GateState::Unlocked);

        // Requested but not unlocked → off. (Deleting the receipt directly, so
        // this exercises the gate rather than the UI's off path.)
        unlock::lock().unwrap();
        assert!(
            !cfg.security_unrestricted_active(),
            "a request with no receipt enabled the feature"
        );
        assert_eq!(cfg.security_gate_state(), unlock::GateState::Locked);

        // Unlocked but not requested → off. A leftover receipt must not silently
        // re-enable a feature the user has turned off.
        unlock::unlock(&pk, &sk).unwrap();
        cfg.security_unrestricted = false;
        assert!(
            !cfg.security_unrestricted_active(),
            "a leftover receipt re-enabled the feature without being asked"
        );
        assert_eq!(cfg.security_gate_state(), unlock::GateState::NotRequested);
    }
}
