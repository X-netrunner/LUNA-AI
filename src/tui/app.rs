//! tui/app.rs — TUI application state and event loop

use crate::config::{LunaConfig, VoiceMode};
use crate::memory::Memory;
use crate::tui::log::LogBuffer;
use crate::tui::widgets::{ChatHistory, DebugPanel, InputLine, StatusBar};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::{Frame, Terminal};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
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
}

impl TuiApp {
    pub fn new(config: LunaConfig, logs: LogBuffer) -> Result<Self> {
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
        })
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
            tokio::spawn(async move {
                loop {
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
                // After a voice reply, keep the wake-word window open so the
                // user can keep talking without repeating "luna".
                self.refresh_conversation_window();
            }
            AppEvent::Voice(text) => {
                let t = text.trim();
                if t.is_empty() {
                    return;
                }
                // A wake-word or window utterance keeps the window extended.
                self.refresh_conversation_window();
                self.add_user_message(t);
                self.submit(t.to_string());
            }
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

    fn handle_key(&mut self, key: KeyEvent) {
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
            (_, KeyCode::Enter) => {
                if !self.input.trim().is_empty() {
                    let user_input = self.input.clone();
                    self.input.clear();
                    self.cursor_pos = 0;
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

    /// Spawn the routed async turn (fast/deep/full) — the UI stays responsive.
    /// The status bar shows the model that actually answered.
    fn submit(&mut self, user_input: String) {
        let lower = user_input.trim().to_lowercase();
        match lower.as_str() {
            "clear" => {
                let _ = self.memory.clear();
                self.messages.push(Msg {
                    role: "assistant".into(),
                    content: "Memory cleared. Fresh start.".into(),
                    thinking: None,
                });
                self.chat_scroll = 0;
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
            let result = crate::agent::run_routed_turn(&user_input, &mut memory, &config).await;
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

        // ── Layout: status on top, then chat+debug side by side, input below
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Status bar
                Constraint::Min(10),    // Chat + debug
                Constraint::Length(3),  // Input line
            ])
            .split(size);

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

        // Cursor
        if let Some(cursor_area) = InputLine::cursor_area(outer[2], self.cursor_pos, &self.input) {
            f.set_cursor_position(cursor_area);
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