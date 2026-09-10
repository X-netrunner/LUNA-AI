//! tui/app.rs — TUI application state and event loop

use crate::config::LunaConfig;
use crate::llm::ollama::OllamaClient;
use crate::llm::react::ReactLoop;
use crate::memory::Memory;
use crate::tui::widgets::{ChatHistory, InputLine, StatusBar};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

type Backend = CrosstermBackend<Stdout>;
type Term = Terminal<Backend>;

#[derive(Debug)]
enum AppEvent {
    Input(Event),
    Response(Msg),
}

#[derive(Debug, Clone)]
pub struct Msg {
    pub role: String,
    pub content: String,
    pub is_tool: bool,
    pub thinking: Option<String>,
}

pub struct TuiApp {
    config: LunaConfig,
    memory: Memory,
    client: Arc<OllamaClient>,
    messages: Vec<Msg>,
    input: String,
    cursor_pos: usize,
    scroll_offset: usize,
    status: String,
    model_name: String,
    gpu_percent: String,
    last_latency: Duration,
    should_quit: bool,
    tx: mpsc::UnboundedSender<AppEvent>,
    rx: mpsc::UnboundedReceiver<AppEvent>,
}

impl TuiApp {
    pub fn new(config: LunaConfig) -> Result<Self> {
        let client = Arc::new(
            OllamaClient::new(
                &config.llm.base_url,
                &config.llm.model,
                config.llm.temperature,
                config.llm.max_tokens,
            )
            .enable_thinking(config.llm.enable_thinking)
            .debug(config.logging.level == "debug"),
        );

        let memory = Memory::new(config.memory.context_window, &config.memory.history_path)?;
        let (tx, rx) = mpsc::unbounded_channel();

        let model_name = config.llm.model.clone();
        let gpu_percent = String::from("?");

        Ok(Self {
            config,
            memory,
            client,
            messages: Vec::new(),
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            status: String::from("Ready"),
            model_name,
            gpu_percent,
            last_latency: Duration::ZERO,
            should_quit: false,
            tx,
            rx,
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

        // Spawn event reader
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

        // Initial draw
        terminal.draw(|f| self.ui(f))?;

        // Main loop
        while !self.should_quit {
            // Handle events
            while let Ok(ev) = self.rx.try_recv() {
                self.handle_app_event(ev);
            }

            // Redraw
            terminal.draw(|f| self.ui(f))?;

            tokio::time::sleep(Duration::from_millis(16)).await; // ~60fps
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
            AppEvent::Input(Event::Resize(_, _)) => {}, // auto-handled by ratatui
            AppEvent::Input(_) => {},
            AppEvent::Response(msg) => {
                self.messages.push(msg);
                self.scroll_to_bottom();
                self.status = String::from("Ready");
            },
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
                    self.add_user_message(&user_input);
                    self.status = String::from("Thinking...");
                    
                    // Create ReactLoop and run it
                    let client = self.client.clone();
                    let config = self.config.clone();
                    let mut memory = self.memory.clone();
                    let system_prompt = crate::agent::build_system_prompt(&config);
                    let response_tx = self.tx.clone(); // We'll reuse the event channel for responses
                    
                    tokio::spawn(async move {
                        let react = ReactLoop::new(&client, config.agent.max_react_iterations, config.agent.native_tools, &config);
                        match react.run(&user_input, &mut memory, &system_prompt).await {
                            Ok((text, _streamed)) => {
                                let msg = Msg {
                                    role: "assistant".into(),
                                    content: text,
                                    is_tool: false,
                                    thinking: None,
                                };
                                let _ = response_tx.send(AppEvent::Response(msg));
                            }
                            Err(e) => {
                                let msg = Msg {
                                    role: "assistant".into(),
                                    content: format!("Error: {}", e),
                                    is_tool: false,
                                    thinking: None,
                                };
                                let _ = response_tx.send(AppEvent::Response(msg));
                            }
                        }
                    });
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
            (_, KeyCode::Up) => {
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                }
            }
            (_, KeyCode::Down) => {
                if self.scroll_offset < self.messages.len().saturating_sub(1) {
                    self.scroll_offset += 1;
                }
            }
            (_, KeyCode::PageUp) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            (_, KeyCode::PageDown) => {
                self.scroll_offset = (self.scroll_offset + 10).min(self.messages.len().saturating_sub(1));
            }
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
            is_tool: false,
            thinking: None,
        });
        self.scroll_to_bottom();
    }

    fn add_assistant_message(&mut self, text: &str, thinking: Option<String>) {
        self.messages.push(Msg {
            role: "assistant".into(),
            content: text.into(),
            is_tool: false,
            thinking,
        });
        self.scroll_to_bottom();
    }

    fn add_tool_message(&mut self, tool_name: &str, result: &str) {
        self.messages.push(Msg {
            role: "tool".into(),
            content: format!("[{}]: {}", tool_name, result),
            is_tool: true,
            thinking: None,
        });
        self.scroll_to_bottom();
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll_offset = self.messages.len().saturating_sub(1);
    }

    fn ui(&mut self, f: &mut Frame) {
        let size = f.size();
        
        // Layout: header (status), chat area, input
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Status bar
                Constraint::Min(10),    // Chat history
                Constraint::Length(3),  // Input line
            ])
            .split(size);

        // Status bar
        let status = StatusBar::new(&self.model_name, &self.gpu_percent, self.last_latency, &self.status);
        f.render_widget(status, chunks[0]);

        // Chat history
        let chat = ChatHistory::new(&self.messages, self.scroll_offset);
        f.render_widget(chat, chunks[1]);

        // Input line
        let input = InputLine::new(&self.input, self.cursor_pos);
        f.render_widget(input, chunks[2]);

        // Set cursor position
        if let Some(cursor_area) = InputLine::cursor_area(chunks[2], self.cursor_pos, &self.input) {
            f.set_cursor_position(cursor_area);
        }
    }
}