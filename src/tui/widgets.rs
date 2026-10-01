//! tui/widgets.rs — Custom minimalist TUI widgets
//!
//! The chat and debug panels render with `Paragraph` (wrap + scroll). Ratatui's
//! Paragraph clears its text area on render, so stale cells never leak through
//! when content shrinks or scrolls.

use crate::tui::app::Msg;
use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

// ── Text wrapping helper ──────────────────────────────────────────────────────
/// Pick a content color that makes the interesting parts of a debug line pop:
/// research sources green, tool calls magenta, thinking blocks yellow, model
/// output cyan. Everything else follows the level color (from the badge).
fn debug_content_color(log: &str, level: Color) -> Color {
    let l = log.to_lowercase();
    if l.contains("sources:") {
        Color::Green
    } else if l.contains("tool call")
        || l.contains("executing tool")
        || l.contains("intercepted freeform")
    {
        Color::Magenta
    } else if l.contains("[think]") || l.contains("rescued answer") {
        Color::Yellow
    } else if l.contains("model output") {
        Color::Cyan
    } else {
        level
    }
}

/// Wrap `text` to `width` columns at word boundaries, preserving newlines.
/// Used so long assistant replies render on multiple lines instead of one.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split('\n') {
        if para.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut line_width = 0usize;
        for word in para.split_whitespace() {
            let w = word.chars().count();
            let sep = if line.is_empty() { 0 } else { 1 };
            if line_width + sep + w > width.max(4) && !line.is_empty() {
                out.push(std::mem::take(&mut line));
                line_width = 0;
            }
            if !line.is_empty() {
                line.push(' ');
                line_width += 1;
            }
            line.push_str(word);
            line_width += w;
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Scroll offset (in wrapped lines, from the top) for a widget that tracks
/// `scroll_from_bottom`. 0 means "follow the newest content".
fn scroll_top(total: usize, view_h: usize, scroll_from_bottom: usize) -> u16 {
    let max = total.saturating_sub(view_h);
    if scroll_from_bottom == 0 {
        max as u16
    } else {
        max.saturating_sub(scroll_from_bottom) as u16
    }
}

// ── Chat history ──────────────────────────────────────────────────────────────

pub struct ChatHistory<'a> {
    messages: &'a [Msg],
    /// How many *lines* up from the bottom we're scrolled. 0 = follow newest.
    scroll_from_bottom: usize,
    focused: bool,
}

impl<'a> ChatHistory<'a> {
    pub fn new(messages: &'a [Msg], scroll_from_bottom: usize, focused: bool) -> Self {
        Self { messages, scroll_from_bottom, focused }
    }

    fn build_lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for msg in self.messages {
            let (prefix, indent, color) = match msg.role.as_str() {
                "user" => ("you › ", "      ", Color::Cyan),
                "assistant" => ("luna › ", "       ", Color::Rgb(180, 140, 255)),
                "tool" => ("tool › ", "       ", Color::Yellow),
                _ => ("", "", Color::White),
            };
            let content_width = width.saturating_sub(prefix.chars().count() + 2);
            let mut first = true;
            for chunk in wrap_text(&msg.content, content_width.max(4)) {
                if first {
                    lines.push(Line::from(vec![
                        Span::styled(
                            prefix.to_string(),
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(chunk, Style::default().fg(Color::White)),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(vec![
                        Span::raw(indent.to_string()),
                        Span::styled(chunk, Style::default().fg(Color::White)),
                    ]));
                }
            }
            if let Some(thinking) = &msg.thinking {
                for chunk in wrap_text(thinking, width.saturating_sub(16)) {
                    lines.push(Line::from(Span::styled(
                        format!("  │ [think] {}", chunk),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    )));
                }
            }
            lines.push(Line::from(""));
        }
        lines
    }
}

impl<'a> Widget for ChatHistory<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" ✦ Chat ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(if self.focused {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::Rgb(60, 60, 60))
            });

        let inner = block.inner(area);
        block.render(area, buf);

        if self.messages.is_empty() {
            let empty = Paragraph::new("No messages yet. Type a message or :config to edit settings.")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            empty.render(inner, buf);
            return;
        }

        let lines = self.build_lines(inner.width as usize);
        let total = lines.len();
        let view_h = inner.height.max(1) as usize;
        let offset = scroll_top(total, view_h, self.scroll_from_bottom);
        Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((offset, 0)).render(inner, buf);
    }
}

// ── Input line ────────────────────────────────────────────────────────────────

pub struct InputLine<'a> {
    input: &'a str,
}

impl<'a> InputLine<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input }
    }

    pub fn cursor_area(area: Rect, cursor_pos: usize, _input: &str) -> Option<(u16, u16)> {
        let x = area.x + 1 + cursor_pos as u16;
        let y = area.y + 1;
        if x < area.x + area.width - 1 && y < area.y + area.height - 1 {
            Some((x, y))
        } else {
            None
        }
    }
}

impl<'a> Widget for InputLine<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" ❯ Prompt ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(80, 200, 140)));

        let inner = block.inner(area);
        block.render(area, buf);

        if self.input.is_empty() {
            let placeholder = Paragraph::new("Type a message or :config to edit luna.toml…")
                .style(Style::default().fg(Color::Rgb(90, 90, 90)));
            placeholder.render(inner, buf);
        } else {
            let paragraph = Paragraph::new(self.input)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: false });
            paragraph.render(inner, buf);
        }
    }
}

// ── Debug / metrics panel ─────────────────────────────────────────────────────

pub struct DebugPanel<'a> {
    logs: &'a [String],
    scroll_from_bottom: usize,
    focused: bool,
}

impl<'a> DebugPanel<'a> {
    pub fn new(logs: &'a [String], scroll_from_bottom: usize, focused: bool) -> Self {
        Self { logs, scroll_from_bottom, focused }
    }

    fn build_lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for log in self.logs {
            let (badge, color) = if log.starts_with("ERROR") {
                ("✖", Color::Red)
            } else if log.starts_with("WARN") {
                ("▲", Color::Yellow)
            } else if log.starts_with("INFO") {
                ("●", Color::Cyan)
            } else if log.starts_with("TRACE") {
                ("·", Color::DarkGray)
            } else {
                ("◆", Color::DarkGray)
            };
            let compact: String = log.chars().take(215).collect();
            let content_color = debug_content_color(&compact, color);
            let badge_span =
                Span::styled(badge, Style::default().fg(color).add_modifier(Modifier::BOLD));
            for chunk in wrap_text(&compact, width.saturating_sub(4).max(4)) {
                lines.push(Line::from(vec![badge_span.clone(), Span::styled(
                    format!(" {}", chunk),
                    Style::default().fg(content_color),
                )]));
            }
        }
        lines
    }
}

impl<'a> Widget for DebugPanel<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" ⚡ Debug ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(if self.focused {
                Style::default().fg(Color::Rgb(220, 180, 60))
            } else {
                Style::default().fg(Color::Rgb(60, 60, 60))
            });

        let inner = block.inner(area);
        block.render(area, buf);

        if self.logs.is_empty() {
            let empty = Paragraph::new("No system logs.")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            empty.render(inner, buf);
            return;
        }

        let lines = self.build_lines(inner.width as usize);
        let total = lines.len();
        let view_h = inner.height.max(1) as usize;
        let offset = scroll_top(total, view_h, self.scroll_from_bottom);
        Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((offset, 0)).render(inner, buf);
    }
}

// ── Status bar ────────────────────────────────────────────────────────────────

pub struct StatusBar<'a> {
    model: &'a str,
    gpu: &'a str,
    cpu: &'a str,
    latency: std::time::Duration,
    status: &'a str,
    thinking: bool,
}

impl<'a> StatusBar<'a> {
    pub fn new(
        model: &'a str,
        gpu: &'a str,
        cpu: &'a str,
        latency: std::time::Duration,
        status: &'a str,
        thinking: bool,
    ) -> Self {
        Self { model, gpu, cpu, latency, status, thinking }
    }
}

impl<'a> Widget for StatusBar<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(60, 60, 60)));
        let inner = block.inner(area);
        block.render(area, buf);

        let latency_ms = self.latency.as_millis();
        let think = if self.thinking { "ON" } else { "OFF" };
        let text = vec![Line::from(vec![
            Span::styled(" ✦ Luna", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Model: ", Style::default().fg(Color::Rgb(140, 140, 140))),
            Span::styled(
                self.model,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("GPU: ", Style::default().fg(Color::Rgb(140, 140, 140))),
            Span::styled(
                self.gpu,
                Style::default().fg(Color::Magenta),
            ),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("CPU: ", Style::default().fg(Color::Rgb(140, 140, 140))),
            Span::styled(
                self.cpu,
                Style::default().fg(Color::Green),
            ),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Latency: ", Style::default().fg(Color::Rgb(140, 140, 140))),
            Span::styled(
                format!("{}ms", latency_ms),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Think: ", Style::default().fg(Color::Rgb(140, 140, 140))),
            Span::styled(
                think,
                Style::default().fg(if self.thinking { Color::Green } else { Color::Red }),
            ),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled(self.status, Style::default().fg(Color::Rgb(180, 180, 255))),
        ])];

        let paragraph = Paragraph::new(text).style(Style::default()).alignment(Alignment::Left);
        paragraph.render(inner, buf);
    }
}

// ── Shortcuts Bar ─────────────────────────────────────────────────────────────

pub struct ShortcutsBar;

impl Widget for ShortcutsBar {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let spans = vec![
            Span::styled(" [F2] ", Style::default().fg(Color::Rgb(80, 200, 140)).add_modifier(Modifier::BOLD)),
            Span::styled("Config  ", Style::default().fg(Color::Rgb(160, 160, 160))),
            Span::styled(" [Tab] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("Switch Focus  ", Style::default().fg(Color::Rgb(160, 160, 160))),
            Span::styled(" [↑↓] ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Scroll  ", Style::default().fg(Color::Rgb(160, 160, 160))),
            Span::styled(" [F1/?] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled("Shortcuts  ", Style::default().fg(Color::Rgb(160, 160, 160))),
            Span::styled(" [Esc] ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Reset  ", Style::default().fg(Color::Rgb(160, 160, 160))),
            Span::styled(" [Ctrl+C] ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled("Quit", Style::default().fg(Color::Rgb(160, 160, 160))),
        ];
        let paragraph = Paragraph::new(Line::from(spans)).alignment(Alignment::Center);
        paragraph.render(area, buf);
    }
}

// ── Shortcuts Help Modal ──────────────────────────────────────────────────────

pub struct ShortcutsModal;

impl Widget for ShortcutsModal {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" ⌨ Keyboard Shortcuts (Press Esc or ? to close) ")
            .title_alignment(Alignment::Center)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));

        let inner = block.inner(area);
        block.render(area, buf);

        let lines = vec![
            Line::from(Span::styled("Navigation & Panel Focus:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("  Tab            ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Toggle focus between Chat and Debug panel", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  ↑ / ↓          ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Scroll focused panel history up / down", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Esc            ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Reset scroll offset & focus Chat input", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
            Line::from(Span::styled("Configuration & Settings:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("  F2 / :config   ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Open interactive luna.toml Config Editor modal", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + S       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Save & reload luna.toml (inside Config Editor)", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + E       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Launch external $EDITOR (nano/nvim/micro) on luna.toml", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
            Line::from(Span::styled("Input Editing:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("  Ctrl + U       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Clear the entire input line", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + W       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Delete the word behind cursor", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + A / E   ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Move cursor to start (Home) or end (End) of line", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
            Line::from(Span::styled("General & System:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("  ? / F1         ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Toggle this Keyboard Shortcuts help dialog", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + B       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Toggle bottom keybindings shortcuts bar visibility", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + C       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Clear input line (or exit Luna if input is empty)", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl + D       ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Exit Luna session immediately", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
            Line::from(Span::styled("Debug & Diagnostics Commands:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("  /debug [all]   ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Run full system diagnostic & stream logs to Debug Panel", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  /test-voice    ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Test Piper TTS, RVC python environment & audio playback", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  /test-model    ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Test Ollama LLM endpoint connectivity & model status", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  /test-stt      ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Test Whisper STT model & audio recording devices", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  /test-memory   ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Test conversation memory & permanent vector storage", Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  /log <message> ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled("Write custom test message directly into Debug Panel", Style::default().fg(Color::White)),
            ]),
        ];

        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        paragraph.render(inner, buf);
    }
}

// ── Config Editor Modal ───────────────────────────────────────────────────────

pub struct ConfigEditorModal<'a> {
    pub state: &'a crate::tui::app::ConfigState,
}

impl<'a> Widget for ConfigEditorModal<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let title = format!(
            " ⚙ Config Editor: ~/.config/luna/luna.toml {} ",
            if self.state.modified { "[Modified]" } else { "[Saved]" }
        );
        let block = Block::default()
            .title(title)
            .title_alignment(Alignment::Center)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(80, 200, 140)).add_modifier(Modifier::BOLD));

        let inner = block.inner(area);
        block.render(area, buf);

        let view_height = inner.height.saturating_sub(2) as usize;
        let mut scroll = self.state.scroll;
        if self.state.cursor_row < scroll {
            scroll = self.state.cursor_row;
        } else if self.state.cursor_row >= scroll + view_height && view_height > 0 {
            scroll = self.state.cursor_row - view_height + 1;
        }

        let mut lines = Vec::new();
        for (idx, line_str) in self.state.lines.iter().enumerate().skip(scroll).take(view_height) {
            let is_cursor_line = idx == self.state.cursor_row;
            let line_num = format!("{:>3} │ ", idx + 1);
            let num_span = Span::styled(
                line_num,
                if is_cursor_line {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                },
            );
            let content_span = Span::styled(
                line_str.clone(),
                if is_cursor_line {
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Rgb(180, 180, 180))
                },
            );
            lines.push(Line::from(vec![num_span, content_span]));
        }

        // Status bar at bottom
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            &self.state.status,
            if self.state.status.starts_with('✓') {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            } else if self.state.status.starts_with('✗') {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            },
        )));

        let paragraph = Paragraph::new(lines);
        paragraph.render(inner, buf);
    }
}

impl<'a> ConfigEditorModal<'a> {
    pub fn cursor_area(area: Rect, state: &crate::tui::app::ConfigState) -> Option<(u16, u16)> {
        let inner_x = area.x + 1 + 6; // 6 cols for "001 │ "
        let view_height = area.height.saturating_sub(4) as usize;
        let mut scroll = state.scroll;
        if state.cursor_row < scroll {
            scroll = state.cursor_row;
        } else if state.cursor_row >= scroll + view_height && view_height > 0 {
            scroll = state.cursor_row - view_height + 1;
        }

        let rel_row = state.cursor_row.saturating_sub(scroll);
        let x = inner_x + state.cursor_col as u16;
        let y = area.y + 1 + rel_row as u16;

        if x < area.x + area.width - 1 && y < area.y + area.height - 2 {
            Some((x, y))
        } else {
            None
        }
    }
}

// ── Slash Command Autocomplete Popover ────────────────────────────────────────

pub struct SlashCommandMenu<'a> {
    pub options: &'a [(&'static str, &'static str)],
    pub selected: usize,
}

impl<'a> Widget for SlashCommandMenu<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" Slash Commands (Tab autocomplete · ↑↓ select) ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();
        for (idx, (cmd, desc)) in self.options.iter().enumerate() {
            let is_selected = idx == self.selected;
            let prefix = if is_selected { " ▶ " } else { "   " };
            let cmd_style = if is_selected {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            };
            let desc_style = if is_selected {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::Rgb(150, 150, 150))
            };
            lines.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(Color::Cyan)),
                Span::styled(format!("{:<12} ", cmd), cmd_style),
                Span::styled(desc.to_string(), desc_style),
            ]));
        }

        let paragraph = Paragraph::new(lines);
        paragraph.render(inner, buf);
    }
}

// ── Structured Settings Menu Modal ────────────────────────────────────────────

pub struct SettingsMenuModal<'a> {
    pub state: &'a crate::tui::app::SettingsMenuState,
}

impl<'a> Widget for SettingsMenuModal<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let title = format!(
            " ⚙ Settings & Controls {} ",
            if self.state.modified { "[Modified]" } else { "[Saved]" }
        );
        let block = Block::default()
            .title(title)
            .title_alignment(Alignment::Center)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(80, 200, 140)).add_modifier(Modifier::BOLD));

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // 1. Category headers / tabs
        let mut cat_spans = Vec::new();
        cat_spans.push(Span::raw(" "));
        for (idx, cat) in self.state.categories.iter().enumerate() {
            let is_active = idx == self.state.cat_index;
            if is_active {
                cat_spans.push(Span::styled(
                    format!(" [ {} ] ", cat.name),
                    Style::default().fg(Color::Yellow).bg(Color::Rgb(40, 40, 60)).add_modifier(Modifier::BOLD),
                ));
            } else {
                cat_spans.push(Span::styled(
                    format!("  {}  ", cat.name),
                    Style::default().fg(Color::Rgb(160, 160, 160)),
                ));
            }
        }
        lines.push(Line::from(cat_spans));
        lines.push(Line::from(""));

        // 2. Setting Items in active category
        if let Some(cat) = self.state.categories.get(self.state.cat_index) {
            for (idx, item) in cat.items.iter().enumerate() {
                let is_selected = idx == self.state.item_index;
                let cursor = if is_selected { " ▶ " } else { "   " };

                let val_span = match &item.setting_type {
                    crate::tui::app::SettingType::Toggle(b) => {
                        if *b {
                            Span::styled("[ ON ] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                        } else {
                            Span::styled("[OFF] ", Style::default().fg(Color::Red))
                        }
                    }
                    crate::tui::app::SettingType::Value(val) => {
                        Span::styled(format!("[ {} ] ", val), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                    }
                };

                let label_style = if is_selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };

                let desc_style = Style::default().fg(Color::Rgb(130, 130, 130));

                lines.push(Line::from(vec![
                    Span::styled(cursor, Style::default().fg(Color::Cyan)),
                    val_span,
                    Span::styled(format!("{:<26} ", item.label), label_style),
                    Span::styled(format!("— {}", item.description), desc_style),
                ]));
            }
        }

        // 3. Status bar & shortcuts footer at bottom
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            &self.state.status,
            if self.state.status.starts_with('✓') {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            } else if self.state.status.starts_with('✗') {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Rgb(180, 180, 255))
            },
        )));
        lines.push(Line::from(vec![
            Span::styled(" [Space/Enter] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("Toggle/Cycle  ", Style::default().fg(Color::Rgb(150, 150, 150))),
            Span::styled(" [Tab] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("Category  ", Style::default().fg(Color::Rgb(150, 150, 150))),
            Span::styled(" [S] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled("Save & Restart Daemons  ", Style::default().fg(Color::Rgb(150, 150, 150))),
            Span::styled(" [T] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::styled("Raw TOML  ", Style::default().fg(Color::Rgb(150, 150, 150))),
            Span::styled(" [Esc] ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("Close", Style::default().fg(Color::Rgb(150, 150, 150))),
        ]));

        let paragraph = Paragraph::new(lines);
        paragraph.render(inner, buf);
    }
}