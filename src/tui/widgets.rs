//! tui/widgets.rs — Custom TUI widgets
//!
//! The chat and debug panels render with `Paragraph` (wrap + scroll). Ratatui's
//! Paragraph clears its text area on render, so stale cells never leak through
//! when content shrinks or scrolls.

use crate::tui::app::Msg;
use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

// ── Text wrapping helper ──────────────────────────────────────────────────────
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
                "user" => ("You:  ", "       ", Color::Green),
                "assistant" => ("Luna: ", "       ", Color::Cyan),
                "tool" => ("Tool: ", "       ", Color::Yellow),
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
                        format!("   [think] {}", chunk),
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
            .title(" Luna ")
            .borders(Borders::ALL)
            .border_style(if self.focused {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            });

        let inner = block.inner(area);
        block.render(area, buf);

        if self.messages.is_empty() {
            let empty = Paragraph::new("No messages yet. Type to start.")
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
            .title(" Input  (Enter send · Tab focus · ↑/↓ scroll · Ctrl-C exit)")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green));

        let inner = block.inner(area);
        block.render(area, buf);

        let paragraph = Paragraph::new(self.input)
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });
        paragraph.render(inner, buf);
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
            // Map the level prefix onto a compact badge + color so the panel
            // reads at a glance: [*] info, [!] warn/error, [+] debug, [~] trace.
            let (badge, color) = if log.starts_with("ERROR") {
                ("[!]", Color::Red)
            } else if log.starts_with("WARN") {
                ("[!]", Color::Yellow)
            } else if log.starts_with("INFO") {
                ("[*]", Color::Cyan)
            } else if log.starts_with("TRACE") {
                ("[~]", Color::DarkGray)
            } else {
                ("[+]", Color::DarkGray)
            };
            // Defensive: collapse long tool-dump lines (HTML/CSS from
            // fetch_page etc.) so one huge result can't flood the panel.
            let compact: String = log.chars().take(215).collect();
            let badge_span =
                Span::styled(badge, Style::default().fg(color).add_modifier(Modifier::BOLD));
            for chunk in wrap_text(&compact, width.saturating_sub(4).max(4)) {
                lines.push(Line::from(vec![badge_span.clone(), Span::styled(
                    format!(" {}", chunk),
                    Style::default().fg(color),
                )]));
            }
        }
        lines
    }
}

impl<'a> Widget for DebugPanel<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" Debug ")
            .borders(Borders::ALL)
            .border_style(if self.focused {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            });

        let inner = block.inner(area);
        block.render(area, buf);

        if self.logs.is_empty() {
            let empty = Paragraph::new("No logs yet.")
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
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(area);
        block.render(area, buf);

        let latency_ms = self.latency.as_millis();
        let think = if self.thinking { "ON " } else { "OFF" };
        let text = vec![Line::from(vec![
            Span::styled("Model: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                self.model,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  |  GPU: "),
            Span::styled(
                self.gpu,
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  |  CPU: "),
            Span::styled(
                self.cpu,
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  |  Latency: "),
            Span::styled(
                format!("{}ms", latency_ms),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  |  Think: "),
            Span::styled(
                think,
                Style::default().fg(if self.thinking { Color::Green } else { Color::Red }),
            ),
            Span::raw("  |  "),
            Span::styled(self.status, Style::default().fg(Color::Cyan)),
        ])];

        let paragraph = Paragraph::new(text).style(Style::default()).alignment(Alignment::Left);
        paragraph.render(inner, buf);
    }
}