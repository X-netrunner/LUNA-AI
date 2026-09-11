//! tui/widgets.rs — Custom TUI widgets

use crate::tui::app::Msg;
use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

// ── Text wrapping helper ──────────────────────────────────────────────────────
/// Wrap `text` to `width` columns at word boundaries, preserving newlines.
/// Used so long assistant replies render on multiple lines instead of one.
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
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

        // Flatten every message into wrapped lines with role-based styling.
        // The role prefix is shown once per message; continuation lines are
        // indented to align under the content.
        let width = inner.width.saturating_sub(2) as usize;
        let mut all_lines: Vec<styled_line::StyledLine> = Vec::new();
        for msg in self.messages {
            let (prefix, indent, color) = match msg.role.as_str() {
                "user" => ("You:  ", "       ", Color::Green),
                "assistant" => ("Luna: ", "       ", Color::Cyan),
                "tool" => ("Tool: ", "       ", Color::Yellow),
                _ => ("", "", Color::White),
            };
            let mut first = true;
            for wrapped in wrap_text(&msg.content, width.saturating_sub(prefix.chars().count())) {
                if first {
                    all_lines.push(styled_line::StyledLine::new(prefix, &wrapped, color));
                    first = false;
                } else {
                    all_lines.push(styled_line::StyledLine::indent(indent, &wrapped));
                }
            }
            if let Some(thinking) = &msg.thinking {
                for wrapped in wrap_text(thinking, width.saturating_sub(4)) {
                    all_lines
                        .push(styled_line::StyledLine::dim("   [think] ", &wrapped));
                }
            }
            // blank spacer line between messages
            all_lines.push(styled_line::StyledLine::blank());
        }

        let total = all_lines.len();
        let view_h = inner.height.saturating_sub(1) as usize;
        // start index: follow bottom unless scrolled up
        let start = if self.scroll_from_bottom == 0 {
            total.saturating_sub(view_h)
        } else {
            total.saturating_sub(view_h).saturating_sub(self.scroll_from_bottom)
        };

        for (i, line) in all_lines.iter().enumerate().skip(start).take(view_h) {
            let y = inner.y + i as u16 - start as u16;
            line.render(buf, inner.x + 1, y, width);
        }
    }
}

mod styled_line {
    use ratatui::buffer::Buffer;
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};

    pub struct StyledLine {
        spans: Vec<Span<'static>>,
        blank: bool,
    }

    impl StyledLine {
        pub fn new(prefix: &str, text: &str, color: Color) -> Self {
            Self {
                spans: vec![
                    Span::styled(
                        prefix.to_string(),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(text.to_string(), Style::default().fg(Color::White)),
                ],
                blank: false,
            }
        }

        pub fn dim(prefix: &str, text: &str) -> Self {
            Self {
                spans: vec![Span::styled(
                    format!("{}{}", prefix, text),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                )],
                blank: false,
            }
        }

        /// Continuation line: no role prefix, indented to match content.
        pub fn indent(indent: &str, text: &str) -> Self {
            let mut spans = vec![Span::raw(indent.to_string())];
            spans.push(Span::styled(text.to_string(), Style::default().fg(Color::White)));
            Self { spans, blank: false }
        }

        pub fn blank() -> Self {
            Self { spans: vec![], blank: true }
        }

        pub fn render(&self, buf: &mut Buffer, x: u16, y: u16, width: usize) {
            if self.blank {
                buf.set_stringn(x, y, " ", 1, Style::default());
            } else {
                let line = Line::from(self.spans.clone());
                buf.set_line(x, y, &line, width as u16);
            }
        }
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

        let width = (inner.width.saturating_sub(2)) as usize;
        let mut wrapped: Vec<(String, Color)> = Vec::new();
        for line in self.logs {
            let color = if line.contains("ERROR") || line.contains("WARN") {
                Color::Red
            } else if line.contains("DEBUG") {
                Color::DarkGray
            } else if line.contains("INFO") {
                Color::Cyan
            } else {
                Color::DarkGray
            };
            for w in wrap_text(line, width) {
                wrapped.push((w, color));
            }
        }

        let view_h = inner.height.saturating_sub(1) as usize;
        let total = wrapped.len();
        let start = if self.scroll_from_bottom == 0 {
            total.saturating_sub(view_h)
        } else {
            total.saturating_sub(view_h).saturating_sub(self.scroll_from_bottom)
        };

        for (i, (line, color)) in wrapped.iter().enumerate().skip(start).take(view_h) {
            let y = inner.y + i as u16 - start as u16;
            buf.set_stringn(
                inner.x + 1,
                y,
                line,
                inner.width.saturating_sub(2) as usize,
                Style::default().fg(*color),
            );
        }
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