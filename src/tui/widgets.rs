//! tui/widgets.rs — Custom TUI widgets

use crate::tui::app::Msg;
use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

pub struct ChatHistory<'a> {
    messages: &'a [Msg],
    scroll_offset: usize,
}

impl<'a> ChatHistory<'a> {
    pub fn new(messages: &'a [Msg], scroll_offset: usize) -> Self {
        Self { messages, scroll_offset }
    }
}

impl<'a> Widget for ChatHistory<'a> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .title(" Luna ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        if self.messages.is_empty() {
            let empty = Paragraph::new("No messages yet. Type to start.")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center);
            empty.render(inner, buf);
            return;
        }

        let items: Vec<ListItem> = self.messages
            .iter()
            .skip(self.scroll_offset)
            .take(inner.height as usize)
            .map(|msg| {
                let (prefix, style) = match msg.role.as_str() {
                    "user" => ("You: ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    "assistant" => ("Luna: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    "tool" => ("Tool: ", Style::default().fg(Color::Yellow)),
                    _ => ("", Style::default()),
                };

                let mut lines = vec![Line::from(vec![
                    Span::styled(prefix, style),
                    Span::raw(&msg.content),
                ])];

                if let Some(thinking) = &msg.thinking {
                    lines.push(Line::from(vec![
                        Span::styled("  [think] ", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
                        Span::styled(thinking, Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
                    ]));
                }

                ListItem::new(lines)
            })
            .collect();

        let list = List::new(items)
            .style(Style::default())
            .highlight_style(Style::default().bg(Color::DarkGray));

        let mut state = ListState::default();
        state.select(Some(0));

        ratatui::widgets::StatefulWidget::render(list, inner, buf, &mut state);
    }
}

pub struct InputLine<'a> {
    input: &'a str,
    cursor_pos: usize,
}

impl<'a> InputLine<'a> {
    pub fn new(input: &'a str, cursor_pos: usize) -> Self {
        Self { input, cursor_pos }
    }

    pub fn cursor_area(area: Rect, cursor_pos: usize, input: &str) -> Option<(u16, u16)> {
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
            .title(" Input ")
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

pub struct StatusBar<'a> {
    model: &'a str,
    gpu: &'a str,
    latency: std::time::Duration,
    status: &'a str,
}

impl<'a> StatusBar<'a> {
    pub fn new(model: &'a str, gpu: &'a str, latency: std::time::Duration, status: &'a str) -> Self {
        Self { model, gpu, latency, status }
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
        let text = vec![
            Line::from(vec![
                Span::styled("Model: ", Style::default().fg(Color::DarkGray)),
                Span::styled(self.model, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw("  |  GPU: "),
                Span::styled(self.gpu, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                Span::raw("  |  Latency: "),
                Span::styled(format!("{}ms", latency_ms), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw("  |  "),
                Span::styled(self.status, Style::default().fg(Color::Cyan)),
            ]),
        ];

        let paragraph = Paragraph::new(text)
            .style(Style::default())
            .alignment(Alignment::Left);
        paragraph.render(inner, buf);
    }
}