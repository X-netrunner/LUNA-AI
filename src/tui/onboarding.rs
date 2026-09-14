//! tui/onboarding.rs — First-run interactive setup screen.
//!
//! Rendered fullscreen by the TUI when Luna is new on this machine. The user
//! walks through optional integrations: web-search keys, Todoist, Spotify
//! (keys + one-time device authorization, streamed live into the panel), then
//! finishes. Secrets are typed masked and stored via `keyring_set` + `keyring:`
//! references written into luna.toml.
//!
//! This file is intentionally UI-state-only: all config/keyring mutations and
//! the async Spotify flow are driven by app.rs (which owns the config).

use crate::config::LunaConfig;
use crate::first_run;
use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::collections::VecDeque;

/// The actions offered on the setup screen, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    SetTavily,
    SetGemini,
    SetTodoist,
    SetSpotifyId,
    SpotifyAuth,
    Finish,
}

/// A rendered action row: status glyph, label, hint, and whether it can run.
pub struct ActionRow {
    pub id: Action,
    pub label: String,
    pub done: bool,
    pub hint: String,
    pub enabled: bool,
}

pub struct Onboarding {
    pub cursor: usize,
    /// Index (into the action list) currently accepting a masked secret.
    pub editing: Option<usize>,
    pub secret: String,
    pub events: VecDeque<String>,
    pub done: bool,
}

impl Onboarding {
    pub fn new() -> Self {
        Self {
            cursor: 0,
            editing: None,
            secret: String::new(),
            events: VecDeque::new(),
            done: false,
        }
    }

    pub fn is_active(&self) -> bool {
        !self.done
    }

    pub fn log(&mut self, line: &str) {
        self.events.push_back(line.to_string());
        while self.events.len() > 20 {
            self.events.pop_front();
        }
    }

    pub fn cursor_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn cursor_down(&mut self, count: usize) {
        self.cursor = (self.cursor + 1).min(count.saturating_sub(1));
    }

    pub fn begin_secret(&mut self, idx: usize) {
        self.editing = Some(idx);
        self.secret.clear();
    }

    pub fn append_char(&mut self, c: char) {
        self.secret.push(c);
    }

    pub fn pop_char(&mut self) {
        self.secret.pop();
    }

    pub fn cancel_secret(&mut self) {
        self.editing = None;
        self.secret.clear();
    }

    pub fn take_secret(&mut self) -> String {
        self.editing = None;
        std::mem::take(&mut self.secret)
    }

    pub fn secret_len(&self) -> usize {
        self.secret.chars().count()
    }

    /// Live action list derived from current config state.
    pub fn actions(&self, config: &LunaConfig) -> Vec<ActionRow> {
        let s = |v: &Option<String>| !v.as_deref().map(str::trim).unwrap_or("").is_empty();
        let rows = vec![
            ActionRow {
                id: Action::SetTavily,
                label: "Web search — Tavily".into(),
                done: s(&config.search.tavily_api_key),
                hint: "free key at tavily.com".into(),
                enabled: true,
            },
            ActionRow {
                id: Action::SetGemini,
                label: "Web search — Google Gemini".into(),
                done: s(&config.search.gemini_api_key),
                hint: "free key at aistudio.google.com".into(),
                enabled: true,
            },
            ActionRow {
                id: Action::SetTodoist,
                label: "Todoist — tasks".into(),
                done: s(&config.todoist.api_token),
                hint: "token at todoist.com/app/settings".into(),
                enabled: true,
            },
            ActionRow {
                id: Action::SetSpotifyId,
                label: "Spotify — client id".into(),
                done: s(&config.spotify.client_id),
                hint: "app at developer.spotify.com/dashboard".into(),
                enabled: true,
            },
            ActionRow {
                id: Action::SpotifyAuth,
                label: "Spotify — authorize luna (one-time)".into(),
                done: s(&config.spotify.refresh_token),
                hint: "opens a URL to approve in the browser".into(),
                enabled: first_run::spotify_auth_ready(config),
            },
            ActionRow {
                id: Action::Finish,
                label: "Done — start chatting".into(),
                done: false,
                hint: "you can also say 'learn about my system' anytime".into(),
                enabled: true,
            },
        ];
        rows
    }
}

/// Keyring entry name an action writes to (None for non-secret actions).
pub fn action_keyring_name(id: Action) -> Option<&'static str> {
    match id {
        Action::SetTavily => Some("tavily"),
        Action::SetGemini => Some("gemini"),
        Action::SetTodoist => Some("todoist"),
        Action::SetSpotifyId => Some("spotify_id"),
        Action::SpotifyAuth | Action::Finish => None,
    }
}

#[allow(clippy::vec_init_then_push)] // fine: lines are built conditionally per row
pub fn render(onb: &Onboarding, f: &mut Frame, config: &LunaConfig, area: Rect) {
    let rows = onb.actions(config);
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(Span::styled(
        "Welcome to Luna — first-time setup",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from("Everything below is optional. Keys are stored in the OS keyring;"));
    lines.push(Line::from("nothing ever needs to sit in plaintext in luna.toml."));
    lines.push(Line::from(""));
    lines.push(Line::from(
        "  Up/Down move · Enter activates · secrets are typed hidden · Esc finishes",
    ));
    lines.push(Line::from(""));

    for (i, row) in rows.iter().enumerate() {
        let selected = i == onb.cursor;
        let mark = if row.done { "✓" } else { "·" };
        let mut spans: Vec<Span> = Vec::new();

        let base = if selected {
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let mark_style = if row.done {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Yellow)
        };

        spans.push(Span::styled(format!("  [{}] ", mark), mark_style.patch(base)));
        if let Some(edit_idx) = onb.editing {
            if edit_idx == i {
                let dots = "•".repeat(onb.secret_len().max(1));
                // Selected row shows the live masked length.
                let part = if onb.secret_len() > 0 {
                    format!("{} ", dots)
                } else {
                    String::new()
                };
                spans.push(Span::styled(
                    format!("{} ({}) ", part, onb.secret_len()),
                    base,
                ));
                spans.push(Span::styled(
                    row.label.clone(),
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled("  — typing secret…  [Enter] save  [Esc] cancel", base));
                lines.push(Line::from(spans));
                continue;
            }
        }

        spans.push(Span::styled(row.label.clone(), base));
        if row.done {
            spans.push(Span::styled("  ✓", Style::default().fg(Color::Green)));
        } else if !row.enabled {
            spans.push(Span::styled(
                format!("  (set the Spotify client id above first)  {}", row.hint),
                Style::default().fg(Color::DarkGray),
            ));
        } else {
            spans.push(Span::styled(
                format!("  — {}", row.hint),
                Style::default().fg(Color::DarkGray),
            ));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));

    if !onb.events.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Setup progress",
            Style::default().fg(Color::Cyan),
        )));
        for ev in onb.events.iter() {
            let color = if ev.starts_with("✓") {
                Color::Green
            } else if ev.starts_with("✗") {
                Color::Red
            } else {
                Color::DarkGray
            };
            lines.push(Line::from(vec![Span::styled(
                format!("  {}", ev),
                Style::default().fg(color),
            )]));
        }
    }

    let mut hint = "Everything is optional — you're set up and ready to chat on the spot.".to_string();
    if onb.editing.is_some() {
        hint = "Type the secret (hidden). Press Enter to save it to the keyring.".into();
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  {}", hint),
        Style::default().fg(Color::DarkGray),
    )));

    let block = Block::default().borders(Borders::ALL).title(" First-time setup ");
    Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .render(area, f.buffer_mut());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LunaConfig;

    #[test]
    fn secret_edit_state_machine() {
        let mut o = Onboarding::new();
        assert!(o.is_active());
        o.begin_secret(2);
        o.append_char('a');
        o.append_char('b');
        o.append_char('3');
        assert_eq!(o.secret_len(), 3);
        o.pop_char();
        let s = o.take_secret();
        assert_eq!(s, "ab");
        assert!(o.editing.is_none());
        assert_eq!(o.secret_len(), 0);
    }

    #[test]
    fn spotify_auth_gated_until_client_id() {
        let o = Onboarding::new();
        let rows = o.actions(&LunaConfig::default());
        let auth = rows.iter().find(|r| r.id == Action::SpotifyAuth).unwrap();
        assert!(!auth.enabled, "needs client id");

        let mut cfg = LunaConfig::default();
        cfg.spotify.client_id = Some("id".into());
        let rows = o.actions(&cfg);
        let auth = rows.iter().find(|r| r.id == Action::SpotifyAuth).unwrap();
        assert!(auth.enabled);
    }

    #[test]
    fn cursor_clamps_to_action_list() {
        let mut o = Onboarding::new();
        o.cursor_down(10_000);
        assert!(o.cursor < 10_000);
        o.cursor_down(0);
        // keeps last valid index usable
        o.cursor_up();
        o.cursor_up();
        o.cursor_up();
        assert_eq!(o.cursor, 0);
    }

    #[test]
    fn keyring_names_map() {
        assert_eq!(action_keyring_name(Action::SetTavily), Some("tavily"));
        assert_eq!(action_keyring_name(Action::SetSpotifyId), Some("spotify_id"));
        assert_eq!(action_keyring_name(Action::SpotifyAuth), None);
        assert_eq!(action_keyring_name(Action::Finish), None);
    }
}