//! first_run.rs — first-time setup detection and the guided setup guide.
//!
//! Luna treats a machine as brand-new when it was never marked as set up
//! (marker `<data>/luna/.onboarded` absent) *and* no external integration is
//! configured yet. On such machines the TUI opens an interactive onboarding
//! screen and the text/voice modes print the same guidance so the user can
//! wire up API keys, system learning, and voice before starting to chat.

use crate::config::LunaConfig;
use anyhow::Result;
use std::path::PathBuf;

/// Marker file written once the user finishes (or dismisses) setup.
fn marker_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("luna")
        .join(".onboarded")
}

/// Does this device still need the guided first-run setup?
pub fn needs_onboarding(config: &LunaConfig) -> bool {
    if marker_path().exists() {
        return false;
    }
    !any_integration_configured(config)
}

/// Has the user configured at least one integration already? If yes, they
/// clearly know their way around — skip the onboarding nag entirely.
pub fn any_integration_configured(config: &LunaConfig) -> bool {
    let s = |v: &Option<String>| !v.as_deref().map(str::trim).unwrap_or("").is_empty();
    s(&config.search.tavily_api_key)
        | s(&config.search.gemini_api_key)
        | s(&config.todoist.api_token)
        | s(&config.spotify.client_id)
        | s(&config.spotify.refresh_token)
}

/// Write the "onboarded" marker so the setup screen never returns.
pub fn mark_done() -> Result<()> {
    let path = marker_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, b"onboarded\n")?;
    Ok(())
}

/// A single setup item: machine name, whether it's already configured, and a
/// one-line "what to do" hint shown to the user.
pub struct Integration<'a> {
    pub name: &'a str,
    pub configured: bool,
    pub hint: &'a str,
    pub keyring_name: Option<&'a str>,
}

/// Status of every optional integration, used by both the TUI screen and the
/// text-mode guide.
pub fn integrations(config: &LunaConfig) -> Vec<Integration<'static>> {
    let s = |v: &Option<String>| !v.as_deref().map(str::trim).unwrap_or("").is_empty();
    vec![
        Integration {
            name: "Web search — Tavily (boosts search)",
            configured: s(&config.search.tavily_api_key),
            hint: "free key at tavily.com",
            keyring_name: Some("tavily"),
        },
        Integration {
            name: "Web search — Google Gemini (boosts search)",
            configured: s(&config.search.gemini_api_key),
            hint: "free key at aistudio.google.com",
            keyring_name: Some("gemini"),
        },
        Integration {
            name: "Todoist (tasks)",
            configured: s(&config.todoist.api_token),
            hint: "token at todoist.com",
            keyring_name: Some("todoist"),
        },
        Integration {
            name: "Spotify (music) — client id",
            configured: s(&config.spotify.client_id),
            hint: "app at developer.spotify.com/dashboard",
            keyring_name: Some("spotify_id"),
        },
    ]
}

/// "spotify auth" entry qualifies once the client id is set. The PKCE flow
/// needs no client secret, only the id.
pub fn spotify_auth_ready(config: &LunaConfig) -> bool {
    let s = |v: &Option<String>| !v.as_deref().map(str::trim).unwrap_or("").is_empty();
    s(&config.spotify.client_id)
}

/// Text-mode one-shot guide (printed at startup on fresh devices, and by the
/// `--setup` flag).
pub fn guide_text(config: &LunaConfig) -> String {
    let mut out = String::new();
    out.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out.push_str("  First-time setup — wire up Luna's integrations\n");
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    out.push_str("  Everything below is optional. Keys are stored in the\n");
    out.push_str("  OS keyring; nothing ever needs to sit in luna.toml.\n\n");

    for i in integrations(config) {
        let mark = if i.configured { "✓" } else { "·" };
        let cmd = match i.keyring_name {
            Some(name) => format!("      luna --set-key {}", name),
            None => String::new(),
        };
        out.push_str(&format!("  [{}] {}  — {}\n", mark, i.name, i.hint));
        if !i.configured && !cmd.is_empty() {
            out.push_str(&format!("{}\n", cmd));
        }
    }

    out.push_str("\n  How to add a key:\n");
    out.push_str("    luna --set-key <name>       prompts without echo\n");
    out.push_str("    luna --spotify-auth          one-time Spotify approval\n\n");

    out.push_str("  Other things worth doing on a fresh box:\n");
    out.push_str("    - \"learn about my system\"   maps your home dir to memory\n");
    out.push_str("    - \"what can you do\"          lists every tool I have\n");
    out.push_str("    - luna --tui                  interactive setup screen\n\n");
    if !spotify_auth_ready(config) {
        out.push_str("  Tip: store your Spotify client id, then run `luna --spotify-auth`\n");
        out.push_str("  to play your liked songs, playlists and more.\n\n");
    }
    out.push_str("  Say \"ok forget it\" or press Ctrl-C to just start chatting.\n");
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LunaConfig;

    #[test]
    fn fresh_config_needs_onboarding() {
        let cfg = LunaConfig::default();
        // Marker must be absent for this test to be meaningful; guard the test
        // so a dev machine that already onboarded doesn't flip the result.
        if marker_path().exists() {
            return;
        }
        assert!(needs_onboarding(&cfg));
    }

    #[test]
    fn any_key_disables_onboarding() {
        let mut cfg = LunaConfig::default();
        cfg.todoist.api_token = Some("keyring:todoist".into());
        assert!(any_integration_configured(&cfg));
    }
}