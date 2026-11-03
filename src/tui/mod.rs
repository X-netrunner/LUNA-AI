//! tui/mod.rs — Ratatui-based TUI for Luna

use crate::config::LunaConfig;
use anyhow::Result;

pub mod app;
pub mod log;
pub mod onboarding;
pub mod widgets;

pub use app::TuiApp;
pub use log::LogBuffer;

/// `force_setup` opens the first-run setup screen even on an already-configured
/// machine (the `luna --setup` flag).
pub async fn run_tui(config: LunaConfig, log: LogBuffer, force_setup: bool) -> Result<()> {
    let app = TuiApp::with_setup(config, log, force_setup)?;
    // The wake daemon hands over a one-breath command ("hey luna, what's the
    // weather") as a pending marker; replay it as the first voice utterance.
    if let Some(pending) = crate::wake::read_pending_voice() {
        if !pending.trim().is_empty() {
            app.enqueue_voice(pending);
        }
    }
    app.run().await
}