//! tui/mod.rs — Ratatui-based TUI for Luna

use crate::config::LunaConfig;
use anyhow::Result;

pub mod app;
pub mod widgets;

pub use app::TuiApp;

pub async fn run_tui(config: LunaConfig) -> Result<()> {
    let mut app = TuiApp::new(config)?;
    app.run().await
}