//! tui/mod.rs — Ratatui-based TUI for Luna

use crate::config::LunaConfig;
use anyhow::Result;

pub mod app;
pub mod log;
pub mod widgets;

pub use app::TuiApp;
pub use log::LogBuffer;

pub async fn run_tui(config: LunaConfig, log: LogBuffer) -> Result<()> {
    let app = TuiApp::new(config, log)?;
    app.run().await
}