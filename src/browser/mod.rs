//! browser/mod.rs — Pure-Rust browser automation using chromiumoxide + Ollama planner.
//!
//! Architecture:
//!   - Types (types.rs): shared ActionResult
//!   - Planner (planner.rs): calls Ollama to break user goal into steps
//!   - Executor (executor.rs): runs steps in Chromium via CDP, handles retries/replanning
//!   - CDP client (cdp.rs): chromiumoxide-based native async DevTools Protocol driver
//!   - DOM (dom.rs): JavaScript expressions for click/type/press/scroll
//!
//! No external Python server, no WebSocket bridge — everything runs in-process.

mod cdp;
mod dom;
mod executor;
mod planner;
mod types;

use anyhow::{Context, Result};
use crate::config::LunaConfig;

use crate::browser::executor::execute_plan;
use crate::browser::planner::Planner;

/// Run one browser-automation task to completion using the native Rust engine.
pub async fn run(task: &str, config: &LunaConfig) -> Result<String> {
    let bc = &config.browser;

    // Initialize the planner (calls Ollama for step generation)
    let planner = Planner::new(
        &config.llm.base_url,
        &config.llm.model,
    );

    // Connect to or launch Chromium
    let browser = cdp::ensure_browser(bc).await.context("ensure chromium")?;

    // Generate initial plan
    let steps = planner.plan(task).await.context("generate plan")?;
    tracing::info!("Generated plan with {} steps", steps.len());

    // Execute with retry/rethink
    let history = execute_plan(&browser, &planner, task, steps)
        .await
        .context("execute plan")?;

    if history.is_empty() {
        return Ok("The browser automation finished without executing any steps.".into());
    }

    let history_str = history.join("\n");
    Ok(format!(
        "Done. The browser automation ({} steps) executed:\n{}\n\nThe browser window is \
         still open so you can see the final state; tell me if you'd like it closed or if the \
         result isn't what you wanted.",
        history.len(),
        history_str
    ))
}

// Re-export for tests and internal use
pub use planner::Step;
pub use types::ActionResult;

#[cfg(test)]
pub use cdp::*;
#[cfg(test)]
pub use dom::*;
#[cfg(test)]
pub use executor::*;
#[cfg(test)]
pub use planner::*;
#[cfg(test)]
pub use types::*;