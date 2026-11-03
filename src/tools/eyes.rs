//! tools/eyes.rs — Luna's eyes (pure Rust)
//!
//! Screenshots the desktop (grim) or the automation Chromium's current page
//! (CDP) and asks a small local vision model — served by the same Ollama the
//! planner uses — to describe what's there. No Python, no external server.
//!
//! The vision model is loaded on demand and swaps with the planner in VRAM,
//! so the caller (the agent) should treat a `see` call as a brief pause of
//! planning. Keep prompts short: qwen2.5vl:3b describes, it doesn't reason.

use crate::config::LunaConfig;
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde_json::{json, Value};
use std::time::Duration;

const SCREENSHOT_TMP: &str = "/tmp/luna-eyes-shot.png";

/// Describe what is currently on the whole desktop (grim screenshot).
pub async fn see_screen(config: &LunaConfig) -> Result<String> {
    let tool = if config.vision.screenshot_cmd.trim().is_empty() {
        "grim"
    } else {
        config.vision.screenshot_cmd.trim()
    };
    let out = crate::tools::shell::run_command(&format!("{tool} {SCREENSHOT_TMP}"), None).await?;
    if !std::path::Path::new(SCREENSHOT_TMP).exists() {
        bail!(
            "screenshot failed ({}): {}",
            tool,
            out.stderr.trim().chars().take(200).collect::<String>()
        );
    }
    let png = std::fs::read(SCREENSHOT_TMP).context("read screenshot PNG")?;
    describe_image(
        config,
        png,
        "You are Luna's eyes looking at the user's desktop. Describe what is \
         on screen: the visible text, windows and their titles, and anything \
         notable. Be concise and factual; if you cannot read something, say so.",
    )
    .await
}

/// Describe the automation Chromium's current page (CDP screenshot).
pub async fn see_browser(config: &LunaConfig) -> Result<String> {
    let png = crate::browser::current_page_png(config)
        .await
        .context("capture browser page")?;
    describe_image(
        config,
        png,
        "You are Luna's eyes looking at a web page in a Chromium window. \
         Describe the page: the main content, headings, visible buttons and \
         links, and anything notable. Be concise and factual; quote visible \
         text where it matters.",
    )
    .await
}

/// Send one PNG to the local Ollama vision model and return its description.
async fn describe_image(config: &LunaConfig, png: Vec<u8>, prompt: &str) -> Result<String> {
    if !config.vision.enabled {
        bail!("Vision is disabled in config ([vision] enabled = false).");
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    let base = config.llm.base_url.trim_end_matches('/').to_string();
    let url = format!("{base}/api/generate");

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&json!({
            "model": config.vision.model,
            "prompt": prompt,
            "images": [b64],
            "stream": false,
        }))
        .timeout(Duration::from_secs(240))
        .send()
        .await
        .context(
            "POST ollama /api/generate failed — is Ollama running and is the vision model \
             pulled? (ollama pull qwen2.5vl:3b)",
        )?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        bail!(
            "vision model call failed ({status}): {}",
            body["error"].as_str().unwrap_or("unknown error")
        );
    }
    body["response"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .ok_or_else(|| anyhow!("vision model returned an empty response"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Live end-to-end against the real Ollama + display. Run only when a
    // human is present and `ollama pull qwen2.5vl:3b` has been done once.
    #[tokio::test]
    #[ignore = "live vision; needs Ollama + a display"]
    async fn live_see_screen() {
        use crate::config::LunaConfig;
        let cfg = LunaConfig::load().expect("config");
        let desc = see_screen(&cfg).await.expect("see screen");
        assert!(!desc.is_empty());
        println!("{desc}");
    }
}